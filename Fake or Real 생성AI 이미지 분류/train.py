from modules.utils import load_yaml, save_yaml, get_logger
from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.datasets import SplitDataset,SplitDataset32, CustomDataset, CustomDataset2
from modules.datasets import  CustomDataset_crop_resize, CustomDataset_padding_resize,CustomDataset_padding_resize_32by32
from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss
from modules.trainer import Trainer

from sklearn.model_selection import StratifiedKFold

from models.utils import get_model
from torch.utils.data import Subset

from torch.utils.data import DataLoader
import torch, wandb
import wandb

from datetime import datetime, timezone, timedelta
import numpy as np
import os, shutil, copy, random


import warnings
warnings.filterwarnings('ignore')

import click
from easydict import EasyDict

@click.command()
# Required.
@click.option('--datadir',      help='Data path',metavar='DIR',type=str,required=True)
@click.option('--model',   help='Model name to train',metavar='STR',type=str,required=True)

# Optional features.
@click.option("--optimizer", default="adam", help="Optimizer name")
@click.option("--lr", default=5.0e-4, help="Learning rate")
@click.option("--loss", default="bce", help="Loss function")
@click.option("--metric", default=["accuracy", "f1macro"], multiple=True, help="Metrics")
@click.option("--n_epochs", default=20, help="Number of epochs")
@click.option("--early_stopping_target", default="val_accuracy", help="Early stopping target")
@click.option("--early_stopping_patience", default=10, help="Early stopping patience")
@click.option("--early_stopping_mode", default="max", help="Early stopping mode")
@click.option("--amp", default=False, help="Enable Automatic Mixed Precision (AMP)")
@click.option("--gpu", default=0, help="GPU device index")
@click.option("--seed", default=42, help="Random seed")
@click.option("--val_size", default=0.3, help="Validation size")
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--num_workers", default=1, help="Number of workers for data loading")
@click.option("--shuffle", default=True, help="Shuffle the data")
@click.option("--pin_memory", default=True, help="Pin memory for faster data transfer")
@click.option("--drop_last", default=True, help="Drop the last incomplete batch")
@click.option("--debug", default=False, help="Enable debug mode")
@click.option("--wandb", default=False, help="Enable Weights & Biases logging")
@click.option("--logging_interval", default=100, help="Logging interval")
@click.option("--plot", default=["loss", "accuracy", "f1macro", "elapsed_time"], multiple=True, help="Plots to display")
@click.option("--elapsed_time", default=True, help="Display elapsed time")
@click.option("--data_size", default=1.0, help="Data 개수")
@click.option("--additional_learning", default=False, help="추가학습 여부")


def main(**kwargs):
    opts = EasyDict(kwargs)
    print(opts)

    # Root Directory
    PROJECT_DIR = os.path.dirname(__file__)
    print(PROJECT_DIR)

    # # Load config
    # config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yaml')
    # config = load_yaml(config_path)

    # 정상 실행이 안된 디렉토리 제거
    target_directory = os.path.join(PROJECT_DIR,'results', 'train')
    for dirpath, dirnames, filenames in os.walk(target_directory):
        for dirname in dirnames:
            target_directory2 = os.path.join(target_directory,dirname)
            for dirpath, dirnames, filenames in os.walk(target_directory2):
              file_count = len(filenames)

              # 현재 디렉토리가 비어있는 경우 삭제
              if file_count < 4:
                  shutil.rmtree(dirpath)
                  print(f"Directory '{dirpath}' is empty and has been deleted.")

    # Train Serial
    kst = timezone(timedelta(hours=9))
    train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

    # Recorder Directory
    if opts['debug']:
        RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', 'debug')
        # remove the record directory if it exists even though directory not empty
        if os.path.exists(RECORDER_DIR): 
          shutil.rmtree(RECORDER_DIR)
    else:
        RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

    os.makedirs(RECORDER_DIR, exist_ok=True)

    # Wandb Setting
    if opts['wandb']:
        run = wandb.init(project='fake_or_real',
                        name=train_serial,
                        config=opts,)

    # Data Directory
    DATA_DIR = opts['datadir']

    # Seed
    torch.manual_seed(opts['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts['seed'])
    random.seed(opts['seed'])
    os.environ['PYTHONHASHSEED'] = str(opts['seed'])
    torch.cuda.manual_seed_all(opts['seed'])  # if use multi-GPU

    # GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Set Logger
    '''
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")

    
    '''
    Load Data
    '''
    # Dataset
    X_train, X_val, Y_train, Y_val = SplitDataset(img_dir = DATA_DIR,
                 val_size = opts['val_size'],
                 seed = opts['seed'],
                 data_size = opts['data_size'])
        
    train_dataset = CustomDataset(X = X_train, y = Y_train)
    val_dataset = CustomDataset2(X = X_val, y = Y_val)

    # # DataLoader
    # train_dataloader = DataLoader(dataset = train_dataset,
    #                               batch_size = opts['batch_size'],
    #                               num_workers = opts['num_workers'],
    #                               shuffle = opts['shuffle'],
    #                               pin_memory = opts['pin_memory'],
    #                               drop_last = opts['drop_last'])
    
    # val_dataloader = DataLoader(dataset = val_dataset,
    #                             batch_size = opts['batch_size'],
    #                             num_workers = opts['num_workers'], 
    #                             shuffle = False,
    #                             pin_memory = opts['pin_memory'],
    #                             drop_last = opts['drop_last'])

    logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")


    '''
    Set model
    '''
    # Load model
    model_name = opts['model']
    model = get_model(model_name = model_name).to(device)
    if opts['additional_learning'] :
      # 저장된 모델 파라미터 로드
      model_path = "/content/drive/MyDrive/github/Fake or Real 판별/Fake_or_Real/results/train/eff_1_F_N2-6_data100%_K-Fold-5_20ep/model.pt"
      model.load_state_dict(torch.load(model_path), strict=False)
    '''
    Set trainer
    '''
    # Optimizer
    optimizer = get_optimizer(optimizer_name=opts['optimizer'])
    optimizer = optimizer(params=model.parameters(),lr=opts['lr'])

    # Loss
    loss = get_loss(loss_name=opts['loss'])
    
    # Metric
    metrics = {metric_name: get_metric(metric_name) for metric_name in opts['metric']}
    
    # Early stoppper
    early_stopper = EarlyStopper(patience=opts['early_stopping_patience'],
                                mode=opts['early_stopping_mode'],
                                logger=logger)

    # AMP
    if opts['amp'] == True:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    
    # Trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      device=device,
                      logger=logger,
                      amp=amp if opts['amp'] else None,
                      interval=opts['logging_interval'])
    
    '''
    Logger
    '''
    # Recorder
    recorder = Recorder(record_dir=RECORDER_DIR,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        amp=amp if opts['amp'] else None,
                        logger=logger)

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), opts)

    '''
    TRAIN
    '''
    # Train
    n_epochs = opts['n_epochs']

    # K-Fold
    # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=opts['seed'])
    # for fold, (train_index, val_index) in enumerate(skf.split(train_dataset.X, train_dataset.y)):
      
    #   train_dataset_fold = Subset(train_dataset, train_index)
    #   val_dataset_fold = Subset(train_dataset, val_index)

    train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size = opts['batch_size'],
                                num_workers = opts['num_workers'],
                                shuffle = opts['shuffle'],
                                pin_memory = opts['pin_memory'],
                                drop_last = opts['drop_last'])
  
    val_dataloader = DataLoader(dataset = val_dataset,
                                batch_size = opts['batch_size'],
                                num_workers = opts['num_workers'], 
                                shuffle = False,
                                pin_memory = opts['pin_memory'],
                                drop_last = opts['drop_last'])


    for epoch_index in range(n_epochs):

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial
        """
        Train
        """
        print(f"Train {epoch_index}/{n_epochs}")
        logger.info(f"--Train {epoch_index}/{n_epochs}")
        trainer.train(dataloader=train_dataloader, epoch_index=epoch_index, mode='train')
        
        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()
        
        """
        Validation
        """
        print(f"Val {epoch_index}/{n_epochs}")
        logger.info(f"--Val {epoch_index}/{n_epochs}")  
        trainer.train(dataloader=val_dataloader, epoch_index=epoch_index, mode='val')
        
        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
        trainer.clear_history()
        
        """
        Record
        """
        # Log results on the local
        recorder.add_row(row_dict)
        recorder.save_plot(opts['plot'])
        
        # Log results on the online (wandb)
        if opts["wandb"]:
            wandb.log(row_dict)
        
        """
        Early stopper
        """
        early_stopping_target = opts['early_stopping_target']
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if (early_stopper.patience_counter == 0) or (epoch_index == n_epochs-1):
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)
        
        if early_stopper.stop == True:
            logger.info(f"Eearly stopped, counter {early_stopper.patience_counter}/{opts['early_stopping_patience']}")
            break


if __name__ == '__main__':
    main()
    
            
