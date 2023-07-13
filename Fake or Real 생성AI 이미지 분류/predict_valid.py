"""Predict
"""
from modules.utils import load_yaml
from models.utils import get_model
from modules.datasets import SplitDataset,SplitDataset32, ValidDataset, ValidDataset_padding_resize

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import random, os, torch
from glob import glob

import click
from easydict import EasyDict

@click.command()
# Required.
@click.option('--datadir',help='Data path',metavar='DIR',type=str,required=True)
@click.option('--train_serial',help='Select to trained model dir',metavar='DIR',type=str,required=True)

@click.option("--batch_size", default=32, help="Batch size")
@click.option("--gpu", default=0, help="GPU device index")
@click.option("--seed", default=42, help="Random seed")
@click.option("--val_size", default=0.3, help="Validation size")

def main(**kwargs):
    opts = EasyDict(kwargs)
    print(opts)

    # # Config
    PROJECT_DIR = os.path.dirname(__file__)
    #predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yaml'))

    # Serial
    train_serial = opts['train_serial']
    kst = timezone(timedelta(hours=9))
    predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    predict_serial = train_serial + '_' + predict_timestamp

    # Predict directory
    PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict_valid', predict_serial)
    os.makedirs(PREDICT_DIR, exist_ok=True)

    # Recorder Directory
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

    # Data Directory
    DATA_DIR = opts['datadir']

    # Dataset
    _, X_val, _, Y_val = SplitDataset(img_dir = DATA_DIR,
                 val_size = opts['val_size'],
                 seed = opts['seed'],)
    print(X_val)
    print(Y_val)
    valid_dataset = ValidDataset(X = X_val)
    valid_df = pd.DataFrame({
      'file_path' : X_val,
      'label' : Y_val,
    })
    print(valid_df)
    print('-'*50)
    print(valid_df['label'].value_counts())
    print('-'*50)

    # Train config
    train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

    # SEED
    torch.manual_seed(opts['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts['seed'])
    random.seed(opts['seed'])
    os.environ['PYTHONHASHSEED'] = str(opts['seed'])
    torch.cuda.manual_seed_all(opts['seed'])  # if use multi-GPU

    # Gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['gpu'])

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=train_config['state']['batch_size'],
                                num_workers=train_config['state']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['state']['pin_memory'],
                                drop_last=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_name = train_config['state']['model']
    model = get_model(model_name=model_name).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    #checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # Make predictions
    y_preds = []
    filenames = []
    # 테스트 데이터로더를 순회하면서 예측 수행
    for batch_index, (x, filename) in enumerate(tqdm(valid_dataloader)):
        # 입력 데이터를 디바이스(GPU 또는 CPU)로 이동하고 데이터 타입을 float으로 설정:
        x = x.to(device, dtype=torch.float)
        # 모델을 통해 입력 데이터에 대한 로짓(logits)을 얻음
        y_logits = model(x).squeeze(-1)
        
        # 로짓을 이진 분류(0 또는 1)로 변환하여 예측값(y_pred)을 얻음
        y_pred = (y_logits > 0.5).to(torch.int).cpu()
        
        # 예측값을 리스트에 저장
        y_preds.append(y_pred)
        
        # 파일명을 리스트에 저장
        filenames.extend(filename)
    
    # 리스트에 저장된 예측값을 텐서로 변환하고 리스트로 변환
    y_preds = torch.cat(y_preds, dim=0).tolist()

    # 예측 결과를 샘플 제출 형식에 맞게 저장
    valid_df['predict'] = y_preds
    valid_df.to_csv(os.path.join(PREDICT_DIR, 'prdict_train.csv'), index=False)
    # pred_df = pd.DataFrame({'ImageId':filenames, 'predict': y_preds})
    # sample_submission = pd.read_csv(opts['sample_submission_path'])
    # result = sample_submission.merge(pred_df, on='ImageId', how='left')
    # result.drop('answer_x', axis=1, inplace=True)
    # result.rename(columns={'answer_y':'answer'}, inplace=True)
    # result.to_csv(os.path.join(PREDICT_DIR, 'result.csv'), index=False)

    print('Done')

if __name__ == '__main__':
  main()