from models.effnet import EffNet
from models.custom import Custom
from models.resnet import ResNet
from models.effnet32 import EffNet32
from models.mobilenet import MobileNet

def get_model(model_name:str):
    if model_name == 'effnet':
        return EffNet()
    elif model_name == 'custom':
        return Custom()
    elif model_name == 'resnet':
        return ResNet()
    elif model_name == 'effnet32':
        return EffNet32()
    elif model_name == 'mobilenet':
        return MobileNet()
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

def get_model_predict(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet()
    
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass