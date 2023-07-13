from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from glob import glob
from sklearn.model_selection import train_test_split
import cv2

def SplitDataset(img_dir:str, val_size:float=0.1, seed=42, data_size:float=1.0):
    fake_images = glob(f'{img_dir}/fake_images/*.png')
    real_images = glob(f'{img_dir}/real_images/*.png')

    # 절반 크기로 fake_images와 real_images 리스트 조정
    len_images = int(len(fake_images) * data_size)

    fake_images = fake_images[:len_images]
    real_images = real_images[:len_images]

    labels = [1] * len(fake_images) + [0] * len(real_images)

    X_train, X_val, y_train, y_val = train_test_split(fake_images + real_images, labels, test_size=val_size, random_state=seed, shuffle=True, stratify=labels)
    print(f"Fake : {y_train.count(1)} / Real : {y_train.count(0)}")
    return X_train, X_val, y_train, y_val

def SplitDataset32(img_dir:str, val_size:float=0.1, seed=42, data_size:float=1.0):
    fake_images = glob(f'{img_dir}/fake*/*.png')
    real_images = glob(f'{img_dir}/real*/*.png')


    # 절반 크기로 fake_images와 real_images 리스트 조정
    len_images = int(len(fake_images) * data_size)

    fake_images = fake_images[:len_images]
    real_images = real_images[:len_images]

    labels = [1] * len(fake_images) + [0] * len(real_images)

    X_train, X_val, y_train, y_val = train_test_split(fake_images + real_images, labels, test_size=val_size, random_state=seed, shuffle=True)

    return X_train, X_val, y_train, y_val

class CustomDataset(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            # transforms.RandomRotation(degrees=(-30,30)), 
            # transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5, -5, 5)),
            # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname

class CustomDataset2(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname
    
class TestDataset(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname
        
    def center_crop(self, img):
        width, height = img.size
        target_width, target_height = (224,224)

        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return img.crop((left, top, right, bottom))


class ValidDataset(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

    def get_data_list(self):
      return self.X


## Crop + Resize 
class CustomDataset_crop_resize(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: self.center_crop(x)),  # center crop 적용
            transforms.Resize((224, 224)),  # resize 적용
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname

    def center_crop(self, img):
        width, height = img.size
        target_width, target_height = (224,224)

        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return img.crop((left, top, right, bottom))


## Padding + Resize 
class CustomDataset_padding_resize(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname

class TestDataset_padding_resize(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

class ValidDataset_padding_resize(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

    def get_data_list(self):
      return self.X

## Padding + 32by32
class CustomDataset_padding_resize_32by32(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname

class TestDataset_padding_resize_32by32(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

class ValidDataset_padding_resize_32by32(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

    def get_data_list(self):
      return self.X

if __name__ == '__main__':
    pass

        