from torch.utils.data import Dataset
from torchvision import transforms
# from PIL import Image

class TINY200Pair(Dataset):
    def __init__(self, data_in, train = True, targets = None, transform = None):
        self.data = data_in
        self.transform = transform
        self.train = train
        if not self.train:
            self.target = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, targets = self.data[idx]
#         img = Image.fromarray(img)
        
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        
        if self.train:
            return pos_1,pos_2,targets
        else:
            return pos_1,pos_2,self.target[idx]
        
        
        
class TINY200(Dataset):
    def __init__(self, data_in, train = True, targets = None, transform = None):
        self.data = data_in
        self.transform = transform
        self.train = train
        if not self.train:
            self.target = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, targets = self.data[idx]
#         img = Image.fromarray(img)
        
        pos_1 = self.transform(img)
        
        if self.train:
            return pos_1,targets
        else:
            return pos_1,self.target[idx]


        
        
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])