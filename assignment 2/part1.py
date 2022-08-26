import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from matplotlib import pyplot as plt
from PIL import Image
from skimage import img_as_ubyte
from torchsummary import summary
# import sklearn
from sklearn import metrics
import matplotlib


base = "./dataset"

######### for visualizing DATASET ##########

# for directory in sorted(os.listdir(base+"/train/")):
#     # print(directory)
#     img = np.load(base+"/train/"+directory+"/img.npy")
#     # info = np.iinfo(img.dtype)
#     # img = img.astype("uint8")
#     # img = img_as_ubyte(img)
#     label = np.load(base+"/train/"+directory+"/label.npy")
#     seg = np.load(base+"/train/"+directory+"/seg.npy")
#     seg_clip = np.clip(seg,0,1)
#     img = np.concatenate((img, np.expand_dims(seg_clip,0)),axis=0)
#     # img = img/img.max()
#     plt.subplot(321),plt.imshow(img[0]),plt.title('1')
#     plt.subplot(322),plt.imshow(img[1]),plt.title('2')
#     plt.subplot(323),plt.imshow(img[2]),plt.title('3')
#     plt.subplot(324),plt.imshow(img[3]),plt.title('4')
#     plt.subplot(325),plt.imshow(seg),plt.title('5')
#     plt.subplot(326),plt.imshow(seg_clip),plt.title('seg_clip')
#     plt.show()

data_transforms_test = transforms.Compose([
                                        # transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        # transforms.ToPILImage(),
                                        # transforms.CenterCrop((224, 224)),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        #                      std=[0.229, 0.224, 0.225])
                                        ])

class DS(Dataset):
    def __init__(self, directory,datatype="train",transform = transforms.Compose([transforms.ToTensor()])):
        self.dir = directory
        self.datatype = datatype
        self.transform = transform
        self.image_files_list = [s for s in sorted(os.listdir(os.path.join(self.dir,self.datatype)))]
        
    def __len__(self):
        return len(self.image_files_list)
    def __getitem__(self, idx):
        path = os.path.join(self.dir,self.datatype,self.image_files_list[idx])
        img = np.load(path+"/img.npy")
        
        label = np.load(path+"/label.npy")
        seg = np.load(path+"/seg.npy")
        seg_clip = np.clip(seg,0,1)
        img = np.concatenate((img, np.expand_dims(seg_clip,0)),axis=0)
        img = img/img.max()
        img = np.transpose(img,(1,2,0)) 
        # img = Image.fromarray(img)
        # label = torch.from_numpy(label)
        img = self.transform(img) 

        return img,label,seg_clip,self.image_files_list[idx]

    # def embedding (self, idx):
    #     if self.datatype == "train":
    #         img = os.path.join("./input/train", self.ds["Image"][idx])
    #         label = self.ds["Y"][idx]
    #     else:
    #         img = os.path.join("./input/test", self.image_files_list[idx])
    #         label = ""
    #     img = Image.open(img).convert('RGB')
    #     img = self.transform(img)
        
    #     return img,label

train_ds = DS(base,datatype="train",transform = data_transforms_test)
val_ds = DS(base,datatype="valid",transform = data_transforms_test)
test_ds = DS(base,datatype="test",transform = data_transforms_test)
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))
# ds = DS(base,datatype="train")
# print(ds.image_files_list)
# for img, label, seg, name in ds:
#     plt.subplot(321),plt.imshow(img[0]),plt.title('1')
#     plt.subplot(322),plt.imshow(img[1]),plt.title('2')
#     plt.subplot(323),plt.imshow(img[2]),plt.title('3')
#     plt.subplot(324),plt.imshow(img[3]),plt.title('4')
#     plt.subplot(325),plt.imshow(img[4]),plt.title('5')
#     plt.subplot(326),plt.imshow(seg),plt.title('seg_clip')
#     plt.show()
    # plt.imshow(img[0])
    # plt.show()
    # print(img.size())
bs = 32
workers=8

train_dl = DataLoader(train_ds,batch_size=bs,num_workers=workers,shuffle= True)
val_dl = DataLoader(val_ds,batch_size=bs,num_workers=workers)
test_dl = DataLoader(test_ds,batch_size=bs,num_workers=workers)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(5,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        # out = self.fc2(out)
        out = self.sigmoid(self.fc2(out))
        return out

def get_accuracy(y_true, y_prob):
    accuracy = metrics.accuracy_score(y_true, y_prob > 0.8)
    return accuracy

model = Cnn()
model = model.cuda()
# summary(model, [(5, 224, 224)])
# exit()
optimizer = optim.Adam(params = model.parameters(),lr=0.001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

tr_los = []
tr_acc = []

val_los = []
val_acc = []

epoch = 40
def train (model):
    for i in range(epoch):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for it,(img,label,_,_) in enumerate(train_dl):
            img = img.float()
            label = label.float()
            # test = torch.max(label, 1)
            img = img.cuda()
            label = label.cuda()
            out = model(img)
            loss = criterion(out,label)

            # acc = ((out.argmax(dim=1) == label).float().mean())
            acc = get_accuracy(label.cpu(), out.cpu())
            epoch_accuracy += acc/len(train_dl)
            epoch_loss += loss/len(train_dl)
            
            # predicted = model(torch.tensor(img,dtype=torch.float32))
            # acc = ((out.argmax(dim=1) == label).float().mean())
            # acc = (predicted.reshape(-1).detach().cpu().numpy().round() == label).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tr_los.append(epoch_loss)
        tr_acc.append(epoch_accuracy)    
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(i, epoch_accuracy,epoch_loss))

        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss =0
            model.eval()
            for img, label,_ ,_ in val_dl:
                img = img.float()
                img = img.cuda()
                label = label.float()
                label = label.cuda()

                val_out = model(img)
                val_loss = criterion(val_out,label)
                
                # acc = ((val_out.argmax(dim=1) == label).float().mean())
                acc = get_accuracy(label.cpu(), val_out.cpu())
                epoch_val_accuracy += acc/ len(val_dl)
                epoch_val_loss += val_loss/ len(val_dl)
            val_los.append(epoch_val_loss)
            val_acc.append(epoch_val_accuracy) 
            print('VAL , val_accuracy : {}, val_loss : {}'.format(epoch_val_accuracy,epoch_val_loss),"\n")

train_model = train(model)

#plotting the loss
plt.plot(tr_los,'r', label='Training loss')
plt.plot(val_los, 'b', label='Validation loss')
plt.title('Loss vs Epochs')
plt.legend(loc=0)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.figure()
plt.plot(tr_acc,'r', label='Training acc')
plt.plot(val_acc, 'b', label='Validation acc')
plt.title('ACC vs Epochs')
plt.legend(loc=0)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()

def test(model):
    epoch_test_accuracy = 0
    epoch_test_loss = 0
    for img, label,_ ,_ in test_dl:
        img = img.float()
        img = img.cuda()
        label = label.float()
        label = label.cuda()

        test_out = model(img)
        test_loss = criterion(test_out,label)
        acc = get_accuracy(label.cpu(), test_out.cpu())
        epoch_test_accuracy += acc/ len(test_dl)
        epoch_test_loss += test_loss/ len(test_dl)
    
    print("Test Accuracy:", epoch_test_accuracy,"Test Loss:", epoch_test_loss)

test(model.eval())




