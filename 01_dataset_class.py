from torch.utils.data import Dataset
import cv2
import os

class mydata(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = cv2.imread(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
cat_label_dir = "cat"
cat_dataset = mydata(root_dir,cat_label_dir)
img,label = cat_dataset[0]

cv2.imshow('cat',img)
cv2.waitKey(0)
cv2.destroyAllWindows()