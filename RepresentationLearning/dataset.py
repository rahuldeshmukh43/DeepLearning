from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, StanfordCars, FER2013
import torchvision.transforms as transforms

def choose_train_dataset(dataset_name="mnist"):
    if dataset_name=="mnist":
        # 28x28 grayscale
        return MNIST(root="./data", train=True, download=True)
    if dataset_name=="cifar10":
        # 32x32 rgb
        return CIFAR10(root="./data", train=True, download=True)
    # if dataset_name=="stanford_cars":
    #     # rgb images
    #     return StanfordCars(root="./data", split='train', download=True)
    # if dataset_name=="fer2013":
    #     # facial expression 
    #     # 48x48 grayscale
    #     # labels: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    #     return FER2013(root="./data", split='train')
    
def choose_test_dataset(dataset_name="mnist"):
    if dataset_name=="mnist":
        # 28x28 grayscale
        return MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    if dataset_name=="cifar10":
        # 32x32 rgb
        return CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    # if dataset_name=="stanford_cars":
    #     # rgb images
    #     return StanfordCars(root="./data", split='test', download=True, transform=transforms.ToTensor())
    # if dataset_name=="fer2013":
    #     # facial expression 
    #     # 48x48 grayscale
    #     # labels: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    #     # testing labels are not available
    #     return FER2013(root="./data", split='test', transform=transforms.ToTensor())

class Classification_Dataset(Dataset):
    def __init__(self,
                 dataset_name="mnist") -> None:
        super().__init__()
        self.dataset = choose_train_dataset(dataset_name=dataset_name)
        self.num_classes = len(self.dataset.classes) #[0-9]
        
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        label1 = torch.tensor(label1, dtype=torch.long)
        img1 = self.transform(img1)
        return img1, label1

# class Contrastive_Dataset(Classification_Dataset):
#     def __init__(self,dataset_name="mnist") -> None:
#         super().__init__(dataset_name=dataset_name)        
#         # make a dictionary of dataset. with keys as class names and corresponding ids in a list
#         self.dataset_dict = defaultdict(list)
#         for i, data in enumerate(self.dataset):
#             _img, label = data
#             self.dataset_dict[label].append(i)

#     def _get_item(self, label,):
#         chosen_idx = np.random.choice(self.dataset_dict[label])
#         img2, label2 = self.dataset[chosen_idx]
#         return img2, label2
        
#     def __getitem__(self, index):
#         """
#         Return an image pair and a integer label indicating if they belong to the same class
#         Args:
#             index (int)
#         Return:
#             img1 (1, 1, H, W)
#             label1 (int) class label
#             img2 (1, 1, H, W)
#             label2 (int) class label
#         """
        
#         img1, label1 = self.dataset[index]

#         # get other image
#         r = np.random.rand(1)
#         if r > 0.5:
#             # get a positive example
#             img2, label2 = self._get_item(label1)
#         else:
#             # get negative example
#             class_list = [i for i in range(self.num_classes)]
#             class_list.remove(label1)
#             neg_label = np.random.choice(class_list)
#             img2, label2 = self._get_item(neg_label)

#         # transform
#         img1 = self.transform(img1)
#         img2 = self.transform(img2)

#         return img1, label1, img2, label2
    
# class Triplet_Dataset(Contrastive_Dataset):
#     def __init__(self, dataset_name="mnist"):
#         super().__init__(dataset_name=dataset_name)
#         return
    
#     def __getitem__(self, index):
#         anchor, anchor_label = self.dataset[index]

#         positive, _ = self._get_item(anchor_label)
#         class_list = [i for i in range(self.num_classes)]
#         class_list.remove(anchor_label)
#         neg_label = np.random.choice(class_list)
#         negative, negative_label = self._get_item(neg_label)

#         # transform images
#         anchor = self.transform(anchor)
#         positive = self.transform(positive)
#         negative = self.transform(negative)

#         return anchor, anchor_label, positive, anchor_label, negative, negative_label
    
# class Quadruplet_Dataset(Contrastive_Dataset):
#     def __init__(self, dataset_name="mnist") -> None:
#         super().__init__(dataset_name=dataset_name)
    
#     def __getitem__(self, index):        
#         anchor, anchor_label = self.dataset[index]

#         positive, _ = self._get_item(anchor_label)
#         class_list = [i for i in range(self.num_classes)]
#         class_list.remove(anchor_label)
#         neg1_label = np.random.choice(class_list)
#         neg1, neg1_label = self._get_item(neg1_label)

#         class_list.remove(neg1_label)
#         neg2_label = np.random.choice(class_list)
#         neg2, neg2_label = self._get_item(neg2_label)

#         # transform images
#         anchor = self.transform(anchor)
#         positive = self.transform(positive)
#         neg1 = self.transform(neg1)
#         neg2 = self.transform(neg2)

#         return anchor, anchor_label, positive, anchor_label, neg1, neg1_label, neg2, neg2_label






