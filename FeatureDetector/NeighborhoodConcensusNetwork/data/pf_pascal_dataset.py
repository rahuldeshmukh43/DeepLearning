import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import scipy.io
import numpy as np
import glob
from PIL import Image


SEED = 4932
np.random.seed(SEED) 
img_resize=400 #400/16 = 25; ncnet has 1/16 factor and final correlation is of 25 spatial res
#dataset_split=(700,300,300) #train, val, test

class PF_PASCAL_DATALOADER():
    def __init__(self, args):
        self.args = args
        self.dataset = PF_PASCAL_DATASET(args)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size = args.batch_size,
                                      num_workers = args.workers,
                                      collate_fn = self.my_collate)
    def load_data(self):
        return self.data_loader
    
    def name(self):
        return 'PFPascalLoader'
    
    def __len__(self):
        return len(self.dataset)

    def my_collate(self, batch):
        tensor_keys = ['img1', 'img2', ] # 'class_idx1', 'class_idx2', 'img1_name', 'img2_name'
        nonuniformtensor_keys = ['img1_ori', 'img2_ori', 'img1_size', 'img2_size', 'img1_annotation', 'img2_annotation', 'common_kp_idx']
        out = {}
        out['label'] = torch.tensor([item['label'] for item in batch])
        for key in tensor_keys:
            d = [item[key] for item in batch]
            out[key] = torch.stack(d)

        for key in nonuniformtensor_keys:
            d = [item[key] for item in batch]
            out[key] = d
        return out


class PF_PASCAL_DATASET(Dataset):
    def __init__(self, args):
        super(PF_PASCAL_DATASET, self).__init__()
        
        self.args = args
        self.data_top_dir = args.datadir
        self.split = args.split
        self.phase = args.phase #train/val/test
                
        #get all pairs
        data_parser = PF_PASCAL_PAIR_PARSER(os.path.join(self.data_top_dir,'parsePascalVOC.mat'))
        self.class_names = data_parser.class_names
        self.num_classes = data_parser.num_classes        
        all_pairs = data_parser.Get_all_pairs()
        
        #spilt into training and testing
        all_pairs = self._split_all_pairs(all_pairs)
        
        # #create negative examples for training and merge -- #how many negatives?: same as positive
        if self.phase == 'train':
            all_negative_pairs = self._make_all_negative_pairs()
            num_negative = len(all_pairs)
            idx = np.arange(len(all_negative_pairs))
            idx = np.random.permutation(idx)[:num_negative]
            all_negative_pairs =  np.array(all_negative_pairs)[idx]
            #combine with all_pairs
            all_pairs = np.concatenate((all_pairs, all_negative_pairs), axis=0)

        # mean and std dev for imagenet -- resnet was trained on this
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]

        # mu = [118.55087744423791, 112.61700425650558, 104.27034932620818]
        # sigma = [69.139815150136, 68.69808586239063, 71.49002337343454]
        
        #define transforms
        # transform should resize so that final image size is 25x25 resolution ie initial one should be 16*25=400x400
        # if self.phase == 'train':
        self.transform = transforms.Compose([ transforms.ToTensor(),
                                              transforms.Resize((img_resize,img_resize)),
                                              transforms.Normalize(mu, sigma) ])
        # else:
        #     self.transform = transforms.Compose([ transforms.ToTensor(),
        #                                           transforms.Resize((img_resize,img_resize)) ])
        self.ori_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((img_resize, img_resize))])
        
        #shuffle data for train
        idx = np.arange(len(all_pairs))
        idx = np.random.permutation(idx)
        self.all_pairs = all_pairs[idx]
        
        return
    
    def _split_all_pairs(self, all_pairs):
        "do a random split"
        num_pairs = len(all_pairs)
        total = np.sum(self.split)
        phase_sizes = np.floor((np.array(self.split)/total)*num_pairs).astype(np.int32)
        idx = np.arange(num_pairs)
        idx = np.random.permutation(idx)        
        if self.phase == 'train':
            start = 0
            stop = phase_sizes[0]
        elif self.phase == 'val':
            start = phase_sizes[0]
            stop = start + phase_sizes[1]
        elif self.phase == 'test':
            start = np.sum(phase_sizes[:-1])
            stop = num_pairs        
        return all_pairs[idx][start:stop]
        
    def _make_all_negative_pairs(self):
        pair_list = []# ((class_idx1,class_idx2), (img_name1,img_name2))
        Mat2img_name = lambda x: os.path.basename(x).replace('.mat','')
        for class_idx1 in range(self.num_classes-1):
            for class_idx2 in range(1,self.num_classes):
                class1_folder = os.path.join(self.data_top_dir,'Annotations', self.class_names[class_idx1])
                class2_folder = os.path.join(self.data_top_dir,'Annotations', self.class_names[class_idx2])
                
                for img1_name in glob.glob(class1_folder+'/*.mat'):
                    for img2_name in glob.glob(class2_folder+'/*.mat'):
                        element = ((class_idx1, class_idx2), (Mat2img_name(img1_name), Mat2img_name(img2_name)))
                        pair_list.append(element)
        return pair_list        
    
    # def _Compute_stats(self):
    #     "read each image in dataset and then compute per channel stats"
    #     imgs = [] #NxHxWxC
    #     for img_file in glob.glob(os.path.join(self.data_top_dir,'JPEGImages')+'/*.jpg'):
    #         img = np.array( Image.open(img_file).resize((img_resize,img_resize)) )
    #         imgs.append(img)
    #     imgs = np.array(imgs)
    #     mu = [np.mean(imgs[:,:,:,c]) for c in range(imgs.shape[-1])]
    #     sigma = [np.std(imgs[:,:,:,c]) for c in range(imgs.shape[-1])]
    #     return mu, sigma
                        
    def __getitem__(self, index):
        #access data
        class_idxs, img_names = self.all_pairs[index]
        class_idx1, class_idx2 = [np.int32(i) for i in class_idxs]
        img1_name, img2_name = img_names
        #read images
        img1 = np.array(Image.open( os.path.join(self.data_top_dir, 'JPEGImages' , img1_name+'.jpg' ) )) #HWC
        img1_size = img1.shape[:-1]
        img2 = np.array(Image.open( os.path.join(self.data_top_dir, 'JPEGImages' , img2_name+'.jpg' ) ))
        img2_size = img2.shape[:-1]
        #read annotations
        img1_annotation = self.read_annotation(os.path.join(self.data_top_dir, 'Annotations' , self.class_names[class_idx1], img1_name+'.mat' ))
        img2_annotation = self.read_annotation(os.path.join(self.data_top_dir, 'Annotations' , self.class_names[class_idx2], img2_name+'.mat' )) 
        common_kp_idx = np.intersect1d(img1_annotation['notnan_idx'], img2_annotation['notnan_idx'])
        
        label = 2*torch.tensor(class_idx1 == class_idx2).float() - 1 #label is y=+1 for positive pair and y=-1 for negative pair
        
        #apply transforms
        img1_t = self.transform(img1) #CHW
        img2_t = self.transform(img2)
        img1 = self.ori_transform(img1)
        img2 = self.ori_transform(img2)
        
        data = {'label': label, #y
                'img1': img1_t,
                'img2': img2_t,
                'img1_ori': img1,
                'img2_ori': img2,
                # 'class_idx1': class_idx1,
                # 'class_idx2': class_idx2,
                # 'img1_name': img1_name,
                # 'img2_name': img2_name,
                'img1_annotation': img1_annotation,
                'img2_annotation': img2_annotation,
                'common_kp_idx': common_kp_idx,
                'img1_size':img1_size,
                'img2_size':img2_size}
        return data
    
    def __len__(self):
        return len(self.all_pairs)

    @staticmethod
    def read_annotation(filename):        
        annotation = scipy.io.loadmat(filename)
        kps = annotation['kps'] # not all keypoints are valid ie some will be nans
        notnan_idx = np.where(~np.isnan(kps[:,0]))[0]
        annotation['notnan_idx'] = notnan_idx
        #annotation['kps'] = kps[ notnan_idx ][:]
        annotation['num_kps'] = len(notnan_idx)
        ## annotation is a dictionary with keys: {'kps', 'num_kps', 'class', 'bbox', 
        #'imsize', 'occluded', 'difficult', 'truncated'}
        # kps are x,y. x= kps[:,0], x is the col index
        
        return annotation
            
class PF_PASCAL_PAIR_PARSER(object):
    "class for parsing the parsePascalVOC.mat file"
    def __init__(self, parsing_mat_file):
        super(PF_PASCAL_PAIR_PARSER, self).__init__()
        
        # Store constructor arguments
        self.parsing_mat_file = parsing_mat_file
        
        #parse dict
        self.parsing_dict = scipy.io.loadmat(self.parsing_mat_file)
        self.Parse() 
        return 
    
    def Parse(self):
        class_names_array, pair_names_array , num_pairs_array=self.parsing_dict['PascalVOC'][0][0]
        self.class_names = [ c[0] for c in class_names_array[0]]
        self.num_classes = len(self.class_names)
        self.num_pairs_per_class = num_pairs_array[0] 
        self.total_pairs = np.sum(self.num_pairs_per_class)        
        self.pair_names = [[ tuple(map(lambda x:x[0], p)) for p in pair_names_array[0][c_idx]] 
                           for c_idx in range(self.num_classes)]
    
    #Getters
    def Get_pairs_for_class(self, class_idx):
        return self.pair_names[class_idx]
        
    def Get_all_pairs(self):
        "returns list with each element of form ((class_idx1,class_idx2), (img_name1,img_name2))"
        all_pairs = [] 
        for c_idx in range(self.num_classes):
            this_pair_list = list(zip(c_idx*np.ones((self.num_pairs_per_class[c_idx],2), dtype=np.uint8),
                                       self.Get_pairs_for_class(c_idx)))            
            all_pairs.extend(this_pair_list)
        return np.array(all_pairs, dtype=object)
    
# #For Code Testing 
# if __name__ =="__main__":
#     parser = argparse.ArgumentParser(description='Testing pf-pascal dataset')
#     parser.add_argument('--phase',type=str,default='test', help='string for phase of network; accepts train or test')
#     parser.add_argument('--top_dir',type=str,default='/mnt/cloudNAS4/Rahul/Projects/Data/PF-dataset-PASCAL/',
#                          help='top directory path to pf pascal dataset')    
#     parser.add_argument('--split', type=np.float64, default=0.7, help='float value indicating percentage of data to be used for training ')
#
#     args = parser.parse_args()
#
#
#     dataset = PF_PASCAL_DATASET(args)
#     print('Num of samples: %d'%(len(dataset)))
#     data = dataset[1]
#     print(data)
#     #dataset tested! it works!
#

    

