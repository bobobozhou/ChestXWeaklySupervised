import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import ipdb


class ChestXray14DataSet(Dataset):
    """"ChestX-Ray14 Data loader"""

    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            txt_labelfile (string): Path to the txt file with labels.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.n_classes = 14
        self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass'
                           'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
                           'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
