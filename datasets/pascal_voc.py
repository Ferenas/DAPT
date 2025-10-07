import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import math
CAT_LIST = ['aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'table',
               'dog',
               'horse',
               'motorbike',
               'person',
               'plant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor']

CAT_LIST_TO_NAME = dict(zip(range(len(CAT_LIST)) ,CAT_LIST))


def _collate(ims, y, c):
    return Datum(impath=ims, label=y, classname=c)

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(data_root,img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = 'voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    data_dtm = []

    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label = cls_labels_dict[img_name]
        label_idx = np.where(label==1)[0]
        class_name = [CAT_LIST[idx]  for  idx in range(len(label_idx))]
        data_dtm.append(_collate(os.path.join(data_root,img_name+'.jpg'),label,class_name))

    return data_dtm









@DATASET_REGISTRY.register()
class VOC12(DatasetBase):
    dataset_dir = "voc12data"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir,'VOCdevkit/VOC2012/JPEGImages')
        train_img_name_list_path = os.path.join('voc12/train_aug_id.txt')
        val_img_name_list_path = os.path.join('voc12/val_id.txt')

        train = load_image_label_list_from_npy(self.image_dir,load_img_name_list(train_img_name_list_path))
        val = load_image_label_list_from_npy(self.image_dir,load_img_name_list(val_img_name_list_path))
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val = self.subsample_classes(train, val, subsample=subsample)

        super().__init__(train_x=train, val=val, test=val)

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            label_idx = random.choices(np.where(item.label == 1)[0])[0]
            labels.add(label_idx)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                label_idx = random.choices(np.where(item.label == 1)[0])[0]
                if label_idx not in selected:
                    continue

                item_new = Datum(
                    impath=item.impath,
                    label=item.label,
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output


    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        return len(CAT_LIST)

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        return CAT_LIST_TO_NAME, CAT_LIST

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            one_hot_label = item.label
            label_idx = random.choices(np.where(one_hot_label==1)[0])[0]
            output[label_idx].append(item)

        return output

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


