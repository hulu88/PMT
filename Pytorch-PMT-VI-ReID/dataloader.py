import numpy as np
from PIL import Image
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform1=None,transform2 = None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # RGB format
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2  = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform1=None,transform2 = None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestData_RegDB(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)

        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def  __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
