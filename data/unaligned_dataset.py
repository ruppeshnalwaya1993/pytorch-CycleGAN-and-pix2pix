import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_Amask = os.path.join(opt.dataroot, opt.phase + 'Amask')
        self.dir_Bmask = os.path.join(opt.dataroot, opt.phase + 'Bmask')


        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.Amask_paths = make_dataset(self.dir_Amask)
        self.Bmask_paths = make_dataset(self.dir_Bmask)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.Amask_paths = sorted(self.Amask_paths)
        self.Bmask_paths = sorted(self.Bmask_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.Amask_size = len(self.Amask_paths)
        self.Bmask_size = len(self.Bmask_paths)
        assert(self.A_size == self.Amask_size)
        assert(self.B_size == self.Bmask_size)
        #self.transform = get_transform(opt)

    def transform(self, image, mask):
        if self.opt.resize_or_crop == 'resize_and_crop':
            osize = [self.opt.loadSize, self.opt.loadSize]
            # Resize
            resize = transforms.Resize(size=osize)
            image = resize(image)
            mask = resize(mask)
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.opt.fineSize,self.opt.fineSize))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        else:
            raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

        # Random horizontal flipping
        if self.opt.isTrain and not self.opt.no_flip:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        mask = TF.normalize(mask, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return image, mask

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        Amask_path = self.Amask_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        Bmask_path = self.Bmask_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        Amask_img = Image.open(Amask_path).convert('RGB')
        Bmask_img = Image.open(Bmask_path).convert('RGB')

        A, Amask = self.transform(A_img, Amask_img)
        B, Bmask = self.transform(B_img, Bmask_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        tmp = Amask[0, ...] * 0.299 + Amask[1, ...] * 0.587 + Amask[2, ...] * 0.114
        Amask = tmp.unsqueeze(0)
        tmp = Bmask[0, ...] * 0.299 + Bmask[1, ...] * 0.587 + Bmask[2, ...] * 0.114
        Bmask = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'Amask': Amask, 'Bmask': Bmask,
                'A_paths': A_path, 'B_paths': B_path,
                'Amask_paths': Amask_path, 'Bmask_paths': Bmask_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
