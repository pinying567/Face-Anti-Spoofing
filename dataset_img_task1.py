import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch

class AntiSpoofingDataset(data.Dataset):

    def __init__(self, data_dir):

        # augmentation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data = []
        self.label = []
        self.fname = []
        self.label_mapping = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0} # 1->real(1), 2,3,4,5->fake(0)
        self.istest = False

        for clip in sorted(os.listdir(data_dir)):
            if len(clip.split('_')) == 4:
                label = self.label_mapping[int(clip[-1])]
            else:
                self.istest = True
            clip_dir = os.path.join(data_dir, clip)
            for x in sorted(os.listdir(clip_dir)):
                img_path = os.path.join(clip_dir, x)
                self.data.append(img_path)
                self.fname.append(clip)
                if not self.istest:
                    self.label.append(label)

    def __getitem__(self, index):
        data = self.transform(Image.open(self.data[index]).convert('RGB'))
        fname = self.fname[index]
        if not self.istest:
            label = self.label[index]
            return data, fname, label
        else:
            return data, fname

    def __len__(self):
        return len(self.data)


"""
import pdb

dataset = AntiSpoofingDataset("oulu_npu_cropped/train")
data_loader = data.DataLoader(dataset, shuffle=True, drop_last=False, pin_memory=True, batch_size=8)

for (step, value) in enumerate(data_loader):

    image = value[0].cuda() # B x 3 x 224 x 224
    if len(value) > 2:
        target = value[2].cuda(async=True)
    pdb.set_trace()
"""

