import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch

class AntiSpoofingDataset(data.Dataset):

    def __init__(self, data_dir):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.seq = []
        self.label = []
        self.fname = []
        self.label_mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2} # 1->real(0), 2,3->print(1), 4,5->replay(2)
        self.istest = False

        for clip in sorted(os.listdir(data_dir)):
            seq_data = []
            clip_dir = os.path.join(data_dir, clip)
            for x in sorted(os.listdir(clip_dir)):
                img_path = os.path.join(clip_dir, x)
                seq_data.append(img_path)
            self.seq.append(seq_data)

            if len(clip.split('_')) == 4:
                label = self.label_mapping[int(clip[-1])]
                self.label.append(label)

            self.fname.append(clip)

        if len(self.label) == 0:
            self.istest = True

    def __getitem__(self, index):
        seq = self.get_seqdata(self.seq[index])
        fname = self.fname[index]
        if not self.istest:
            label = self.label[index]
            return seq, fname, label
        else:
            return seq, fname

    def __len__(self):
        return len(self.seq)

    def get_seqdata(self, img_list):

        def load_data(x):
            img = Image.open(x)
            img = self.transform(img.convert('RGB'))
            return img

        return torch.stack(list(map(load_data, img_list)))


"""
import pdb

dataset = AntiSpoofingDataset("oulu_npu_cropped/test")
data_loader = data.DataLoader(dataset, shuffle=True, drop_last=False, pin_memory=True, batch_size=8)

for (step, value) in enumerate(data_loader):

    seq = value[0].cuda() # B x 11 x 6 x 224 x 224
    if len(value) > 2:
        target = value[2].cuda(async=True)
    pdb.set_trace()
"""

