import argparse
import os
import json

import timm
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageFolderDataset(Dataset):
    """Simple dataloader that returns an image and its name """
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.img_list = [os.path.join(img_dir, x) for x in os.listdir(img_dir)
                        if os.path.splitext(x)[1].lower() in valid_extensions]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        with Image.open(img_path) as pil_im:
            image = self.transform(pil_im.convert('RGB'))
        return image, os.path.basename(self.img_list[idx])


def parse_args():
    """Read and returns our input arguments """
    parser = argparse.ArgumentParser(
        'Run a batch of images on a timm backbone and save their features')
    parser.add_argument('--images', help='Images path', required=True)
    parser.add_argument('--output', help='Output path', required=True)
    parser.add_argument('--backbone', help='timm backbone', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size. Usually it is better to '
                        'choose the highest value that your GPU can handle', default=8)
    return parser.parse_args()

def get_eval_dataset(model:str, images_path:str, batch_size:int, image_size:tuple=(224,224)):
    """Gets a dataloader for our images folder respecting the selected backbone transformations"""
     # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config['mean']
    data_std = data_config['std']

    eval_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )
    eval_ds = ImageFolderDataset(images_path, eval_transforms)

    return torch.utils.data.DataLoader(eval_ds, batch_size=batch_size)


if __name__ == '__main__':
    args = parse_args()
    # https://huggingface.co/docs/timm/en/feature_extraction#create-with-no-classifier
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    eval_dataset = get_eval_dataset(args.backbone, args.images, args.batch_size)

    os.makedirs(args.output, exist_ok=True)

    model.eval()
    res = {}
    for batch in eval_dataset:
        imgs, paths = batch
        preds = model(imgs)

        for idx in range(len(paths)):
            res[paths[idx]] = preds[idx].tolist()

    with open(os.path.join(args.output, 'results.json'), 'wt') as fid:
        json.dump(res, fid)