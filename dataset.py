import os
import re

from typing import List, Dict
from skimage import io
from torch.utils.data import Dataset


class CNRParkDataset(Dataset):
    def __init__(self, patches_dir, labels_dir, transform=None):
        self.cam_dates_spots: Dict[str, Dict[str, int]] = dict()
        with open(f"{labels_dir}/all.txt", "r") as f:
            for line in f.readlines():
                line_splt: List[str] = line.split()
                img_loc, label = line_splt[0], int(line_splt[1])
                matches = re.match("[A-Z]+\/[\d|-]+\/camera\d+\/[A-Z]_([\d|-]+)_([\d|.]+)_([A-Z]\d+)_(\d+)", img_loc)
                date, time, cam, spot = matches.groups()
                d_str = f"{cam}_{date}_{spot}"
                if self.cam_dates_spots.get(d_str) is None:
                    self.cam_dates_spots[d_str] = dict()
                self.cam_dates_spots[d_str][img_loc] = label
        self.patches_dir = patches_dir
        self.transform = transform

    def __len__(self):
        return len(self.cam_dates_spots)

    def __getitem__(self, idx):
        cam_date_spot = next(val for i, val in enumerate(self.cam_dates_spots.values()) if i == idx)
        images = [io.imread(os.path.join(self.patches_dir, img_loc)) for img_loc in cam_date_spot.keys()]
        labels = [label for label in cam_date_spot.values()]
        sample = {'images': images, 'labels': labels}

        if self.transform:
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample
