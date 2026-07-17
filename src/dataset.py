import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset 
import numpy as np  

class dataset(Dataset) : 

    CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHARS_to_index = {c: i+1 for i,c in enumerate(CHARS)}
    index_to_CHARS = {i: c for c,i in CHARS_to_index.items()}

    def __init__(self, dataset_name, split_dataset ='train', text_column  ='label', image_height=32 , image_width=100, limit=5000):
        self.image_height = image_height
        self.image_width = image_width
        self.text_column = text_column
        self.data = load_dataset(dataset_name, split=f"{split_dataset}[:{limit}]")

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample['image'].convert('L') # Grayscales the image 
        text = sample[self.text_column]

        image = image.resize((self.image_width, self.image_height))
        image = np.array(image, dtype=np.float32)
        image = (image/127.5) - 1.0
        image = torch.from_numpy(image).unsqueeze(0)

        target = [self.CHARS_to_index[c] for c in text if c in self.CHARS_to_index]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor([len(target)])

        return image, target, target_length
    def __len__(self):
        return len(self.data)
    
def collate_fn(batch) : 
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths


def _get_dataloader() : 
    mjsynth = dataset(dataset_name="priyank-m/MJSynth_text_recognition", split_dataset='train', text_column='label', limit=200000) # change it to 500k
    svhn = dataset(dataset_name="MiXaiLL76/SVHN_OCR", split_dataset='train', text_column='text', limit=28265)
    iiit5k_train = dataset(dataset_name="MiXaiLL76/IIIT5K_OCR", split_dataset='train', text_column='text', limit=2000 )

    train_dataset = ConcatDataset([mjsynth, svhn, iiit5k_train])

    iiit5k_test = dataset(dataset_name="MiXaiLL76/IIIT5K_OCR", split_dataset='test', text_column='text', limit=3000 )
    mjsynth_valid = dataset(dataset_name="priyank-m/MJSynth_text_recognition", split_dataset='val', text_column='label', limit=5000)

    valid_dataset = ConcatDataset([mjsynth_valid, iiit5k_test])

# changed num_workers and batch size and pin memory 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=6, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader 

        