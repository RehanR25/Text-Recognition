import torch
checkpoint = torch.load('checkpoints/crnn.pt', map_location='cpu')
print('val_loss:', checkpoint['val_loss'])
print('fc shape:', checkpoint['model_state_dict']['fc.weight'].shape)

from datasets import load_dataset
ds = load_dataset("yunusserhat/TextOCR-Dataset")
sample  = next(iter(ds['train']))
print(sample)