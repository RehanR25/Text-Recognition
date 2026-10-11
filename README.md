## CRNN Text Recognition 

A PyTorch implementation of a CRNN (CNN + bidirectional LSTM + CTC loss) that reads text from images. It can be trained on public scene-text datasets and run live on a webcam feed. 

### How it works 
  1. A CNN extracts features from a 32 x 100 grayscale image
  2. The feature map is turned into a sequence along the image width
  3. Two bidirectional LSTM layers model the sequence
  4. A linear layer outputs per-timestamp class scores (62 characters plus a CTC blank, 63 classes)
  5. CTC loss is used for training, and greedy decoding (collapse repeats, drop blanks) is used for prediction

 Supported characters: 0-9, a-z, A-Z (case sensitive)

### Project Structure

| File | Purpose | 
|--- | --- | 
| ```model.py``` | CRNN architecture (7 conv layers, 2 BiLSTM layers, linear head) |
| ```dataset.py``` | Dataset class, collate function and dataloader setup |
| ```training.py``` | Training loop with validation, checkpointing and early stopping | 
| ```prediction.py``` | Live webcam inference | 

### Datasets
Loaded from Hugging Face with the datasets library: 
  * Train: MJSynth (pritank-m/MJSynth_text_recognition), SVHN OCR (MiXaiLL76/SVHN_OCR), IIIT5K (MiXaiLL76/IIIT5K_OCR)
  * Validation: MJSynth validation split and IIIT5K test split

Images are converted to grayscale, resized to 32x100 and normalised to [-1,1]. Samples per source are capped by the ```limit``` argument in ```dataset.py```

### Setup

```sh
pip install torch datasets numpy pillow opencv-python
```
A CUDA GPU is used automatically if available 

### Training 

```bash
python training.py
```
  * Loss: CTC loss
  * Optimiser: Adam ( lr: 1e-4, weight decay: 1e-5) with cosine annealing warm restarts
  * Gradient clipping at norm 1
  * Validation every 300 batches
  * Best model (lowest validation loss) saved to ```checkpoints/crnn.pt```
  * Early stopping after 5 validations without improvement
  * Resumes from an existing checkpoint if one is found. If the final layer shape has changed, only the matching weights are loaded and best loss is reset

### Live prediction 

```bash
python prediction.py
```
Requires a trained ```checkpoints/crnn.pt```
  * Place text inside the rectangle shown on the webcam feed
  * Press ```p``` to run recognition on the boxed region
  * Press ```q``` to quit
The crop region is set by ```x1, y1, x2, y2``` at the top of ```prediction.py```

### Acknowledgements 
This project was developed with reference to GitYCC/crnn-pytorch by GitYCC






 
