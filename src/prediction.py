import torch
import os 
from model import model
from PIL import Image
import numpy as np 
import torch.nn.functional as F 
import dataset 
import cv2 

cap = cv2.VideoCapture(0)

# Define the crop region
x1,y1,x2,y2 = 200, 150, 500, 250

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CRNN = model(in_channels=1, num_classes=37)
checkpoint_path = 'checkpoints/crnn.pt'
if not os.path.exists(checkpoint_path): 
    print('No checkpoints found, Training . . .')
    import training
    CRNN = training.CRNN
else : 
    checkpoint = torch.load('checkpoints/crnn.pt', map_location=device)
    CRNN.load_state_dict(checkpoint['model_state_dict'])
# This ensures that if training.py saves a checkpoint when validation improves and the first validation check never triggers or never improves, 
# the file might not exists even after training runs which would lead to FileNotFoundError
CRNN.to(device)
CRNN.eval()
predicted_text = ''


# wrap it 
while True: 
    ret, frame = cap.read()
    if not ret: 
        break 

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,0,0), 2)
    cv2.putText(display_frame, predicted_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    cv2.imshow('Live feed', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break 
    elif key == ord('p'): 
        # Cropping the region insdie the border 
        cropped_frame = frame[y1:y2, x1:x2] 
        image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100,32)) 
        image = image.astype(np.float32)
        image = (image/127.5) - 1.0
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad(): 
            logits = CRNN(image)
            log_probs = F.log_softmax(logits, dim=2)

        log_probs = log_probs.squeeze(1)

        predicted_indices= torch.argmax(log_probs, dim=1)

        collapsed = []
        previous = None
        for index in predicted_indices: 
            if index != previous and index != 0 : 
                collapsed.append(index)
            previous = index 

        predicted_chars = [dataset.dataset.index_to_CHARS[int(idx)] for idx in collapsed]
        predicted_text = ''.join(predicted_chars)

        print('Your word is',predicted_text)

cap.release()
cv2.destroyAllWindows()