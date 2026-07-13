import torch
from torch import nn 
from model import model
from torch import optim
import torch.nn.functional as F
from dataset import _get_dataloader
import os 
import time 

start = time.time()
def main():

    # consider setting learning rate conditionally 

    # Creating an instance of model
    CRNN = model(in_channels=1, num_classes=63)

    criterion = nn.CTCLoss(zero_infinity=True)
    # The optimiser updates model weights using gradients computed by the loss.backward() in training loop 
    # Adam converges faster and requires less manual tuning of the learning parameter than RMSprop 
    optimiser = optim.Adam(CRNN.parameters(), lr=0.001)

    # Learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=3
    )

    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('CUDA available')
    CRNN.to(device)

    criterion.to(device)

    train_loader, valid_loader = _get_dataloader()

    i = 0 
    valid_interval = 500
    epochs = 30

    # Periodic validation checks
    # to track accuracy and can be used to decide when to save checkpoints
    def evaluation(i) : 
        nonlocal best_val_loss

        if i % valid_interval == 0 : 
            CRNN.eval()
            valid_loss = 0 
            with torch.no_grad(): 
                for valid_images, valid_targets , valid_target_lengths in valid_loader:
                    valid_images = valid_images.to(device)
                    valid_targets = valid_targets.to(device)
                    valid_target_lengths = valid_target_lengths.to(device)

                    valid_logits = CRNN(valid_images)
                    valid_log_probs = F.log_softmax(valid_logits, dim=2)

                    valid_batch_size = valid_images.size(0)
                    valid_input_lengths = torch.full((valid_batch_size,), valid_log_probs.size(0), dtype=torch.long)
                    
                    loss = criterion(valid_log_probs, valid_targets, valid_input_lengths, valid_target_lengths)
                    valid_loss += loss.item()
            
            avg_val_loss = valid_loss/ len(valid_loader)
            scheduler.step(avg_val_loss)
            print(f'Validation loss :{avg_val_loss}')

            if avg_val_loss < best_val_loss: 
                best_val_loss = avg_val_loss 
                save_path = os.path.join(checkpoints_dir, 'crnn.pt')
                checkpoint = {
                'model_state_dict': CRNN.state_dict(),   
                'val_loss': avg_val_loss
                }
                torch.save(checkpoint, save_path)
                print(f'New best model saved (val_loss= {avg_val_loss:.4f}) -> {save_path}')
        
            CRNN.train()

    checkpoint_path = os.path.join(checkpoints_dir, 'crnn.pt')

    if os.path.exists(checkpoint_path): 
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_state = checkpoint['model_state_dict']
        model_state = CRNN.state_dict()


        filtered_state = { 
            k : v for k,v in checkpoint_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        model_state.update(filtered_state)
        CRNN.load_state_dict(model_state)

        skipped = checkpoint_state.keys() - filtered_state.keys()

        if skipped: 
            best_val_loss = float('inf')
            print('Resumed from checkpoint with new final layer, resetting loss value')
        else: 
            best_val_loss = checkpoint['val_loss']
            print(f'Resumed from checkpoint, best_val_loss={best_val_loss:.4f}')
        
    else: 
        best_val_loss = float('inf')
        print(f'No checkpoint found')
        
    """

    if os.path.exists(checkpoint_path): 
        checkpoint = torch.load(checkpoint_path, map_location=device)
        CRNN.load_state_dict(checkpoint['model_state_dict'], strict=False)
        best_val_loss = checkpoint['val_loss']
        print(f'Resumed from checkpoint, best_val_loss={best_val_loss:.4f}')
    else : 
        best_val_loss = float('inf') # to track the best score seen so far - 0.4
    
        """

    for epoch in range(epochs) : 
        for images, targets, target_lengths in train_loader: 
            images = images.to(device)
            targets = targets.to(device)

            # Length of each label from dataset so CTC knows where to split targets 
            target_lengths = target_lengths.to(device)

            logits = CRNN(images)
            # raw, unnormalised output scores from last layer from activated (softwax) is applied 
            # self.fc(x) outputs logits (one score per class at each timestamp)
            log_probs = F.log_softmax(logits, dim=2)

            # Number of samples (images) processed
            batch_size = images.size(0)
            # CTC requires input_lengths so that it knows how many timesteps to consider per sample
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Clears old gradients
            optimiser.zero_grad()
            # Computes new gradients 
            loss.backward()
            # Important for RNNs, since prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(CRNN.parameters(), 5)
            # Apply the update onto the optimiser
            optimiser.step()

            i += 1
            evaluation(i)
        print("Epoch done")

if __name__ == '__main__':
    main()
    print(f'Execution time: {time.time() - start:.6f} seconds')