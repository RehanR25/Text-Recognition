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


    # Creating an instance of model
    CRNN = model(in_channels=1, num_classes=63)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CRNN.to(device)
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

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

    criterion = nn.CTCLoss(zero_infinity=True)
    # The optimiser updates model weights using gradients computed by the loss.backward() in training loop 
    # Adam converges faster and requires less manual tuning of the learning parameter than RMSprop 
    optimiser = optim.Adam(CRNN.parameters(), lr= 0.0001 , weight_decay=1e-5)

    # Learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=1000, T_mult=2
    )

    criterion.to(device)

    train_loader, valid_loader = _get_dataloader()

    i = 0 
    valid_interval = 300
    epochs = 20
    # too many epochs is a bad thing if the training data does not learn 
    # since the model constantly updates its weights, it overfits ultimately 
    patience_counter = 0
    patience_threshold = 5
    

    # Periodic validation checks
    # to track accuracy and can be used to decide when to save checkpoints
    # This allows the model to save the best model till now
    def evaluation(i) : 
        nonlocal best_val_loss, patience_counter

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
            print(f'Validation loss :{avg_val_loss}')

            CRNN.train()

            if avg_val_loss < best_val_loss: 
                best_val_loss = avg_val_loss 
                save_path = os.path.join(checkpoints_dir, 'crnn.pt')
                checkpoint = {
                'model_state_dict': CRNN.state_dict(),   
                'val_loss': avg_val_loss
                }
                torch.save(checkpoint, save_path)
                print(f'New best model saved (val_loss= {avg_val_loss:.4f}) -> {save_path}')
                patience_counter = 0 
            else: 
                patience_counter += 1
            if patience_counter >= patience_threshold: 
                return True 
        return False 
        

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
            torch.nn.utils.clip_grad_norm_(CRNN.parameters(), 1)
            # Apply the update onto the optimiser
            optimiser.step()

            scheduler.step()

            i += 1
            if i % 100 == 0 : 
                print(f'Batch {i}, Train loss: {loss.item():.4f}')
            stop = evaluation(i)
            if stop: 
                break
        if stop: 
            break
        print("Epoch done")


if __name__ == '__main__':
    main()
    print(f'Execution time: {time.time() - start:.6f} seconds')