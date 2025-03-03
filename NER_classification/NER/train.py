import pandas as pd
import torch
import ast
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Deactivate warnings
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 3

# Load DataFrames
train = pd.read_csv("dataset/train.csv")
val = pd.read_csv("dataset/val.csv")
test = pd.read_csv("dataset/test.csv")

# Download BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def format_data(df: pd.DataFrame):
    input_ids = []
    attention_masks = []
    labels = []

    # for adding paddings to have equal lengths arrays
    max_length = max(len(tokens) for tokens in df['tokens'])

    # converting tokens into indices and labels
    for _, row in df.iterrows():
        tokens = row['tokens']

        # convert labels from a string to a list (if necessary)
        if isinstance(row['labels'], str):
            # using ast.literal_eval to convert a string to a list
            label = ast.literal_eval(row['labels'])
        else:
            label = row['labels']

        # tokenization from padding to max_length
        encoding = tokenizer(tokens, padding='max_length', max_length=max_length,
                             truncation=True, is_split_into_words=True, return_tensors='pt')

        input_ids.append(encoding['input_ids'][0])
        attention_masks.append(encoding['attention_mask'][0])

        # adjusting labels to max_length with adding padding (0)
        label_padded = label + [0] * (max_length - len(label))
        labels.append(label_padded)

    # convert to tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)

    print(f"Data shape - input_ids: {input_ids.shape}, attention_masks: {attention_masks.shape}, labels: {labels.shape}")

    return TensorDataset(input_ids, attention_masks, labels)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch, labels_batch = [x.to(device) for x in batch]
            
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                labels=labels_batch
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Only consider non-padded tokens
            active_mask = attention_mask_batch.view(-1) == 1
            predictions = predictions.view(-1)[active_mask].cpu().numpy()
            true_labels = labels_batch.view(-1)[active_mask].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    avg_loss = total_loss / len(dataloader)
    
    # Calculate precision, recall, and f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, 
                                                             all_predictions, 
                                                             average='weighted',
                                                             zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Prepare dataloaders
train_ds = format_data(train)
val_ds = format_data(val)
test_ds = format_data(test)

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Initialize model
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler
num_training_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop with validation
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids_batch, attention_mask_batch, labels_batch = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            labels=labels_batch
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Validation
    val_metrics = evaluate(model, val_dataloader)
    
    print(f"\nEpoch {epoch + 1}")
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Validation metrics:")
    print(f"- Loss: {val_metrics['loss']:.4f}")
    print(f"- Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"- Precision: {val_metrics['precision']:.4f}")
    print(f"- Recall: {val_metrics['recall']:.4f}")
    print(f"- F1-score: {val_metrics['f1']:.4f}")
    
    # Early stopping and model checkpoint
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered!")
        break

# Load best model for final testing
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Final test evaluation
test_metrics = evaluate(model, test_dataloader)
print("\nFinal Test Results:")
print(f"Test metrics:")
print(f"- Loss: {test_metrics['loss']:.4f}")
print(f"- Accuracy: {test_metrics['accuracy']:.4f}")
print(f"- Precision: {test_metrics['precision']:.4f}")
print(f"- Recall: {test_metrics['recall']:.4f}")
print(f"- F1-score: {test_metrics['f1']:.4f}")

# Save the best model
model.save_pretrained('./ner_best_model')