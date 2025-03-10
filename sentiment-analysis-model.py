import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1. Define the BERT-based sentiment analysis model
class BertSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout_rate=0.1):
        super(BertSentimentClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classify
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        return logits

# 2. Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Progress bar
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        # Get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)

# 3. Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

# 4. Plotting functions
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_training_history(train_losses, val_metrics):
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy and F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], 'g-', label='Accuracy')
    plt.plot(epochs, [m['f1'] for m in val_metrics], 'p-', label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# 5. Main training function
def train_bert_sentiment_model(
    train_loader,
    val_loader,
    epochs=4,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = BertSentimentClassifier()
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * epochs
    
    # Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Track training progress
    train_losses = []
    val_metrics = []
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        
        # Evaluate epoch
        val_results = evaluate(model, val_loader, device)
        val_metrics.append(val_results)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        print(f"Val Accuracy: {val_results['accuracy']:.4f}")
        print(f"Val F1 Score: {val_results['f1']:.4f}")
    
    # Plot training history
    plot_training_history(train_losses, val_metrics)
    
    # Plot final confusion matrix
    plot_confusion_matrix(val_metrics[-1]['confusion_matrix'])
    
    return model, train_losses, val_metrics

# 6. Function to predict sentiment for new descriptions
def predict_sentiment(model, tokenizer, texts, device=None, max_length=256):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    predictions = []
    
    for text in texts:
        # Tokenize
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
        
        # Convert to sentiment label
        sentiment = "Positive" if preds.item() == 1 else "Negative"
        predictions.append(sentiment)
    
    return predictions

if __name__ == "__main__":
    # This would be used after running the preprocessing script
    from preprocessing import preprocess_data
    
    # 1. Load and preprocess data
    train_loader, val_loader, tokenizer = preprocess_data(
        'AppleStore.csv',
        'appleStore_description.csv',
        max_length=256,
        batch_size=16
    )
    
    # 2. Train model
    model, train_losses, val_metrics = train_bert_sentiment_model(
        train_loader,
        val_loader,
        epochs=4
    )
    
    # 3. Save model
    torch.save(model.state_dict(), 'bert_apple_sentiment.pt')
    
    # 4. Example prediction
    sample_texts = [
        "This app is amazing and works perfectly. I love using it every day!",
        "Terrible app, crashes constantly and drains battery. Do not download."
    ]
    
    predictions = predict_sentiment(model, tokenizer, sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {pred}\n")
