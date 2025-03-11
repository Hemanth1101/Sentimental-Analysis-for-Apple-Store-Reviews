import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# 1. Load and Merge Datasets
def load_and_merge_data(app_file, desc_file):
    """
    Load and merge the Apple Store datasets
    """
    # Load the datasets
    app_data = pd.read_csv(app_file)
    desc_data = pd.read_csv(desc_file)
    
    # Merge datasets on id and track_name for verification
    merged_data = pd.merge(app_data, desc_data, on=['id', 'track_name'])
    
    print(f"Loaded {len(merged_data)} apps with descriptions")
    return merged_data

# 2. Create labels for sentiment analysis
def create_sentiment_labels(df, rating_threshold=3.5):
    """
    Create sentiment labels based on user ratings
    Positive: rating >= threshold
    Negative: rating < threshold
    """
    df['sentiment'] = (df['user_rating'] >= rating_threshold).astype(int)
    
    # Print distribution of sentiment labels
    sentiment_counts = df['sentiment'].value_counts()
    print(f"Positive reviews: {sentiment_counts.get(1, 0)}")
    print(f"Negative reviews: {sentiment_counts.get(0, 0)}")
    
    return df

# 3. Clean and preprocess text
def clean_text(text):
    """
    Clean and preprocess app descriptions
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# 4. Create a PyTorch Dataset for BERT
class AppleStoreDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length=512):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        text = self.descriptions[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Main preprocessing pipeline
def preprocess_data(app_file, desc_file, test_size=0.2, max_length=256, batch_size=16):
    """
    Main preprocessing pipeline for Apple Store data
    """
    # 1. Load and merge data
    df = load_and_merge_data(app_file, desc_file)
    
    # 2. Create sentiment labels
    df = create_sentiment_labels(df)
    
    # 3. Clean descriptions
    print("Cleaning app descriptions...")
    df['cleaned_desc'] = df['app_desc'].apply(clean_text)
    df['cleaned_desc'] = df['cleaned_desc'].apply(remove_stopwords)
    
    # 4. Split data into train and validation sets
    train_df, val_df = train_test_split(
        df[['cleaned_desc', 'sentiment']], 
        test_size=test_size, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # 5. Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 6. Create datasets
    train_dataset = AppleStoreDataset(
        train_df['cleaned_desc'].values,
        train_df['sentiment'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = AppleStoreDataset(
        val_df['cleaned_desc'].values,
        val_df['sentiment'].values,
        tokenizer,
        max_length
    )
    
    # 7. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, tokenizer = preprocess_data(
        'AppleStore.csv',
        'appleStore_description.csv',
        max_length=256,  # Most descriptions are well under this limit
        batch_size=16    # Adjust based on available GPU memory
    )
    
    # Check a sample batch
    sample_batch = next(iter(train_loader))
    print(f"Input shape: {sample_batch['input_ids'].shape}")
    print(f"Attention mask shape: {sample_batch['attention_mask'].shape}")
    print(f"Labels shape: {sample_batch['label'].shape}")
