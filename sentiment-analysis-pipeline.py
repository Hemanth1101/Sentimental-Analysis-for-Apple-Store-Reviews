import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import classification_report

# Import the preprocessing and model modules
from preprocessing import preprocess_data, clean_text, remove_stopwords
from model import BertSentimentClassifier, train_bert_sentiment_model, predict_sentiment

def main():
    parser = argparse.ArgumentParser(description='BERT Sentiment Analysis for Apple Store App Reviews')
    parser.add_argument('--app_data', type=str, default='AppleStore.csv', help='Path to the Apple Store data CSV')
    parser.add_argument('--desc_data', type=str, default='appleStore_description.csv', help='Path to the app descriptions CSV')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for BERT')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and results')
    parser.add_argument('--analyze_genres', action='store_true', help='Analyze sentiment by app genre')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Preprocess data
    print("Preprocessing data...")
    train_loader, val_loader, tokenizer = preprocess_data(
        args.app_data,
        args.desc_data,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # 2. Train model
    print("\nTraining BERT sentiment model...")
    model, train_losses, val_metrics = train_bert_sentiment_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # 3. Save model and tokenizer
    model_path = os.path.join(args.output_dir, 'bert_apple_sentiment.pt')
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
    print(f"Model saved to {model_path}")
    
    # 4. Generate detailed evaluation report
    print("\nGenerating evaluation report...")
    
    # Get validation predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(args.output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # 5. Optional: Analyze sentiment by app genre
    if args.analyze_genres:
        print("\nAnalyzing sentiment by app genre...")
        
        # Load and merge data again to get genre information
        app_data = pd.read_csv(args.app_data)
        desc_data = pd.read_csv(args.desc_data)
        merged_data = pd.merge(app_data, desc_data, on=['id', 'track_name'])
        
        # Clean descriptions
        merged_data['cleaned_desc'] = merged_data['app_desc'].apply(clean_text)
        merged_data['cleaned_desc'] = merged_data['cleaned_desc'].apply(remove_stopwords)
        
        # Group by genre
        genres = merged_data['prime_genre'].unique()
        genre_sentiments = {}
        
        for genre in genres:
            genre_descs = merged_data[merged_data['prime_genre'] == genre]['cleaned_desc'].tolist()
            
            # Skip if no descriptions
            if not genre_descs:
                continue
                
            # Predict sentiment for each genre
            predictions = predict_sentiment(model, tokenizer, genre_descs, device, args.max_length)
            positive_ratio = predictions.count('Positive') / len(predictions)
            genre_sentiments[genre] = positive_ratio
        
        # Plot genre sentiment analysis
        genre_df = pd.DataFrame({
            'Genre': list(genre_sentiments.keys()),
            'Positive Ratio': list(genre_sentiments.values())
        }).sort_values('Positive Ratio', ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.bar(genre_df['Genre'], genre_df['Positive Ratio'], color='skyblue')
        plt.xlabel('App Genre')
        plt.ylabel('Positive Review Ratio')
        plt.title('Sentiment Analysis by App Genre')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        genre_plot_path = os.path.join(args.output_dir, 'genre_sentiment.png')
        plt.savefig(genre_plot_path)
        print(f"Genre sentiment analysis saved to {genre_plot_path}")
    
    # 6. Final information
    print("\nSentiment analysis pipeline completed successfully!")
    print(f"Results saved to {args.output_dir}/")
    print(f"Final validation accuracy: {val_metrics[-1]['accuracy']:.4f}")
    print(f"Final validation F1 score: {val_metrics[-1]['f1']:.4f}")

if __name__ == "__main__":
    main()
