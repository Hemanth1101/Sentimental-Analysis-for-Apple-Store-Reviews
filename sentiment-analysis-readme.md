# BERT Sentiment Analysis for Apple Store App Reviews

This project implements an end-to-end BERT-based sentiment analysis pipeline for Apple Store app reviews. The analysis uses app descriptions and user ratings to train a sentiment classifier that can predict whether an app will receive positive or negative reviews based on its description..

## Project Structure

- `preprocessing.py`: Data loading, cleaning, and preparation for BERT
- `model.py`: BERT sentiment classifier implementation and training logic
- `pipeline.py`: End-to-end pipeline combining preprocessing and model training
- `requirements.txt`: Required Python packages

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.0+
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- tqdm

Install the required packages:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

## Quick Start

1. Prepare your data files:
   - `AppleStore.csv`: Contains app information and ratings
   - `appleStore_description.csv`: Contains app descriptions

2. Run the end-to-end pipeline:

```bash
python pipeline.py
```

This will:
- Preprocess the data
- Train a BERT sentiment classifier
- Evaluate the model
- Save the trained model and results to the `output` directory

## Advanced Usage

The pipeline script supports several command-line arguments:

```bash
python pipeline.py --app_data AppleStore.csv --desc_data appleStore_description.csv --epochs 4 --batch_size 16 --learning_rate 2e-5 --max_length 256 --output_dir output --analyze_genres
```

Arguments:
- `--app_data`: Path to the Apple Store data CSV (default: 'AppleStore.csv')
- `--desc_data`: Path to the app descriptions CSV (default: 'appleStore_description.csv')
- `--epochs`: Number of training epochs (default: 4)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length for BERT (default: 256)
- `--output_dir`: Directory to save model and results (default: 'output')
- `--analyze_genres`: Enable genre-based sentiment analysis (optional flag)

## Data Preprocessing

The preprocessing module performs the following steps:
1. Loads and merges the Apple Store datasets
2. Creates sentiment labels based on user ratings
3. Cleans and preprocesses app descriptions
4. Removes stopwords
5. Creates PyTorch datasets for BERT
6. Splits data into training and validation sets

## Model Architecture

The sentiment analysis model uses a pre-trained BERT model with a classification head:
- Base model: bert-base-uncased
- Dropout layer with 0.1 rate
- Linear classification layer for binary sentiment

## Using the Trained Model

After training, you can use the model for sentiment prediction:

```python
from transformers import BertTokenizer
from model import BertSentimentClassifier, predict_sentiment
import torch

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertSentimentClassifier()
model.load_state_dict(torch.load('output
