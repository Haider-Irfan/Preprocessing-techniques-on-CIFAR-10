# GPT-2 Sentiment Analysis using Fine-Tuning

## 📌 Overview
This project fine-tunes a GPT-2 (DistilGPT-2) language model for sentiment analysis on real-world Twitter data. The model learns to generate sentiment labels (Positive, Neutral, Negative) based on input sentences using a prompt-based learning approach.

## Objective
To explore the use of generative language models (GPT-2) for sentiment classification by framing the task as a text-generation problem rather than a traditional classifier.

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- GPT-2 (DistilGPT-2)
- NLTK
- Scikit-learn
- Google Colab (GPU)

## Dataset
- **Dataset:** TweetEval – Sentiment Analysis
- **Classes:** Positive, Neutral, Negative
- **Samples Used:** 15,000 tweets
- **Source:** Hugging Face Datasets

## Methodology
1. Text preprocessing (cleaning, tokenization, stopword removal, stemming)
2. Prompt-based formatting:
3. Fine-tuning DistilGPT-2 using causal language modeling
4. Inference via text generation
5. Evaluation using validation accuracy

## Results
- Validation Accuracy (sample-based): ~XX%
- Model successfully generates correct sentiment labels for unseen sentences

## How to Run

### Install dependencies
```bash

pip install -r requirements.txt

Train the model
Run the notebook:

notebooks/gpt2-training-sentiment.ipynb

Inference Example
predict_sentiment("I really enjoyed this product!")

