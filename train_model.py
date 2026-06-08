import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'AlWarraq_Final_Version.csv')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_engine_advanced():
    print("Starting Advanced Training (N-Grams Enabled)...")

    # 1.load data
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: File not found at {RAW_DATA_PATH}")
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
   
    df['context'] = df['context'].fillna('')
    
    
    if 'author' in df.columns:
        df.rename(columns={'author': 'authors'}, inplace=True)

    print(f"Loaded {len(df)} books.")

    # 2.TF-IDF 
    print("Vectorizing with N-Grams (1,2) and Min-DF (2)...")
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        
      
        ngram_range=(1, 2),   
        min_df=2,            
        
        max_features=10000,  
        norm='l2'            
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(df['context'])
    except ValueError as e:
        print(f"Error during vectorization: {e}")
        print("   -> Check if 'context' column contains valid text.")
        sys.exit(1)

    # 3. save
    print("Saving advanced models...")
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(tfidf_matrix, os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))
    joblib.dump(df, os.path.join(MODEL_DIR, 'books_processed.pkl'))

    print("-" * 30)
    print("TRAINING SUCCESSFUL!")
    print(f"   - Matrix Shape: {tfidf_matrix.shape}")
    print(f"     (Rows: {tfidf_matrix.shape[0]} books, Cols: {tfidf_matrix.shape[1]} features)")
    print("   - N-gram range: (1, 2)")
    print("   - Min Document Frequency: 2")
    print("-" * 30)

if __name__ == "__main__":
    train_engine_advanced()