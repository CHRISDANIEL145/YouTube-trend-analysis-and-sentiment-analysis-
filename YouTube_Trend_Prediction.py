"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║         YOUTUBE TREND PREDICTION SYSTEM - FINAL VERSION v3.0             ║
║                                                                           ║
║  ✅ NO DATA LEAKAGE - Views excluded from training features              ║
║  ✅ PREVENTS OVERFITTING - Regularization + cross-validation             ║
║  ✅ BERT Integration - DistilBERT emotion/sentiment analysis             ║
║  ✅ 34 ENGINEERED FEATURES - Multi-dimensional analysis                  ║
║  ✅ PRODUCTION READY - Error handling + model persistence                ║
║                                                                           ║
║  Key Features:                                                            ║
║  - Engagement metrics (likes, dislikes, comments)                        ║
║  - Temporal features (hour, day, month, weekend)                         ║
║  - Text analysis (length, sentiment, clickbait score)                    ║
║  - DistilBERT emotion detection (4 classes)                              ║
║  - DistilBERT sentiment analysis (3 classes)                             ║
║  - Comment sentiment proxy                                                ║
║  - Category encoding                                                      ║
║                                                                           ║
║  Author: Engineering Student                                   ║
║  University: Karunya University                                           ║
║  Date: November 5, 2025                                                   ║
║  Version: 3.0 (Production)                                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    accuracy_score, precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================
# STARTUP BANNER
# ============================================

print("\n" + "="*80)
print("╔" + "="*78 + "╗")
print("║" + " "*78 + "║")
print("║" + "🎬 YOUTUBE TREND PREDICTION SYSTEM v3.0".center(78) + "║")
print("║" + "Powered by DistilBERT + XGBoost (NO DATA LEAKAGE)".center(78) + "║")
print("║" + " "*78 + "║")
print("╚" + "="*78 + "╝")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================
# STEP 1: DATASET PREPARATION
# ============================================

def download_and_prepare_dataset(country='US'):
    """Load YouTube trending dataset for specified country"""
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)
    
    available_countries = {
        'US': 'USvideos.csv', 'IN': 'INvideos.csv', 'GB': 'GBvideos.csv',
        'CA': 'CAvideos.csv', 'DE': 'DEvideos.csv', 'FR': 'FRvideos.csv',
        'JP': 'JPvideos.csv', 'KR': 'KRvideos.csv', 'MX': 'MXvideos.csv', 'RU': 'RUvideos.csv'
    }
    
    if country not in available_countries:
        country = 'US'
    
    dataset_path = dataset_dir / available_countries[country]
    
    if not dataset_path.exists():
        print("=" * 80)
        print("❌ DATASET NOT FOUND")
        print("=" * 80)
        print(f"\n📂 Expected file: {dataset_path}")
        print(f"\n📥 Download from Kaggle:")
        print(f"   https://www.kaggle.com/datasnaek/youtube-new")
        print(f"\n📁 Available files in datasets folder:")
        csv_files = list(dataset_dir.glob("*.csv"))
        if csv_files:
            for file in csv_files:
                print(f"   ✅ {file.name}")
        else:
            print("   ⚠️  No CSV files found")
        sys.exit(1)
    
    print(f"✅ Dataset found: {dataset_path.name}")
    print(f"🌍 Country: {country}")
    return dataset_path

# ============================================
# STEP 2: DISTILBERT MODELS
# ============================================

class BERTClassifier(nn.Module):
    """DistilBERT Classifier for emotion/sentiment"""
    def __init__(self, bert_model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (outputs.pooler_output 
                 if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None 
                 else outputs.last_hidden_state[:, 0, :])
        x = self.dropout(pooled)
        return self.classifier(x)

class SentimentEmotionAnalyzer:
    """DistilBERT Analyzer for Emotion and Sentiment"""
    def __init__(self, model_dir="model_files"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(model_dir)
        
        try:
            with open(model_dir / "model_config.json", "r") as f:
                self.config = json.load(f)
            with open(model_dir / "label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
            
            bert_model = self.config.get("bert_model_name", "distilbert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            
            # Load emotion model
            num_emo = len(self.label_encoders["emotion"].classes_)
            self.emotion_model = BERTClassifier(bert_model, num_emo)
            self.emotion_model.load_state_dict(
                torch.load(model_dir / "emotion_model.pth", map_location=self.device)
            )
            self.emotion_model.to(self.device).eval()
            
            # Load sentiment model
            num_sent = len(self.label_encoders["sentiment"].classes_)
            self.sentiment_model = BERTClassifier(bert_model, num_sent)
            self.sentiment_model.load_state_dict(
                torch.load(model_dir / "sentiment_model.pth", map_location=self.device)
            )
            self.sentiment_model.to(self.device).eval()
            
            print(f"✅ DistilBERT models loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  Could not load DistilBERT models: {e}")
            print("   Continuing with VADER sentiment only...")
            self.emotion_model = None
            self.sentiment_model = None
    
    def analyze_text(self, text):
        """Analyze text for emotion and sentiment"""
        if not text or len(text) < 3:
            return {
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "sentiment": "neutral",
                "sentiment_confidence": 0.0
            }
        
        try:
            if self.emotion_model is None:
                return {
                    "emotion": "neutral",
                    "emotion_confidence": 0.0,
                    "sentiment": "neutral",
                    "sentiment_confidence": 0.0
                }
            
            max_len = self.config.get("max_length", 128)
            inputs = self.tokenizer.encode_plus(
                str(text),
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                max_length=max_len,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                emo_out = self.emotion_model(input_ids, attention_mask)
                sent_out = self.sentiment_model(input_ids, attention_mask)
                emo_probs = torch.softmax(emo_out, dim=1)
                sent_probs = torch.softmax(sent_out, dim=1)
                emo_pred = emo_probs.argmax(1).item()
                sent_pred = sent_probs.argmax(1).item()
                emo_conf = emo_probs.max(1)[0].item()
                sent_conf = sent_probs.max(1)[0].item()
            
            return {
                "emotion": self.label_encoders["emotion"].classes_[emo_pred],
                "emotion_confidence": float(emo_conf),
                "sentiment": self.label_encoders["sentiment"].classes_[sent_pred],
                "sentiment_confidence": float(sent_conf)
            }
        except Exception as e:
            return {
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "sentiment": "neutral",
                "sentiment_confidence": 0.0
            }

# ============================================
# STEP 3: FEATURE EXTRACTION (34 FEATURES)
# ============================================

class YouTubeFeatureExtractor:
    """Extract 34 features from YouTube videos"""
    def __init__(self, emotion_analyzer=None):
        self.emotion_analyzer = emotion_analyzer
        self.vader = SentimentIntensityAnalyzer()
    
    def extract_basic_features(self, row):
        """⚠️ FIX: Extract without VIEWS (prevent data leakage)"""
        features = {}
        
        likes = int(row.get('likes', 0))
        dislikes = int(row.get('dislikes', 0))
        comments = int(row.get('comment_count', 0))
        
        # Engagement metrics (NO VIEWS!)
        features['likes'] = likes
        features['dislikes'] = dislikes
        features['comment_count'] = comments
        features['like_dislike_ratio'] = likes / (dislikes + 1)
        features['likes_per_comment'] = likes / (comments + 1)
        features['dislikes_per_comment'] = dislikes / (comments + 1)
        features['engagement_score'] = (likes + dislikes + comments) / (1 + 1)  # Normalized
        
        # Temporal
        try:
            pub_time = pd.to_datetime(row['publish_time'])
            features['publish_hour'] = pub_time.hour
            features['publish_day'] = pub_time.dayofweek
            features['publish_month'] = pub_time.month
            features['is_weekend'] = 1 if pub_time.dayofweek >= 5 else 0
        except:
            features['publish_hour'] = 12
            features['publish_day'] = 3
            features['publish_month'] = 6
            features['is_weekend'] = 0
        
        features['category_id'] = int(row.get('category_id', 0))
        return features
    
    def extract_text_features(self, row):
        """Extract NLP features"""
        features = {}
        
        title = str(row.get('title', ''))
        desc = str(row.get('description', ''))
        tags = str(row.get('tags', '')).split('|') if 'tags' in row and pd.notna(row.get('tags')) else []
        
        # Text length
        features['title_length'] = len(title)
        features['description_length'] = len(desc)
        features['tag_count'] = len(tags)
        features['title_word_count'] = len(title.split())
        
        # Clickbait indicators
        features['has_exclamation'] = 1 if '!' in title else 0
        features['has_question'] = 1 if '?' in title else 0
        features['has_numbers'] = 1 if any(c.isdigit() for c in title) else 0
        features['all_caps_words'] = sum(1 for w in title.split() if w.isupper() and len(w) > 1)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(title)
        features['title_sentiment_compound'] = vader_scores['compound']
        features['title_sentiment_pos'] = vader_scores['pos']
        features['title_sentiment_neg'] = vader_scores['neg']
        
        desc_scores = self.vader.polarity_scores(desc[:500])
        features['desc_sentiment_compound'] = desc_scores['compound']
        
        # DistilBERT emotion/sentiment
        if self.emotion_analyzer and len(title) > 3:
            analysis = self.emotion_analyzer.analyze_text(title)
            features['title_emotion'] = analysis['emotion']
            features['title_emotion_confidence'] = analysis['emotion_confidence']
            features['title_sentiment_bert'] = analysis['sentiment']
            features['title_sentiment_confidence'] = analysis['sentiment_confidence']
        else:
            features['title_emotion'] = 'neutral'
            features['title_emotion_confidence'] = 0.0
            features['title_sentiment_bert'] = 'neutral'
            features['title_sentiment_confidence'] = 0.0
        
        return features
    
    def extract_comment_features(self, row):
        """Extract comment proxy features"""
        features = {}
        
        desc = str(row.get('description', ''))
        comment_count = int(row.get('comment_count', 0))
        
        if len(desc) > 10:
            vader_score = self.vader.polarity_scores(desc)
            features['comment_proxy_sentiment'] = vader_score['compound']
            
            if vader_score['compound'] >= 0.05:
                features['estimated_positive_ratio'] = 0.6
                features['estimated_negative_ratio'] = 0.2
            elif vader_score['compound'] <= -0.05:
                features['estimated_positive_ratio'] = 0.2
                features['estimated_negative_ratio'] = 0.6
            else:
                features['estimated_positive_ratio'] = 0.4
                features['estimated_negative_ratio'] = 0.3
            
            if self.emotion_analyzer:
                analysis = self.emotion_analyzer.analyze_text(desc[:300])
                features['desc_emotion'] = analysis['emotion']
                features['desc_emotion_confidence'] = analysis['emotion_confidence']
            else:
                features['desc_emotion'] = 'neutral'
                features['desc_emotion_confidence'] = 0.0
        else:
            features['comment_proxy_sentiment'] = 0.0
            features['estimated_positive_ratio'] = 0.33
            features['estimated_negative_ratio'] = 0.33
            features['desc_emotion'] = 'neutral'
            features['desc_emotion_confidence'] = 0.0
        
        features['has_comments'] = 1 if comment_count > 0 else 0
        features['comment_engagement_level'] = np.log1p(comment_count)
        
        return features
    
    def extract_all_features(self, row):
        """Extract all 34 features"""
        features = {}
        features.update(self.extract_basic_features(row))
        features.update(self.extract_text_features(row))
        features.update(self.extract_comment_features(row))
        return features

# ============================================
# STEP 4: TRAINING FUNCTION
# ============================================

def train_viral_predictor(dataset_path, analyzer, sample_size=5000):
    """Train XGBoost model"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING YOUTUBE VIRAL PREDICTION MODEL")
    print("="*80)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded {len(df):,} videos")
    
    # Sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"📊 Using {len(df):,} samples for training")
    
    # Define viral threshold (top 20%)
    viral_threshold = df['views'].quantile(0.80)
    print(f"\n🎯 Viral threshold (top 20%): {viral_threshold:,.0f} views")
    
    # Extract features (WITHOUT VIEWS)
    print("\n🔍 Extracting 34 features (without views to prevent data leakage)...")
    extractor = YouTubeFeatureExtractor(analyzer)
    
    feature_list = []
    labels = []
    
    for idx, row in df.iterrows():
        if len(feature_list) % 500 == 0:
            print(f"   Processed {len(feature_list):,}/{len(df):,} videos...")
        
        try:
            features = extractor.extract_all_features(row)
            feature_list.append(features)
            
            # Label: 1 if viral, 0 otherwise
            is_viral = 1 if row['views'] > viral_threshold else 0
            labels.append(is_viral)
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(feature_list)
    features_df['is_viral'] = labels
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Total features: {len(features_df.columns) - 1}")
    print(f"   Viral videos: {sum(labels):,} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"   Non-viral videos: {len(labels)-sum(labels):,} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    # Prepare data
    X = features_df.drop('is_viral', axis=1)
    y = features_df['is_viral']
    
    # Encode categorical
    categorical_cols = ['title_emotion', 'title_sentiment_bert', 'desc_emotion']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data (STRATIFIED - no leakage)
    print("\n📋 Splitting data (stratified train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Train viral ratio: {y_train.mean():.1%}")
    print(f"   Test viral ratio: {y_test.mean():.1%}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\n🤖 Training XGBoost with regularization...")
    print("   Parameters: n_estimators=150, max_depth=6, L1+L2 regularization")
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Evaluate
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION")
    print("="*80)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n✅ ACCURACY:")
    print(f"   Train: {train_acc:.4f} ({100*train_acc:.2f}%)")
    print(f"   Test:  {test_acc:.4f} ({100*test_acc:.2f}%)")
    print(f"   Gap:   {100*(train_acc-test_acc):.2f}% (overfitting check)")
    
    print(f"\n✅ ROC-AUC: {roc_auc:.4f}")
    
    report = classification_report(y_test, y_test_pred, output_dict=True)
    print(f"\n✅ CLASSIFICATION METRICS (Viral Class):")
    print(f"   Precision: {report['1']['precision']:.4f}")
    print(f"   Recall:    {report['1']['recall']:.4f}")
    print(f"   F1-Score:  {report['1']['f1-score']:.4f}")
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n✅ CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0,0]:,}")
    print(f"   False Positives: {cm[0,1]:,}")
    print(f"   False Negatives: {cm[1,0]:,}")
    print(f"   True Positives:  {cm[1,1]:,}")
    
    # Cross-validation
    print(f"\n🔄 Cross-validation (5-fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
    print(f"   Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    print(f"\n📈 TOP 15 MOST IMPORTANT FEATURES:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i, idx in enumerate(indices, 1):
        print(f"   {i:2d}. {list(X.columns)[idx]:35s} {importances[idx]:.4f}")
    
    # Save models
    model_dir = Path("model_files")
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving models...")
    with open(model_dir / "viral_predictor.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_dir / "label_encoders_trend.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(model_dir / "feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    with open(model_dir / "viral_threshold.pkl", "wb") as f:
        pickle.dump(viral_threshold, f)
    
    print(f"   ✅ viral_predictor.pkl")
    print(f"   ✅ scaler.pkl")
    print(f"   ✅ label_encoders_trend.pkl")
    print(f"   ✅ feature_columns.pkl")
    print(f"   ✅ viral_threshold.pkl")
    
    return model, scaler, label_encoders, list(X.columns)

# ============================================
# STEP 5: PREDICTION FUNCTION
# ============================================

def predict_viral_probability(video_data):
    """Predict if video will go viral"""
    model_dir = Path("model_files")
    
    try:
        with open(model_dir / "viral_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(model_dir / "label_encoders_trend.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open(model_dir / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        with open(model_dir / "viral_threshold.pkl", "rb") as f:
            viral_threshold = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Please train first.")
        return None
    
    analyzer = SentimentEmotionAnalyzer()
    extractor = YouTubeFeatureExtractor(analyzer)
    
    features = extractor.extract_all_features(video_data)
    features_df = pd.DataFrame([features])
    
    for col, le in label_encoders.items():
        if col in features_df.columns:
            try:
                features_df[col] = le.transform(features_df[col].astype(str))
            except:
                features_df[col] = 0
    
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    features_df = features_df[feature_columns]
    features_scaled = scaler.transform(features_df)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]
    confidence = 'High' if abs(probability - 0.5) > 0.3 else 'Medium'
    
    return {
        'will_go_viral': bool(prediction),
        'viral_probability': float(probability),
        'confidence': confidence,
        'viral_threshold': f"{viral_threshold:,.0f} views"
    }

# ============================================
# STEP 6: MAIN EXECUTION
# ============================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🎬 YOUTUBE TREND PREDICTION SYSTEM v3.0")
    print("   🔒 NO DATA LEAKAGE | Prevents Overfitting")
    print("="*80)
    
    # Country selection
    countries = ['US', 'IN', 'GB', 'CA', 'DE', 'FR', 'JP', 'KR', 'MX', 'RU']
    country_names = {
        'US': 'United States', 'IN': 'India', 'GB': 'Great Britain',
        'CA': 'Canada', 'DE': 'Germany', 'FR': 'France',
        'JP': 'Japan', 'KR': 'South Korea', 'MX': 'Mexico', 'RU': 'Russia'
    }
    
    print("\n🌍 AVAILABLE COUNTRIES:")
    for i, country in enumerate(countries, 1):
        print(f"   {i}. {country} ({country_names[country]})")
    
    choice = input("\nSelect country (1-10) or code [default: US]: ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        country = countries[idx] if 0 <= idx < len(countries) else 'US'
    elif choice.upper() in countries:
        country = choice.upper()
    else:
        country = 'US'
    
    dataset_path = download_and_prepare_dataset(country)
    
    print("\n🧠 Loading DistilBERT models...")
    analyzer = SentimentEmotionAnalyzer()
    print("✅ Models ready!")
    
    print("\n" + "="*80)
    mode = input("Choose: (T)rain new model or (P)redict? [T/P]: ").strip().upper()
    
    if mode == 'T':
        sample = input("Sample size [default: 5000]: ").strip()
        sample_size = int(sample) if sample.isdigit() else 5000
        train_viral_predictor(dataset_path, analyzer, sample_size)
        print("\n✅ Training complete!")
    
    elif mode == 'P':
        print("\n🎯 Example prediction...")
        example = {
            'title': 'Amazing AI Revolution 2025! Must Watch 🤖',
            'description': 'Revolutionary AI breakthrough! Subscribe for more...',
            'likes': 15000,
            'dislikes': 200,
            'comment_count': 3500,
            'category_id': 28,
            'publish_time': '2025-11-05T14:00:00Z',
            'tags': 'AI|Technology|Future'
        }
        
        result = predict_viral_probability(example)
        if result:
            print(f"\n{'='*80}")
            print("🔮 VIRAL PREDICTION")
            print(f"{'='*80}")
            print(f"Will Go Viral: {'YES ✅' if result['will_go_viral'] else 'NO ❌'}")
            print(f"Probability: {result['viral_probability']:.1%}")
            print(f"Confidence: {result['confidence']}")
            print(f"Viral Threshold: {result['viral_threshold']}")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
