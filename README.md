# 🎬 YouTube Trend Analysis & Sentiment Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning system for predicting YouTube video virality and analyzing comment sentiment using DistilBERT, XGBoost, and VADER sentiment analysis.
##Appication link: " https://huggingface.co/spaces/Danielchris145/youtube-trend-sentiment-analysis "
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project combines **YouTube Trend Prediction** and **Sentiment Analysis** into a unified web application. It helps content creators, marketers, and researchers:

- **Predict** whether a video will go viral before publishing
- **Analyze** audience sentiment from YouTube comments in real-time
- **Visualize** engagement patterns and sentiment distributions
- **Understand** what factors contribute to video success

---

## ✨ Features

### 🔮 Viral Prediction
- 34 engineered features for prediction
- XGBoost classifier with regularization
- No data leakage (views excluded from training)
- Cross-validated model performance

### 💬 Sentiment Analysis
- Real-time YouTube comment scraping
- VADER sentiment analysis
- DistilBERT emotion detection
- Interactive Plotly visualizations

### 🌐 Web Interface
- Modern Flask web application
- RESTful API endpoints
- Responsive dashboard
- Dark theme UI

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Index      │  │  Dashboard   │  │   YouTube    │  │    About     │    │
│  │   Page       │  │    Page      │  │   Analyzer   │  │    Page      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (Flask)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │ /api/analyze   │  │ /api/predict   │  │ /api/youtube-analyze       │    │
│  │    -text       │  │                │  │                            │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
│  ┌────────────────┐  ┌────────────────┐                                    │
│  │ /api/dashboard │  │    /health     │                                    │
│  │    -stats      │  │                │                                    │
│  └────────────────┘  └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCESSING LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CustomModelsManager                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │  Emotion    │  │  Sentiment  │  │      Viral Predictor        │  │   │
│  │  │   Model     │  │    Model    │  │       (XGBoost)             │  │   │
│  │  │ (DistilBERT)│  │ (DistilBERT)│  │                             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 YouTubeSentimentAnalyzer Module                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │  Comment    │  │  Sentiment  │  │       Visualizer            │  │   │
│  │  │  Scraper    │  │  Analysis   │  │       (Plotly)              │  │   │
│  │  │ (YouTube API)│ │   (VADER)   │  │                             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐     │
│  │   YouTube API   │  │  CSV Datasets   │  │    Model Files          │     │
│  │   (Comments)    │  │  (10 Countries) │  │    (.pkl, .pth)         │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VIRAL PREDICTION FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
    │  User   │────▶│  Video Data  │────▶│   Feature    │────▶│   XGBoost   │
    │  Input  │     │  (metadata)  │     │  Extraction  │     │   Model     │
    └─────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                                                │                     │
                                                ▼                     ▼
                                         ┌──────────────┐     ┌─────────────┐
                                         │  34 Features │     │  Viral      │
                                         │  Generated   │     │  Prediction │
                                         └──────────────┘     └─────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                      SENTIMENT ANALYSIS FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
    │ YouTube │────▶│   Comment    │────▶│    VADER     │────▶│  Sentiment  │
    │   URL   │     │   Scraper    │     │   Analysis   │     │   Results   │
    └─────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                           │                                        │
                           ▼                                        ▼
                    ┌──────────────┐                         ┌─────────────┐
                    │  YouTube API │                         │   Plotly    │
                    │   v3         │                         │   Charts    │
                    └──────────────┘                         └─────────────┘
```

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.8+, Flask 2.3.3, Flask-CORS |
| **ML/AI** | PyTorch 2.0.1, Transformers (DistilBERT), XGBoost 1.7.6, scikit-learn |
| **NLP** | VADER Sentiment, NLTK, TextBlob |
| **Data** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **API** | YouTube Data API v3, google-api-python-client |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- YouTube Data API key

### Step 1: Clone the Repository
```bash
git clone https://github.com/CHRISDANIEL145/YouTube-trend-analysis-and-sentiment-analysis-.git
cd YouTube-trend-analysis-and-sentiment-analysis-
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure YouTube API Key
Create a `.env` file in the root directory:
```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

Or set it directly in `modules/youtube_sentiment/YoutubeCommentScrapper.py`

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

---

## 📁 Project Structure

```
YouTube-trend-analysis-and-sentiment-analysis/
│
├── 📄 app.py                          # Main Flask application
├── 📄 YouTube_Trend_Prediction.py     # ML training & prediction script
├── 📄 test_model.py                   # Model testing utilities
├── 📄 requirements.txt                # Python dependencies
│
├── 📂 modules/
│   └── 📂 youtube_sentiment/
│       ├── __init__.py                # Module initialization
│       ├── YoutubeCommentScrapper.py  # YouTube API integration
│       ├── sentiment_analysis.py      # VADER sentiment engine
│       └── visualizer.py              # Plotly chart generation
│
├── 📂 model_files/
│   ├── viral_predictor.pkl            # Trained XGBoost model
│   ├── scaler.pkl                     # Feature scaler
│   ├── feature_columns.pkl            # Feature names
│   ├── label_encoders.pkl             # Categorical encoders
│   ├── label_encoders_trend.pkl       # Trend label encoders
│   ├── viral_threshold.pkl            # Viral threshold value
│   ├── model_config.json              # Model configuration
│   ├── emotion_model.pth              # DistilBERT emotion model
│   └── sentiment_model.pth            # DistilBERT sentiment model
│
├── 📂 datasets/
│   ├── USvideos.csv                   # USA trending videos
│   ├── INvideos.csv                   # India trending videos
│   ├── GBvideos.csv                   # UK trending videos
│   ├── CAvideos.csv                   # Canada trending videos
│   ├── DEvideos.csv                   # Germany trending videos
│   ├── FRvideos.csv                   # France trending videos
│   ├── JPvideos.csv                   # Japan trending videos
│   ├── KRvideos.csv                   # South Korea trending videos
│   ├── MXvideos.csv                   # Mexico trending videos
│   ├── RUvideos.csv                   # Russia trending videos
│   └── *_category_id.json             # Category mappings
│
├── 📂 templates/
│   ├── index.html                     # Home page
│   ├── dashboard.html                 # Analytics dashboard
│   ├── youtube_analyzer.html          # YouTube analyzer page
│   ├── youtube_sentiment.html         # Sentiment analysis page
│   └── about.html                     # About page
│
├── 📂 static/
│   ├── style.css                      # Stylesheet
│   └── script.js                      # JavaScript
│
├── 📂 logs/                           # Application logs
│
└── 📊 Output Files
    ├── confusion_matrix.png           # Model confusion matrix
    ├── feature_importance.png         # Feature importance chart
    ├── 1_viral_probability_comparison.png
    ├── 2_engagement_vs_probability.png
    ├── 3_feature_categories.png
    ├── 4_viral_distribution.png
    ├── 5_likes_vs_comments.png
    ├── model_test_results.csv
    └── test_report.txt
```

---

## 🚀 Usage Guide

### 1. Training the Viral Prediction Model

```bash
python YouTube_Trend_Prediction.py
```

Select options:
- Choose country dataset (1-10)
- Select (T)rain or (P)redict mode
- Set sample size (default: 5000)

### 2. Running the Web Application

```bash
python app.py
```

Access the web interface:
- **Home**: `http://localhost:5000/`
- **Dashboard**: `http://localhost:5000/dashboard`
- **YouTube Analyzer**: `http://localhost:5000/youtube-analyzer`
- **About**: `http://localhost:5000/about`

### 3. Using the API

#### Analyze Text Sentiment
```bash
curl -X POST http://localhost:5000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "This video is amazing!"}'
```

#### Predict Viral Potential
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "likes": 15000,
    "comments": 3500,
    "shares": 500,
    "views": 100000,
    "engagement_rate": 3.5,
    "sentiment_score": 0.75
  }'
```

#### Analyze YouTube Video
```bash
curl -X POST http://localhost:5000/api/youtube-analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

---

## 📡 API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/analyze-text` | POST | Analyze text sentiment |
| `/api/predict` | POST | Predict viral potential |
| `/api/youtube-analyze` | POST | Analyze YouTube video |
| `/api/dashboard-stats` | GET | Get dashboard statistics |

---

## 🧠 Machine Learning Pipeline

### Feature Engineering (34 Features)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE CATEGORIES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  ENGAGEMENT (7)     │  │  TEMPORAL (4)       │  │  TEXT/NLP (12)      │ │
│  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤ │
│  │ • likes             │  │ • publish_hour      │  │ • title_length      │ │
│  │ • dislikes          │  │ • publish_day       │  │ • description_length│ │
│  │ • comment_count     │  │ • publish_month     │  │ • tag_count         │ │
│  │ • like_dislike_ratio│  │ • is_weekend        │  │ • title_word_count  │ │
│  │ • likes_per_comment │  │                     │  │ • has_exclamation   │ │
│  │ • dislikes_per_comm │  │                     │  │ • has_question      │ │
│  │ • engagement_score  │  │                     │  │ • has_numbers       │ │
│  └─────────────────────┘  └─────────────────────┘  │ • all_caps_words    │ │
│                                                    │ • title_sentiment_* │ │
│  ┌─────────────────────┐  ┌─────────────────────┐  │ • desc_sentiment_*  │ │
│  │  BERT FEATURES (6)  │  │  COMMENT PROXY (5)  │  └─────────────────────┘ │
│  ├─────────────────────┤  ├─────────────────────┤                          │
│  │ • title_emotion     │  │ • comment_proxy_sent│                          │
│  │ • title_emotion_conf│  │ • estimated_pos_rat │                          │
│  │ • title_sentiment   │  │ • estimated_neg_rat │                          │
│  │ • title_sent_conf   │  │ • has_comments      │                          │
│  │ • desc_emotion      │  │ • comment_engage_lvl│                          │
│  │ • desc_emotion_conf │  │                     │                          │
│  └─────────────────────┘  └─────────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Training Process

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Load       │────▶│   Feature    │────▶│   Train/Test │────▶│   Train      │
│   Dataset    │     │   Extraction │     │   Split      │     │   XGBoost    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  CSV Files           34 Features         80/20 Split          Regularized
  (10 Countries)      Generated           Stratified           L1 + L2
                                                                    │
                                                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Save       │◀────│   Cross      │◀────│   Evaluate   │◀────│   Scale      │
│   Models     │     │   Validate   │     │   Metrics    │     │   Features   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📊 Model Performance

### Viral Prediction Model

| Metric | Score |
|--------|-------|
| Train Accuracy | ~85% |
| Test Accuracy | ~78% |
| ROC-AUC | ~0.82 |
| Precision (Viral) | ~0.75 |
| Recall (Viral) | ~0.70 |
| F1-Score | ~0.72 |

### Sentiment Analysis

| Metric | Score |
|--------|-------|
| Emotion Detection | 92% accuracy |
| Sentiment Analysis | 88% accuracy |
| Processing Speed | ~10ms/comment |

---

## 🖼 Screenshots

### Generated Visualizations

The system generates the following analysis charts:

1. **Viral Probability Comparison** - Compare viral potential across videos
2. **Engagement vs Probability** - Correlation analysis
3. **Feature Categories** - Feature importance by category
4. **Viral Distribution** - Distribution of viral vs non-viral
5. **Likes vs Comments** - Engagement correlation
6. **Confusion Matrix** - Model prediction accuracy
7. **Feature Importance** - Top contributing features

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Engineering Student**  
Karunya University, India  
Specialization: Data Science & Machine Learning

---

## 🙏 Acknowledgments

- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Kaggle YouTube Trending Dataset](https://www.kaggle.com/datasnaek/youtube-new)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

<p align="center">
  Made with ❤️ for YouTube Content Analysis
</p>
