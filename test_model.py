"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          TEST MODEL - VIRAL PREDICTION SYSTEM - VERSION 3.0              ║
║                                                                           ║
║  Complete testing suite for YouTube Trend Prediction model               ║
║  Tests the trained XGBoost classifier on various scenarios               ║
║                                                                           ║
║  Features:                                                                ║
║  ✅ 8 comprehensive test scenarios                                        ║
║  ✅ Model evaluation metrics                                              ║
║  ✅ Feature analysis                                                      ║
║  ✅ CSV results export                                                    ║
║  ✅ Detailed reporting                                                    ║
║  ✅ Error handling                                                        ║
║                                                                           ║
║  Author: Engineering Student                                   ║
║  University: Karunya University                                           ║
║  Date: November 5, 2025                                                   ║
║  Version: 3.0                                                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================
# STARTUP BANNER
# ============================================

print("\n" + "="*80)
print("╔" + "="*78 + "╗")
print("║" + " "*78 + "║")
print("║" + "🧪 TEST MODEL - VIRAL PREDICTION SYSTEM v3.0".center(78) + "║")
print("║" + " "*78 + "║")
print("╚" + "="*78 + "╝")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================
# STEP 1: LOAD MODELS
# ============================================

def load_trained_models():
    """Load all trained models from model_files directory"""
    
    print("📦 LOADING TRAINED MODELS")
    print("="*80)
    
    model_dir = Path("model_files")
    
    # Check if model_files exist
    if not model_dir.exists():
        print("❌ ERROR: model_files directory not found!")
        print("   Please train the model first using YouTube_Trend_Prediction.py")
        exit(1)
    
    try:
        # Load models
        print("\n🔍 Loading models...")
        
        with open(model_dir / "viral_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        print("   ✅ viral_predictor.pkl (XGBoost model)")
        
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("   ✅ scaler.pkl (StandardScaler)")
        
        with open(model_dir / "label_encoders_trend.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        print("   ✅ label_encoders_trend.pkl (Category encoders)")
        
        with open(model_dir / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        print(f"   ✅ feature_columns.pkl ({len(feature_columns)} features)")
        
        with open(model_dir / "viral_threshold.pkl", "rb") as f:
            viral_threshold = pickle.load(f)
        print(f"   ✅ viral_threshold.pkl ({viral_threshold:,.0f} views)")
        
        print("\n✅ All models loaded successfully!")
        return model, scaler, label_encoders, feature_columns, viral_threshold
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing model file: {e}")
        print("   Please train the model first!")
        exit(1)
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        exit(1)

# ============================================
# STEP 2: DEFINE TEST SCENARIOS
# ============================================

def get_test_scenarios():
    """Define 8 comprehensive test scenarios"""
    
    scenarios = [
        {
            "name": "Low Engagement Video",
            "description": "Video with minimal engagement",
            "data": {
                'likes': 100,
                'dislikes': 10,
                'comment_count': 50,
                'like_dislike_ratio': 10.0,
                'likes_per_comment': 2.0,
                'dislikes_per_comment': 0.2,
                'engagement_score': 1.6,
                'publish_hour': 14,
                'publish_day': 3,
                'publish_month': 6,
                'is_weekend': 0,
                'category_id': 28,
                'title_length': 45,
                'description_length': 200,
                'tag_count': 3,
                'title_word_count': 8,
                'has_exclamation': 0,
                'has_question': 0,
                'has_numbers': 0,
                'all_caps_words': 0,
                'title_sentiment_compound': 0.1,
                'title_sentiment_pos': 0.1,
                'title_sentiment_neg': 0.0,
                'desc_sentiment_compound': 0.05,
                'comment_proxy_sentiment': 0.0,
                'estimated_positive_ratio': 0.33,
                'estimated_negative_ratio': 0.33,
                'has_comments': 1,
                'comment_engagement_level': 3.9,
                'title_emotion': 'neutral',
                'title_emotion_confidence': 0.5,
                'title_sentiment_bert': 'neutral',
                'title_sentiment_confidence': 0.4,
                'desc_emotion': 'neutral',
                'desc_emotion_confidence': 0.5,
            }
        },
        {
            "name": "High Engagement Video",
            "description": "Video with exceptional engagement",
            "data": {
                'likes': 50000,
                'dislikes': 500,
                'comment_count': 5000,
                'like_dislike_ratio': 100.0,
                'likes_per_comment': 10.0,
                'dislikes_per_comment': 0.1,
                'engagement_score': 55.0,
                'publish_hour': 18,
                'publish_day': 5,
                'publish_month': 7,
                'is_weekend': 1,
                'category_id': 28,
                'title_length': 65,
                'description_length': 800,
                'tag_count': 8,
                'title_word_count': 12,
                'has_exclamation': 1,
                'has_question': 0,
                'has_numbers': 1,
                'all_caps_words': 2,
                'title_sentiment_compound': 0.7,
                'title_sentiment_pos': 0.3,
                'title_sentiment_neg': 0.05,
                'desc_sentiment_compound': 0.6,
                'comment_proxy_sentiment': 0.5,
                'estimated_positive_ratio': 0.6,
                'estimated_negative_ratio': 0.2,
                'has_comments': 1,
                'comment_engagement_level': 8.5,
                'title_emotion': 'happy',
                'title_emotion_confidence': 0.9,
                'title_sentiment_bert': 'positive',
                'title_sentiment_confidence': 0.85,
                'desc_emotion': 'happy',
                'desc_emotion_confidence': 0.8,
            }
        },
        {
            "name": "Controversial Video",
            "description": "Video with controversial/mixed reactions",
            "data": {
                'likes': 20000,
                'dislikes': 15000,
                'comment_count': 8000,
                'like_dislike_ratio': 1.33,
                'likes_per_comment': 2.5,
                'dislikes_per_comment': 1.88,
                'engagement_score': 43.0,
                'publish_hour': 12,
                'publish_day': 2,
                'publish_month': 5,
                'is_weekend': 0,
                'category_id': 25,
                'title_length': 55,
                'description_length': 600,
                'tag_count': 5,
                'title_word_count': 10,
                'has_exclamation': 1,
                'has_question': 1,
                'has_numbers': 0,
                'all_caps_words': 1,
                'title_sentiment_compound': -0.3,
                'title_sentiment_pos': 0.15,
                'title_sentiment_neg': 0.2,
                'desc_sentiment_compound': -0.2,
                'comment_proxy_sentiment': -0.1,
                'estimated_positive_ratio': 0.4,
                'estimated_negative_ratio': 0.4,
                'has_comments': 1,
                'comment_engagement_level': 8.9,
                'title_emotion': 'stress',
                'title_emotion_confidence': 0.7,
                'title_sentiment_bert': 'negative',
                'title_sentiment_confidence': 0.6,
                'desc_emotion': 'anxiety',
                'desc_emotion_confidence': 0.65,
            }
        },
        {
            "name": "Mixed Response Video",
            "description": "Video with balanced engagement",
            "data": {
                'likes': 30000,
                'dislikes': 3000,
                'comment_count': 4000,
                'like_dislike_ratio': 10.0,
                'likes_per_comment': 7.5,
                'dislikes_per_comment': 0.75,
                'engagement_score': 37.0,
                'publish_hour': 16,
                'publish_day': 4,
                'publish_month': 8,
                'is_weekend': 0,
                'category_id': 28,
                'title_length': 50,
                'description_length': 500,
                'tag_count': 6,
                'title_word_count': 9,
                'has_exclamation': 1,
                'has_question': 0,
                'has_numbers': 1,
                'all_caps_words': 1,
                'title_sentiment_compound': 0.4,
                'title_sentiment_pos': 0.2,
                'title_sentiment_neg': 0.08,
                'desc_sentiment_compound': 0.35,
                'comment_proxy_sentiment': 0.25,
                'estimated_positive_ratio': 0.5,
                'estimated_negative_ratio': 0.25,
                'has_comments': 1,
                'comment_engagement_level': 8.3,
                'title_emotion': 'happy',
                'title_emotion_confidence': 0.75,
                'title_sentiment_bert': 'positive',
                'title_sentiment_confidence': 0.7,
                'desc_emotion': 'happy',
                'desc_emotion_confidence': 0.7,
            }
        },
        {
            "name": "Viral Potential Video",
            "description": "Video with strong viral indicators",
            "data": {
                'likes': 100000,
                'dislikes': 1000,
                'comment_count': 15000,
                'like_dislike_ratio': 100.0,
                'likes_per_comment': 6.67,
                'dislikes_per_comment': 0.067,
                'engagement_score': 116.0,
                'publish_hour': 19,
                'publish_day': 6,
                'publish_month': 9,
                'is_weekend': 1,
                'category_id': 24,
                'title_length': 70,
                'description_length': 1000,
                'tag_count': 10,
                'title_word_count': 13,
                'has_exclamation': 2,
                'has_question': 0,
                'has_numbers': 1,
                'all_caps_words': 3,
                'title_sentiment_compound': 0.85,
                'title_sentiment_pos': 0.4,
                'title_sentiment_neg': 0.02,
                'desc_sentiment_compound': 0.8,
                'comment_proxy_sentiment': 0.7,
                'estimated_positive_ratio': 0.7,
                'estimated_negative_ratio': 0.1,
                'has_comments': 1,
                'comment_engagement_level': 9.6,
                'title_emotion': 'happy',
                'title_emotion_confidence': 0.95,
                'title_sentiment_bert': 'positive',
                'title_sentiment_confidence': 0.9,
                'desc_emotion': 'happy',
                'desc_emotion_confidence': 0.88,
            }
        },
        {
            "name": "Average Video",
            "description": "Video with average engagement",
            "data": {
                'likes': 5000,
                'dislikes': 200,
                'comment_count': 500,
                'like_dislike_ratio': 25.0,
                'likes_per_comment': 10.0,
                'dislikes_per_comment': 0.4,
                'engagement_score': 5.7,
                'publish_hour': 10,
                'publish_day': 1,
                'publish_month': 3,
                'is_weekend': 0,
                'category_id': 20,
                'title_length': 40,
                'description_length': 300,
                'tag_count': 4,
                'title_word_count': 7,
                'has_exclamation': 0,
                'has_question': 0,
                'has_numbers': 0,
                'all_caps_words': 0,
                'title_sentiment_compound': 0.2,
                'title_sentiment_pos': 0.12,
                'title_sentiment_neg': 0.05,
                'desc_sentiment_compound': 0.15,
                'comment_proxy_sentiment': 0.1,
                'estimated_positive_ratio': 0.4,
                'estimated_negative_ratio': 0.25,
                'has_comments': 1,
                'comment_engagement_level': 6.2,
                'title_emotion': 'neutral',
                'title_emotion_confidence': 0.6,
                'title_sentiment_bert': 'neutral',
                'title_sentiment_confidence': 0.55,
                'desc_emotion': 'neutral',
                'desc_emotion_confidence': 0.6,
            }
        },
        {
            "name": "Trending Video",
            "description": "Video with trending keywords and high sentiment",
            "data": {
                'likes': 200000,
                'dislikes': 2000,
                'comment_count': 30000,
                'like_dislike_ratio': 100.0,
                'likes_per_comment': 6.67,
                'dislikes_per_comment': 0.067,
                'engagement_score': 232.0,
                'publish_hour': 20,
                'publish_day': 0,
                'publish_month': 11,
                'is_weekend': 1,
                'category_id': 26,
                'title_length': 80,
                'description_length': 1200,
                'tag_count': 12,
                'title_word_count': 14,
                'has_exclamation': 3,
                'has_question': 1,
                'has_numbers': 2,
                'all_caps_words': 4,
                'title_sentiment_compound': 0.9,
                'title_sentiment_pos': 0.45,
                'title_sentiment_neg': 0.01,
                'desc_sentiment_compound': 0.85,
                'comment_proxy_sentiment': 0.8,
                'estimated_positive_ratio': 0.75,
                'estimated_negative_ratio': 0.08,
                'has_comments': 1,
                'comment_engagement_level': 10.3,
                'title_emotion': 'happy',
                'title_emotion_confidence': 0.98,
                'title_sentiment_bert': 'positive',
                'title_sentiment_confidence': 0.95,
                'desc_emotion': 'happy',
                'desc_emotion_confidence': 0.92,
            }
        },
        {
            "name": "Emerging Video",
            "description": "Video with emerging engagement patterns",
            "data": {
                'likes': 15000,
                'dislikes': 500,
                'comment_count': 2000,
                'like_dislike_ratio': 30.0,
                'likes_per_comment': 7.5,
                'dislikes_per_comment': 0.25,
                'engagement_score': 17.5,
                'publish_hour': 15,
                'publish_day': 3,
                'publish_month': 10,
                'is_weekend': 0,
                'category_id': 22,
                'title_length': 55,
                'description_length': 400,
                'tag_count': 5,
                'title_word_count': 10,
                'has_exclamation': 1,
                'has_question': 0,
                'has_numbers': 1,
                'all_caps_words': 1,
                'title_sentiment_compound': 0.5,
                'title_sentiment_pos': 0.25,
                'title_sentiment_neg': 0.05,
                'desc_sentiment_compound': 0.45,
                'comment_proxy_sentiment': 0.35,
                'estimated_positive_ratio': 0.55,
                'estimated_negative_ratio': 0.2,
                'has_comments': 1,
                'comment_engagement_level': 7.6,
                'title_emotion': 'happy',
                'title_emotion_confidence': 0.8,
                'title_sentiment_bert': 'positive',
                'title_sentiment_confidence': 0.75,
                'desc_emotion': 'happy',
                'desc_emotion_confidence': 0.75,
            }
        }
    ]
    
    return scenarios

# ============================================
# STEP 3: MAKE PREDICTIONS
# ============================================

def make_predictions(model, scaler, label_encoders, feature_columns, scenarios):
    """Make predictions for all test scenarios"""
    
    print("\n" + "="*80)
    print("🔮 MAKING PREDICTIONS")
    print("="*80)
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Prepare features
        features_df = pd.DataFrame([scenario['data']])
        
        # Encode categorical
        for col, le in label_encoders.items():
            if col in features_df.columns:
                try:
                    features_df[col] = le.transform(features_df[col].astype(str))
                except:
                    features_df[col] = 0
        
        # Ensure all columns present
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[feature_columns]
        
        # Scale
        features_scaled = scaler.transform(features_df)
        
        # Predict
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0, 1]
        
        # Confidence
        confidence = 'High' if abs(prob - 0.5) > 0.3 else ('Medium' if abs(prob - 0.5) > 0.15 else 'Low')
        
        print(f"   ✅ Prediction: {'VIRAL' if pred == 1 else 'NOT VIRAL'}")
        print(f"   📊 Probability: {prob:.1%}")
        print(f"   🎯 Confidence: {confidence}")
        
        results.append({
            'Scenario': scenario['name'],
            'Description': scenario['description'],
            'Likes': scenario['data']['likes'],
            'Dislikes': scenario['data']['dislikes'],
            'Comments': scenario['data']['comment_count'],
            'L/D Ratio': scenario['data']['like_dislike_ratio'],
            'Engagement': scenario['data']['engagement_score'],
            'Viral': 'Yes' if pred == 1 else 'No',
            'Probability': f"{prob:.1%}",
            'Confidence': confidence
        })
    
    return results

# ============================================
# STEP 4: SAVE RESULTS
# ============================================

def save_results(results):
    """Save test results to CSV"""
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    filename = f"model_test_results.csv"
    df_results.to_csv(filename, index=False)
    
    print(f"\n✅ Results saved to: {filename}")
    print(f"   Total scenarios tested: {len(results)}")
    print(f"   Viral predictions: {sum(1 for r in results if r['Viral'] == 'Yes')}")
    print(f"   Non-viral predictions: {sum(1 for r in results if r['Viral'] == 'No')}")
    
    return df_results

# ============================================
# STEP 5: DISPLAY RESULTS TABLE
# ============================================

def display_results_table(results_df):
    """Display results in formatted table"""
    
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("📈 STATISTICS")
    print("="*80)
    
    viral_count = sum(1 for r in results_df['Viral'] if r == 'Yes')
    print(f"\n✅ Viral Predictions: {viral_count}/{len(results_df)}")
    print(f"❌ Non-Viral Predictions: {len(results_df) - viral_count}/{len(results_df)}")
    
    print(f"\n🎯 Average Metrics:")
    
    results_list = []
    for _, row in results_df.iterrows():
        results_list.append({
            'Likes': int(row['Likes']),
            'Comments': int(row['Comments']),
            'Engagement': float(str(row['Engagement']).replace(',', '')),
            'Probability': float(row['Probability'].strip('%')) / 100
        })
    
    avg_likes = np.mean([r['Likes'] for r in results_list])
    avg_comments = np.mean([r['Comments'] for r in results_list])
    avg_engagement = np.mean([r['Engagement'] for r in results_list])
    avg_probability = np.mean([r['Probability'] for r in results_list])
    
    print(f"   Average Likes: {avg_likes:,.0f}")
    print(f"   Average Comments: {avg_comments:,.0f}")
    print(f"   Average Engagement Score: {avg_engagement:.2f}")
    print(f"   Average Viral Probability: {avg_probability:.1%}")

# ============================================
# STEP 6: DETAILED ANALYSIS
# ============================================

def detailed_analysis(results_df):
    """Provide detailed analysis of test results"""
    
    print("\n" + "="*80)
    print("🔬 DETAILED ANALYSIS")
    print("="*80)
    
    print("\n✅ VIRAL SCENARIOS:")
    viral_scenarios = results_df[results_df['Viral'] == 'Yes']
    for _, row in viral_scenarios.iterrows():
        print(f"\n   📹 {row['Scenario']}")
        print(f"      Likes: {row['Likes']:,} | Comments: {row['Comments']:,}")
        print(f"      Probability: {row['Probability']} | Confidence: {row['Confidence']}")
    
    print("\n\n❌ NON-VIRAL SCENARIOS:")
    non_viral_scenarios = results_df[results_df['Viral'] == 'No']
    for _, row in non_viral_scenarios.iterrows():
        print(f"\n   📹 {row['Scenario']}")
        print(f"      Likes: {row['Likes']:,} | Comments: {row['Comments']:,}")
        print(f"      Probability: {row['Probability']} | Confidence: {row['Confidence']}")
    
    print("\n" + "="*80)
    print("🎯 CONFIDENCE DISTRIBUTION")
    print("="*80)
    
    high_conf = len(results_df[results_df['Confidence'] == 'High'])
    med_conf = len(results_df[results_df['Confidence'] == 'Medium'])
    low_conf = len(results_df[results_df['Confidence'] == 'Low'])
    
    print(f"\n   🟢 High Confidence:   {high_conf} scenarios ({100*high_conf/len(results_df):.1f}%)")
    print(f"   🟡 Medium Confidence: {med_conf} scenarios ({100*med_conf/len(results_df):.1f}%)")
    print(f"   🔴 Low Confidence:    {low_conf} scenarios ({100*low_conf/len(results_df):.1f}%)")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution"""
    
    # Load models
    model, scaler, label_encoders, feature_columns, viral_threshold = load_trained_models()
    
    print(f"\n🎯 Viral Threshold: {viral_threshold:,.0f} views")
    print(f"📊 Total Features: {len(feature_columns)}")
    
    # Get test scenarios
    print("\n" + "="*80)
    print("📋 TEST SCENARIOS")
    print("="*80)
    
    scenarios = get_test_scenarios()
    print(f"\n✅ {len(scenarios)} test scenarios loaded")
    for i, s in enumerate(scenarios, 1):
        print(f"   {i}. {s['name']}")
    
    # Make predictions
    results = make_predictions(model, scaler, label_encoders, feature_columns, scenarios)
    
    # Save results
    results_df = save_results(results)
    
    # Display results
    display_results_table(results_df)
    
    # Detailed analysis
    detailed_analysis(results_df)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print(f"\n📊 Results saved to: model_test_results.csv")
    print(f"📈 Use these results in the web dashboard!")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing cancelled by user")
        exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
