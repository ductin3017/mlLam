import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def prepare_data(file_path):
    """
    Chuẩn bị dữ liệu cho classification
    """
    df = pd.read_csv(file_path)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['SeasonPreference'] = le.fit_transform(df['SeasonPreference'])
    df['RecommendationType'] = le.fit_transform(df['RecommendationType'])
    
    # Tách features và target
    X = df[['UserAge', 'Budget', 'Distance', 'InterestScore', 'TravelFrequency', 'SeasonPreference']]
    y = df['RecommendationType']
    
    # Chia train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def create_hybrid_ann_svm(input_shape, num_classes):
    """
    Tạo mô hình hybrid ANN-SVM
    ANN được sử dụng để trích xuất features, SVM để phân loại
    """
    # Feature extraction layers
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    features = Dropout(0.2)(x)
    
    # Tạo model trích xuất features
    feature_extractor = Model(inputs=inputs, outputs=features)
    
    return feature_extractor

def evaluate_hybrid_ann_svm(X_train, X_test, y_train, y_test):
    """
    Đánh giá mô hình hybrid ANN-SVM
    """
    # Tạo feature extractor
    feature_extractor = create_hybrid_ann_svm(X_train.shape[1], len(np.unique(y_train)))
    
    # Train feature extractor
    feature_extractor.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    feature_extractor.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Trích xuất features
    train_features = feature_extractor.predict(X_train)
    test_features = feature_extractor.predict(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(train_features, y_train)
    
    # Predict
    y_pred = svm.predict(test_features)
    
    # Tính metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # In kết quả
    print("\nResults for Hybrid ANN-SVM:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Hybrid ANN-SVM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return {
        'model': 'Hybrid ANN-SVM',
        'f1_score': f1,
        'recall': recall
    }

def evaluate_hybrid_adaboost_svm(X_train, X_test, y_train, y_test):
    """
    Đánh giá mô hình hybrid AdaBoost-SVM
    """
    # Train AdaBoost để trích xuất features
    base_estimator = DecisionTreeClassifier(max_depth=3)
    ada = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=0.5
    )
    ada.fit(X_train, y_train)
    
    # Sử dụng predict_proba như features cho SVM
    train_features = ada.predict_proba(X_train)
    test_features = ada.predict_proba(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(train_features, y_train)
    
    # Predict
    y_pred = svm.predict(test_features)
    
    # Tính metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # In kết quả
    print("\nResults for Hybrid AdaBoost-SVM:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Hybrid AdaBoost-SVM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return {
        'model': 'Hybrid AdaBoost-SVM',
        'f1_score': f1,
        'recall': recall
    }

def visualize_comparison(results):
    """
    So sánh kết quả của các mô hình hybrid
    """
    df_results = pd.DataFrame(results)
    
    # Plot metrics comparison
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(results))
    width = 0.35
    
    plt.bar(x - width/2, df_results['f1_score'], width, label='F1 Score')
    plt.bar(x + width/2, df_results['recall'], width, label='Recall')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Hybrid Models Performance Comparison')
    plt.xticks(x, df_results['model'], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nSummary of Results:")
    print(df_results[['model', 'f1_score', 'recall']])

def main():
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('Travel_Recommendation_Dataset.csv')
    
    # Đánh giá hybrid ANN-SVM
    print("\nEvaluating Hybrid ANN-SVM...")
    ann_svm_results = evaluate_hybrid_ann_svm(X_train, X_test, y_train, y_test)
    
    # Đánh giá hybrid AdaBoost-SVM
    print("\nEvaluating Hybrid AdaBoost-SVM...")
    ada_svm_results = evaluate_hybrid_adaboost_svm(X_train, X_test, y_train, y_test)
    
    # So sánh kết quả
    print("\nComparing hybrid models...")
    visualize_comparison([ann_svm_results, ada_svm_results])

if __name__ == "__main__":
    main() 