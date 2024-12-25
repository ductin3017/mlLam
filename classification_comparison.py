import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.metrics import log_loss  # for cross entropy

def prepare_data(file_path):
    """
    Chuẩn bị dữ liệu cho phân loại
    """
    # Đọc dữ liệu
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

def evaluate_knn(X_train, X_test, y_train, y_test, k_values):
    """
    Đánh giá mô hình kNN với nhiều giá trị k
    """
    results = []
    
    for k in k_values:
        # Train model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Predict
        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)
        
        # Tính metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        entropy = log_loss(y_test, y_pred_proba)
        
        # Lưu kết quả
        results.append({
            'model': f'kNN (k={k})',
            'f1_score': f1,
            'recall': recall,
            'cross_entropy': entropy,
            'y_pred': y_pred
        })
        
        # In kết quả
        print(f"\nResults for kNN with k={k}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - kNN (k={k})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    return results

def evaluate_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Đánh giá mô hình Naive Bayes
    """
    # Train model
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # Predict
    y_pred = nb.predict(X_test)
    y_pred_proba = nb.predict_proba(X_test)
    
    # Tính metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    entropy = log_loss(y_test, y_pred_proba)
    
    # In kết quả
    print("\nResults for Naive Bayes")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Naive Bayes')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return {
        'model': 'Naive Bayes',
        'f1_score': f1,
        'recall': recall,
        'cross_entropy': entropy,
        'y_pred': y_pred
    }

def visualize_results(knn_results, nb_result):
    """
    Trực quan hóa kết quả so sánh
    """
    # Combine results
    all_results = knn_results + [nb_result]
    df_results = pd.DataFrame(all_results)
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    
    # Plot F1 Score and Recall
    x = np.arange(len(all_results))
    width = 0.35
    
    plt.bar(x - width/2, df_results['f1_score'], width, label='F1 Score')
    plt.bar(x + width/2, df_results['recall'], width, label='Recall')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, df_results['model'], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot Cross Entropy
    plt.figure(figsize=(12, 6))
    plt.bar(df_results['model'], df_results['cross_entropy'])
    plt.title('Cross Entropy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Cross Entropy Loss')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nSummary of Results:")
    print(df_results[['model', 'f1_score', 'recall', 'cross_entropy']])

def main():
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('Travel_Recommendation_Dataset.csv')
    
    # Định nghĩa các giá trị k cho kNN
    k_values = [3, 5, 7, 9, 11]
    
    # Đánh giá kNN
    print("\nEvaluating kNN models...")
    knn_results = evaluate_knn(X_train, X_test, y_train, y_test, k_values)
    
    # Đánh giá Naive Bayes
    print("\nEvaluating Naive Bayes model...")
    nb_result = evaluate_naive_bayes(X_train, X_test, y_train, y_test)
    
    # Visualize kết quả
    print("\nVisualizing results...")
    visualize_results(knn_results, nb_result)

if __name__ == "__main__":
    main() 