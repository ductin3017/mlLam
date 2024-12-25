import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import log_loss

def prepare_data(file_path, task='classification'):
    """
    Chuẩn bị dữ liệu cho cả classification và regression
    """
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['SeasonPreference'] = le.fit_transform(df['SeasonPreference'])
    df['RecommendationType'] = le.fit_transform(df['RecommendationType'])
    
    # Tách features
    X = df[['UserAge', 'Budget', 'Distance', 'InterestScore', 'TravelFrequency', 'SeasonPreference']]
    
    if task == 'classification':
        y = df['RecommendationType']
    else:  # regression
        y = df['Budget']  # Dự đoán Budget
    
    # Chia train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_decision_tree_classifier(X_train, X_test, y_train, y_test):
    """
    Đánh giá Decision Tree Classifier với các tham số khác nhau
    """
    # Các tham số thử nghiệm
    params = [
        {'max_depth': 5, 'min_samples_split': 2},
        {'max_depth': 10, 'min_samples_split': 5}
    ]
    
    results = []
    
    for param in params:
        # Train model
        dt = DecisionTreeClassifier(**param, random_state=42)
        dt.fit(X_train, y_train)
        
        # Predict
        y_pred = dt.predict(X_test)
        y_pred_proba = dt.predict_proba(X_test)
        
        # Tính metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        entropy = log_loss(y_test, y_pred_proba)
        
        # Lưu kết quả
        results.append({
            'model': f'Decision Tree (depth={param["max_depth"]}, split={param["min_samples_split"]})',
            'f1_score': f1,
            'recall': recall,
            'cross_entropy': entropy,
            'y_pred': y_pred
        })
        
        # In kết quả
        print(f"\nResults for Decision Tree with parameters: {param}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Decision Tree\n{param}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    return results

def evaluate_decision_tree_regressor(X_train, X_test, y_train, y_test):
    """
    Đánh giá Decision Tree Regressor và Linear Regression
    """
    # Decision Tree Regression
    dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_reg.fit(X_train, y_train)
    dt_pred = dt_reg.predict(X_test)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # Tính metrics
    results = {
        'Decision Tree': {
            'MSE': mean_squared_error(y_test, dt_pred),
            'R2': r2_score(y_test, dt_pred),
            'predictions': dt_pred
        },
        'Linear Regression': {
            'MSE': mean_squared_error(y_test, lr_pred),
            'R2': r2_score(y_test, lr_pred),
            'predictions': lr_pred
        }
    }
    
    # Visualize kết quả
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, dt_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Decision Tree Regression')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, lr_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression')
    
    plt.tight_layout()
    plt.show()
    
    return results

def visualize_classification_comparison(dt_results, knn_results, nb_result):
    """
    So sánh kết quả của Decision Tree, kNN và Naive Bayes
    """
    # Combine results
    all_results = dt_results + knn_results + [nb_result]
    df_results = pd.DataFrame(all_results)
    
    # Plot metrics comparison
    plt.figure(figsize=(15, 6))
    
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
    
    # Print summary table
    print("\nSummary of Classification Results:")
    print(df_results[['model', 'f1_score', 'recall', 'cross_entropy']])

def main():
    # Classification
    print("Running Classification Analysis...")
    X_train, X_test, y_train, y_test = prepare_data('Travel_Recommendation_Dataset.csv', 'classification')
    
    # Decision Tree Classification
    dt_results = evaluate_decision_tree_classifier(X_train, X_test, y_train, y_test)
    
    # Load previous results (you need to modify this based on how you store previous results)
    # For demonstration, I'll create dummy results
    k_values = [3, 5, 7, 9, 11]
    knn_results = [{'model': f'kNN (k={k})', 'f1_score': 0.8, 'recall': 0.7, 'cross_entropy': 0.5} for k in k_values]
    nb_result = {'model': 'Naive Bayes', 'f1_score': 0.75, 'recall': 0.73, 'cross_entropy': 0.6}
    
    # Compare classification results
    visualize_classification_comparison(dt_results, knn_results, nb_result)
    
    # Regression
    print("\nRunning Regression Analysis...")
    X_train, X_test, y_train, y_test = prepare_data('Travel_Recommendation_Dataset.csv', 'regression')
    
    # Compare regression results
    regression_results = evaluate_decision_tree_regressor(X_train, X_test, y_train, y_test)
    
    # Print regression results
    print("\nRegression Results:")
    for model, metrics in regression_results.items():
        print(f"\n{model}:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"R2 Score: {metrics['R2']:.4f}")

if __name__ == "__main__":
    main() 