import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def prepare_data(file_path):
    """
    Chuẩn bị dữ liệu cho thí nghiệm
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def experiment_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Thực hiện thí nghiệm với các tham số khác nhau
    """
    # Định nghĩa các tham số thí nghiệm
    solvers = ['lbfgs', 'newton-cg']
    tolerances = [1e-4, 1e-3]
    max_iters = [100, 200]
    
    # Lưu kết quả
    results = []
    
    # Thực hiện thí nghiệm
    for solver in solvers:
        for tol in tolerances:
            for max_iter in max_iters:
                # Train model
                model = LogisticRegression(
                    solver=solver,
                    tol=tol,
                    max_iter=max_iter,
                    multi_class='multinomial'
                )
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Tính accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Lưu kết quả
                results.append({
                    'solver': solver,
                    'tolerance': tol,
                    'max_iterations': max_iter,
                    'accuracy': accuracy,
                    'y_pred': y_pred
                })
                
                # In kết quả
                print(f"\nResults for: Solver={solver}, Tolerance={tol}, Max Iterations={max_iter}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                # Vẽ confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix\nSolver={solver}, Tol={tol}, MaxIter={max_iter}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.show()
    
    return results

def visualize_results(results):
    """
    Trực quan hóa kết quả thí nghiệm
    """
    # Tạo DataFrame từ kết quả
    df_results = pd.DataFrame([
        {
            'solver': r['solver'],
            'tolerance': r['tolerance'],
            'max_iterations': r['max_iterations'],
            'accuracy': r['accuracy']
        }
        for r in results
    ])
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    
    # Grouped bar plot
    x = np.arange(len(df_results))
    plt.bar(x, df_results['accuracy'])
    plt.xticks(x, [
        f"{row['solver']}\n{row['tolerance']}\n{row['max_iterations']}"
        for _, row in df_results.iterrows()
    ], rotation=45)
    
    plt.title('Model Accuracy for Different Parameters')
    plt.xlabel('Parameters (Solver/Tolerance/MaxIter)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nSummary of Results:")
    print(df_results.sort_values('accuracy', ascending=False))

def main():
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('Travel_Recommendation_Dataset.csv')
    
    # Thực hiện thí nghiệm
    print("\nRunning experiments...")
    results = experiment_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Visualize kết quả
    print("\nVisualizing results...")
    visualize_results(results)

if __name__ == "__main__":
    main() 