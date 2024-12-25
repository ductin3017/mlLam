import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def prepare_data(file_path):
    """
    Chuẩn bị dữ liệu cho classification
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
    
    # Chuyển đổi target thành one-hot encoding cho ANN
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            y_train_cat, y_test_cat)

def evaluate_adaboost(X_train, X_test, y_train, y_test):
    """
    Đánh giá AdaBoost với các tham số khác nhau
    """
    # Các tham số thử nghiệm
    params = [
        {'n_estimators': 50, 'learning_rate': 1.0, 'estimator': DecisionTreeClassifier(max_depth=1)},
        {'n_estimators': 100, 'learning_rate': 0.5, 'estimator': DecisionTreeClassifier(max_depth=2)},
        {'n_estimators': 200, 'learning_rate': 0.1, 'estimator': DecisionTreeClassifier(max_depth=3)}
    ]
    
    results = []
    
    for param in params:
        # Train model
        ada = AdaBoostClassifier(
            estimator=param['estimator'],
            n_estimators=param['n_estimators'],
            learning_rate=param['learning_rate'],
            random_state=42
        )
        ada.fit(X_train, y_train)
        
        # Predict
        y_pred = ada.predict(X_test)
        
        # Tính metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Lưu kết quả
        results.append({
            'model': f'AdaBoost (n={param["n_estimators"]}, lr={param["learning_rate"]}, depth={param["estimator"].max_depth})',
            'f1_score': f1,
            'recall': recall
        })
        
        # In kết quả
        print(f"\nResults for AdaBoost with parameters:")
        print(f"n_estimators: {param['n_estimators']}")
        print(f"learning_rate: {param['learning_rate']}")
        print(f"max_depth: {param['estimator'].max_depth}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - AdaBoost\nn={param["n_estimators"]}, lr={param["learning_rate"]}, depth={param["estimator"].max_depth}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    return results

def create_ann_model(input_shape, num_classes, architecture):
    """
    Tạo mô hình ANN với kiến trúc khác nhau
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    
    # Hidden layers theo kiến trúc
    for units in architecture:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    model.compile(optimizer=Adam(0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def evaluate_ann(X_train, X_test, y_train_cat, y_test_cat):
    """
    Đánh giá các kiến trúc ANN khác nhau
    """
    # Các kiến trúc khác nhau (số node trong mỗi hidden layer)
    architectures = [
        [32],  # 1 hidden layer
        [64, 32],  # 2 hidden layers
    ]
    
    results = []
    histories = []
    
    for arch in architectures:
        # Create and train model
        model = create_ann_model(X_train.shape[1], y_train_cat.shape[1], arch)
        
        # Train
        history = model.fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        histories.append(history)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test_cat, axis=1)
        
        # Tính metrics
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        
        # Lưu kết quả
        results.append({
            'model': f'ANN (layers={len(arch)}, architecture={arch})',
            'f1_score': f1,
            'recall': recall,
            'history': history.history
        })
        
        # In kết quả
        print(f"\nResults for ANN with architecture: {arch}")
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes))
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - ANN\nArchitecture: {arch}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        # Plot learning curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results

def visualize_comparison(adaboost_results, ann_results):
    """
    So sánh kết quả của tất cả các mô hình
    """
    # Combine results
    all_results = adaboost_results + ann_results
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
    print("\nSummary of Results:")
    print(df_results[['model', 'f1_score', 'recall']])

def main():
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = prepare_data('Travel_Recommendation_Dataset.csv')
    
    # AdaBoost
    print("\nEvaluating AdaBoost models...")
    adaboost_results = evaluate_adaboost(X_train, X_test, y_train, y_test)
    
    # ANN
    print("\nEvaluating ANN models...")
    ann_results = evaluate_ann(X_train, X_test, y_train_cat, y_test_cat)
    
    # So sánh kết quả
    print("\nComparing all models...")
    visualize_comparison(adaboost_results, ann_results)

if __name__ == "__main__":
    main() 