# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Đọc và xử lý dữ liệu
def load_and_preprocess_data(file_path):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Label Encoding cho biến categorical
    le = LabelEncoder()
    df['SeasonPreference'] = le.fit_transform(df['SeasonPreference'])
    df['RecommendationType'] = le.fit_transform(df['RecommendationType'])
    
    # Tách features và target
    X = df.drop('RecommendationType', axis=1)
    y = df['RecommendationType']
    
    # Chuyển y thành one-hot encoding
    y = to_categorical(y)
    
    # Chia train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape dữ liệu cho CNN (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, scaler

# 2. Xây dựng mô hình CNN
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First Convolutional Layer
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        
        # Second Convolutional Layer
        Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        
        # Flatten layer
        Flatten(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model với Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# 3. Train và đánh giá mô hình
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Training
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=1)
    
    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Vẽ đồ thị learning curves
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return history

# 4. Function để dự đoán cho người dùng mới
def predict_for_new_user(model, scaler, user_input):
    # Chuẩn hóa input
    user_input_scaled = scaler.transform([user_input])
    # Reshape cho CNN
    user_input_reshaped = user_input_scaled.reshape((1, user_input_scaled.shape[1], 1))
    # Dự đoán
    prediction = model.predict(user_input_reshaped)
    return np.argmax(prediction, axis=1)[0]

def analyze_data(df):
    """Phân tích và trực quan hóa dữ liệu"""
    plt.figure(figsize=(15, 10))
    
    # 1. Phân phối của các biến số
    plt.subplot(2, 3, 1)
    sns.histplot(data=df, x='UserAge', bins=30)
    plt.title('Phân phối độ tuổi')
    
    plt.subplot(2, 3, 2)
    sns.histplot(data=df, x='Budget', bins=30)
    plt.title('Phân phối ngân sách')
    
    plt.subplot(2, 3, 3)
    sns.histplot(data=df, x='Distance', bins=30)
    plt.title('Phân phối khoảng cách')
    
    plt.subplot(2, 3, 4)
    sns.histplot(data=df, x='InterestScore', bins=30)
    plt.title('Phân phối điểm hứng thú')
    
    plt.subplot(2, 3, 5)
    sns.countplot(data=df, x='SeasonPreference')
    plt.title('Phân phối mùa ưa thích')
    
    plt.subplot(2, 3, 6)
    sns.countplot(data=df, x='RecommendationType')
    plt.title('Phân phối loại du lịch')
    
    plt.tight_layout()
    plt.show()

    # 2. Correlation matrix
    plt.figure(figsize=(10, 8))
    numeric_cols = ['UserAge', 'Budget', 'Distance', 'InterestScore', 'TravelFrequency']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Ma trận tương quan')
    plt.show()

    # 3. Box plots để phát hiện outliers
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, y='Budget')
    plt.title('Box Plot - Budget')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, y='Distance')
    plt.title('Box Plot - Distance')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, y='InterestScore')
    plt.title('Box Plot - Interest Score')
    
    plt.tight_layout()
    plt.show()

    # 4. Scatter plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='UserAge', y='Budget', hue='RecommendationType', alpha=0.6)
    plt.title('Age vs Budget')
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='InterestScore', y='Budget', hue='RecommendationType', alpha=0.6)
    plt.title('Interest Score vs Budget')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='Distance', y='Budget', hue='RecommendationType', alpha=0.6)
    plt.title('Distance vs Budget')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Đọc dữ liệu
    df = pd.read_csv('Travel_Recommendation_Dataset.csv')
    
    # Phân tích dữ liệu
    analyze_data(df)
    
    # Load và xử lý dữ liệu
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('Travel_Recommendation_Dataset.csv')
    
    # Xây dựng mô hình
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]
    model = build_cnn_model(input_shape, num_classes)
    
    # In tổng quan mô hình
    model.summary()
    
    # Train và đánh giá mô hình
    history = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Ví dụ dự đoán cho người dùng mới
    new_user = [30, 3000, 500, 7.5, 2, 5]  # [Age, Budget, Distance, InterestScore, SeasonPreference, TravelFrequency]
    prediction = predict_for_new_user(model, scaler, new_user)
    print(f"\nRecommended type for new user: {prediction}")