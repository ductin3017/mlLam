import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def plot_histogram(df, column_name, bins=30):
    """Plot histogram của một trường dữ liệu"""
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def plot_heatmap(df):
    """Plot heatmap cho tất cả các trường số"""
    # Chọn các cột số
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Tính correlation matrix
    correlation = df[numeric_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def clean_data(df):
    """Làm sạch dữ liệu bằng cách điền giá trị vào các ô trống"""
    df_cleaned = df.copy()
    
    # Hiển thị số lượng giá trị NaN trong mỗi cột
    print("\nMissing values before cleaning:")
    print(df_cleaned.isnull().sum())
    
    # Điền giá trị cho các cột số
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Điền bằng giá trị trung bình
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    # Điền giá trị cho các cột categorical
    categorical_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        # Điền bằng mode (giá trị phổ biến nhất)
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum())
    
    return df_cleaned

def transform_and_plot(df, column_name, new_range=(0, 1)):
    """Chuyển đổi một cột sang range mới và vẽ đồ thị so sánh"""
    # Tạo bản sao của DataFrame
    df_transformed = df.copy()
    
    # Min-Max scaling
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df_transformed[f'{column_name}_transformed'] = (
        (df[column_name] - min_val) * (new_range[1] - new_range[0]) / 
        (max_val - min_val) + new_range[0]
    )
    
    # Plot so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original data
    ax1.hist(df[column_name], bins=30, edgecolor='black')
    ax1.set_title(f'Original {column_name}')
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Frequency')
    
    # Transformed data
    ax2.hist(df_transformed[f'{column_name}_transformed'], bins=30, edgecolor='black')
    ax2.set_title(f'Transformed {column_name} (range {new_range})')
    ax2.set_xlabel(f'{column_name}_transformed')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df_transformed[f'{column_name}_transformed']

def train_and_plot_learning_curves(X, y, epochs=50):
    """Train mô hình và vẽ learning curves"""
    # Chia dữ liệu
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Xây dựng mô hình
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(optimizer=Adam(0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train
    history = model.fit(X_train_scaled, y_train,
                       epochs=epochs,
                       validation_data=(X_val_scaled, y_val),
                       verbose=0)
    
    # Plot learning curves
    plt.figure(figsize=(15, 5))
    
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

def main():
    # Đọc dữ liệu
    df = pd.read_csv('Travel_Recommendation_Dataset.csv')
    
    # 1. Plot histogram của UserAge
    print("Plotting histogram of UserAge...")
    plot_histogram(df, 'UserAge')
    
    # 2. Plot heatmap
    print("\nPlotting correlation heatmap...")
    plot_heatmap(df)
    
    # 3. Clean data
    print("\nCleaning data...")
    df_cleaned = clean_data(df)
    
    # 4. Transform và plot Budget
    print("\nTransforming and plotting Budget...")
    transformed_budget = transform_and_plot(df_cleaned, 'Budget', new_range=(0, 1))
    
    # 5. Train model và plot learning curves
    print("\nTraining model and plotting learning curves...")
    X = df_cleaned[['UserAge', 'Budget', 'Distance', 'InterestScore', 'TravelFrequency']]
    y = (df_cleaned['RecommendationType'] == 'Premium').astype(int)  # Binary classification example
    train_and_plot_learning_curves(X, y, epochs=50)
    
    # Hiển thị thống kê
    print("\nOriginal Budget statistics:")
    print(df_cleaned['Budget'].describe())
    print("\nTransformed Budget statistics:")
    print(transformed_budget.describe())

if __name__ == "__main__":
    main() 