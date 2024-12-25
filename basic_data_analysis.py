import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def plot_histogram(df, column_name, bins=30):
    """
    Vẽ histogram cho một trường dữ liệu
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # In thống kê cơ bản
    print(f"\nBasic statistics for {column_name}:")
    print(df[column_name].describe())

def plot_heatmap(df):
    """
    Vẽ heatmap cho tất cả các trường số
    """
    # Lấy các cột số
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Tính correlation matrix
    correlation = df[numeric_columns].corr()
    
    # Vẽ heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, 
                annot=True,  # Hiển thị giá trị
                cmap='coolwarm',  # Màu sắc
                center=0,  # Giá trị trung tâm cho colormap
                fmt='.2f')  # Format số
    plt.title('Correlation Heatmap of Numerical Fields')
    plt.tight_layout()
    plt.show()
    
    # In các tương quan mạnh
    print("\nStrong correlations (|correlation| > 0.5):")
    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 < col2:  # Tránh in trùng lặp
                corr = correlation.loc[col1, col2]
                if abs(corr) > 0.5:
                    print(f"{col1} vs {col2}: {corr:.3f}")

def clean_data(df):
    """
    Làm sạch dữ liệu bằng cách điền giá trị vào các ô trống
    """
    df_cleaned = df.copy()
    
    # Hiển thị số lượng giá trị thiếu ban đầu
    print("\nMissing values before cleaning:")
    print(df_cleaned.isnull().sum())
    
    # Xử lý cho từng loại dữ liệu
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns
    
    # Điền giá trị cho các cột số
    for col in numeric_columns:
        # Điền bằng giá trị trung bình
        mean_value = df_cleaned[col].mean()
        df_cleaned[col].fillna(mean_value, inplace=True)
    
    # Điền giá trị cho các cột category
    for col in categorical_columns:
        # Điền bằng giá trị phổ biến nhất
        mode_value = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(mode_value, inplace=True)
    
    # Hiển thị số lượng giá trị thiếu sau khi xử lý
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum())
    
    return df_cleaned

def transform_and_plot(df, column_name, new_range=(0, 1)):
    """
    Chuyển đổi một cột sang range mới và vẽ biểu đồ so sánh
    """
    # Tạo scaler
    scaler = MinMaxScaler(feature_range=new_range)
    
    # Transform dữ liệu
    original_data = df[column_name].values.reshape(-1, 1)
    transformed_data = scaler.fit_transform(original_data).flatten()
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.hist(original_data, bins=30, edgecolor='black')
    plt.title(f'Original {column_name} Distribution')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    
    # Transformed data
    plt.subplot(1, 2, 2)
    plt.hist(transformed_data, bins=30, edgecolor='black')
    plt.title(f'Transformed {column_name} Distribution\n(Range: {new_range})')
    plt.xlabel(f'Transformed {column_name}')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # In thống kê so sánh
    print(f"\nComparison of {column_name} statistics:")
    print("\nOriginal data:")
    print(pd.Series(original_data.flatten()).describe())
    print("\nTransformed data:")
    print(pd.Series(transformed_data).describe())
    
    return pd.Series(transformed_data, name=f'Transformed_{column_name}')

def main():
    # Đọc dữ liệu
    print("Reading data...")
    df = pd.read_csv('Travel_Recommendation_Dataset.csv')
    
    # 1. Plot histogram của UserAge
    print("\nPlotting histogram of UserAge...")
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

if __name__ == "__main__":
    main() 