import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import statistics
import time
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

# Load MovieLens dataset
print("Loading MovieLens dataset...")
df = movielens.load_pandas_df(
    size="100k",
    header=["userId", "movieId", "rating", "timestamp"]
)
df = df[["userId", "movieId", "rating"]]
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Data preprocessing
print("\nPreprocessing data...")
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["userId_enc"] = user_enc.fit_transform(df["userId"])
df["movieId_enc"] = item_enc.fit_transform(df["movieId"])

scaler = StandardScaler()
df[["userId_enc", "movieId_enc"]] = scaler.fit_transform(df[["userId_enc", "movieId_enc"]])

print("Data after preprocessing:")
print(df.head())

# Prepare features and target
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df[["userId_enc", "movieId_enc"]]
y = df["rating"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Experiment 1: Run 10 times for k = 5
print("\n" + "="*50)
print("EXPERIMENT 1: Running 10 iterations for k = 5")
print("="*50)

rmse_l = []
mae_l = []
mse_l = []

for i in range(10):
    combined = list(zip(X.values, y.values))
    random.shuffle(combined)
    dulieu_X_shuffled, dulieu_Y_shuffled = zip(*combined)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(dulieu_X_shuffled, dulieu_Y_shuffled, test_size=1/3, random_state=125)
    
    model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_Test)
        
    mse = mean_squared_error(Y_Test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_Test, y_pred)
    
    mse_l.append(mse)
    rmse_l.append(rmse)
    mae_l.append(mae)
    
    print(f"Lan {i+1}: RMSE = {rmse:.4f} | MAE = {mae:.4f} | MSE = {mse:.4f}")

mean_rmse = np.mean(rmse_l)
mean_mae = np.mean(mae_l)
mean_mse = np.mean(mse_l)

print(f"\nTrung binh sau 10 lan:")
print(f"RMSE trung binh: {mean_rmse:.4f}")
print(f"MAE trung binh: {mean_mae:.4f}")
print(f"MSE trung binh: {mean_mse:.4f}")

# Experiment 2: Test different k values
print("\n" + "="*50)
print("EXPERIMENT 2: Testing different k values (1-20)")
print("="*50)

k_results = []

for k_neighbor in range(1, 21):
    time_train = []
    time_test = []
    rmse_list = []
    mae_list = []
    mse_list = []
    
    for each in range(1, 11):  # Chạy 10 lần thử nghiệm
        combined = list(zip(X.values, y.values))
        random.shuffle(combined)
        dulieu_X_shuffled, dulieu_Y_shuffled = zip(*combined)
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(dulieu_X_shuffled, dulieu_Y_shuffled, test_size=1/3, random_state=125)

        # Huấn luyện mô hình KNN
        st = time.time()
        model = KNeighborsRegressor(n_neighbors=k_neighbor)
        model.fit(X_Train, Y_Train)
        et = time.time() - st
        time_train.append(et)

        # Dự đoán và tính toán thời gian
        st = time.time()
        Y_Pred = model.predict(X_Test)
        et = time.time() - st
        time_test.append(et)

        knn_mse = mean_squared_error(Y_Test, Y_Pred)
        mse_list.append(knn_mse)
        knn_rmse = np.sqrt(knn_mse)
        rmse_list.append(knn_rmse)
        knn_mae = mean_absolute_error(Y_Test, Y_Pred)
        mae_list.append(knn_mae)

    # Tính trung bình thời gian train, test và MSE
    Ketqua_timeTrain = statistics.mean(time_train)
    Ketqua_timeTest = statistics.mean(time_test)
    Ketqua_mse = statistics.mean(mse_list)
    Ketqua_rmse = statistics.mean(rmse_list)
    Ketqua_mae = statistics.mean(mae_list)
    
    k_results.append({
        'k': k_neighbor,
        'train_time': Ketqua_timeTrain,
        'test_time': Ketqua_timeTest,
        'rmse': Ketqua_rmse,
        'mae': Ketqua_mae,
        'mse': Ketqua_mse
    })
    
    print(f"k = {k_neighbor}")
    print(f"Train Time: {Ketqua_timeTrain:.4f}")
    print(f"Test Time: {Ketqua_timeTest:.4f}")
    print(f"Trung binh: RMSE = {Ketqua_rmse:.4f} | MAE = {Ketqua_mae:.4f} | MSE = {Ketqua_mse:.4f}")
    print("-" * 40)

# Visualization
print("\n" + "="*50)
print("VISUALIZATION")
print("="*50)

# Plot for Experiment 1 (k=5 iterations)
iterations = np.arange(1, 11)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(iterations, rmse_l, marker='o', label='RMSE')
plt.plot(iterations, mae_l, marker='s', label='MAE')
plt.plot(iterations, mse_l, marker='^', label='MSE')
plt.xlabel('Lần lặp')
plt.ylabel('Giá trị')
plt.title('Biểu đồ Bias và Variance qua các lần lặp của K = 5')
plt.legend()
plt.grid(True)

# Plot for Experiment 2 (different k values)
plt.subplot(1, 2, 2)
k_values = [result['k'] for result in k_results]
rmse_values = [result['rmse'] for result in k_results]
mae_values = [result['mae'] for result in k_results]

plt.plot(k_values, rmse_values, marker='o', label='RMSE')
plt.plot(k_values, mae_values, marker='s', label='MAE')
plt.xlabel('K values')
plt.ylabel('Error')
plt.title('Performance across different K values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find best k
best_k_result = min(k_results, key=lambda x: x['rmse'])
print(f"\nBest k value: {best_k_result['k']}")
print(f"Best RMSE: {best_k_result['rmse']:.4f}")
print(f"Best MAE: {best_k_result['mae']:.4f}")

# Save the best model
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

# Train final model with best k
final_model = KNeighborsRegressor(n_neighbors=best_k_result['k'])
final_model.fit(X, y)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Luu mo hinh thanh cong!")
print(f"Model saved with k = {best_k_result['k']}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Dataset: MovieLens 100k")
print(f"Total samples: {len(df)}")
print(f"Best k value: {best_k_result['k']}")
print(f"Best RMSE: {best_k_result['rmse']:.4f}")
print(f"Best MAE: {best_k_result['mae']:.4f}")
print(f"Average training time: {best_k_result['train_time']:.4f} seconds")
print(f"Average testing time: {best_k_result['test_time']:.4f} seconds") 