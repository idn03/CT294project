import pandas as pd
import zipfile
import os

def load_movielens_100k():
    """Load MovieLens 100k dataset from local zip file"""
    if os.path.exists('ml-100k.zip'):
        print("Loading dataset từ file local ml-100k.zip...")
        with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
            # Tìm file ratings trong zip
            ratings_file = None
            for file in zip_ref.namelist():
                if 'u.data' in file:
                    ratings_file = file
                    break
            
            if ratings_file:
                # Đọc file ratings với delimiter tab
                df = pd.read_csv(zip_ref.open(ratings_file), 
                               delimiter='\t', 
                               header=None,
                               names=["userId", "movieId", "rating", "timestamp"])
                print(f"Đã load dataset từ {ratings_file}")
                return df
            else:
                print("Không tìm thấy file ratings trong zip")
                return None
    else:
        print("File ml-100k.zip không tồn tại")
        return None

if __name__ == "__main__":
    df = load_movielens_100k()
    if df is not None:
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
