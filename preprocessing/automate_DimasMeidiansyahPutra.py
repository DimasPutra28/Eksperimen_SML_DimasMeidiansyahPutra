import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_path, output_path):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis.
    
    Args:
        input_path (str): Path ke file dataset mentah
        output_path (str): Path untuk menyimpan dataset yang sudah diproses
    """
    # 1. Load data
    data = pd.read_csv(input_path)
    
    # 2. Preprocessing
    # Cek missing values
    if data.isnull().sum().sum() > 0:
        print("Peringatan: Terdapat missing values. Mengisi dengan median.")
        data = data.fillna(data.median(numeric_only=True))
    
    # Cek duplikasi
    if data.duplicated().sum() > 0:
        print(f"Menghapus {data.duplicated().sum()} data duplikat.")
        data = data.drop_duplicates()
    
    # Normalisasi fitur numerik (kecuali GradeClass)
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'GradeClass' in numeric_columns:
        numeric_columns.remove('GradeClass')  # Jangan normalisasi target
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # 3. Simpan data hasil preprocessing
    data.to_csv(output_path, index=False)
    print(f"Dataset telah diproses dan disimpan di: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    input_path = os.path.join(root_dir, 'Student_performance_raw_data.csv')
    output_path = os.path.join(script_dir, 'preprocessing/Student_performance_processed_data.csv')

    preprocess_data(input_path, output_path)