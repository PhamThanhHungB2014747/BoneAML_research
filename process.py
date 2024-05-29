import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Bước 1: Đọc dữ liệu
data = pd.read_csv('Bone_AML_8390_1004.csv')
data = data.drop(columns=['Replicate','ID','Barcode'])
# Bước 2: Làm sạch dữ liệu (nếu cần)
# Ví dụ: Loại bỏ các hàng có giá trị bị thiếu
data = data.dropna()

# Bước 3: Chia dữ liệu thành các đặc trưng (X) và nhãn (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Bước 4: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Chuẩn hóa dữ liệu (nếu cần)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 6: Lưu dữ liệu đã xử lý vào các file mới
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data['Class'] = y_train.reset_index(drop=True)
test_data = pd.DataFrame(X_test, columns=X.columns)
test_data['Class'] = y_test.reset_index(drop=True)

# train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
