import base64
import io
import os
from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import joblib
from sklearn.calibration import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Sử dụng Agg backend cho Matplotlib để tránh xung đột với tkinter
matplotlib.use('Agg')

app = Flask(__name__)

# Đường dẫn đến thư mục chứa các mô hình
MODEL_DIR = "models/"

#Đặt giá trị cho nhãn Class để tính metrics
label_mapping = {'AML': 1, 'healthy': 0}
label = ['AML', 'Healthy']
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận file từ request
    file = request.files['file']
    # Đọc dữ liệu từ file
    test_data = pd.read_csv(file)
    x = test_data.drop('Class', axis=1).astype(float)
    y = test_data['Class'].map(label_mapping).astype(int)
    z = test_data.drop(columns='Class')
    # Tiền xử lý dữ liệu: Chuyển nhãn "Class" sang dạng số
    label_encoder = LabelEncoder()
    label_encoder.fit(test_data['Class'])
    class_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    test_data.drop(columns=['Class'], inplace=True)
    # Lấy tên mô hình từ request
    model_name = request.form['model']
    # Đường dẫn đến file mô hình
    model_path = MODEL_DIR + model_name
    # Load mô hình
    model = joblib.load(model_path)
    #Xủ lý cho ra tên mô hình
    name = os.path.basename(model_path)
    name = os.path.splitext(name)[0]
    # Thực hiện dự đoán
    predictions = model.predict(test_data)
    # Dự đoán với cross-validation
    prediction = cross_val_predict(model, x, y, cv=5)  # Sử dụng cross-validation với số fold là 5
    # Tính các giá trị đánh giá cho nhãn 'AML'
    accuracy_aml = accuracy_score(y[y == 1], prediction[y == 1])
    precision_aml = precision_score(y[y == 1], prediction[y == 1], pos_label=1)
    recall_aml = recall_score(y[y == 1], prediction[y == 1], pos_label=1)
    f1_aml = f1_score(y[y == 1], prediction[y == 1], pos_label=1)
    # Tính các giá trị đánh giá cho nhãn 'healthy'
    accuracy_healthy = accuracy_score(y[y == 0], prediction[y == 0])
    precision_healthy = precision_score(y[y == 0], prediction[y == 0], pos_label=0)
    recall_healthy = recall_score(y[y == 0], prediction[y == 0], pos_label=0)
    f1_healthy = f1_score(y[y == 0], prediction[y == 0], pos_label=0)
    # Kết hợp các giá trị đánh giá từ cả hai nhãn
    accuracy = round(((accuracy_aml + accuracy_healthy) / 2)*100,2)
    precision = (precision_aml + precision_healthy) / 2
    recall = (recall_aml + recall_healthy) / 2
    f1 = (f1_aml + f1_healthy) / 2
    # Thay đổi đoạn code hiển thị kết quả dự đoán
    predictions = [class_map[prediction] for prediction in predictions]
    results = pd.DataFrame(test_data, columns=z.columns)
    results['Prediction'] = predictions
    results['Data'] = range(1, len(test_data) + 1)
    #Thống kê số lượng nhãn AML và healthy theo từng mô hình
    aml_count = (prediction[y == 1] == 1).sum()
    healthy_count = (prediction[y == 0] == 0).sum()
    count = [aml_count, healthy_count]
    #Vẽ biểu đồ
    plt.bar(label, count, color=['blue', 'green'])
    plt.title("Statistical")
    plt.xlabel("Class")
    plt.ylabel("Count")
    # Chuyển đổi biểu đồ thành định dạng dữ liệu có thể nhúng vào HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()  # Đóng biểu đồ để tránh sự cố bộ nhớ

    return render_template(
        'result.html', 
        predictions=predictions, 
        results=results.to_dict(orient='records'), 
        name=name, 
        accuracy=accuracy, 
        precision=precision, 
        f1=f1, 
        recall=recall,
        graphic=graphic
    )

if __name__ == '__main__':
    app.run(debug=True)
