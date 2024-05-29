import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import joblib

# Đọc dữ liệu từ file CSV
file_path = 'train_data_Bone_AML.csv'
df = pd.read_csv(file_path)

# Chuyển đổi nhãn Class từ kiểu object thành kiểu số
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Tách các đặc trưng và nhãn
X = df.drop('Class', axis=1)
y = df['Class']
# Huấn luyện mô hình KNN trên toàn bộ tập dữ liệu
# num_samples = len(df)
# k = min(5, num_samples - 1) 
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'knn_model.pkl'
# joblib.dump(knn, model_path)

# Huấn luyện mô hình Decision Tree trên toàn bộ tập dữ liệu
# decision_tree = DecisionTreeClassifier(random_state=42)
# decision_tree.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'decision_tree_model.pkl'
# joblib.dump(decision_tree, model_path)

# Huấn luyện mô hình Native Bayes trên toàn bộ tập dữ liệu
# naive_bayes = GaussianNB()
# naive_bayes.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'naive_bayes_model.pkl'
# joblib.dump(naive_bayes, model_path)

# Huấn luyện mô hình Random Forest trên toàn bộ tập dữ liệu
# random_forest = RandomForestClassifier(random_state=42)
# random_forest.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'random_forest_model.pkl'
# joblib.dump(random_forest, model_path)

# Huấn luyện mô hình SVM trên toàn bộ tập dữ liệu
# svm_model = SVC(kernel='linear', random_state=42)
# svm_model.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'svm_model.pkl'
# joblib.dump(svm_model, model_path)

# Huấn luyện mô hình Logistic Regression trên toàn bộ tập dữ liệu
# logistic_regression_model = LogisticRegression(random_state=42)
# logistic_regression_model.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'logistic_regression_model.pkl'
# joblib.dump(logistic_regression_model, model_path)

# Huấn luyện mô hình Multinomial Logistic Regression trên toàn bộ tập dữ liệu
# multinomial_logistic_regression_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
# multinomial_logistic_regression_model.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'multinomial_logistic_regression_model.pkl'
# joblib.dump(multinomial_logistic_regression_model, model_path)

# Huấn luyện mô hình AdaBoost trên toàn bộ tập dữ liệu
# adaboost_model = AdaBoostClassifier(random_state=42)
# adaboost_model.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'adaboost_model.pkl'
# joblib.dump(adaboost_model, model_path)

# Huấn luyện mô hình Gradient Boosting trên toàn bộ tập dữ liệu
# gradient_boosting_model = GradientBoostingClassifier(random_state=42)
# gradient_boosting_model.fit(X, y)

# Lưu mô hình đã huấn luyện
# model_path = 'gradient_boosting_model.pkl'
# joblib.dump(gradient_boosting_model, model_path)

# Huấn luyện mô hình XGBoost trên toàn bộ tập dữ liệu
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X, y)

# Lưu mô hình đã huấn luyện
model_path = 'xgboost_model.pkl'
joblib.dump(xgb_model, model_path)
# print(f'Model saved to {model_path}')
