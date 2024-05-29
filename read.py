import joblib

# Đọc nội dung của tệp model.pkl
with open('./models/knn_model.pkl', 'rb') as file:
    content = joblib.load(file)
# with open('./models/adaboost_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/decision_tree_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/gradient_boosting_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/logistic_regression_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/multinomial_logistic_regression_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/nb_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/random_forest_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/svm_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# with open('./models/xgboost_model.pkl', 'rb') as file:
#     content = joblib.load(file)
# In ra nội dung của mô hình
print(content)
