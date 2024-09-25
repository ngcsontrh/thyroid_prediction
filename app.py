from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def PointScale():
    svm = joblib.load('./ml_models/svc_model.pkl')
    tree = joblib.load('./ml_models/tree_model.pkl')
    nn = joblib.load('./ml_models/nn_model.pkl')
    stacking = joblib.load('./ml_models/stacking_model.pkl')
    label_encoders = joblib.load('./artifacts/label_encoders.pkl')
    scalers = joblib.load('./artifacts/scalers.pkl')

    data = pd.DataFrame([request.get_json()])
    print("Dữ liệu đầu vào:", data)

    for column in data.columns:
            try:
                data[column] = label_encoders[column].transform(data[column])
            except ValueError:
                data[column] = label_encoders[column].fit_transform(data[column])

    for column in data.columns:
        if column in ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']:
            try:
                mm = scalers[column]
                data[column] = mm.transform(data[[column]])
            except ValueError:
                data[column] = mm.fit_transform(data[column])

    print("Dữ liệu mã hóa:\n", data.to_string())

    svm_pred = svm.predict(data)
    tree_pred = tree.predict(data)
    nn_predict = nn.predict(data)
    stacking_predict = stacking.predict(data)
    result = {
        'svm': int(svm_pred[0]),
        'tree': int(tree_pred[0]),
        'nn': int(nn_predict[0]),
        'stacking': int(stacking_predict[0])
    }

    return result, 200

if __name__ == "__main__":
    app.run(debug=True)