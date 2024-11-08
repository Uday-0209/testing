import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
data = pd.read_csv("C:\\Users\\SMDDC-02\\Desktop\\shuffled_features.csv")

X = data.drop(['Label','label_num'], axis=1)
Y = data['label_num']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, train_size=0.9)
print(X_train.shape, Y_train.shape)
models = [(
    'Logisitic Regression',
    {'max_iter':1000, 'solver':'lbfgs'},
    LogisticRegression(),
    (X_train, Y_train),
    (X_test, Y_test)
    ),
    (
        'Random forest classifier',
        {'n_estimators':10, 'max_depth':4},
        RandomForestClassifier(),
        (X_train, Y_train),
        (X_test, Y_test)
    ),
    (
        'DecisionTreeClassifier',
        {'random_state':4},
        DecisionTreeClassifier(),
        (X_train, Y_train),
        (X_test, Y_test)
    ),
    (
        'Gradient boosting classifier',
        {'n_estimators':15, 'max_depth':6},
        GradientBoostingClassifier(),
        (X_train, Y_train),
        (X_test, Y_test)
    )
]

reports = []

for model_name, params, model, train_set, test_set in models:
    X_train = train_set[0]
    X_test = test_set[0]
    Y_train = train_set[1]
    Y_test = test_set[1]

    model.set_params(**params)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, output_dict=True)

    reports.append(report)

print(reports)

# params = {
#     'max_iter' : 1000,
#     'solver': 'lbfgs',
# }
#
# le = LogisticRegression(**params)
#
# model = le.fit(X_train, Y_train)
#
# y_pred = model.predict(X_test)
#
# print(classification_report(Y_test, y_pred))
# report = classification_report(Y_test, y_pred, output_dict=True)
# print(confusion_matrix(Y_test, y_pred))
#
# print(report)
#
# mlflow.set_experiment("First dataset")
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#
# with mlflow.start_run():
#     mlflow.log_params(params)
#     mlflow.log_metrics({
#         'accuracy': report['accuracy'],
#         'recall_class_0': report['1']['recall'],
#         'recall_class_1' : report['2']['recall'],
#         'f1_score_macro' : report['macro avg']['f1-score']
#     })
#     mlflow.sklearn.log_model(le, "Logistic Regression")

mlflow.set_experiment("Anomaly detection")
mlflow.set_tracking_uri('http://172.18.100.83:5000')

for i , element in enumerate(models):
    model_name = element[0]
    params = element[1]
    model = element[2]
    report = reports[i]

    with mlflow.start_run(run_name = model_name):
        mlflow.log_params(params)
        mlflow.log_metrics({
            'accuracy':report['accuracy'],
            'recall_class_0': report['1']['recall'],
            'recall_class_1':report['2']['recall'],
            'recall_class_2': report['3']['recall'],
            'recall_class_3': report['4']['recall'],
            'recall_class_4': report['5']['recall'],
            'recall_class_5': report['6']['recall'],
            'recall_class_6': report['7']['recall'],
            'recall_class_7': report['8']['recall'],
            'recall_class_8': report['9']['recall'],
            'recall_class_9': report['10']['recall'],
            'f1_score_macro': report['macro avg']['f1-score'],
            'weighted avg':report['weighted avg']['f1-score']

        })

        mlflow.sklearn.log_model(model, "model")
