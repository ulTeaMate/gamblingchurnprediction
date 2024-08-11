import joblib
def predict(df):
    clf = joblib.load('churn_model.sav')
    return clf.predict(df)