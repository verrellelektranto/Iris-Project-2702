import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Iris Classifier")
st.write("This is a simple Iris Classifier app")

# Try these paths (current folder first, then Downloads)
MODEL_PATHS = [Path("model.joblib"), Path.home() / "Downloads" / "model.joblib"]

def load_or_train_model():
    for p in MODEL_PATHS:
        if p.exists():
            try:
                model = joblib.load(p)
                st.success(f"Loaded model from: {p}")
                return model
            except Exception as e:
                st.error(f"Failed to load model from {p}: {e}")
                break
    # Fallback: train small model and save to local model.joblib
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")
    st.info("No valid model found — trained a new model and saved as model.joblib")
    return model

model = load_or_train_model()

def get_prediction(data: pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# -- UI inputs
left, right = st.columns(2, gap="medium", border=True)
left.subheader("Sepal")
sepal_length = left.slider("Sepal Length", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
sepal_width = left.slider("Sepal Width", min_value=1.0, max_value=10.0, value=3.4, step=0.1)

right.subheader("Petal")
petal_length = right.slider("Petal Length", min_value=1.0, max_value=10.0, value=1.3, step=0.1)
petal_width = right.slider("Petal Width", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

data = pd.DataFrame({
    "sepal length (cm)": [sepal_length],
    "sepal width (cm)" : [sepal_width],
    "petal length (cm)": [petal_length],
    "petal width (cm)" : [petal_width]
})
st.dataframe(data, use_container_width=True)

if st.button("Predict", use_container_width=True):
    pred, pred_proba = get_prediction(data)
    label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    idx = int(pred[0])
    label_pred = label_map[idx]
    proba = float(pred_proba[0][idx])
    st.write(f"Iris Anda diklasifikasikan sebagai {label_pred} ({proba:.0%} confidence)")
