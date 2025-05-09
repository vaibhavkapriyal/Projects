import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Load training and testing data
train = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")

# Clean prognosis column
train['prognosis'] = train['prognosis'].str.strip()
test['prognosis'] = test['prognosis'].str.strip()

# Encode target
disease_list = train['prognosis'].unique()
disease_dict = {d: i for i, d in enumerate(disease_list)}
reverse_disease_dict = {i: d for d, i in disease_dict.items()}

train['prognosis'] = train['prognosis'].map(disease_dict)
test['prognosis'] = test['prognosis'].map(disease_dict)

# Features and target
X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]
X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]

symptom_list = X_train.columns.tolist()

# Model
model = BernoulliNB()
model.fit(X_train, y_train)

# GUI
class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Predictor")
        self.root.geometry("500x500")
        self.root.configure(bg="#f0f4f7")

        ttk.Style().configure("TLabel", font=("Segoe UI", 11))
        ttk.Style().configure("TButton", font=("Segoe UI", 11))
        ttk.Style().configure("TCombobox", font=("Segoe UI", 10))

        title = ttk.Label(root, text="Disease Predictor", font=("Segoe UI", 18, "bold"))
        title.pack(pady=20)

        self.symptom_vars = []
        self.comboboxes = []

        for i in range(5):
            label = ttk.Label(root, text=f"Select Symptom {i+1}:")
            label.pack(pady=(10 if i == 0 else 5, 2))

            var = tk.StringVar()
            combo = ttk.Combobox(root, textvariable=var, values=["None"] + symptom_list, state="readonly", width=40)
            combo.current(0)
            combo.pack(pady=2)

            self.symptom_vars.append(var)
            self.comboboxes.append(combo)

        self.result_label = ttk.Label(root, text="", font=("Segoe UI", 12, "bold"), foreground="green")
        self.result_label.pack(pady=20)

        predict_btn = ttk.Button(root, text="Predict Disease", command=self.predict_disease)
        predict_btn.pack(pady=10)

    def predict_disease(self):
        input_symptoms = [var.get() for var in self.symptom_vars if var.get() != "None"]

        if not input_symptoms:
            messagebox.showwarning("Input Error", "Please select at least one symptom.")
            return

        input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptom_list]
        prediction = model.predict([input_vector])[0]
        predicted_disease = reverse_disease_dict[prediction]
        self.result_label.config(text=f"Predicted Disease: {predicted_disease}")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()
