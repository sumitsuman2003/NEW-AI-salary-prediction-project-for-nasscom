import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
import tkinter as tk
from tkinter import messagebox

# ---------------------------
# 1️⃣ Load ANN dataset
# ---------------------------
df_ann = pd.read_csv('salary_prediction_dataset_80k.csv')

# ---------------------------
# 2️⃣ Preprocessing ANN
# ---------------------------
le_education_ann = LabelEncoder()
df_ann['Education'] = le_education_ann.fit_transform(df_ann['Education'])

le_job_ann = LabelEncoder()
df_ann['JobRole_encoded'] = le_job_ann.fit_transform(df_ann['JobRole'])

le_company_ann = LabelEncoder()
df_ann['CompanySize'] = le_company_ann.fit_transform(df_ann['CompanySize'])
le_location_ann = LabelEncoder()
df_ann['Location'] = le_location_ann.fit_transform(df_ann['Location'])
le_gender_ann = LabelEncoder()
df_ann['Gender'] = le_gender_ann.fit_transform(df_ann['Gender'])

X_ann = df_ann[['Experience','Age','Education','CompanySize','Location','Gender']].values
y_salary_ann = df_ann['Salary(INR)'].values
y_jobrole_ann = df_ann['JobRole_encoded'].values

scaler_ann = StandardScaler()
X_ann_scaled = scaler_ann.fit_transform(X_ann)

# Compute class weights for JobRole
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_jobrole_ann), y=y_jobrole_ann)
class_weights_dict = dict(enumerate(weights))

# ---------------------------
# 3️⃣ Build ANN Salary Regression Model (6-7 layers)
# ---------------------------
model_salary = Sequential()
model_salary.add(Dense(256, activation='relu', input_shape=(X_ann_scaled.shape[1],)))
model_salary.add(Dense(128, activation='relu'))
model_salary.add(Dense(128, activation='relu'))
model_salary.add(Dense(64, activation='relu'))
model_salary.add(Dense(64, activation='relu'))
model_salary.add(Dense(32, activation='relu'))
model_salary.add(Dense(1, activation='linear'))
model_salary.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_salary.fit(X_ann_scaled, y_salary_ann, epochs=20, batch_size=256)

# ---------------------------
# 4️⃣ Build ANN JobRole Classification Model (6 layers + class weight)
# ---------------------------
model_job = Sequential()
model_job.add(Dense(256, activation='relu', input_shape=(X_ann_scaled.shape[1],)))
model_job.add(Dense(128, activation='relu'))
model_job.add(Dense(128, activation='relu'))
model_job.add(Dense(64, activation='relu'))
model_job.add(Dense(64, activation='relu'))
model_job.add(Dense(len(np.unique(y_jobrole_ann)), activation='softmax'))
model_job.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_job.fit(X_ann_scaled, y_jobrole_ann, epochs=20, batch_size=256, class_weight=class_weights_dict)

# ---------------------------
# 5️⃣ Load RNN dataset
# ---------------------------
df_rnn = pd.read_csv('salary_rnn_dataset_80k.csv')

# ---------------------------
# 6️⃣ Preprocessing RNN
# ---------------------------
le_education_rnn = LabelEncoder()
df_rnn['Education'] = le_education_rnn.fit_transform(df_rnn['Education'])
le_company_rnn = LabelEncoder()
df_rnn['CompanySize'] = le_company_rnn.fit_transform(df_rnn['CompanySize'])
le_location_rnn = LabelEncoder()
df_rnn['Location'] = le_location_rnn.fit_transform(df_rnn['Location'])
le_gender_rnn = LabelEncoder()
df_rnn['Gender'] = le_gender_rnn.fit_transform(df_rnn['Gender'])

X_rnn = df_rnn[['Experience','Age','Education','CompanySize','Location','Gender']].values
X_rnn = np.expand_dims(X_rnn, axis=1)

salary_columns = [f'Salary_Year_{i+1}' for i in range(5)]
y_rnn = df_rnn[salary_columns].values

input_rnn = Input(shape=(1, X_rnn.shape[2]))
x_rnn = LSTM(128, activation='relu')(input_rnn)
x_rnn = Dense(128, activation='relu')(x_rnn)
x_rnn = Dense(64, activation='relu')(x_rnn)
x_rnn = Dense(64, activation='relu')(x_rnn)
x_rnn = Dense(32, activation='relu')(x_rnn)
x_rnn = Dense(16, activation='relu')(x_rnn)
output_rnn = Dense(5, activation='linear')(x_rnn)

model_rnn = Model(inputs=input_rnn, outputs=output_rnn)
model_rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_rnn.fit(X_rnn, y_rnn, epochs=20, batch_size=256)

# ---------------------------
# 7️⃣ Decision Tree (Regression + Classification)
# ---------------------------
dt_salary = DecisionTreeRegressor(max_depth=10)
dt_salary.fit(X_ann, y_salary_ann)

dt_job = DecisionTreeClassifier(max_depth=10)
dt_job.fit(X_ann, y_jobrole_ann)

# ---------------------------
# 8️⃣ GUI
# ---------------------------
root = tk.Tk()
root.title("Salary & Job Role Prediction")
root.geometry('500x750')

labels = [
    'Experience (0-20 years)',
    'Age (20-60 years)',
    'Education (Diploma=0, B.Tech=1, M.Tech=2, PhD=3)',
    'CompanySize (Small=0, Medium=1, Large=2)',
    'Location (Non-Metro=0, Metro=1)',
    'Gender (Male=0, Female=1, Other=2)'
]
entries = []
for label in labels:
    tk.Label(root, text=label).pack()
    e = tk.Entry(root)
    e.pack()
    entries.append(e)

info_label = tk.Label(root, text="ANN Salary & JobRole | RNN Salary Sequence | Decision Tree Salary & JobRole | Multi-class issue fixed!", font=('Helvetica', 10, 'italic'))
info_label.pack(pady=10)

result_label = tk.Label(root, text="", font=('Helvetica', 12, 'bold'))
result_label.pack(pady=20)

def predict():
    try:
        input_vals = [float(e.get()) for e in entries]
        # Input validation
        if not (0 <= input_vals[0] <= 20): raise ValueError("Experience must be 0-20")
        if not (20 <= input_vals[1] <= 60): raise ValueError("Age must be 20-60")
        if not (0 <= input_vals[2] <=3): raise ValueError("Education must be 0-3")
        if not (0 <= input_vals[3] <=2): raise ValueError("CompanySize must be 0-2")
        if not (0 <= input_vals[4] <=1): raise ValueError("Location must be 0-1")
        if not (0 <= input_vals[5] <=2): raise ValueError("Gender must be 0-2")

        # ANN Salary
        ann_scaled = scaler_ann.transform([input_vals])
        pred_salary = model_salary.predict(ann_scaled)[0][0]

        # ANN JobRole
        pred_job = model_job.predict(ann_scaled)
        job_role = np.argmax(pred_job[0])
        job_role_name = le_job_ann.inverse_transform([job_role])[0]

        # RNN Salary Sequence
        rnn_input_val = np.expand_dims([input_vals], axis=1)
        pred_salary_seq = model_rnn.predict(rnn_input_val)[0]
        pred_salary_seq = [int(s) for s in pred_salary_seq]

        # Decision Tree
        dt_salary_val = int(dt_salary.predict([input_vals])[0])
        dt_job_val = le_job_ann.inverse_transform([dt_job.predict([input_vals])[0]])[0]

        result_label.config(text=f"ANN Salary: ₹{int(pred_salary)}\nANN JobRole: {job_role_name}\nRNN Salary Sequence: {pred_salary_seq}\nDT Salary: ₹{dt_salary_val}\nDT JobRole: {dt_job_val}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn = tk.Button(root, text="Predict", command=predict)
predict_btn.pack(pady=20)

root.mainloop()