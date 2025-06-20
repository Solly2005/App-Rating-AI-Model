import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
sample = pd.read_csv("/content/SampleSubmission.csv")

# Drop unnecessary columns
drop_cols = ["X0", "X9", "X10", "X11"]
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Label encode categorical columns
cat_cols = ["X1", "X5", "X7", "X8"]
label_encoders = {}
for col in cat_cols:
    train[col] = train[col].astype(str)
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le

# Clean numeric columns
train["X2"] = pd.to_numeric(train["X2"], errors="coerce")
train["X3"] = train["X3"].replace("Varies with device", pd.NA).str.replace("M", "", regex=False)
train["X3"] = pd.to_numeric(train["X3"], errors="coerce") * 1e6
train["X4"] = train["X4"].str.replace(r"[+,]", "", regex=True)
train["X4"] = pd.to_numeric(train["X4"], errors="coerce")
train["X5"] = train["X5"].astype(str).str.replace("Free", "0").str.replace("$", "")
train["X5"] = pd.to_numeric(train["X5"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")

if "X6" in train.columns and (train["X6"].dtype == object or train["X6"].nunique() <= 1):
    train.drop(columns="X6", inplace=True)

# Drop missing rows
train.dropna(inplace=True)

# Split data
X = train.drop(columns="Y")
y = train["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error on training data: {mse:.4f}")

# Preprocess test set
for col in cat_cols:
    test[col] = test[col].astype(str)
    le = label_encoders[col]
    if 'Unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Unknown')
    test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    test[col] = le.transform(test[col])

test["X2"] = pd.to_numeric(test["X2"], errors="coerce")
test["X3"] = test["X3"].replace("Varies with device", pd.NA).str.replace("M", "", regex=False)
test["X3"] = pd.to_numeric(test["X3"], errors="coerce") * 1e6
test["X4"] = test["X4"].str.replace(r"[+,]", "", regex=True)
test["X4"] = pd.to_numeric(test["X4"], errors="coerce")
test["X5"] = test["X5"].astype(str).str.replace("Free", "0").str.replace("$", "")
test["X5"] = pd.to_numeric(test["X5"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")

if "X6" in test.columns and (test["X6"].dtype == object or test["X6"].nunique() <= 1):
    test.drop(columns="X6", inplace=True)

test.dropna(inplace=True)

# Predict on test set
preds = model.predict(test)

# Submission
submission = pd.DataFrame({
    "row_id": sample["row_id"].iloc[:len(preds)],
    "Y": preds
})
submission.to_csv("submission_decisiontree.csv", index=False)
