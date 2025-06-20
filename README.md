📊 App Performance Prediction using Decision Tree Regressor
This repository contains a Python script for building a Decision Tree Regression model that predicts the performance (denoted as Y) of mobile applications using a variety of features. It includes data preprocessing, model training, evaluation, and prediction on test data for submission.

📁 Project Structure
bash
Copy
Edit
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── SampleSubmission.csv    # Sample submission format
├── script.py               # Main Python script (this code)
├── submission_decisiontree.csv  # Final predictions file (output)
🧠 Features of the Script
Loads training and test data from CSV files.

Cleans and transforms both numeric and categorical features.

Encodes categorical features using LabelEncoder.

Trains a DecisionTreeRegressor from sklearn.

Evaluates model performance using Mean Squared Error (MSE).

Predicts target values for the test set and saves a submission file.

🛠️ Libraries Used
pandas – for data manipulation

numpy – for numerical operations

scikit-learn – for machine learning models and utilities:

DecisionTreeRegressor

train_test_split

mean_squared_error

LabelEncoder

🧹 Data Preprocessing Steps
Dropped columns: X0, X9, X10, X11 (non-informative or redundant).

Encoded categorical columns: X1, X5, X7, X8 using LabelEncoder.

Cleaned numeric columns:

Converted X2 to numeric.

Removed "M" from X3, replaced "Varies with device" with NaN, and scaled values to bytes.

Cleaned X4 by removing +, ,.

Cleaned X5 by replacing "Free" with 0 and removing $ symbols.

Dropped columns like X6 if they were uninformative (e.g., single unique value).

Removed rows with missing values after processing.

📊 Model Details
Model: DecisionTreeRegressor

Parameters:

max_depth=5

random_state=42

Evaluation Metric: Mean Squared Error (MSE) on training data.

📤 Output
submission_decisiontree.csv:
A CSV file in the format required for submission, containing:

python-repl
Copy
Edit
row_id,Y
0,0.123
1,0.456
...
🧪 How to Run the Script
Place the following files in the working directory:

train.csv

test.csv

SampleSubmission.csv

Install required packages:

bash
Copy
Edit
pip install pandas numpy scikit-learn
Run the script:

bash
Copy
Edit
python script.py
Find the output file: submission_decisiontree.csv

🔍 Notes
Ensure that the categorical columns in test.csv contain values seen in train.csv. If not, unknown labels are safely handled with the addition of a placeholder "Unknown" class.

The script drops any rows with missing values after processing, which may affect the number of test predictions generated.

📧 Contact
For questions or suggestions, feel free to open an issue or contact the author.
