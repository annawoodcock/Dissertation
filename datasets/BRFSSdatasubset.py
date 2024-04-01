from sklearn.utils import resample
import pandas as pd

file_path = 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
diabetes_5050_data = pd.read_csv(file_path)

# Selecting the most relevant columns
selected_columns = ['Diabetes_binary', 'BMI', 'HighBP', 'HighChol', 'PhysActivity', 'Smoker', 'Age', 'GenHlth', 'CholCheck', 'HeartDiseaseorAttack']
reduced_data = diabetes_5050_data[selected_columns]

diabetes_positive = reduced_data[reduced_data['Diabetes_binary'] == 1]
diabetes_negative = reduced_data[reduced_data['Diabetes_binary'] == 0]

subset_positive = resample(diabetes_positive, n_samples=500, random_state=1)
subset_negative = resample(diabetes_negative, n_samples=500, random_state=1)

subset_1000 = pd.concat([subset_positive, subset_negative])

subset_1000 = subset_1000.sample(frac=1, random_state=1).reset_index(drop=True)

subset_1000.to_csv('diabetes_subset_1000.csv', index=False)

