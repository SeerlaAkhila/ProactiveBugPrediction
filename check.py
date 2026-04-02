import pandas as pd

# STEP 1: Load dataset with correct separator
ck = pd.read_csv("single-version-ck-oo.csv", sep=';')

# STEP 2: Remove extra spaces in column names
ck.columns = ck.columns.str.strip()

# STEP 3: Remove empty column (if exists)
ck = ck.loc[:, ck.columns != '']

# STEP 4: Select required features
data = ck[['numberOfLinesOfCode', 'cbo', 'rfc', 'wmc', 'bugs']]

# STEP 5: Rename columns
data.columns = ['LOC', 'CBO', 'RFC', 'WMC', 'bugs']

# STEP 6: Convert bugs → binary defect
data['defect'] = data['bugs'].apply(lambda x: 1 if x > 0 else 0)

# STEP 7: Drop original bugs column
data = data.drop('bugs', axis=1)

# STEP 8: Save cleaned dataset
data.to_csv("cleaned_dataset.csv", index=False)

# STEP 9: Check output
print(data.head())
print("\nDataset saved as cleaned_dataset.csv")