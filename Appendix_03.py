# Test-Train Split
from sklearn.model_selection import train_test_split


# Count the number of occurrences for each drug class
class_counts = df['Drugs'].value_counts()
print(class_counts)


# Filter to keep only drugs with at least 5 samples
valid_classes = class_counts[class_counts >= 5].index


# Keep only rows in df with those valid classes
df_filtered = df[df['Drugs'].isin(valid_classes)].reset_index(drop=True)


print(f"Original dataset size: {df.shape[0]}")
print(f"Filtered dataset size: {df_filtered.shape[0]}")
print(f"Remaining drug classes: {df_filtered['Drugs'].nunique()}")




# Proceed with the split
from sklearn.model_selection import train_test_split


X = df_filtered.drop(columns=['Drugs'])
y = df_filtered['Drugs']


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.3, random_state=42, stratify=y
)


# Step 3: After the split, remove rare classes from both train and test
train_class_counts = y_train.value_counts()
test_class_counts = y_test.value_counts()


# Find classes with <5 samples in the test set
rare_classes_in_test = test_class_counts[(test_class_counts <= 5) | (test_class_counts >= 100)].index


# Remove those classes from both y_train and y_test
y_train_filtered = y_train[~y_train.isin(rare_classes_in_test)]
X_train_filtered = X_train.loc[y_train_filtered.index]


y_test_filtered = y_test[~y_test.isin(rare_classes_in_test)]
X_test_filtered = X_test.loc[y_test_filtered.index]


# Now check the class distribution
print(f"Class distribution in y_train after filtering: \n{y_train_filtered.value_counts()}")
print(f"Class distribution in y_test after filtering: \n{y_test_filtered.value_counts()}")


print(f"Number of features in X_train_filtered: {X_train_filtered.shape[1]}")
print(f"Number of features in X_test_filtered: {X_test_filtered.shape[1]}")

# Saving test and train dataframes and ensuring no null values
# Create the folder if it doesn't exist
if not os.path.exists(dataset_folder):
   os.makedirs(dataset_folder)
   print(f'Created folder: {dataset_folder}')
else:
   print(f'Folder already exists: {dataset_folder}')


# Save the final datasets to the new folder
X_train_filtered.to_csv(os.path.join(dataset_folder, 'X_train_filtered.csv'), index=False)
y_train_filtered.to_csv(os.path.join(dataset_folder, 'y_train_filtered.csv'), index=False)
X_test_filtered.to_csv(os.path.join(dataset_folder, 'X_test_filtered.csv'), index=False)
y_test_filtered.to_csv(os.path.join(dataset_folder, 'y_test_filtered.csv'), index=False)


print("Filtered train and test datasets have been saved as CSV files in the 'train_and_test_dataset' folder.")


# checking for null values
print("Nulls in X_train_filtered:", X_train_filtered.isnull().sum().sum())
print("Nulls in X_test_filtered:", X_test_filtered.isnull().sum().sum())


print("Shape before filtering:", X_train.shape, y_train.shape)
print("Shape after filtering:", X_train_filtered.shape, y_train_filtered.shape)


print("Index aligned?", all(X_train_filtered.index == y_train_filtered.index))


print("Any nulls in original df_filtered?", df_filtered.isnull().sum().sum())
