# Predictor
import json
import joblib
import pandas as pd
import numpy as np


# Load necessary components (model, label encoding, etc.)
model_path = "/content/drive/MyDrive/Thesis/predictor/model/xgboost_model.pkl"
label_mapping_path = '/content/drive/MyDrive/Thesis/drug_label_mapping.json'
feature_columns_path = '/content/drive/MyDrive/Thesis/predictor/model/xgboost_feature_columns.json'
encoding_mapping_path = '/content/drive/MyDrive/Thesis/encoding_mapping.json'


# Load the model
model = joblib.load(model_path)


# Load label encoding mapping (drug label → drug name)
with open(label_mapping_path, 'r') as f:
    drug_label_mapping = json.load(f)
label_to_drug = {v: k for k, v in drug_label_mapping.items()}


# Load feature columns used in the model
with open(feature_columns_path, 'r') as f:
    feature_columns = json.load(f)


# Load encoding mappings
with open(encoding_mapping_path, 'r') as f:
    encoding_mapping = json.load(f)


# Reverse Reference and Alternate mappings
encoding_mapping['Reference'] = {v: int(k) for k, v in encoding_mapping['Reference'].items()}
encoding_mapping['Alternate'] = {v: int(k) for k, v in encoding_mapping['Alternate'].items()}


def preprocess_input(chrom, pos, ref, alt):
    # Bin the Position feature (bin size of 1,000,000)
    binned_pos = pos // 1_000_000  # Binning by dividing the position by 1,000,000


    # Prepare the input data dictionary
    encoded_data = {
        'Chromosome': int(chrom) if isinstance(chrom, str) and chrom.isdigit() else chrom,
        'Position': binned_pos,
        'Reference': encoding_mapping['Reference'][ref],
        'Alternate': encoding_mapping['Alternate'][alt]
    }
   
    # Convert to DataFrame with correct feature order
    input_df = pd.DataFrame([encoded_data], columns=feature_columns)


    # Force everything to be float (matching model input)
    input_df = input_df.astype(float)


    return input_df


# Function to predict top 5 drugs
def predict_top_5_drugs(chrom, pos, ref, alt):
    # Preprocess the input data
    input_df = preprocess_input(chrom, pos, ref, alt)
    print(input_df)
   
    # Get prediction probabilities
    #y_pred_prob = model.predict_proba(input_df)[0]
    y_pred_prob = model.predict_proba(input_df.values)[0]
   
    # Get top 5 drug indices based on probabilities
    top_5_indices = np.argsort(y_pred_prob)[::-1][:5]
   
    # Convert indices to drug names and fetch their corresponding probabilities
    top_5_drugs_with_confidence = [
        (label_to_drug[i], y_pred_prob[i]) for i in top_5_indices
    ]
   
    return top_5_drugs_with_confidence




# Example usage
# Example usage
def main():
    chrom = input("Enter Chromosome (e.g., 17 or chr17): ")
   
    # Validate Position input
    while True:
        pos_input = input("Enter Position (e.g., 12345678): ")
        if pos_input.isdigit():
            pos = int(pos_input)
            break
        else:
            print("Please enter a valid numeric Position.")
   
    ref = input("Enter Reference base (A/C/G/T): ").strip().upper()
    alt = input("Enter Alternate base (A/C/G/T): ").strip().upper()


    try:
        top_5_drugs_with_confidence = predict_top_5_drugs(chrom, pos, ref, alt)
        print("\nTop 5 drug predictions with confidence:")
        for i, (drug, confidence) in enumerate(top_5_drugs_with_confidence, 1):
            print(f"{i}. {drug} (Confidence: {confidence:.2%})")
    except Exception as e:
        print(f"Error during prediction: {e}")




if __name__ == "__main__":
    main()
