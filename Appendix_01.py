# Merge Duplicate rsID Entries and Consolidate Drug and Phenotype Information
    import pandas as pd


    # Load the dataset
    df = pd.read_excel('C:/Users/hp/Desktop/uni/Thesis/Cancer/retracing/PharmGKB.xlsx')


    # Ensure rsID is treated as a string
    df['rsID'] = df['rsID'].astype(str)


    # Function to convert values to string and handle NaNs
    def safe_join(values):
        return ', '.join(sorted(set(map(str, values.dropna()))))


    # Group by 'rsID' and aggregate 'Drugs' and 'Phenotype Categories' as comma-separated values
    df_grouped = df.groupby('rsID', as_index=False).agg({
        'Disease': 'first',  # Assuming the disease remains the same
        'PharmGKB ID': 'first',  # Keeping the first occurrence
        'Genes': 'first',  # Keeping the first occurrence
        'Drugs': safe_join,  # Merge unique drugs, handling NaN values
        'Significance': 'first',  # Keeping the first occurrence
        'Phenotype Categories': safe_join,  # Merge unique phenotype categories, handling NaN values
        'Pediatric': 'first'  # Keeping the first occurrence
    })


    # Reorder columns to ensure rsID remains correctly placed
    columns_order = ['rsID', 'Disease', 'PharmGKB ID', 'Genes', 'Drugs', 'Significance', 'Phenotype Categories', 'Pediatric']
    df_grouped = df_grouped[columns_order]


    # Save the cleaned dataframe
    df_grouped.to_excel('C:/Users/hp/Desktop/uni/Thesis/Cancer/retracing/merged_output.xlsx', index=False)


    print("Duplicate rsID records have been merged successfully.")

# HGVS Retrieval 
import pandas as pd
import requests
import time
import random


# Load the Excel file
file_path = "02- HGVS retrieval input.xlsx"  # Updated input file name
xls = pd.ExcelFile(file_path)


# Load the "Variant Annotations" sheet into a DataFrame
df = xls.parse(sheet_name="Variant Annotations")


# Function to retrieve HGVS notations using dbSNP API with retries
def get_hgvs_notations(rsid, max_retries=5):
    print(f"Fetching data for {rsid}...")  # Debug message
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/{rsid.lstrip('rs')}"
   
    retries = 0
    backoff = 1  # Start with 1 second delay


    while retries < max_retries:
        try:
            response = requests.get(url, timeout=10)  # Timeout to prevent hanging
            if response.status_code == 200:
                data = response.json()
                try:
                    # Extract all HGVS notations
                    alleles = data['primary_snapshot_data']['placements_with_allele'][0]['alleles']
                    hgvs_list = [allele['hgvs'] for allele in alleles if not allele['hgvs'].endswith('=')]


                    if hgvs_list:
                        return hgvs_list  # Return valid HGVS notations
               
                except (KeyError, IndexError):
                    pass  # Handle cases where data is missing


            print(f"⚠️ No valid HGVS notation found for {rsid}.")
            return ["HGVS not available"]
       
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {rsid}: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
            retries += 1


    print(f"❌ Max retries reached. Skipping {rsid}.")
    return ["HGVS not available"]  # Return if all retries fail


# Apply the function to each rsID and expand duplicated rows
expanded_rows = []
for _, row in df.iterrows():
    rsid = str(row['rsID'])
    hgvs_notations = get_hgvs_notations(rsid)
   
    for notation in hgvs_notations:
        new_row = row.copy()
        new_row['HGVS notation'] = notation
        expanded_rows.append(new_row)


    # Random delay between 1-3s to prevent API rate-limiting
    time.sleep(random.uniform(1, 3))


# Convert expanded data to a new DataFrame
df_expanded = pd.DataFrame(expanded_rows)


# Save the updated DataFrame to a new Excel file
df_expanded.to_excel("02- HGVS retrieval output.xlsx", index=False)  # Updated output file name


print("✅ Processing complete! Updated file saved as '02- HGVS retrieval output.xlsx'.")



# SMILES Retrieval
import pandas as pd
import pubchempy as pcp


# Load the Excel file
file_path = '/Users/oviyachandrasekar/Documents/SAISH/LEUKEMIA/drug descriptors/padel/lates.xlsx'
print(f" Loading Excel file from: {file_path}")


df = pd.read_excel(file_path)
print(f"✅ Loaded {len(df)} drug names from the file.")


# Function to get SMILES from PubChem
def get_smiles(drug_name):
    try:
        print(f"Searching for SMILES of: {drug_name}")
        compound = pcp.get_compounds(drug_name, 'name')
        if compound:
            smiles = compound[0].isomeric_smiles
            print(f"✅ Found SMILES: {smiles}")
            return smiles
        else:
            print(f" No SMILES found for: {drug_name}")
            return 'Not Found'
    except Exception as e:
        print(f"⚠️ Error retrieving {drug_name}: {e}")
        return 'Error'


# Apply the function to get SMILES for each drug
print(" Fetching SMILES for all drugs...")


df['SMILES'] = df['drug name'].apply(get_smiles)


# Save the updated file (overwrite or save as a new file)
updated_file_path = file_path  # Overwrite
df.to_excel(updated_file_path, index=False)


print(f" File updated successfully: {updated_file_path}")



# Converting SMILES excel to .smi
import pandas as pd


# Load Excel file
file_path = "/Users/oviyachandrasekar/Documents/SAISH/LEUKEMIA/drug descriptors/padel/lates copy.xlsx"
df = pd.read_excel(file_path)


# Define the columns (Update column names accordingly)
smiles_col = "SMILES"  # Column with SMILES
drug_col = "drug name"  # Optional: Column with drug names


# Create .smi file
smi_file = "/Users/oviyachandrasekar/Documents/SAISH/LEUKEMIA/drug descriptors/final3.smi"
df[[smiles_col, drug_col]].to_csv(smi_file, sep="\t", index=False, header=False)


print(f"SMILES file saved to: {smi_file} ✅")



# Duplicating rows to accommodate for multiple drugs associated with each record
import pandas as pd
import gspread
from google.colab import auth
from google.auth import default


# Authenticate Google Sheets API
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


# Define Google Sheets filename
dataset_name = "dataset_for_encoding"


# Load dataset from Google Sheets
spreadsheet = gc.open(dataset_name)
worksheet_dataset = spreadsheet.worksheet("dataset")
df_dataset = pd.DataFrame(worksheet_dataset.get_all_records())


# Function to expand rows where multiple drugs are listed
def expand_drug_rows(df):
    expanded_rows = []
    for _, row in df.iterrows():
        drugs = row["Drugs"].split(";")  # Split drugs by semicolon
        for drug in drugs:
            new_row = row.copy()
            new_row["Drugs"] = drug.strip()  # Assign only one drug per row
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)


# Apply transformation
df_adjusted = expand_drug_rows(df_dataset)


# Save the adjusted data to a new sheet 'drugs_adjusted'
worksheet_adjusted = spreadsheet.add_worksheet(title="drugs_adjusted", rows=df_adjusted.shape[0]+1, cols=df_adjusted.shape[1])
worksheet_adjusted.update([df_adjusted.columns.values.tolist()] + df_adjusted.values.tolist())


print("✅ Updated Google Sheet with 'drugs_adjusted' successfully!")

