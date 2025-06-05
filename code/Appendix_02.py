# PharmGKB encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# === Load dataset ===
input_file = "pharmgkb_unencoded.xlsx"
output_file = "encoded_pharmgkb_v2.xlsx"


print("Reading data from 'drugs_adjusted' sheet...")
df = pd.read_excel(input_file, sheet_name="drugs_adjusted")
df.columns = df.columns.str.strip()


label_mappings = {}


# === Drop rows where 'Genes' is missing ===
initial_rows = df.shape[0]
df = df.dropna(subset=['Genes'])
print(f"Dropped {initial_rows - df.shape[0]} rows where 'Genes' was missing.")


def label_encode_from_one(series, column_name):
    le = LabelEncoder()
    encoded = le.fit_transform(series)  # removed +1 for 0-based encoding
    label_mappings[column_name] = {i: label for i, label in enumerate(le.classes_)}
    return encoded


# === Label Encode 'Genes' ===
print("Label encoding 'Genes'...")
df['Genes'] = label_encode_from_one(df['Genes'], 'Genes')


# === Label Encode 'Phenotype Categories' ===
print("Label encoding 'Phenotype Categories'...")
df['Phenotype Categories'] = df['Phenotype Categories'].fillna("Other").astype(str)
df['Phenotype Categories'] = df['Phenotype Categories'].apply(lambda x: x.split(",")[0].strip())  # Use first if multiple
df['Phenotype Categories'] = label_encode_from_one(df['Phenotype Categories'], 'Phenotype Categories')


# === Encode 'Significance' column to binary ===
if 'Significance' in df.columns:
    print("Encoding 'Significance' column to binary...")
    df['Significance'] = df['Significance'].str.lower().map(lambda x: 1 if x == 'yes' else 0)
else:
    print("⚠️ 'Significance' column not found!")


# === Filter and label encode 'Disease' column ===
valid_diseases = {
    "Acute Lymphoid Leukemia",
    "Chronic myelogenous leukemia, BCR-ABL1 positive",
    "Leukemia",
    "Leukemia, Lymphocytic, Chronic, B-Cell",
    "Leukemia, Myeloid",
    "Leukemia, Myeloid, Acute",
    "Precursor T-Cell Lymphoblastic Leukemia-Lymphoma",
    "Therapy-related acute myeloid leukemia (t-AML)",
    "Burkitt Lymphoma",
    "Hodgkin Disease",
    "Lymphoma",
    "Lymphoma, B-Cell",
    "Lymphoma, T-Cell",
    "Lymphoma, Large B-Cell, Diffuse",
    "Mantle cell lymphoma",
    "Non-Hodgkin Lymphoma",
    "Primary central nervous system lymphoma"
}
print("Filtering and label encoding 'Disease'...")
df = df[df["Disease"].isin(valid_diseases)]
df['Disease'] = label_encode_from_one(df['Disease'], 'Disease')


# === Label Encode 'Reference' and 'Alternate' columns ===
for col in ['Reference', 'Alternate']:
    if col in df.columns:
        print(f"Label encoding '{col}'...")
        df[col] = label_encode_from_one(df[col].astype(str).str.upper().fillna("-"), col)
    else:
        print(f"⚠️ Column '{col}' not found in the dataset!")


# === Save to Excel ===
df.to_excel(output_file, sheet_name="encoded_dataset", index=False)
print("✅ Saved all encoded data to 'encoded_dataset'.")


# === Print label encodings ===
print("\n Label Encodings Used:")
for column, mapping in label_mappings.items():
    print(f"\n{column}:")
    for k, v in mapping.items():
        print(f"  {k} = {v}")



# Ensembl encoding 
# S# Starting with UNIQUE_HGVS_INPUT from ensembl_HGVSINPUT_deleted_columns


input_file = "ensembl_unencoded.xlsx"
input_sheet = "UNIQUE_HGVS_INPUT"
output_file = "encoded_ensembl_v2.xlsx"
output_sheet = "ENCODED"


import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import re


# Loading the Excel file
print("Loading Excel file...")
df = pd.read_excel(input_file, sheet_name=input_sheet, dtype=str).fillna('')
print(f"Loaded sheet '{input_sheet}' with shape: {df.shape}")
df.isnull().sum()


# FORMAT FOR COMMENTS:[step no.] [column being encoded] [encoding method] [new column name]


# 01. Converting all scientific notation to decimal
def convert_sci_to_float(x):
    try:
        return format(float(x), 'f')
    except:
        return x


df = df.apply(lambda col: col.apply(convert_sci_to_float))
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except (ValueError, TypeError):
        pass


    # Convert 'Chromosome' column to integer (if present)
if 'Chromosome' in df.columns:
    try:
        df['Chromosome'] = pd.to_numeric(df['Chromosome'], errors='coerce').fillna(0).astype(int)
    except Exception as e:
        print(f"Warning: Could not convert 'Chromosome' column to integer. {e}")


print("Step 01: Conversion complete. Data added to DataFrame.")


# display and save all your numbers in decimal notation
pd.set_option('display.float_format', '{:.10f}'.format)


#==============================================================================


# 02. Extracting CHROM_POS_REF_ALT


    # This part uses EnsemblAPI, which sometimes has network errors.
    # This has been ignored for now, performed separately
    # InsteadCHROM_POS_ALT_REF has been manually added into the input_file.


# Ensembl API URL for variant annotation
ENSEMBL_API_URL = "https://rest.ensembl.org/vep/human/hgvs/{}?content-type=application/json"


# Read HGVS notations from the input Excel file
print("Extracting CHROM_POS_REF_ALT...")
df = pd.read_excel(input_file, sheet_name=input_sheet, dtype=str).fillna('')
hgvs_list = df["#Uploaded_variation"].dropna().tolist()


# List to store variant data for Excel
variant_data = []


# Process HGVS notations and extract variant info
for hgvs in hgvs_list:
    try:
        # Request data from Ensembl API
        response = requests.get(ENSEMBL_API_URL.format(hgvs))
        if response.status_code == 200:
            data = response.json()
            for variant in data:
                chrom = variant["seq_region_name"]
                pos = variant["start"]
                ref, alt = variant["allele_string"].split("/")


                # Store data for DataFrame
                variant_data.append([hgvs, chrom, pos, ref, alt])


                print(f"Added variant: {hgvs} -> {chrom}:{pos} {ref}>{alt}")
        else:
            print(f"Failed to retrieve {hgvs}: {response.status_code}")
    except Exception as e:
        print(f"Error processing {hgvs}: {e}")


    # Convert the collected data into a DataFrame
df_variant_output = pd.DataFrame(variant_data, columns=["HGVS Notation", "Chromosome", "Position", "Reference", "Alternate"])


    # Save results to the main DataFrame
df = pd.merge(df, df_variant_output, left_on="#Uploaded_variation", right_on="HGVS Notation", how="left")


print("Step 02: Extraction complete. Data added to DataFrame.")


#==============================================================================


# 03. Encoding Location (POS) using Bin (discretizing) encoding


# Ensure the Position column is numeric
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')


bin_size = 1_000_000


    # Create bin feature based on position
df['Position_bin'] = df['Position'] // bin_size


    # Replace the 'Position' column with the 'Position_bin' column
df['Position'] = df['Position_bin']
df.drop(columns=['Position_bin'], inplace=True)


    # Delete the original 'Location' column if it exists
if 'Location' in df.columns:
    df.drop(columns=['Location'], inplace=True)
   
# Preview the result
#print(df[['Position']].head())


print("Step 03: Encoding Location (Position) complete. Data added to DataFrame.")


#==============================================================================


# === Step 04. Label Encoding Allele (REF and ALT) ===


# Ensure 'Ref' and 'Alt' columns are safe for encoding
df["Ref"] = df["Ref"].fillna("").astype(str).str.upper()
df["Alt"] = df["Alt"].fillna("").astype(str).str.upper()


# Label Encoding for 'Ref' and 'Alt', starting from 0
allele_mappings = {}


for column in ["Ref", "Alt"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column]) # Removed +1 to ensure 0-based encoding
    mapping = {i: label for i, label in enumerate(le.classes_)}  # Correct mapping
    allele_mappings[column] = mapping


# Print allele mappings
print("Step 04: Label Encoding Allele (REF and ALT) complete.")
print("Allele Label Mappings:")
for col, mapping in allele_mappings.items():
    print(f"\n{col}:")
    for label, base in mapping.items():
        print(f"  {label} = {base}")
       
#==============================================================================


# 05. Encoding Consequence - Label Encoding


def clean_and_label_encode_consequence(df, column="Consequence", new_col="Consequence_Encoded"):
    # Step 1: Standardize consequence entries
    df[column] = (
        df[column]
        .fillna("")
        .apply(lambda x: ",".join([
            part.strip().lower().replace(" ", "_")
            for part in x.split(",") if part.strip() != ""
        ]))
    )


    # Step 2: If multiple consequences, take the first one
    df[column] = df[column].apply(lambda x: x.split(",")[0] if x else "")


    # Step 3: Apply Label Encoding
    le = LabelEncoder()
    encoded_values = le.fit_transform(df[column]) # Removed +1 to ensure 0-based encoding
    df[new_col] = pd.to_numeric(encoded_values, downcast='integer')
   
    # Step 4: Print mapping
    mapping = {i: label for i, label in enumerate(le.classes_)}
    print("Step 05: Encoding Consequence complete. Label mapping:")
    for num, label in mapping.items():
        print(f"  {num} = {label}")


    # Step 5: Drop original column
    df.drop(columns=[column], inplace=True)


    return df


# Apply the function
df = clean_and_label_encode_consequence(df)


#==============================================================================


# 06. Encoding IMPACT - Ordinal Encoding


# Define ordinal mapping for IMPACT
impact_mapping = {
    "MODIFIER": 0,
    "LOW": 1,
    "MODERATE": 5,
    "HIGH": 10
}


print("Ordinal Mapping for IMPACT:")
for key, value in impact_mapping.items():
    print(f"  {key}: {value}")


# Replace IMPACT column with encoded version in same position and keep same name
if "IMPACT" in df.columns:
    print("Step 06: Encoding 'IMPACT' with ordinal encoding...")
    impact_index = df.columns.get_loc("IMPACT")
    encoded_series = df["IMPACT"].map(impact_mapping).fillna(0).astype(int)
    df.drop(columns=["IMPACT"], inplace=True)
    df.insert(impact_index, "IMPACT", encoded_series)


    print("✅ Step 06: Encoding IMPACT complete. Data added to DataFrame.")
else:
    print("⚠️ Step 06: 'IMPACT' column not found. Skipping.")


#==============================================================================


# 07. Encoding SYMBOL - Label Encoding (numeric, starting from 1)
if "SYMBOL" in df.columns:
    print("Step 07: Encoding 'SYMBOL' with label encoding...")
   
    df["SYMBOL"] = df["SYMBOL"].fillna("").astype(str)
    symbol_index = df.columns.get_loc("SYMBOL")


    label_encoder = LabelEncoder()
    encoded_values = label_encoder.fit_transform(df["SYMBOL"]) # Removed +1 to ensure 0-based encoding
    encoded_series = pd.Series(encoded_values, name="SYMBOL")


    # Drop and replace the original column with encoded values
    df.drop(columns=["SYMBOL"], inplace=True)
    df.insert(symbol_index, "SYMBOL", encoded_series.astype(int))


    # Mapping
    symbol_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("✅ Step 07: Encoding SYMBOL complete. Data added to DataFrame.")
    print("Label Mapping for SYMBOL:")
    for symbol, code in symbol_mapping.items():
        print(f"  {code} = {symbol}")
else:
    print("⚠️ Step 07: 'SYMBOL' column not found. Skipping.")
   
#==============================================================================


# 08. Encoding Feature_type - Label Encoding
if "Feature_type" in df.columns:
    print("Step 08: Encoding 'Feature_type' with label encoding...")
   
    feature_index = df.columns.get_loc("Feature_type")
    df["Feature_type"] = df["Feature_type"].fillna("").astype(str)


    le_feature = LabelEncoder()
    df["Feature_type_Encoded"] = le_feature.fit_transform(df["Feature_type"]) # Removed +1 to ensure 0-based encoding


    # Save mapping
    feature_mapping = {label: idx for idx, label in enumerate(le_feature.classes_)}
   
    '''
    # Save mapping
    feature_mapping = dict(zip(le_feature.classes_, le_feature.transform(le_feature.classes_) + 1))
    '''
   
    # Drop original column
    df.drop(columns=["Feature_type"], inplace=True)


    # Move encoded column to original position
    encoded_col = df.pop("Feature_type_Encoded")
    df.insert(feature_index, "Feature_type_Encoded", encoded_col)


    # Print mapping
    print("✅ Step 08: Encoding Feature_type complete. Data added to DataFrame.")
    print("Label Mapping for Feature_type:")
    for label, code in feature_mapping.items():
        print(f"  {code} = {label}")
else:
    print("⚠️ Step 08: 'Feature_type' column not found. Skipping.")


#==============================================================================


# 09. Encoding BIOTYPE - Label Encoding


# Get index of 'BIOTYPE' column
if 'BIOTYPE' in df.columns:
    biotype_index = df.columns.get_loc("BIOTYPE")


    # Label encode (including '-' as a valid category)
    le = LabelEncoder()
    encoded_biotype = le.fit_transform(df['BIOTYPE']) # Removed +1 to ensure 0-based encoding


    df.drop(columns=["BIOTYPE"], inplace=True)
    df.insert(biotype_index, "BIOTYPE", pd.to_numeric(encoded_biotype, downcast='integer'))


    # Mapping dictionary
    mapping = {i: label for i, label in enumerate(le.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 09: Encoding BIOTYPE complete. Label mapping:")
    for k, v in mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 09: 'BIOTYPE' column not found. Skipping.")


#==============================================================================


# 10. Encoding EXON/INTRON - Label Encoding


# Get index of 'EXON' column
if 'EXON' in df.columns:
    exon_index = df.columns.get_loc("EXON")


    # Fill NaNs with '-'
    df[["EXON", "INTRON"]] = df[["EXON", "INTRON"]].fillna("-")


    # Define classification logic
    def classify_region(row):
        if row['EXON'] != '-' and row['EXON'].strip() != '':
            return 1  # EXON = 1
        elif row['INTRON'] != '-' and row['INTRON'].strip() != '':
            return 0  # INTRON = 0
        else:
            return -1  # Handle empty or invalid cases (this is optional)


    # Apply classification
    df["EXON/INTRON"] = df.apply(classify_region, axis=1)


    # Label encode the EXON/INTRON column (ensure integer values)
    le = LabelEncoder()
    df["EXON/INTRON"] = le.fit_transform(df["EXON/INTRON"]) # Removed +1 to ensure 0-based encoding


    # Insert 'EXON/INTRON' in place of 'EXON', and drop 'EXON' and 'INTRON'
    col_data = df.pop("EXON/INTRON")
    df.drop(columns=["EXON", "INTRON"], inplace=True)
    df.insert(exon_index, "EXON/INTRON", col_data)


    # Mapping dictionary
    mapping = {i: label for i, label in enumerate(le.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 10: Encoding EXON/INTRON complete. Label mapping:")
    for k, v in mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 10: 'EXON' column not found. Skipping.")


#==============================================================================


# 11. Splitting Amino_acids into Ref_AA and Alt_AA
if "Amino_acids" in df.columns:
    # Split 'Amino_acids' column into 'Ref_AA' and 'Alt_AA' based on '/' separator
    df[['Ref_AA', 'Alt_AA']] = df['Amino_acids'].str.split('/', expand=True)
   
    # Ensure columns are in the same place as 'Amino_acids'
    amino_acids_index = df.columns.get_loc('Amino_acids')
    df.insert(amino_acids_index, 'Ref_AA', df.pop('Ref_AA'))
    df.insert(amino_acids_index + 1, 'Alt_AA', df.pop('Alt_AA'))
   
    # Drop the original 'Amino_acids' column
    df.drop(columns=["Amino_acids"], inplace=True)


    print("Step 11: Splitting Amino_acids into Ref_AA and Alt_AA complete. Data added to DataFrame.")
else:
    print("Step 11: 'Amino_acids' column not found. Skipping.")


#==============================================================================


# 12. Encoding Ref_AA and Alt_AA - Label Encoding


# Clean up '-' and empty entries
df['Ref_AA'] = df['Ref_AA'].replace(['-', ''], pd.NA)
df['Alt_AA'] = df['Alt_AA'].replace(['-', ''], pd.NA)


# Encode Ref_AA
if 'Ref_AA' in df.columns:
    ref_index = df.columns.get_loc("Ref_AA")
    le_ref = LabelEncoder()


    # Drop NaN values and apply label encoding
    ref_non_null = df['Ref_AA'].dropna()
    df.loc[ref_non_null.index, 'Ref_AA'] = le_ref.fit_transform(ref_non_null) # Removed +1 to ensure 0-based encoding
    df['Ref_AA'] = df['Ref_AA'].astype('Int64')  # Preserve nullable int type
   
    # Print mapping for Ref_AA
    ref_mapping = {i: label for i, label in enumerate(le_ref.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 12a: Encoding Ref_AA complete. Label mapping:")
    for k, v in ref_mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 12a: 'Ref_AA' column not found. Skipping.")


# Encode Alt_AA
if 'Alt_AA' in df.columns:
    alt_index = df.columns.get_loc("Alt_AA")
    le_alt = LabelEncoder()


    # Drop NaN values and apply label encoding
    alt_non_null = df['Alt_AA'].dropna()
    df.loc[alt_non_null.index, 'Alt_AA'] = le_alt.fit_transform(alt_non_null) # Removed +1 to ensure 0-based encoding
    df['Alt_AA'] = df['Alt_AA'].astype('Int64')  # Preserve nullable int type
   
    # Print mapping for Alt_AA
    alt_mapping = {i: label for i, label in enumerate(le_alt.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 12b: Encoding Alt_AA complete. Label mapping:")
    for k, v in alt_mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 12b: 'Alt_AA' column not found. Skipping.")


#==============================================================================


# 13. Splitting Codons into FromCodon and ToCodon
    # Insert FromCodon and ToCodon after Codons column
if 'Codons' in df.columns:
    codons_index = df.columns.get_loc("Codons") + 1
    from_to = df['Codons'].str.split('/', expand=True)
    df.insert(codons_index, 'FromCodon', from_to[0])
    df.insert(codons_index + 1, 'ToCodon', from_to[1])
    print("Step 13: Codons split into FromCodon and ToCodon (case-sensitive).")
else:
    print("Step 13: 'Codons' column not found. Skipping.")


#==============================================================================


# 14. Extracting Mutation_Type (transition/transversion), Codon_Positon, AA_Change (synonymous/nonsynonymous)
    # Codon Table
codon_table = {
    'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'TGT':'C', 'TGC':'C',
    'GAT':'D', 'GAC':'D',
    'GAA':'E', 'GAG':'E',
    'TTT':'F', 'TTC':'F',
    'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
    'CAT':'H', 'CAC':'H',
    'ATA':'I', 'ATT':'I', 'ATC':'I',
    'AAA':'K', 'AAG':'K',
    'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
    'ATG':'M',
    'AAT':'N', 'AAC':'N',
    'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'CAA':'Q', 'CAG':'Q',
    'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R', 'AGA':'R', 'AGG':'R',
    'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S',
    'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
    'TGG':'W',
    'TAT':'Y', 'TAC':'Y',
    'TAA':'*', 'TAG':'*', 'TGA':'*'
}


# Function to classify mutation
def classify_change(from_codon, to_codon):
    # Handle cases where codons are invalid or empty
    if from_codon == '-' or to_codon == '-':
        return -1, -1, -1  # return -1 for invalid cases
    if not from_codon or not to_codon:
        return None, None, None


    # Convert to uppercase to ensure consistency
    from_codon = from_codon.upper()
    to_codon = to_codon.upper()


    # Find positions where the codons differ
    position = [i for i in range(3) if from_codon[i] != to_codon[i]]
   
    if not position:
        return 1, -1, 1  # no change: return 1 for mutation type and amino acid change


    # Identify the position of mutation
    pos = position[0] + 1  # Position should be 1-indexed


    # Bases in the codon
    base_from = from_codon[position[0]]
    base_to = to_codon[position[0]]


    # Purines and Pyrimidines for mutation type classification
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}


    # Mutation type: 1 for transition (purine to purine or pyrimidine to pyrimidine), 2 for transversion
    if (base_from in purines and base_to in purines) or (base_from in pyrimidines and base_to in pyrimidines):
        mutation_type = 1  # Transition
    else:
        mutation_type = 2  # Transversion


    # Using a codon table (ensure you have the 'codon_table' dictionary available)
    aa_from = codon_table.get(from_codon, '?')
    aa_to = codon_table.get(to_codon, '?')


    # Amino acid change: 1 if no change, -1 if change
    aa_change = 1 if aa_from == aa_to else -1


    return mutation_type, pos, aa_change


    # Compute features
if 'FromCodon' in df.columns and 'ToCodon' in df.columns:
    codon_insert_index = df.columns.get_loc("ToCodon") + 1
    mutation_df = df.apply(
        lambda row: pd.Series(classify_change(row['FromCodon'], row['ToCodon'])),
        axis=1
    )
    mutation_df.columns = ['MutationType', 'CodonPosition', 'AA_Change']


    # Insert each column right after ToCodon
    for i, col in enumerate(mutation_df.columns):
        df.insert(codon_insert_index + i, col, mutation_df[col])    


    print("Step 14: MutationType, CodonPosition, and AA_Change extracted and inserted.")
else:
    print("Step 14: FromCodon or ToCodon not found. Skipping.")


#==============================================================================


# 15. Deleting FromCodon and ToCodon
if 'FromCodon' in df.columns and 'ToCodon' in df.columns:
    df.drop(columns=["FromCodon", "ToCodon"], inplace=True)


    print("Step 15: FromCodon and ToCodon deleted")
else:
    print("Step 15: FromCodon or ToCodon not found. Skipping.")


#==============================================================================


# 16. Encoding CANONICAL
df['CANONICAL'] = df['CANONICAL'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else -1)
print("Step 16: Encoding CANONICAL complete.")


#==============================================================================


# 17. Encoding APPRIS - Ordinal Encoding
# Replace '-' with NaN, then fill with 'Unknown'
df['APPRIS'] = df['APPRIS'].replace('-', pd.NA).fillna('Unknown')


# Get index of 'APPRIS' column
if 'APPRIS' in df.columns:
    appris_index = df.columns.get_loc("APPRIS")


    # Define category order with 'Unknown' first to ensure it is encoded as 0
    category_order = ["Unknown", "P1", "P2", "P3", "P4", "A1", "A2"]


    # Ordinal encoding
    encoder = OrdinalEncoder(categories=[category_order])
    encoded = encoder.fit_transform(df[['APPRIS']]).astype(int)


    # Drop original and insert encoded at same position
    df.drop(columns=["APPRIS"], inplace=True)
    df.insert(appris_index, "APPRIS", encoded)


    # Print mapping
    print("Step 17: Ordinal encoding of APPRIS complete. Label mapping:")
    for i, label in enumerate(category_order):
        print(f"  {i} = {label}")
else:
    print("Step 17: 'APPRIS' column not found. Skipping.")


#==============================================================================


# 18 + 19. Function to extract score inside parentheses and convert to float
def extract_score(text):
    match = re.search(r'\(([^)]+)', str(text))  # Capture text inside ()
    if match:
        try:
            return float(match.group(1))  # Convert to float
        except ValueError:
            return 0.0
    return 0.0


# 18. Encoding SIFT
if "SIFT" in df.columns:
    df["SIFT"] = df["SIFT"].apply(extract_score).astype(float)
    print("Step 18: Extracted SIFT scores added to DataFrame.")
else:
    print("Step 18: 'SIFT' column not found. Skipping.")


# 19. Encoding PolyPhen
if "PolyPhen" in df.columns:
    df["PolyPhen"] = df["PolyPhen"].apply(extract_score).astype(float)
    print("Step 19: Extracted PolyPhen scores added to DataFrame.")
else:
    print("Step 19: 'PolyPhen' column not found. Skipping.")


#==============================================================================


# Step 20. Encoding CLIN_SIG - Ordinal Encoding using custom priority


# Fill NA with empty string
df['CLIN_SIG'] = df['CLIN_SIG'].fillna("")


# Define your exact label priority
label_priority = {
    "not_provided": 0,
    "no_classifications_from_unflagged_records": 1,
    "benign": 2,
    "likely_benign": 3,
    "benign/likely_benign": 3,
    "uncertain_significance": 4,
    "conflicting_interpretations_of_pathogenicity": 5,
    "uncertain_risk_allele": 6,
    "likely_risk_allele": 7,
    "risk_factor": 8,
    "affects": 9,
    "association": 10,
    "likely_pathogenic": 11,
    "pathogenic/likely_pathogenic": 11,
    "pathogenic": 12,
    "drug_response": 13  # Optional: treat separately if needed
}


# Clean and parse multiple labels, selecting the one with the highest priority
def select_most_severe(entry):
    if not entry or entry.strip() in {"", "-"}:
        return "not_provided"
    labels = re.split(r'[,/]', entry.lower())
    labels = [re.sub(r'\s+', '_', lbl.strip()) for lbl in labels if lbl.strip()]
    ranked = [(label, label_priority.get(label, -1)) for label in labels]
    if not ranked:
        return "not_provided"
    most_severe = max(ranked, key=lambda x: x[1])
    return most_severe[0]


if "CLIN_SIG" in df.columns:
    # Get index of the 'CLIN_SIG' column
    clin_sig_index = df.columns.get_loc("CLIN_SIG")
   
    # Apply most severe label selector
    cleaned = df["CLIN_SIG"].apply(select_most_severe)


    # Map to numeric values using the label_priority
    encoded = cleaned.map(label_priority).fillna(0).astype(int)


    # Replace original column at the same index
    df.drop(columns=['CLIN_SIG'], inplace=True)
    df.insert(clin_sig_index, "CLIN_SIG", encoded)


    # Print mapping
    print("Step 20: CLIN_SIG encoding complete using custom priority.")
    print("Label mapping used (0-based):")
    for label, value in sorted(label_priority.items(), key=lambda x: x[1]):
        print(f"  {value}: {label}")
else:
    print("Step 20: 'CLIN_SIG' column not found. Skipping.")


# ==============================================================================


#ENCODING SOMATIC AND PHENO - MEAN ENCODING
# Shared function for mean encoding, now treating '-' as a data value
def encode_mean_from_comma_sep(val):
    try:
        val_str = str(val)
        values = []
        for x in val_str.split(','):
            x = x.strip()
            if x == "-":
                values.append(-1)  # Encode '-' as -1
            elif x.isdigit():
                values.append(int(x))
        return float(np.mean(values)) if values else 0.0
    except Exception as e:
        print(f"⚠️ Error processing value '{val}': {e}")
        return 0.0


# 21. Encoding SOMATIC
if "SOMATIC" in df.columns:
    somatic_index = df.columns.get_loc("SOMATIC")
    somatic_encoded = df["SOMATIC"].apply(encode_mean_from_comma_sep).astype(float)
    df.drop(columns=["SOMATIC"], inplace=True)
    df.insert(somatic_index, "SOMATIC", somatic_encoded)
    print("Step 21 ✅: 'SOMATIC' mean-encoded (hyphens encoded as -1).")
else:
    print("Step 21 ⚠️: 'SOMATIC' column not found. Skipping.")


# 22. Encoding PHENO
if "PHENO" in df.columns:
    pheno_index = df.columns.get_loc("PHENO")
    pheno_encoded = df["PHENO"].apply(encode_mean_from_comma_sep).astype(float)
    df.drop(columns=["PHENO"], inplace=True)
    df.insert(pheno_index, "PHENO", pheno_encoded)
    print("Step 22 ✅: 'PHENO' mean-encoded (hyphens encoded as -1).")
else:
    print("Step 22 ⚠️: 'PHENO' column not found. Skipping.")


# ==============================================================================
# 23. Encoding MaveDB_score - Mean Encoding (treat '-' as -1.0)
def compute_mean_score(cell):
    try:
        values = []
        for val in str(cell).split(','):
            val = val.strip()
            if val == "-":
                values.append(-1.0)
            elif val != "":
                values.append(float(val))
        return float(sum(values) / len(values)) if values else 0.0
    except Exception as e:
        print(f"⚠️ Error processing MaveDB_score value '{cell}': {e}")
        return 0.0


if "MaveDB_score" in df.columns:
    mavedb_index = df.columns.get_loc("MaveDB_score")
    print("Step 23: Encoding 'MaveDB_score' column...")


    encoded_scores = df["MaveDB_score"].apply(compute_mean_score).astype(float)
    df.drop(columns=["MaveDB_score"], inplace=True)
    df.insert(mavedb_index, "MaveDB_score", encoded_scores)


    print("Step 23 ✅: 'MaveDB_score' mean-encoded (hyphens encoded as -1.0).")
else:
    print("Step 23 ⚠️: 'MaveDB_score' column not found. Skipping.")


#==============================================================================
# 24. Encoding EVE_CLASS - Ordinal Encoding
if 'EVE_CLASS' in df.columns:
    # Normalize the 'EVE_CLASS' column: fill empty, strip, and set to consistent case
    df['EVE_CLASS'] = df['EVE_CLASS'].astype(str).str.strip().replace("-", "").replace("nan", "")


    # Define mapping
    eve_mapping = {
        "Benign": 0,
        "Uncertain": 1,
        "Pathogenic": 2
    }


    # Apply encoding based on mapping
    df['EVE_CLASS'] = df['EVE_CLASS'].map(eve_mapping)  # Replace 'EVE_CLASS' directly with the encoded values


    # Ensure no NaN values after mapping (replacing NaN with 0, or another value if needed)
    df['EVE_CLASS'] = df['EVE_CLASS'].fillna(0).astype(int)


    # Save to Excel
    df.to_excel(output_file, sheet_name=output_sheet, index=False)


    print("Step 24 ✅: 'EVE_CLASS' column replaced with encoded values (saved as numbers).")
else:
    print("Step 24 ⚠️: 'EVE_CLASS' column not found. Skipping.")


#==============================================================================


# 25. Encoding am_class - Ordinal Encoding
# Clean and prepare data
df['am_class'] = df['am_class'].replace('-', pd.NA)


# Get index of 'am_class' column
if 'am_class' in df.columns:
    am_class_index = df.columns.get_loc("am_class")


    # Define the severity order
    severity_order = {
        'likely_benign': 0,
        'ambiguous': 1,
        'likely_pathogenic': 2
    }


    # Apply encoding
    df['am_class'] = df['am_class'].map(severity_order)


    # Ensure no NaN values after mapping (replacing NaN with 0, or another value if needed)
    df['am_class'] = df['am_class'].fillna(0).astype(int)


    print("Step 25 ✅: am_class encoding complete. Data added to DataFrame (saved as numbers).")
else:
    print("Step 25 ⚠️: 'am_class' column not found. Skipping.")


#==============================================================================
# DEFINING COMMON FUNCTIONS USED FOR THE REMAINING STEPS


# Encoding function to be applied to columns
def encode_column(df, col, encoding_map, missing_value=''):
    if col in df.columns:
        encoded = df[col].map(encoding_map)
        encoded = encoded.where(df[col].notna(), missing_value)
        col_index = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        df.insert(col_index, col, encoded)


        print(f"Step 26. -31. : Encoding '{col}' complete. Label mapping:")
        for label, value in encoding_map.items():
            print(f"  {value} = {label}")
    else:
        print(f"Step 26. -31. : '{col}' column not found. Skipping.")
   
def clean_and_encode_predictions(df):
    # Define columns and encoding maps
    lrt_col = 'LRT_pred'
    lrt_encoding = {'D': 1, 'N': 0}


    other_pred_cols = [
        'LIST-S2_pred',
        'MetaLR_pred',
        'M-CAP_pred',
        'MetaRNN_pred',
        'MetaSVM_pred'
    ]
    other_encoding = {'D': 1, 'T': 0}


    additional_pred_cols = {
        'PROVEAN_pred': {'D': 1, 'N': 0},
        'PrimateAI_pred': {'D': 1, 'T': 0},
        'fathmm-MKL_coding_pred': {'D': 1, 'N': 0},
        'fathmm-XF_coding_pred': {'D': 1, 'N': 0},
    }


#==============================================================================


# 26. Encoding LRT_pred
    encode_column(df, lrt_col, lrt_encoding)
   
#==============================================================================
# 27. - 31. Encoding other prediction columns


    # 27. LIST-S2_pred
    # 28. Encoding M-CAP_pred
    # 29. Encoding MetaLR_pred
    # 30. Encoding MetaRNN_pred
    # 31. Encoding MetaSVM_pred


        # Encode other prediction columns
    for col in other_pred_cols:
        encode_column(df, col, other_encoding)
   
#==============================================================================
# 32. - 35. Encoding additional prediction columns
    # 32. Encoding PROVEAN_pred
    # 33. Encoding PrimateAI_pred
    # 34. Encoding fathmm-MKL_coding_pred
    # 35. Encoding fathmm-XF_coding_pred


    for col, mapping in additional_pred_cols.items():
        encode_column(df, col, mapping)
    return df
#==============================================================================


# Apply COMMON FUNCTIONS USED FOR THE REMAINING STEPS directly to loaded df
df = clean_and_encode_predictions(df)


#==============================================================================


# 36. Encoding MutationAssessor_pred
if 'MutationAssessor_pred' in df.columns:
    ma_encoding = {'H': 1.0, 'M': 0.75, 'L': 0.5, 'N': 0.0, '0': 0.0}
    encoded_ma = df['MutationAssessor_pred'].map(ma_encoding)
    encoded_ma = encoded_ma.where(df['MutationAssessor_pred'].notna(), '')
    col_index = df.columns.get_loc('MutationAssessor_pred')
    df.drop(columns=['MutationAssessor_pred'], inplace=True)
    df.insert(col_index, 'MutationAssessor_pred', encoded_ma)
    print("Step 36: 'MutationAssessor_pred' encoding complete. Label mapping:")
    for label, value in ma_encoding.items():
        print(f"  {value} = {label}")
else:
    print("Step 36: 'MutationAssessor_pred' column not found. Skipping.")
         
#==============================================================================


# 37. Encoding MutationTaster_pred


def encode_mutationtaster(val):
    if pd.isna(val) or val in ['-', '0', '']:
        return ''
    scores = {'D': 1.0, 'P': 0.75, 'N': 0.25}
    split_vals = val.split(',')
    valid_scores = [scores[v] for v in split_vals if v in scores]
    return np.mean(valid_scores) if valid_scores else ''


if 'MutationTaster_pred' in df.columns:
    encoded_mt = df['MutationTaster_pred'].apply(encode_mutationtaster)
    col_index = df.columns.get_loc('MutationTaster_pred')
    df.drop(columns=['MutationTaster_pred'], inplace=True)
    df.insert(col_index, 'MutationTaster_pred', encoded_mt)
    print("Step 37: 'MutationTaster_pred' encoding complete. Label mapping:")
    print("  1.0 = D\n  0.75 = P\n  0.25 = N")
else:
    print("Step 37: 'MutationTaster_pred' column not found. Skipping.")
   
#==============================================================================


# 38. Encoding MutationTaster_score (some values are separated by commas)


def encode_mutation_taster_score(score):
    if pd.isna(score) or score == '-':
        return np.nan
    try:
        score_list = [float(x) for x in str(score).split(',') if x]
        return np.mean(score_list) if score_list else np.nan
    except Exception as e:
        print(f"⚠️ Error processing score: {score} ({e})")
        return np.nan


if 'MutationTaster_score' in df.columns:
    encoded_mt_score = df['MutationTaster_score'].apply(encode_mutation_taster_score)
    col_index = df.columns.get_loc('MutationTaster_score')
    df.drop(columns=['MutationTaster_score'], inplace=True)
    df.insert(col_index, 'MutationTaster_score', encoded_mt_score)
    # Convert to decimal format (string)
    df['MutationTaster_score'] = df['MutationTaster_score'].apply(lambda x: f"{x:.10f}" if pd.notna(x) else x)
    print("Step 38: 'MutationTaster_score' encoding complete. Data added to DataFrame.")
else:
    print("Step 38: 'MutationTaster_score' column not found. Skipping.")


   
#==============================================================================


# 39. Encoding SiPhy_29way_pi into 4 columns (values separated by colons)
def split_siphy(row):
    if pd.isna(row) or row == '-' or row.count(':') != 3:
        return [np.nan, np.nan, np.nan, np.nan]
    try:
        return [float(x) for x in row.split(':')]
    except:
        return [np.nan, np.nan, np.nan, np.nan]


if 'SiPhy_29way_pi' in df.columns:
    siphy_cols = df['SiPhy_29way_pi'].apply(split_siphy).tolist()
    siphy_split_df = pd.DataFrame(siphy_cols, columns=['SiPhy_A', 'SiPhy_C', 'SiPhy_G', 'SiPhy_T'])


    # Find the column index of the original column
    col_index = df.columns.get_loc('SiPhy_29way_pi')


    # Drop the original column
    df.drop(columns=['SiPhy_29way_pi'], inplace=True)


    # Insert the new columns at the original column's position
    for i, col_name in enumerate(['SiPhy_A', 'SiPhy_C', 'SiPhy_G', 'SiPhy_T']):
        df.insert(col_index + i, col_name, siphy_split_df[col_name])


    print("Step 39: 'SiPhy_29way_pi' encoding complete. Columns replaced in original position.")
else:
    print("Step 39: 'SiPhy_29way_pi' column not found. Skipping.")


#==============================================================================


# 40. Encoding clinvar_clnsig - Ordinal Encoding using label_priority_clnsig
print("Step 40: Encoding 'clinvar_clnsig' using ordinal encoding...")


# Define the priority dictionary
label_priority_clnsig = {
    "clinvar_clnsig_no_classification_for_the_single_variant": 0,
    "clinvar_clnsig_other": 0,
    "clinvar_clnsig_Benign": 1,
    "clinvar_clnsig_Likely_benign": 2,
    "clinvar_clnsig_Benign/Likely_benign": 2,
    "clinvar_clnsig_Uncertain_significance": 3,
    "clinvar_clnsig_Conflicting_classifications_of_pathogenicity": 4,
    "clinvar_clnsig_risk_factor": 5,
    "clinvar_clnsig_association": 6,
    "clinvar_clnsig_Affects": 7,
    "clinvar_clnsig_Likely_pathogenic": 8,
    "clinvar_clnsig_Pathogenic": 9,
    "clinvar_clnsig_drug_response": 10
}


if 'clinvar_clnsig' in df.columns:
    # Clean the 'clinvar_clnsig' column by replacing '-' and NaN with empty strings
    df['clinvar_clnsig_clean'] = df['clinvar_clnsig'].replace('-', '').fillna('')


    # Split the entries and find max priority per row
    def get_max_priority(entry):
        labels = [label.strip() for label in entry.split(',') if label.strip()]
        if not labels:
            return 0  # Treat empty or missing as lowest priority
        priorities = [label_priority_clnsig.get(label, 0) for label in labels]
        return max(priorities)


    # Apply ordinal encoding
    encoded_values = df['clinvar_clnsig_clean'].apply(get_max_priority)


    # Get the original column index
    col_index = df.columns.get_loc('clinvar_clnsig')


    # Drop original columns and insert encoded one
    df.drop(columns=['clinvar_clnsig', 'clinvar_clnsig_clean'], inplace=True)
    df.insert(col_index, 'clinvar_clnsig', pd.to_numeric(encoded_values, downcast='integer'))


    print("Step 40: Ordinal encoding complete. Sample mapping:")
    for k, v in sorted(label_priority_clnsig.items(), key=lambda x: x[1]):
        print(f"{v} = {k}")
else:
    print("Step 40: 'clinvar_clnsig' column not found. Skipping.")


#==============================================================================


# 41. CONVERTING NUMBERS SAVED AS TEXT TO NUMBERS
def convert_text_to_numbers(df):
    # Convert any columns with numbers stored as text to actual numeric values.
    # Non-numeric entries will be left unchanged.
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception:
            pass  # Keep original column if it can't be converted


    print("Step 41: Text-formatted numbers converted to numeric, non-numeric entries left unchanged.")
    return df


df = convert_text_to_numbers(df)


#==============================================================================
#==============================================================================


# 43.Process OpenTargets_l2g column
def encode_l2g_column(df, column='OpenTargets_l2g'):
    import numpy as np


    def parse_values(val):
        try:
            return [float(x) for x in str(val).split(',')]
        except:
            return [np.nan]


    # Parse the values
    parsed = df[column].apply(parse_values)


    # Compute stats
    col_mean = parsed.apply(np.mean)
    col_max = parsed.apply(np.max)
    col_min = parsed.apply(np.min)
    col_std = parsed.apply(np.std)


    mean_col = f"{column}_mean"
    max_col = f"{column}_max"
    min_col = f"{column}_min"
    std_col = f"{column}_std"


    # Find the column index
    col_idx = df.columns.get_loc(column)


    df.drop(columns=[column], inplace=True)
    df.insert(col_idx, mean_col, col_mean)
    df.insert(col_idx + 1, max_col, col_max)
    df.insert(col_idx + 2, min_col, col_min)
    df.insert(col_idx + 3, std_col, col_std)


    print(f"✅ Replaced '{column}' with: {mean_col}, {max_col}, {min_col}, {std_col}")


    return df


df = encode_l2g_column(df)


#==============================================================================


# 43. Saving the DF as an Excel File


# Converting all scientific notation to decimal once again
df = df.apply(lambda col: col.apply(convert_sci_to_float))


# Convert all '-' and 'nan' values in the entire dataframe to 0
def replace_dash_and_nan_with_zero(df):
    # Replace '-' and 'nan' strings with 0 in the entire dataframe
    df = df.replace(['-', 'nan', 'NaN'], 0)
   
    # replacing applymap with DataFrame.apply
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if isinstance(x, str) and x.strip().lower() == 'nan' else x)


    # Optional: Infer correct data types (ensure numeric columns stay numeric)
    df = df.infer_objects(copy=False)


    print("All '-' and 'nan' strings have been replaced with 0 values.")
    return df


# Apply the function to your dataframe
df = replace_dash_and_nan_with_zero(df)


# Drop the specified columns if they exist
columns_to_drop = ['ALLELE_AFTER_SLASH', 'REF_ALLELE', 'Codons']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


print("✅ Specified columns dropped (if present).")


print("Writing dataframe to a new file...")
with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name=output_sheet, index=False, na_rep='', float_format="%.10f")


print(f"✅ All encodings complete. Output saved to '{output_sheet}' in '{output_file}'")


tarting with UNIQUE_HGVS_INPUT from ensembl_HGVSINPUT_deleted_columns


input_file = "ensembl_unencoded.xlsx"
input_sheet = "UNIQUE_HGVS_INPUT"
output_file = "encoded_ensembl_v2.xlsx"
output_sheet = "ENCODED"


import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import re


# Loading the Excel file
print("Loading Excel file...")
df = pd.read_excel(input_file, sheet_name=input_sheet, dtype=str).fillna('')
print(f"Loaded sheet '{input_sheet}' with shape: {df.shape}")
df.isnull().sum()


# FORMAT FOR COMMENTS:[step no.] [column being encoded] [encoding method] [new column name]


# 01. Converting all scientific notation to decimal
def convert_sci_to_float(x):
    try:
        return format(float(x), 'f')
    except:
        return x


df = df.apply(lambda col: col.apply(convert_sci_to_float))
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except (ValueError, TypeError):
        pass


    # Convert 'Chromosome' column to integer (if present)
if 'Chromosome' in df.columns:
    try:
        df['Chromosome'] = pd.to_numeric(df['Chromosome'], errors='coerce').fillna(0).astype(int)
    except Exception as e:
        print(f"Warning: Could not convert 'Chromosome' column to integer. {e}")


print("Step 01: Conversion complete. Data added to DataFrame.")


# display and save all your numbers in decimal notation
pd.set_option('display.float_format', '{:.10f}'.format)


#==============================================================================


# 02. Extracting CHROM_POS_REF_ALT


    # This part uses EnsemblAPI, which sometimes has network errors.
    # This has been ignored for now, performed separately
    # InsteadCHROM_POS_ALT_REF has been manually added into the input_file.


# Ensembl API URL for variant annotation
ENSEMBL_API_URL = "https://rest.ensembl.org/vep/human/hgvs/{}?content-type=application/json"


# Read HGVS notations from the input Excel file
print("Extracting CHROM_POS_REF_ALT...")
df = pd.read_excel(input_file, sheet_name=input_sheet, dtype=str).fillna('')
hgvs_list = df["#Uploaded_variation"].dropna().tolist()


# List to store variant data for Excel
variant_data = []


# Process HGVS notations and extract variant info
for hgvs in hgvs_list:
    try:
        # Request data from Ensembl API
        response = requests.get(ENSEMBL_API_URL.format(hgvs))
        if response.status_code == 200:
            data = response.json()
            for variant in data:
                chrom = variant["seq_region_name"]
                pos = variant["start"]
                ref, alt = variant["allele_string"].split("/")


                # Store data for DataFrame
                variant_data.append([hgvs, chrom, pos, ref, alt])


                print(f"Added variant: {hgvs} -> {chrom}:{pos} {ref}>{alt}")
        else:
            print(f"Failed to retrieve {hgvs}: {response.status_code}")
    except Exception as e:
        print(f"Error processing {hgvs}: {e}")


    # Convert the collected data into a DataFrame
df_variant_output = pd.DataFrame(variant_data, columns=["HGVS Notation", "Chromosome", "Position", "Reference", "Alternate"])


    # Save results to the main DataFrame
df = pd.merge(df, df_variant_output, left_on="#Uploaded_variation", right_on="HGVS Notation", how="left")


print("Step 02: Extraction complete. Data added to DataFrame.")


#==============================================================================


# 03. Encoding Location (POS) using Bin (discretizing) encoding


# Ensure the Position column is numeric
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')


bin_size = 1_000_000


    # Create bin feature based on position
df['Position_bin'] = df['Position'] // bin_size


    # Replace the 'Position' column with the 'Position_bin' column
df['Position'] = df['Position_bin']
df.drop(columns=['Position_bin'], inplace=True)


    # Delete the original 'Location' column if it exists
if 'Location' in df.columns:
    df.drop(columns=['Location'], inplace=True)
   
# Preview the result
#print(df[['Position']].head())


print("Step 03: Encoding Location (Position) complete. Data added to DataFrame.")


#==============================================================================


# === Step 04. Label Encoding Allele (REF and ALT) ===


# Ensure 'Ref' and 'Alt' columns are safe for encoding
df["Ref"] = df["Ref"].fillna("").astype(str).str.upper()
df["Alt"] = df["Alt"].fillna("").astype(str).str.upper()


# Label Encoding for 'Ref' and 'Alt', starting from 0
allele_mappings = {}


for column in ["Ref", "Alt"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column]) # Removed +1 to ensure 0-based encoding
    mapping = {i: label for i, label in enumerate(le.classes_)}  # Correct mapping
    allele_mappings[column] = mapping


# Print allele mappings
print("Step 04: Label Encoding Allele (REF and ALT) complete.")
print("Allele Label Mappings:")
for col, mapping in allele_mappings.items():
    print(f"\n{col}:")
    for label, base in mapping.items():
        print(f"  {label} = {base}")
       
#==============================================================================


# 05. Encoding Consequence - Label Encoding


def clean_and_label_encode_consequence(df, column="Consequence", new_col="Consequence_Encoded"):
    # Step 1: Standardize consequence entries
    df[column] = (
        df[column]
        .fillna("")
        .apply(lambda x: ",".join([
            part.strip().lower().replace(" ", "_")
            for part in x.split(",") if part.strip() != ""
        ]))
    )


    # Step 2: If multiple consequences, take the first one
    df[column] = df[column].apply(lambda x: x.split(",")[0] if x else "")


    # Step 3: Apply Label Encoding
    le = LabelEncoder()
    encoded_values = le.fit_transform(df[column]) # Removed +1 to ensure 0-based encoding
    df[new_col] = pd.to_numeric(encoded_values, downcast='integer')
   
    # Step 4: Print mapping
    mapping = {i: label for i, label in enumerate(le.classes_)}
    print("Step 05: Encoding Consequence complete. Label mapping:")
    for num, label in mapping.items():
        print(f"  {num} = {label}")


    # Step 5: Drop original column
    df.drop(columns=[column], inplace=True)


    return df


# Apply the function
df = clean_and_label_encode_consequence(df)


#==============================================================================


# 06. Encoding IMPACT - Ordinal Encoding


# Define ordinal mapping for IMPACT
impact_mapping = {
    "MODIFIER": 0,
    "LOW": 1,
    "MODERATE": 5,
    "HIGH": 10
}


print("Ordinal Mapping for IMPACT:")
for key, value in impact_mapping.items():
    print(f"  {key}: {value}")


# Replace IMPACT column with encoded version in same position and keep same name
if "IMPACT" in df.columns:
    print("Step 06: Encoding 'IMPACT' with ordinal encoding...")
    impact_index = df.columns.get_loc("IMPACT")
    encoded_series = df["IMPACT"].map(impact_mapping).fillna(0).astype(int)
    df.drop(columns=["IMPACT"], inplace=True)
    df.insert(impact_index, "IMPACT", encoded_series)


    print("✅ Step 06: Encoding IMPACT complete. Data added to DataFrame.")
else:
    print("⚠️ Step 06: 'IMPACT' column not found. Skipping.")


#==============================================================================


# 07. Encoding SYMBOL - Label Encoding (numeric, starting from 1)
if "SYMBOL" in df.columns:
    print("Step 07: Encoding 'SYMBOL' with label encoding...")
   
    df["SYMBOL"] = df["SYMBOL"].fillna("").astype(str)
    symbol_index = df.columns.get_loc("SYMBOL")


    label_encoder = LabelEncoder()
    encoded_values = label_encoder.fit_transform(df["SYMBOL"]) # Removed +1 to ensure 0-based encoding
    encoded_series = pd.Series(encoded_values, name="SYMBOL")


    # Drop and replace the original column with encoded values
    df.drop(columns=["SYMBOL"], inplace=True)
    df.insert(symbol_index, "SYMBOL", encoded_series.astype(int))


    # Mapping
    symbol_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("✅ Step 07: Encoding SYMBOL complete. Data added to DataFrame.")
    print("Label Mapping for SYMBOL:")
    for symbol, code in symbol_mapping.items():
        print(f"  {code} = {symbol}")
else:
    print("⚠️ Step 07: 'SYMBOL' column not found. Skipping.")
   
#==============================================================================


# 08. Encoding Feature_type - Label Encoding
if "Feature_type" in df.columns:
    print("Step 08: Encoding 'Feature_type' with label encoding...")
   
    feature_index = df.columns.get_loc("Feature_type")
    df["Feature_type"] = df["Feature_type"].fillna("").astype(str)


    le_feature = LabelEncoder()
    df["Feature_type_Encoded"] = le_feature.fit_transform(df["Feature_type"]) # Removed +1 to ensure 0-based encoding


    # Save mapping
    feature_mapping = {label: idx for idx, label in enumerate(le_feature.classes_)}
   
    '''
    # Save mapping
    feature_mapping = dict(zip(le_feature.classes_, le_feature.transform(le_feature.classes_) + 1))
    '''
   
    # Drop original column
    df.drop(columns=["Feature_type"], inplace=True)


    # Move encoded column to original position
    encoded_col = df.pop("Feature_type_Encoded")
    df.insert(feature_index, "Feature_type_Encoded", encoded_col)


    # Print mapping
    print("✅ Step 08: Encoding Feature_type complete. Data added to DataFrame.")
    print("Label Mapping for Feature_type:")
    for label, code in feature_mapping.items():
        print(f"  {code} = {label}")
else:
    print("⚠️ Step 08: 'Feature_type' column not found. Skipping.")


#==============================================================================


# 09. Encoding BIOTYPE - Label Encoding


# Get index of 'BIOTYPE' column
if 'BIOTYPE' in df.columns:
    biotype_index = df.columns.get_loc("BIOTYPE")


    # Label encode (including '-' as a valid category)
    le = LabelEncoder()
    encoded_biotype = le.fit_transform(df['BIOTYPE']) # Removed +1 to ensure 0-based encoding


    df.drop(columns=["BIOTYPE"], inplace=True)
    df.insert(biotype_index, "BIOTYPE", pd.to_numeric(encoded_biotype, downcast='integer'))


    # Mapping dictionary
    mapping = {i: label for i, label in enumerate(le.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 09: Encoding BIOTYPE complete. Label mapping:")
    for k, v in mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 09: 'BIOTYPE' column not found. Skipping.")


#==============================================================================


# 10. Encoding EXON/INTRON - Label Encoding


# Get index of 'EXON' column
if 'EXON' in df.columns:
    exon_index = df.columns.get_loc("EXON")


    # Fill NaNs with '-'
    df[["EXON", "INTRON"]] = df[["EXON", "INTRON"]].fillna("-")


    # Define classification logic
    def classify_region(row):
        if row['EXON'] != '-' and row['EXON'].strip() != '':
            return 1  # EXON = 1
        elif row['INTRON'] != '-' and row['INTRON'].strip() != '':
            return 0  # INTRON = 0
        else:
            return -1  # Handle empty or invalid cases (this is optional)


    # Apply classification
    df["EXON/INTRON"] = df.apply(classify_region, axis=1)


    # Label encode the EXON/INTRON column (ensure integer values)
    le = LabelEncoder()
    df["EXON/INTRON"] = le.fit_transform(df["EXON/INTRON"]) # Removed +1 to ensure 0-based encoding


    # Insert 'EXON/INTRON' in place of 'EXON', and drop 'EXON' and 'INTRON'
    col_data = df.pop("EXON/INTRON")
    df.drop(columns=["EXON", "INTRON"], inplace=True)
    df.insert(exon_index, "EXON/INTRON", col_data)


    # Mapping dictionary
    mapping = {i: label for i, label in enumerate(le.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 10: Encoding EXON/INTRON complete. Label mapping:")
    for k, v in mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 10: 'EXON' column not found. Skipping.")


#==============================================================================


# 11. Splitting Amino_acids into Ref_AA and Alt_AA
if "Amino_acids" in df.columns:
    # Split 'Amino_acids' column into 'Ref_AA' and 'Alt_AA' based on '/' separator
    df[['Ref_AA', 'Alt_AA']] = df['Amino_acids'].str.split('/', expand=True)
   
    # Ensure columns are in the same place as 'Amino_acids'
    amino_acids_index = df.columns.get_loc('Amino_acids')
    df.insert(amino_acids_index, 'Ref_AA', df.pop('Ref_AA'))
    df.insert(amino_acids_index + 1, 'Alt_AA', df.pop('Alt_AA'))
   
    # Drop the original 'Amino_acids' column
    df.drop(columns=["Amino_acids"], inplace=True)


    print("Step 11: Splitting Amino_acids into Ref_AA and Alt_AA complete. Data added to DataFrame.")
else:
    print("Step 11: 'Amino_acids' column not found. Skipping.")


#==============================================================================


# 12. Encoding Ref_AA and Alt_AA - Label Encoding


# Clean up '-' and empty entries
df['Ref_AA'] = df['Ref_AA'].replace(['-', ''], pd.NA)
df['Alt_AA'] = df['Alt_AA'].replace(['-', ''], pd.NA)


# Encode Ref_AA
if 'Ref_AA' in df.columns:
    ref_index = df.columns.get_loc("Ref_AA")
    le_ref = LabelEncoder()


    # Drop NaN values and apply label encoding
    ref_non_null = df['Ref_AA'].dropna()
    df.loc[ref_non_null.index, 'Ref_AA'] = le_ref.fit_transform(ref_non_null) # Removed +1 to ensure 0-based encoding
    df['Ref_AA'] = df['Ref_AA'].astype('Int64')  # Preserve nullable int type
   
    # Print mapping for Ref_AA
    ref_mapping = {i: label for i, label in enumerate(le_ref.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 12a: Encoding Ref_AA complete. Label mapping:")
    for k, v in ref_mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 12a: 'Ref_AA' column not found. Skipping.")


# Encode Alt_AA
if 'Alt_AA' in df.columns:
    alt_index = df.columns.get_loc("Alt_AA")
    le_alt = LabelEncoder()


    # Drop NaN values and apply label encoding
    alt_non_null = df['Alt_AA'].dropna()
    df.loc[alt_non_null.index, 'Alt_AA'] = le_alt.fit_transform(alt_non_null) # Removed +1 to ensure 0-based encoding
    df['Alt_AA'] = df['Alt_AA'].astype('Int64')  # Preserve nullable int type
   
    # Print mapping for Alt_AA
    alt_mapping = {i: label for i, label in enumerate(le_alt.classes_)} # Removed +1 to ensure 0-based encoding
    print("Step 12b: Encoding Alt_AA complete. Label mapping:")
    for k, v in alt_mapping.items():
        print(f"  {k} = {v}")
else:
    print("Step 12b: 'Alt_AA' column not found. Skipping.")


#==============================================================================


# 13. Splitting Codons into FromCodon and ToCodon
    # Insert FromCodon and ToCodon after Codons column
if 'Codons' in df.columns:
    codons_index = df.columns.get_loc("Codons") + 1
    from_to = df['Codons'].str.split('/', expand=True)
    df.insert(codons_index, 'FromCodon', from_to[0])
    df.insert(codons_index + 1, 'ToCodon', from_to[1])
    print("Step 13: Codons split into FromCodon and ToCodon (case-sensitive).")
else:
    print("Step 13: 'Codons' column not found. Skipping.")


#==============================================================================


# 14. Extracting Mutation_Type (transition/transversion), Codon_Positon, AA_Change (synonymous/nonsynonymous)
    # Codon Table
codon_table = {
    'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'TGT':'C', 'TGC':'C',
    'GAT':'D', 'GAC':'D',
    'GAA':'E', 'GAG':'E',
    'TTT':'F', 'TTC':'F',
    'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
    'CAT':'H', 'CAC':'H',
    'ATA':'I', 'ATT':'I', 'ATC':'I',
    'AAA':'K', 'AAG':'K',
    'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
    'ATG':'M',
    'AAT':'N', 'AAC':'N',
    'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'CAA':'Q', 'CAG':'Q',
    'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R', 'AGA':'R', 'AGG':'R',
    'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S',
    'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
    'TGG':'W',
    'TAT':'Y', 'TAC':'Y',
    'TAA':'*', 'TAG':'*', 'TGA':'*'
}


# Function to classify mutation
def classify_change(from_codon, to_codon):
    # Handle cases where codons are invalid or empty
    if from_codon == '-' or to_codon == '-':
        return -1, -1, -1  # return -1 for invalid cases
    if not from_codon or not to_codon:
        return None, None, None


    # Convert to uppercase to ensure consistency
    from_codon = from_codon.upper()
    to_codon = to_codon.upper()


    # Find positions where the codons differ
    position = [i for i in range(3) if from_codon[i] != to_codon[i]]
   
    if not position:
        return 1, -1, 1  # no change: return 1 for mutation type and amino acid change


    # Identify the position of mutation
    pos = position[0] + 1  # Position should be 1-indexed


    # Bases in the codon
    base_from = from_codon[position[0]]
    base_to = to_codon[position[0]]


    # Purines and Pyrimidines for mutation type classification
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}


    # Mutation type: 1 for transition (purine to purine or pyrimidine to pyrimidine), 2 for transversion
    if (base_from in purines and base_to in purines) or (base_from in pyrimidines and base_to in pyrimidines):
        mutation_type = 1  # Transition
    else:
        mutation_type = 2  # Transversion


    # Using a codon table (ensure you have the 'codon_table' dictionary available)
    aa_from = codon_table.get(from_codon, '?')
    aa_to = codon_table.get(to_codon, '?')


    # Amino acid change: 1 if no change, -1 if change
    aa_change = 1 if aa_from == aa_to else -1


    return mutation_type, pos, aa_change


    # Compute features
if 'FromCodon' in df.columns and 'ToCodon' in df.columns:
    codon_insert_index = df.columns.get_loc("ToCodon") + 1
    mutation_df = df.apply(
        lambda row: pd.Series(classify_change(row['FromCodon'], row['ToCodon'])),
        axis=1
    )
    mutation_df.columns = ['MutationType', 'CodonPosition', 'AA_Change']


    # Insert each column right after ToCodon
    for i, col in enumerate(mutation_df.columns):
        df.insert(codon_insert_index + i, col, mutation_df[col])    


    print("Step 14: MutationType, CodonPosition, and AA_Change extracted and inserted.")
else:
    print("Step 14: FromCodon or ToCodon not found. Skipping.")


#==============================================================================


# 15. Deleting FromCodon and ToCodon
if 'FromCodon' in df.columns and 'ToCodon' in df.columns:
    df.drop(columns=["FromCodon", "ToCodon"], inplace=True)


    print("Step 15: FromCodon and ToCodon deleted")
else:
    print("Step 15: FromCodon or ToCodon not found. Skipping.")


#==============================================================================


# 16. Encoding CANONICAL
df['CANONICAL'] = df['CANONICAL'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else -1)
print("Step 16: Encoding CANONICAL complete.")


#==============================================================================


# 17. Encoding APPRIS - Ordinal Encoding
# Replace '-' with NaN, then fill with 'Unknown'
df['APPRIS'] = df['APPRIS'].replace('-', pd.NA).fillna('Unknown')


# Get index of 'APPRIS' column
if 'APPRIS' in df.columns:
    appris_index = df.columns.get_loc("APPRIS")


    # Define category order with 'Unknown' first to ensure it is encoded as 0
    category_order = ["Unknown", "P1", "P2", "P3", "P4", "A1", "A2"]


    # Ordinal encoding
    encoder = OrdinalEncoder(categories=[category_order])
    encoded = encoder.fit_transform(df[['APPRIS']]).astype(int)


    # Drop original and insert encoded at same position
    df.drop(columns=["APPRIS"], inplace=True)
    df.insert(appris_index, "APPRIS", encoded)


    # Print mapping
    print("Step 17: Ordinal encoding of APPRIS complete. Label mapping:")
    for i, label in enumerate(category_order):
        print(f"  {i} = {label}")
else:
    print("Step 17: 'APPRIS' column not found. Skipping.")


#==============================================================================


# 18 + 19. Function to extract score inside parentheses and convert to float
def extract_score(text):
    match = re.search(r'\(([^)]+)', str(text))  # Capture text inside ()
    if match:
        try:
            return float(match.group(1))  # Convert to float
        except ValueError:
            return 0.0
    return 0.0


# 18. Encoding SIFT
if "SIFT" in df.columns:
    df["SIFT"] = df["SIFT"].apply(extract_score).astype(float)
    print("Step 18: Extracted SIFT scores added to DataFrame.")
else:
    print("Step 18: 'SIFT' column not found. Skipping.")


# 19. Encoding PolyPhen
if "PolyPhen" in df.columns:
    df["PolyPhen"] = df["PolyPhen"].apply(extract_score).astype(float)
    print("Step 19: Extracted PolyPhen scores added to DataFrame.")
else:
    print("Step 19: 'PolyPhen' column not found. Skipping.")


#==============================================================================


# Step 20. Encoding CLIN_SIG - Ordinal Encoding using custom priority


# Fill NA with empty string
df['CLIN_SIG'] = df['CLIN_SIG'].fillna("")


# Define your exact label priority
label_priority = {
    "not_provided": 0,
    "no_classifications_from_unflagged_records": 1,
    "benign": 2,
    "likely_benign": 3,
    "benign/likely_benign": 3,
    "uncertain_significance": 4,
    "conflicting_interpretations_of_pathogenicity": 5,
    "uncertain_risk_allele": 6,
    "likely_risk_allele": 7,
    "risk_factor": 8,
    "affects": 9,
    "association": 10,
    "likely_pathogenic": 11,
    "pathogenic/likely_pathogenic": 11,
    "pathogenic": 12,
    "drug_response": 13  # Optional: treat separately if needed
}


# Clean and parse multiple labels, selecting the one with the highest priority
def select_most_severe(entry):
    if not entry or entry.strip() in {"", "-"}:
        return "not_provided"
    labels = re.split(r'[,/]', entry.lower())
    labels = [re.sub(r'\s+', '_', lbl.strip()) for lbl in labels if lbl.strip()]
    ranked = [(label, label_priority.get(label, -1)) for label in labels]
    if not ranked:
        return "not_provided"
    most_severe = max(ranked, key=lambda x: x[1])
    return most_severe[0]


if "CLIN_SIG" in df.columns:
    # Get index of the 'CLIN_SIG' column
    clin_sig_index = df.columns.get_loc("CLIN_SIG")
   
    # Apply most severe label selector
    cleaned = df["CLIN_SIG"].apply(select_most_severe)


    # Map to numeric values using the label_priority
    encoded = cleaned.map(label_priority).fillna(0).astype(int)


    # Replace original column at the same index
    df.drop(columns=['CLIN_SIG'], inplace=True)
    df.insert(clin_sig_index, "CLIN_SIG", encoded)


    # Print mapping
    print("Step 20: CLIN_SIG encoding complete using custom priority.")
    print("Label mapping used (0-based):")
    for label, value in sorted(label_priority.items(), key=lambda x: x[1]):
        print(f"  {value}: {label}")
else:
    print("Step 20: 'CLIN_SIG' column not found. Skipping.")


# ==============================================================================


#ENCODING SOMATIC AND PHENO - MEAN ENCODING
# Shared function for mean encoding, now treating '-' as a data value
def encode_mean_from_comma_sep(val):
    try:
        val_str = str(val)
        values = []
        for x in val_str.split(','):
            x = x.strip()
            if x == "-":
                values.append(-1)  # Encode '-' as -1
            elif x.isdigit():
                values.append(int(x))
        return float(np.mean(values)) if values else 0.0
    except Exception as e:
        print(f"⚠️ Error processing value '{val}': {e}")
        return 0.0


# 21. Encoding SOMATIC
if "SOMATIC" in df.columns:
    somatic_index = df.columns.get_loc("SOMATIC")
    somatic_encoded = df["SOMATIC"].apply(encode_mean_from_comma_sep).astype(float)
    df.drop(columns=["SOMATIC"], inplace=True)
    df.insert(somatic_index, "SOMATIC", somatic_encoded)
    print("Step 21 ✅: 'SOMATIC' mean-encoded (hyphens encoded as -1).")
else:
    print("Step 21 ⚠️: 'SOMATIC' column not found. Skipping.")


# 22. Encoding PHENO
if "PHENO" in df.columns:
    pheno_index = df.columns.get_loc("PHENO")
    pheno_encoded = df["PHENO"].apply(encode_mean_from_comma_sep).astype(float)
    df.drop(columns=["PHENO"], inplace=True)
    df.insert(pheno_index, "PHENO", pheno_encoded)
    print("Step 22 ✅: 'PHENO' mean-encoded (hyphens encoded as -1).")
else:
    print("Step 22 ⚠️: 'PHENO' column not found. Skipping.")


# ==============================================================================
# 23. Encoding MaveDB_score - Mean Encoding (treat '-' as -1.0)
def compute_mean_score(cell):
    try:
        values = []
        for val in str(cell).split(','):
            val = val.strip()
            if val == "-":
                values.append(-1.0)
            elif val != "":
                values.append(float(val))
        return float(sum(values) / len(values)) if values else 0.0
    except Exception as e:
        print(f"⚠️ Error processing MaveDB_score value '{cell}': {e}")
        return 0.0


if "MaveDB_score" in df.columns:
    mavedb_index = df.columns.get_loc("MaveDB_score")
    print("Step 23: Encoding 'MaveDB_score' column...")


    encoded_scores = df["MaveDB_score"].apply(compute_mean_score).astype(float)
    df.drop(columns=["MaveDB_score"], inplace=True)
    df.insert(mavedb_index, "MaveDB_score", encoded_scores)


    print("Step 23 ✅: 'MaveDB_score' mean-encoded (hyphens encoded as -1.0).")
else:
    print("Step 23 ⚠️: 'MaveDB_score' column not found. Skipping.")


#==============================================================================
# 24. Encoding EVE_CLASS - Ordinal Encoding
if 'EVE_CLASS' in df.columns:
    # Normalize the 'EVE_CLASS' column: fill empty, strip, and set to consistent case
    df['EVE_CLASS'] = df['EVE_CLASS'].astype(str).str.strip().replace("-", "").replace("nan", "")


    # Define mapping
    eve_mapping = {
        "Benign": 0,
        "Uncertain": 1,
        "Pathogenic": 2
    }


    # Apply encoding based on mapping
    df['EVE_CLASS'] = df['EVE_CLASS'].map(eve_mapping)  # Replace 'EVE_CLASS' directly with the encoded values


    # Ensure no NaN values after mapping (replacing NaN with 0, or another value if needed)
    df['EVE_CLASS'] = df['EVE_CLASS'].fillna(0).astype(int)


    # Save to Excel
    df.to_excel(output_file, sheet_name=output_sheet, index=False)


    print("Step 24 ✅: 'EVE_CLASS' column replaced with encoded values (saved as numbers).")
else:
    print("Step 24 ⚠️: 'EVE_CLASS' column not found. Skipping.")


#==============================================================================


# 25. Encoding am_class - Ordinal Encoding
# Clean and prepare data
df['am_class'] = df['am_class'].replace('-', pd.NA)


# Get index of 'am_class' column
if 'am_class' in df.columns:
    am_class_index = df.columns.get_loc("am_class")


    # Define the severity order
    severity_order = {
        'likely_benign': 0,
        'ambiguous': 1,
        'likely_pathogenic': 2
    }


    # Apply encoding
    df['am_class'] = df['am_class'].map(severity_order)


    # Ensure no NaN values after mapping (replacing NaN with 0, or another value if needed)
    df['am_class'] = df['am_class'].fillna(0).astype(int)


    print("Step 25 ✅: am_class encoding complete. Data added to DataFrame (saved as numbers).")
else:
    print("Step 25 ⚠️: 'am_class' column not found. Skipping.")


#==============================================================================
# DEFINING COMMON FUNCTIONS USED FOR THE REMAINING STEPS


# Encoding function to be applied to columns
def encode_column(df, col, encoding_map, missing_value=''):
    if col in df.columns:
        encoded = df[col].map(encoding_map)
        encoded = encoded.where(df[col].notna(), missing_value)
        col_index = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        df.insert(col_index, col, encoded)


        print(f"Step 26. -31. : Encoding '{col}' complete. Label mapping:")
        for label, value in encoding_map.items():
            print(f"  {value} = {label}")
    else:
        print(f"Step 26. -31. : '{col}' column not found. Skipping.")
   
def clean_and_encode_predictions(df):
    # Define columns and encoding maps
    lrt_col = 'LRT_pred'
    lrt_encoding = {'D': 1, 'N': 0}


    other_pred_cols = [
        'LIST-S2_pred',
        'MetaLR_pred',
        'M-CAP_pred',
        'MetaRNN_pred',
        'MetaSVM_pred'
    ]
    other_encoding = {'D': 1, 'T': 0}


    additional_pred_cols = {
        'PROVEAN_pred': {'D': 1, 'N': 0},
        'PrimateAI_pred': {'D': 1, 'T': 0},
        'fathmm-MKL_coding_pred': {'D': 1, 'N': 0},
        'fathmm-XF_coding_pred': {'D': 1, 'N': 0},
    }


#==============================================================================


# 26. Encoding LRT_pred
    encode_column(df, lrt_col, lrt_encoding)
   
#==============================================================================
# 27. - 31. Encoding other prediction columns


    # 27. LIST-S2_pred
    # 28. Encoding M-CAP_pred
    # 29. Encoding MetaLR_pred
    # 30. Encoding MetaRNN_pred
    # 31. Encoding MetaSVM_pred


        # Encode other prediction columns
    for col in other_pred_cols:
        encode_column(df, col, other_encoding)
   
#==============================================================================
# 32. - 35. Encoding additional prediction columns
    # 32. Encoding PROVEAN_pred
    # 33. Encoding PrimateAI_pred
    # 34. Encoding fathmm-MKL_coding_pred
    # 35. Encoding fathmm-XF_coding_pred


    for col, mapping in additional_pred_cols.items():
        encode_column(df, col, mapping)
    return df
#==============================================================================


# Apply COMMON FUNCTIONS USED FOR THE REMAINING STEPS directly to loaded df
df = clean_and_encode_predictions(df)


#==============================================================================


# 36. Encoding MutationAssessor_pred
if 'MutationAssessor_pred' in df.columns:
    ma_encoding = {'H': 1.0, 'M': 0.75, 'L': 0.5, 'N': 0.0, '0': 0.0}
    encoded_ma = df['MutationAssessor_pred'].map(ma_encoding)
    encoded_ma = encoded_ma.where(df['MutationAssessor_pred'].notna(), '')
    col_index = df.columns.get_loc('MutationAssessor_pred')
    df.drop(columns=['MutationAssessor_pred'], inplace=True)
    df.insert(col_index, 'MutationAssessor_pred', encoded_ma)
    print("Step 36: 'MutationAssessor_pred' encoding complete. Label mapping:")
    for label, value in ma_encoding.items():
        print(f"  {value} = {label}")
else:
    print("Step 36: 'MutationAssessor_pred' column not found. Skipping.")
         
#==============================================================================


# 37. Encoding MutationTaster_pred


def encode_mutationtaster(val):
    if pd.isna(val) or val in ['-', '0', '']:
        return ''
    scores = {'D': 1.0, 'P': 0.75, 'N': 0.25}
    split_vals = val.split(',')
    valid_scores = [scores[v] for v in split_vals if v in scores]
    return np.mean(valid_scores) if valid_scores else ''


if 'MutationTaster_pred' in df.columns:
    encoded_mt = df['MutationTaster_pred'].apply(encode_mutationtaster)
    col_index = df.columns.get_loc('MutationTaster_pred')
    df.drop(columns=['MutationTaster_pred'], inplace=True)
    df.insert(col_index, 'MutationTaster_pred', encoded_mt)
    print("Step 37: 'MutationTaster_pred' encoding complete. Label mapping:")
    print("  1.0 = D\n  0.75 = P\n  0.25 = N")
else:
    print("Step 37: 'MutationTaster_pred' column not found. Skipping.")
   
#==============================================================================


# 38. Encoding MutationTaster_score (some values are separated by commas)


def encode_mutation_taster_score(score):
    if pd.isna(score) or score == '-':
        return np.nan
    try:
        score_list = [float(x) for x in str(score).split(',') if x]
        return np.mean(score_list) if score_list else np.nan
    except Exception as e:
        print(f"⚠️ Error processing score: {score} ({e})")
        return np.nan


if 'MutationTaster_score' in df.columns:
    encoded_mt_score = df['MutationTaster_score'].apply(encode_mutation_taster_score)
    col_index = df.columns.get_loc('MutationTaster_score')
    df.drop(columns=['MutationTaster_score'], inplace=True)
    df.insert(col_index, 'MutationTaster_score', encoded_mt_score)
    # Convert to decimal format (string)
    df['MutationTaster_score'] = df['MutationTaster_score'].apply(lambda x: f"{x:.10f}" if pd.notna(x) else x)
    print("Step 38: 'MutationTaster_score' encoding complete. Data added to DataFrame.")
else:
    print("Step 38: 'MutationTaster_score' column not found. Skipping.")


   
#==============================================================================


# 39. Encoding SiPhy_29way_pi into 4 columns (values separated by colons)
def split_siphy(row):
    if pd.isna(row) or row == '-' or row.count(':') != 3:
        return [np.nan, np.nan, np.nan, np.nan]
    try:
        return [float(x) for x in row.split(':')]
    except:
        return [np.nan, np.nan, np.nan, np.nan]


if 'SiPhy_29way_pi' in df.columns:
    siphy_cols = df['SiPhy_29way_pi'].apply(split_siphy).tolist()
    siphy_split_df = pd.DataFrame(siphy_cols, columns=['SiPhy_A', 'SiPhy_C', 'SiPhy_G', 'SiPhy_T'])


    # Find the column index of the original column
    col_index = df.columns.get_loc('SiPhy_29way_pi')


    # Drop the original column
    df.drop(columns=['SiPhy_29way_pi'], inplace=True)


    # Insert the new columns at the original column's position
    for i, col_name in enumerate(['SiPhy_A', 'SiPhy_C', 'SiPhy_G', 'SiPhy_T']):
        df.insert(col_index + i, col_name, siphy_split_df[col_name])


    print("Step 39: 'SiPhy_29way_pi' encoding complete. Columns replaced in original position.")
else:
    print("Step 39: 'SiPhy_29way_pi' column not found. Skipping.")


#==============================================================================


# 40. Encoding clinvar_clnsig - Ordinal Encoding using label_priority_clnsig
print("Step 40: Encoding 'clinvar_clnsig' using ordinal encoding...")


# Define the priority dictionary
label_priority_clnsig = {
    "clinvar_clnsig_no_classification_for_the_single_variant": 0,
    "clinvar_clnsig_other": 0,
    "clinvar_clnsig_Benign": 1,
    "clinvar_clnsig_Likely_benign": 2,
    "clinvar_clnsig_Benign/Likely_benign": 2,
    "clinvar_clnsig_Uncertain_significance": 3,
    "clinvar_clnsig_Conflicting_classifications_of_pathogenicity": 4,
    "clinvar_clnsig_risk_factor": 5,
    "clinvar_clnsig_association": 6,
    "clinvar_clnsig_Affects": 7,
    "clinvar_clnsig_Likely_pathogenic": 8,
    "clinvar_clnsig_Pathogenic": 9,
    "clinvar_clnsig_drug_response": 10
}


if 'clinvar_clnsig' in df.columns:
    # Clean the 'clinvar_clnsig' column by replacing '-' and NaN with empty strings
    df['clinvar_clnsig_clean'] = df['clinvar_clnsig'].replace('-', '').fillna('')


    # Split the entries and find max priority per row
    def get_max_priority(entry):
        labels = [label.strip() for label in entry.split(',') if label.strip()]
        if not labels:
            return 0  # Treat empty or missing as lowest priority
        priorities = [label_priority_clnsig.get(label, 0) for label in labels]
        return max(priorities)


    # Apply ordinal encoding
    encoded_values = df['clinvar_clnsig_clean'].apply(get_max_priority)


    # Get the original column index
    col_index = df.columns.get_loc('clinvar_clnsig')


    # Drop original columns and insert encoded one
    df.drop(columns=['clinvar_clnsig', 'clinvar_clnsig_clean'], inplace=True)
    df.insert(col_index, 'clinvar_clnsig', pd.to_numeric(encoded_values, downcast='integer'))


    print("Step 40: Ordinal encoding complete. Sample mapping:")
    for k, v in sorted(label_priority_clnsig.items(), key=lambda x: x[1]):
        print(f"{v} = {k}")
else:
    print("Step 40: 'clinvar_clnsig' column not found. Skipping.")


#==============================================================================


# 41. CONVERTING NUMBERS SAVED AS TEXT TO NUMBERS
def convert_text_to_numbers(df):
    # Convert any columns with numbers stored as text to actual numeric values.
    # Non-numeric entries will be left unchanged.
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception:
            pass  # Keep original column if it can't be converted


    print("Step 41: Text-formatted numbers converted to numeric, non-numeric entries left unchanged.")
    return df


df = convert_text_to_numbers(df)


#==============================================================================
#==============================================================================


# 43.Process OpenTargets_l2g column
def encode_l2g_column(df, column='OpenTargets_l2g'):
    import numpy as np


    def parse_values(val):
        try:
            return [float(x) for x in str(val).split(',')]
        except:
            return [np.nan]


    # Parse the values
    parsed = df[column].apply(parse_values)


    # Compute stats
    col_mean = parsed.apply(np.mean)
    col_max = parsed.apply(np.max)
    col_min = parsed.apply(np.min)
    col_std = parsed.apply(np.std)


    mean_col = f"{column}_mean"
    max_col = f"{column}_max"
    min_col = f"{column}_min"
    std_col = f"{column}_std"


    # Find the column index
    col_idx = df.columns.get_loc(column)


    df.drop(columns=[column], inplace=True)
    df.insert(col_idx, mean_col, col_mean)
    df.insert(col_idx + 1, max_col, col_max)
    df.insert(col_idx + 2, min_col, col_min)
    df.insert(col_idx + 3, std_col, col_std)


    print(f"✅ Replaced '{column}' with: {mean_col}, {max_col}, {min_col}, {std_col}")


    return df


df = encode_l2g_column(df)


#==============================================================================


# 43. Saving the DF as an Excel File


# Converting all scientific notation to decimal once again
df = df.apply(lambda col: col.apply(convert_sci_to_float))


# Convert all '-' and 'nan' values in the entire dataframe to 0
def replace_dash_and_nan_with_zero(df):
    # Replace '-' and 'nan' strings with 0 in the entire dataframe
    df = df.replace(['-', 'nan', 'NaN'], 0)
   
    # replacing applymap with DataFrame.apply
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if isinstance(x, str) and x.strip().lower() == 'nan' else x)


    # Optional: Infer correct data types (ensure numeric columns stay numeric)
    df = df.infer_objects(copy=False)


    print("All '-' and 'nan' strings have been replaced with 0 values.")
    return df


# Apply the function to your dataframe
df = replace_dash_and_nan_with_zero(df)


# Drop the specified columns if they exist
columns_to_drop = ['ALLELE_AFTER_SLASH', 'REF_ALLELE', 'Codons']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


print("✅ Specified columns dropped (if present).")


print("Writing dataframe to a new file...")
with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name=output_sheet, index=False, na_rep='', float_format="%.10f")


print(f"✅ All encodings complete. Output saved to '{output_sheet}' in '{output_file}'")



# Combining data from all databases
import pandas as pd


# File & sheet names
ensembl_file = "encoded_ensembl_v2.xlsx"
ensembl_sheet = "ENCODED"


pharmgkb_file = "encoded_pharmgkb_v2.xlsx"
pharmgkb_sheet = "encoded_dataset"


drug_desc_file = "DRUG_DESCRIPTORS.xlsx"
drug_desc_sheet = "Padel_descriptors"


output_file = "all_features_except_drugs_encoded.xlsx"
output_sheet = "all_features"


# Load Ensembl data and rename for merge compatibility
df_ensembl = pd.read_excel(ensembl_file, sheet_name=ensembl_sheet)
df_ensembl = df_ensembl.rename(columns={"#Uploaded_variation": "HGVS notation"})


# Load PharmGKB data
df_pharmgkb = pd.read_excel(pharmgkb_file, sheet_name=pharmgkb_sheet)


# Drop the nucleotide columns from PharmGKB
columns_to_drop = ['C', 'T', 'A', 'G']
df_pharmgkb.drop(columns=[col for col in columns_to_drop if col in df_pharmgkb.columns], inplace=True)


# Merge: add Ensembl info to PharmGKB using left join on 'HGVS notation'
df_combined = df_pharmgkb.merge(df_ensembl, on="HGVS notation", how="left")


# Keep Ensembl's Chromosome and Position, drop PharmGKB's
df_combined.drop(columns=['Chromosome_x', 'Position_x'], inplace=True)
df_combined.rename(columns={'Chromosome_y': 'Chromosome', 'Position_y': 'Position'}, inplace=True)


# Load drug descriptors
df_drug_desc = pd.read_excel(drug_desc_file, sheet_name=drug_desc_sheet)


# Ensure 'Drugs' column is lowercase in both DataFrames for consistent merge
df_combined['Drugs'] = df_combined['Drugs'].str.lower()
df_drug_desc['Drugs'] = df_drug_desc['Drugs'].str.lower()


# Merge drug descriptors with combined data
df_all_features = df_combined.merge(df_drug_desc, on="Drugs", how="left")


# ✅ Drop 'HGVS notation' column from final dataset if it exists
if 'HGVS notation' in df_all_features.columns:
    df_all_features.drop(columns=['HGVS notation'], inplace=True)
    print("Column 'HGVS notation' removed from final dataset.")


# Save final output
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_all_features.to_excel(writer, sheet_name=output_sheet, index=False)


print(f"✅ Final dataset saved as '{output_file}' with {len(df_all_features)} rows and {len(df_all_features.columns)} columns.")




# Drive Mounting (common to all code from Google Colab Notebook)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# Making a copy of the base files in the drive in Colab workspace
import shutil


src = '/content/drive/My Drive/Thesis'
dst = '/content'  # Destination inside Colab workspace


shutil.copytree(src, dst, dirs_exist_ok=True)


import os
# Define path to the "Thesis" folder
thesis_folder = '/content/drive/My Drive/Thesis'


# Create the folder if it doesn't exist
if not os.path.exists(thesis_folder):
   os.makedirs(thesis_folder)
   print(f'Created folder: {thesis_folder}')
else:
   print(f'Folder already exists: {thesis_folder}')

# Cleaning: Dropping Null Values and Duplicates
# read the file
import pandas as pd
df = pd.read_csv("/content/all_features_except_drugs_encoded.csv")
df.isnull().sum()
null_columns = df.columns[df.isnull().any()]
print("Columns with null values:")
print(null_columns.tolist())
# cleaning up columns based on if they are null in CHROM-POS-REF-ALT
original_rows = df.shape[0]
df = df.dropna(subset=['Chromosome', 'Position', 'Reference', 'Alternate'])
print(f"Rows before cleanup: {original_rows}")
print(f"Rows after cleanup: {df.shape[0]}")
df.dropna
df.drop_duplicates

# Encoding drug names
from sklearn.preprocessing import LabelEncoder
import json


# Check for drugs not in your reference list
all_drugs = [
   "methotrexate", "thioguanine", "mercaptopurine", "etoposide", "vincristine",
   "prednisone", "prednisolone", "azathioprine", "l-asparagine", "cyclophosphamide",
   "cytarabine", "doxorubicin", "daunorubicin", "dexamethasone", "fluorouracil",
   "leucovorin", "voriconazole", "teniposide", "bilirubin", "imatinib",
   "nilotinib", "omacetaxine", "azacitidine", "dasatinib", "chlorambucil",
   "ibrutinib", "fludarabine", "idarubicin", "ondansetron", "busulfan",
   "cladribine", "venetoclax", "dexrazoxane", "dacarbazine", "procarbazine",
   "bleomycin", "lenalidomide", "vindesine", "hydrocortisone", "mitoxantrone",
   "arabinofuranosylcytosine triphosphate", "irinotecan", "tacrolimus"
]
unknown_drugs = df[~df['Drugs'].isin(all_drugs)]
print(f"Unknown drug names: {unknown_drugs['Drugs'].unique()}")
print(f"Count of unknown drugs: {unknown_drugs.shape[0]}")


# Reference list of all possible drugs
all_drugs = [
   "methotrexate", "thioguanine", "mercaptopurine", "etoposide", "vincristine",
   "prednisone", "prednisolone", "azathioprine", "l-asparagine", "cyclophosphamide",
   "cytarabine", "doxorubicin", "daunorubicin", "dexamethasone", "fluorouracil",
   "leucovorin", "voriconazole", "teniposide", "bilirubin", "imatinib",
   "nilotinib", "omacetaxine", "azacitidine", "dasatinib", "chlorambucil",
   "ibrutinib", "fludarabine", "idarubicin", "ondansetron", "busulfan",
   "cladribine", "venetoclax", "dexrazoxane", "dacarbazine", "procarbazine",
   "bleomycin", "lenalidomide", "vindesine", "hydrocortisone", "mitoxantrone",
   "arabinofuranosylcytosine triphosphate", "irinotecan", "tacrolimus"
]


# Initialize the label encoder and fit it to your master list
le = LabelEncoder()
le.fit(all_drugs)


# Replace the column in your dataframe 
df['Drugs'] = le.transform(df['Drugs'])


# Create and save the mapping to JSON
drug_mapping = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}


# Save to JSON
with open('/content/drive/MyDrive/Thesis/drug_label_mapping.json', 'w') as f:
   json.dump(drug_mapping, f, indent=4)


print("Drug column successfully label-encoded and mapping saved to 'drug_label_mapping.json' in Thesis folder.")
print(drug_mapping)


