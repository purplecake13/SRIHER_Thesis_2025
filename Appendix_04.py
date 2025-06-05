# Feature Preprocessing & Selection
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd


def preprocess_features(X_train, X_test, y_train, genetic_columns, k_genetic=130, k_drug=170, missing_threshold=0.5):
   # Define base columns
   base_columns = ["Chromosome", "Position", "Reference", "Alternate"]


   # Extract base columns
   X_train_base = X_train[base_columns]
   X_test_base = X_test[base_columns]


   # Extract and clean genetic columns
   X_train_genetic = X_train[genetic_columns]
   X_test_genetic = X_test[genetic_columns]


   # Drop columns with too many missing values
   missing_train = X_train_genetic.isnull().mean()
   missing_test = X_test_genetic.isnull().mean()
   columns_to_keep = missing_train[missing_train < missing_threshold].index.intersection(missing_test[missing_test < missing_threshold].index)


   X_train_genetic = X_train_genetic[columns_to_keep]
   X_test_genetic = X_test_genetic[columns_to_keep]


   # Re-identify drug features after excluding base and genetic columns
   drug_columns = [col for col in X_train.columns if col not in genetic_columns and col not in base_columns and col != "Drugs"]


   X_train_drug = X_train[drug_columns]
   X_test_drug = X_test[drug_columns]




   # Drop drug columns with too many missing values (same threshold as genetics)
   missing_train_drug = X_train_drug.isnull().mean()
   missing_test_drug = X_test_drug.isnull().mean()
   drug_columns_to_keep = missing_train_drug[missing_train_drug < missing_threshold].index.intersection(
       missing_test_drug[missing_test_drug < missing_threshold].index)


   X_train_drug = X_train_drug[drug_columns_to_keep]
   X_test_drug = X_test_drug[drug_columns_to_keep]




   # Impute missing values
   imputer = SimpleImputer(strategy='mean')


   X_train_genetic_imputed = pd.DataFrame(imputer.fit_transform(X_train_genetic), columns=X_train_genetic.columns, index=X_train.index)
   X_test_genetic_imputed = pd.DataFrame(imputer.transform(X_test_genetic), columns=X_train_genetic.columns, index=X_test.index)


   X_train_drug_array = imputer.fit_transform(X_train_drug)
   X_train_drug_imputed = pd.DataFrame(X_train_drug_array, columns=X_train_drug.columns, index=X_train.index)


   X_test_drug_array = imputer.transform(X_test_drug)
   X_test_drug_imputed = pd.DataFrame(X_test_drug_array, columns=X_train_drug.columns, index=X_test.index)


   # Feature selection - Genetic
   selector_genetic = SelectKBest(score_func=f_classif, k=k_genetic)
   X_train_genetic_selected = selector_genetic.fit_transform(X_train_genetic_imputed, y_train)
   X_test_genetic_selected = selector_genetic.transform(X_test_genetic_imputed)
   selected_genetic_features = X_train_genetic_imputed.columns[selector_genetic.get_support()]


   # Feature selection - Drug
   selector_drug = SelectKBest(score_func=f_classif, k=k_drug)
   X_train_drug_selected = selector_drug.fit_transform(X_train_drug_imputed, y_train)
   X_test_drug_selected = selector_drug.transform(X_test_drug_imputed)
   selected_drug_features = X_train_drug_imputed.columns[selector_drug.get_support()]


   # Combine all parts
   X_train_final = pd.concat([
       X_train_base.reset_index(drop=True),
       pd.DataFrame(X_train_genetic_selected, columns=selected_genetic_features, index=X_train.index).reset_index(drop=True),
       pd.DataFrame(X_train_drug_selected, columns=selected_drug_features, index=X_train.index).reset_index(drop=True)
   ], axis=1)


   X_test_final = pd.concat([
       X_test_base.reset_index(drop=True),
       pd.DataFrame(X_test_genetic_selected, columns=selected_genetic_features, index=X_test.index).reset_index(drop=True),
       pd.DataFrame(X_test_drug_selected, columns=selected_drug_features, index=X_test.index).reset_index(drop=True)
   ], axis=1)


   return X_train_final, X_test_final, selector_genetic, selector_drug


# Usage
genetic_columns = ["Disease", "Genes", "Reference", "Alternate", "Significance", "Phenotype Categories", "Pediatric", "Chromosome", "Position", "Ref",
   "Alt", "IMPACT", "SYMBOL", "Feature_type_Encoded", "BIOTYPE", "EXON/INTRON", "Ref_AA", "Alt_AA", "MutationType", "CodonPosition",
   "AA_Change", "STRAND", "CANONICAL", "TSL", "APPRIS", "SIFT", "PolyPhen", "gnomADe_AF", "gnomADg_AF", "CLIN_SIG",
   "SOMATIC", "PHENO", "CADD_PHRED", "CADD_RAW", "MaxEntScan_alt", "MaxEntScan_diff", "MaxEntScan_ref", "MaveDB_score", "REVEL", "ada_score",
   "rf_score", "ClinPred", "mutfunc_exp", "mutfunc_int", "mutfunc_mod", "LOEUF", "OpenTargets_l2g_mean", "OpenTargets_l2g_max", "OpenTargets_l2g_min", "OpenTargets_l2g_std",
   "Enformer_SAD", "Enformer_SAR", "Existing_InFrame_oORFs", "Existing_OutOfFrame_oORFs", "Existing_uORFs", "EVE_CLASS", "EVE_SCORE", "am_class", "am_pathogenicity", "1000Gp3_AC",
   "1000Gp3_AF", "ALFA_Total_AC", "ALFA_Total_AF", "ALFA_Total_AN", "BayesDel_addAF_score", "BayesDel_noAF_score", "DANN_score", "DEOGEN2_score", "ESM1b_score", "EVE_score.1",
   "Eigen-PC-phred_coding", "Eigen-PC-raw_coding", "Eigen-phred_coding", "Eigen-raw_coding", "FATHMM_score", "GERP++_NR", "GERP++_RS", "GM12878_fitCons_score", "H1-hESC_fitCons_score", "HUVEC_confidence_value",
   "HUVEC_fitCons_score", "LIST-S2_pred", "LIST-S2_score", "LRT_Omega", "LRT_pred", "LRT_score", "M-CAP_pred", "M-CAP_score", "MPC_score", "MVP_score",
   "MetaLR_pred", "MetaLR_score", "MetaRNN_pred", "MetaRNN_score", "MetaSVM_pred", "MetaSVM_score", "MutPred_score", "MutationAssessor_pred", "MutationAssessor_score", "MutationTaster_pred",
   "MutationTaster_score", "PROVEAN_pred", "PROVEAN_score", "PrimateAI_pred", "PrimateAI_score", "Reliability_index", "SiPhy_29way_logOdds", "SiPhy_A", "SiPhy_C", "SiPhy_G",
   "SiPhy_T", "VARITY_ER_LOO_score", "VARITY_ER_score", "VARITY_R_LOO_score", "VARITY_R_score", "bStatistic", "clinvar_clnsig", "codon_degeneracy", "fathmm-MKL_coding_pred", "fathmm-MKL_coding_score",
   "fathmm-XF_coding_pred", "fathmm-XF_coding_score", "gMVP_score", "gnomAD_exomes_AC", "gnomAD_exomes_AF", "gnomAD_exomes_AN", "gnomAD_genomes_AC", "gnomAD_genomes_AF", "gnomAD_genomes_AN", "integrated_confidence_value",
   "integrated_fitCons_score", "phastCons17way_primate", "Geno2MP_HPO_count", "BLOSUM62", "pHaplo", "pTriplo", "Consequence_Encoded"
]


X_train_final, X_test_final, shared_selector_genetic, shared_selector_drug = preprocess_features(
   X_train_filtered, X_test_filtered, y_train_filtered, genetic_columns, k_genetic=130, k_drug=170
)


# Create subfolder for final datasets
final_dataset_folder = os.path.join(thesis_folder, 'train_and_test_final_dataset')


# Create the folder if it doesn't exist
if not os.path.exists(final_dataset_folder):
   os.makedirs(final_dataset_folder)
   print(f"Created folder: {final_dataset_folder}")
else:
   print(f"Folder already exists: {final_dataset_folder}")


# Save final train and test datasets as CSV files
X_train_final.to_csv(os.path.join(final_dataset_folder, 'X_train_final.csv'), index=False)
X_test_final.to_csv(os.path.join(final_dataset_folder, 'X_test_final.csv'), index=False)
y_train_filtered.to_csv(os.path.join(final_dataset_folder, 'y_train_final.csv'), index=False)
y_test_filtered.to_csv(os.path.join(final_dataset_folder, 'y_test_final.csv'), index=False)


print("Final train and test datasets have been saved as CSV files in 'train_and_test_final_dataset'.")


# Running checks:
print("Train shape:", X_train_final.shape)
print("Test shape:", X_test_final.shape)


# Make sure your resulting DataFrames have named columns and not default numeric indices:
print("First few column names in X_train_final:")
print(X_train_final.columns[:10])


# Spot-check for NaNs
print("Missing values in train:", X_train_final.isnull().sum().sum())
print("Missing values in test:", X_test_final.isnull().sum().sum())


# confirm count of selected features
print("Genetic features selected:", shared_selector_genetic.get_support().sum())
print("Drug features selected:", shared_selector_drug.get_support().sum())


