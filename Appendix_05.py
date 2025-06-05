# Models
# XGBoost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score, roc_auc_score,
                             f1_score, matthews_corrcoef, roc_curve, auc, top_k_accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shap
import joblib


# Define model name and output directory
model_name = "xgboost"
output_dir = f"/content/drive/MyDrive/Thesis/evaluation/{model_name}_output/"
os.makedirs(output_dir, exist_ok=True)


model_save_path = "/content/drive/MyDrive/Thesis/predictor/model"
os.makedirs(model_save_path, exist_ok=True)


# === Label Encoding ===
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_filtered)
y_test_encoded = label_encoder.transform(y_test_filtered)


# Save label encoding mapping
label_encoding_mapping = {str(label): int(i) for i, label in enumerate(label_encoder.classes_)}
with open(os.path.join(model_save_path, 'xgboost_label_encoding_mapping.json'), 'w') as f:
    json.dump(label_encoding_mapping, f, indent=4)


# Save feature columns used in the model
feature_columns = list(X_train_final.columns)
with open(os.path.join(model_save_path, 'xgboost_feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f, indent=4)
print("Feature columns saved.")


#ensure all features are numeric
X_train_final = X_train_final.apply(pd.to_numeric)
X_test_final = X_test_final.apply(pd.to_numeric)


# === Train XGBoost Model ===
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(label_encoder.classes_)
)
#model.fit(X_train_final, y_train_encoded)
model.fit(X_train_final.values, y_train_encoded)


# Save the trained model
joblib.dump(model, os.path.join(model_save_path, 'xgboost_model.pkl'))
print(f"Trained XGBoost model saved to {os.path.join(model_save_path, 'xgboost_model.pkl')}")




# === Predict ===
y_pred_encoded = model.predict(X_test_final.values)
y_pred_prob_encoded = model.predict_proba(X_test_final.values)


# === Inverse transform predictions ===
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
y_test_original = label_encoder.inverse_transform(y_test_encoded)


# reverse-map class numbers back to drug names for all your plots.
import json
# Load original mapping (drug name → label)
with open('/content/drive/MyDrive/Thesis/drug_label_mapping.json', 'r') as f:
    drug_label_mapping = json.load(f)
# Invert to get label → drug name
label_to_drug = {v: k for k, v in drug_label_mapping.items()}
y_pred_original = [label_to_drug[i] for i in y_pred_encoded]
y_test_original = [label_to_drug[i] for i in y_test_encoded]


# create short name for long drug names
friendly_drug_names = {
    "arabinofuranosylcytosine triphosphate": "ara-CTP",
}


# === Metrics ===
def mean_reciprocal_rank(y_true, y_proba):
    ranks = []
    for true_label, probs in zip(y_true, y_proba):
        sorted_indices = np.argsort(probs)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        ranks.append(1.0 / rank)
    return np.mean(ranks)


def ndcg_at_k(y_true, y_proba, k=5):
    def dcg(relevance):
        return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance)])


    ndcg_scores = []
    for true_label, probs in zip(y_true, y_proba):
        top_k_indices = np.argsort(probs)[::-1][:k]
        relevance = [1 if i == true_label else 0 for i in top_k_indices]
        ideal_relevance = sorted(relevance, reverse=True)
        ndcg_scores.append(dcg(relevance) / dcg(ideal_relevance) if dcg(ideal_relevance) != 0 else 0.0)
    return np.mean(ndcg_scores)


metrics = {
    'Accuracy': accuracy_score(y_test_original, y_pred_original),
    'Balanced Accuracy': balanced_accuracy_score(y_test_original, y_pred_original),
    'Precision (macro)': precision_score(y_test_original, y_pred_original, average='macro'),
    'Precision (weighted)': precision_score(y_test_original, y_pred_original, average='weighted'),
    'Recall (macro)': recall_score(y_test_original, y_pred_original, average='macro'),
    'Recall (weighted)': recall_score(y_test_original, y_pred_original, average='weighted'),
    'F1 Score (macro)': f1_score(y_test_original, y_pred_original, average='macro'),
    'F1 Score (weighted)': f1_score(y_test_original, y_pred_original, average='weighted'),
    'MCC': matthews_corrcoef(y_test_original, y_pred_original),
    'Top-5 Accuracy': top_k_accuracy_score(y_test_encoded, y_pred_prob_encoded, k=5),
    'NDCG@5': ndcg_at_k(y_test_encoded, y_pred_prob_encoded),
    'MRR': mean_reciprocal_rank(y_test_encoded, y_pred_prob_encoded),
    'ROC-AUC': roc_auc_score(y_test_encoded, y_pred_prob_encoded, average='macro', multi_class='ovr')
}


# Save metrics to CSV
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)


# Print metrics
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")


# === Classification Report & Confusion Matrix ===


# Generate classification report
classification_rep = classification_report(y_test_original, y_pred_original, output_dict=True)


# Convert the report to a pandas DataFrame
classification_rep_df = pd.DataFrame(classification_rep).transpose()


# Rename the index (drug names) to friendly versions where applicable
friendly_index = [friendly_drug_names.get(label, label) for label in classification_rep_df.index]
classification_rep_df.index = friendly_index


# Print the DataFrame for better alignment
print(classification_rep_df)


# Save the DataFrame to a file
classification_rep_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))


# === confusion matrix ===
# Find which drug names actually appear
all_drugs_in_test_or_pred = sorted(list(set(y_test_original) | set(y_pred_original)))


# Apply friendly names
friendly_labels = [friendly_drug_names.get(name, name) for name in all_drugs_in_test_or_pred]


# Create confusion matrix with only these labels
conf_matrix = confusion_matrix(y_test_original, y_pred_original, labels=all_drugs_in_test_or_pred)


# Now plot
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=friendly_labels, yticklabels=friendly_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


# Save the confusion matrix plot to a file
conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)


# === ROC Curves ===
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_original)


plt.figure()
for i, drug_name in enumerate(all_drugs_in_test_or_pred):
    friendly_name = friendly_drug_names.get(drug_name, drug_name)
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob_encoded[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{friendly_name} (AUC = {roc_auc:.2f})')


plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right", fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.show()


# === SHAP Analysis ===
explainer = shap.Explainer(model, X_train_final)
shap_values = explainer(X_test_final)


# Fix class names shown in SHAP legend
friendly_class_names = [friendly_drug_names.get(label_to_drug[i], label_to_drug[i]) for i in range(len(label_to_drug))]
shap_values._names = friendly_class_names  # override internal names for better legends


# Aggregate the SHAP values across classes (if multi-class) and calculate importance
shap_importance = np.abs(shap_values.values).mean(axis=2).mean(axis=0)


# Ensure the number of features in shap_importance matches the columns in X_train_final
if shap_importance.shape[0] != X_train_final.shape[1]:
    raise ValueError(f"The number of SHAP importances ({shap_importance.shape[0]}) does not match the number of features ({X_train_final.shape[1]}) in X_train_final.")


# Create a DataFrame to store the feature importances
shap_importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,  # This should now contain the correct feature names
    'SHAP Importance': shap_importance
})


# Sort the importance values in descending order
shap_importance_df = shap_importance_df.sort_values(by='SHAP Importance', ascending=False)


# Save SHAP feature importances to CSV
shap_importance_df.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)


# Optionally, print the top 10 features to the console
print(shap_importance_df.head(10))


# === SHAP Summary Plot ===
# Generate the summary plot for SHAP feature importance
#shap.summary_plot(shap_values, X_test_final, show=False)
shap.summary_plot(shap_values, X_test_final, class_names=friendly_class_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
plt.show()  # Ensure the plot is displayed


print("\nModel evaluation and analysis completed.")

# Random Forest
import json
import numpy as np
import shap
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix,
    top_k_accuracy_score, roc_auc_score, roc_curve
)
from scipy.stats import rankdata
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})


# === Load Drug Label Mapping ===
with open("drug_label_mapping.json", "r") as f:
    label_mapping = json.load(f)
id_to_label = {int(v): k for k, v in label_mapping.items()}  # integer class label to drug name


# === Model Training ===
print("Training Random Forest...")
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train_filtered)


# === Predictions ===
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)
class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}


# === Metrics ===
accuracy = accuracy_score(y_test_filtered, y_pred)
precision_macro = precision_score(y_test_filtered, y_pred, average='macro', zero_division=1)
precision_weighted = precision_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
recall_macro = recall_score(y_test_filtered, y_pred, average='macro', zero_division=1)
recall_weighted = recall_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
f1_macro = f1_score(y_test_filtered, y_pred, average='macro', zero_division=1)
f1_weighted = f1_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
mcc = matthews_corrcoef(y_test_filtered, y_pred)


# Binarize y_test for ROC-AUC
y_test_bin = label_binarize(y_test_filtered, classes=np.unique(y_train_filtered))
try:
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba[:, :y_test_bin.shape[1]], average='macro', multi_class='ovr')
except ValueError:
    roc_auc = None


def mean_reciprocal_rank(y_true, y_scores, class_indices):
    ranks = []
    for true, score in zip(y_true, y_scores):
        ranking = rankdata(-score, method='ordinal')
        true_idx = class_indices[true]
        rank = ranking[true_idx]
        ranks.append(1.0 / rank)
    return np.mean(ranks)


mrr = mean_reciprocal_rank(y_test_filtered.to_numpy(), y_pred_proba, class_indices)


def ndcg_score(y_true, y_score, k=5):
    dcg = 0
    for true, scores in zip(y_true, y_score):
        order = np.argsort(scores)[::-1][:k]
        if true in order:
            index = list(order).index(true)
            dcg += 1 / np.log2(index + 2)
    idcg = 1.0
    return dcg / len(y_true) / idcg


ndcg = ndcg_score(y_test_filtered.to_numpy(), y_pred_proba)


try:
    top_k_acc = top_k_accuracy_score(y_test_filtered, y_pred_proba, k=5, labels=np.unique(y_train_filtered))
except ValueError:
    top_k_acc = None


# === Classification Report ===
classification_rep = classification_report(y_test_filtered, y_pred, zero_division=1, target_names=[id_to_label[int(cls)] for cls in model.classes_])


# === Output Directory Setup ===
model_name = "random_forest"
eval_dir = f"/content/drive/MyDrive/Thesis/evaluation/{model_name}_output/"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs("/content/drive/MyDrive/Thesis/predictor/model/", exist_ok=True)


# === Save Metrics ===
metrics = {
    "Accuracy": accuracy,
    "Precision (Macro)": precision_macro,
    "Precision (Weighted)": precision_weighted,
    "Recall (Macro)": recall_macro,
    "Recall (Weighted)": recall_weighted,
    "F1 Score (Macro)": f1_macro,
    "F1 Score (Weighted)": f1_weighted,
    "MCC": mcc,
    "ROC-AUC": roc_auc if roc_auc is not None else 'N/A',
    "MRR": mrr,
    "NDCG@5": ndcg,
    "Top-5 Accuracy": top_k_acc if top_k_acc is not None else 'Skipped'
}
pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, f"{model_name}_metrics.csv"), index=False)


# === Display Metrics ===
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision (Macro): {precision_macro}")
print(f"Precision (Weighted): {precision_weighted}")
print(f"Recall (Macro): {recall_macro}")
print(f"Recall (Weighted): {recall_weighted}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Weighted): {f1_weighted}")
print(f"MCC: {mcc}")
print(f"ROC-AUC: {roc_auc if roc_auc is not None else 'N/A'}")
print(f"MRR: {mrr}")
print(f"NDCG@5: {ndcg}")
print(f"Top-5 Accuracy: {top_k_acc if top_k_acc is not None else 'Skipped'}")




# === Display Classification Report ===
print("Classification Report:")
print(classification_rep)


# === Save Classification Report === (already done)
with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
    f.write(classification_rep)


# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test_filtered, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[id_to_label[int(cls)] for cls in model.classes_],
            yticklabels=[id_to_label[int(cls)] for cls in model.classes_])
plt.title("Confusion Matrix")
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("True", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
plt.show()


# === ROC Curve (Multiclass, One-vs-Rest) ===
plt.figure(figsize=(10, 7))
for i, class_label in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"{id_to_label[int(class_label)]}")
plt.title("ROC Curve (OvR)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "roc_curve.png"))
plt.show()


# === SHAP Summary Plot ===
try:
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_train_final)
    shap_values = explainer(X_test_final[:100])  # Using only the first 100 samples for analysis


    # Fix class names shown in SHAP legend using the friendly names
    friendly_class_names = [friendly_drug_names.get(label_to_drug[i], label_to_drug[i]) for i in range(len(label_to_drug))]
    shap_values._names = friendly_class_names  # Override internal names for better legends


    # Aggregate the SHAP values across classes and calculate importance
    shap_importance = np.abs(shap_values.values).mean(axis=2).mean(axis=0)


    # Ensure the number of features in shap_importance matches the columns in X_train_final
    if shap_importance.shape[0] != X_train_final.shape[1]:
        raise ValueError(f"The number of SHAP importances ({shap_importance.shape[0]}) does not match the number of features ({X_train_final.shape[1]}) in X_train_final.")


    # Create a DataFrame to store the feature importances
    shap_importance_df = pd.DataFrame({
        'Feature': X_train_final.columns,  # This should now contain the correct feature names
        'SHAP Importance': shap_importance
    })


    # Sort the importance values in descending order
    shap_importance_df = shap_importance_df.sort_values(by='SHAP Importance', ascending=False)


    # Save SHAP feature importances to CSV
    shap_importance_df.to_csv(os.path.join(eval_dir, 'shap_feature_importance.csv'), index=False)


    # Optionally, print the top 10 features to the console
    print(shap_importance_df.head(10))


    # === SHAP Summary Plot ===
    # Generate the summary plot for SHAP feature importance with the friendly class names
    shap.summary_plot(shap_values, X_test_final[:100], class_names=friendly_class_names, show=False)


    # Set title and save the plot
    plt.title("SHAP Summary Plot", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'shap_summary_plot.png'))
    plt.show()  # Ensure the plot is displayed


except Exception as e:
    print("⚠️ SHAP analysis skipped due to:", e)


# === Save Trained Model ===
joblib.dump(model, f"/content/drive/MyDrive/Thesis/predictor/model/{model_name}_model.pkl")


print("✅ Evaluation, plots, and model saved successfully.")
# Logistic Regression 
import numpy as np
import pandas as pd
import os
import json
import shap
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_curve, auc)
from sklearn.preprocessing import LabelBinarizer
from IPython.display import Image, display


# === SETTINGS ===
model_name = "logistic_regression"
output_dir = f"/content/drive/MyDrive/Thesis/evaluation/{model_name}_output"
model_path = f"/content/drive/MyDrive/Thesis/predictor/model/{model_name}.pkl"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("/content/drive/MyDrive/Thesis/predictor/model", exist_ok=True)


# === DRUG NAME DECODING ===
with open("drug_label_mapping.json") as f:
    label_mapping = json.load(f)
inv_label_mapping = {v: k for k, v in label_mapping.items()}


friendly_drug_names = {
    name: name[:15] + '...' if len(name) > 18 else name
    for name in inv_label_mapping.values()
}


# === Data Preprocessing ===
# Assuming X_train_final and X_test_final have already been preprocessed in preprocess_features function
X_train_processed, X_test_processed, _, _ = preprocess_features(
    X_train_filtered, X_test_filtered, y_train_filtered, genetic_columns, k_genetic=130, k_drug=170
)


# === Model Training ===
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs'))
])


pipeline.fit(X_train_final, y_train_filtered)
y_pred = pipeline.predict(X_test_final)
y_pred_prob = pipeline.predict_proba(X_test_final)


# === Decode Predictions ===
y_test_names = [inv_label_mapping[i] for i in y_test_filtered]
y_pred_names = [inv_label_mapping[i] for i in y_pred]




# === Classification Report ===
# Determine only used class indices
used_classes = sorted(set(y_test_filtered) | set(y_pred))


# Get friendly class names (label index → drug name → short name)
used_class_names = [
    friendly_drug_names.get(label_to_drug[cls], label_to_drug[cls])
    for cls in used_classes
]


# Generate classification report
clf_report = classification_report(
    y_test_names,
    y_pred_names,
    target_names=used_class_names,
    output_dict=False
)


# Print the classification report to the console
print("\n=== Classification Report ===")
print(clf_report)


# Save classification report to file
with open(os.path.join(output_dir, f"{model_name}_classification_report.txt"), "w") as f:
    f.write(clf_report)


# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test_names, y_pred_names)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[friendly_drug_names[inv_label_mapping[i]] for i in np.unique(y_test_filtered)],
            yticklabels=[friendly_drug_names[inv_label_mapping[i]] for i in np.unique(y_test_filtered)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
conf_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_path)
plt.show()


# === ROC Curves ===
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_filtered)
plt.figure()
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    label = friendly_drug_names[inv_label_mapping[lb.classes_[i]]]
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc='lower right')
roc_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_path)
plt.show()


# === Custom Metrics ===
def top_5_accuracy(y_true, y_pred_prob):
    top_5_preds = np.argsort(y_pred_prob, axis=1)[:, -5:]
    return np.mean([y in top_5 for y, top_5 in zip(y_true, top_5_preds)])


def ndcg_at_k(y_true, y_pred_prob, k=5):
    dcg = 0
    idcg = 0
    for i, true_class in enumerate(y_true):
        rank = np.where(np.argsort(-y_pred_prob[i]) == true_class)[0]
        if rank.size > 0:
            dcg += 1 / np.log2(rank[0] + 2)
        idcg += 1 / np.log2(2)
    return dcg / idcg if idcg != 0 else 0


def mrr(y_true, y_pred_prob):
    score = 0
    for i, true_class in enumerate(y_true):
        rank = np.where(np.argsort(-y_pred_prob[i]) == true_class)[0]
        if rank.size > 0:
            score += 1 / (rank[0] + 1)
    return score / len(y_true)


metrics = {
    "Accuracy": accuracy_score(y_test_filtered, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_test_filtered, y_pred),
    "Precision (macro)": precision_score(y_test_filtered, y_pred, average='macro'),
    "Precision (weighted)": precision_score(y_test_filtered, y_pred, average='weighted'),
    "Recall (macro)": recall_score(y_test_filtered, y_pred, average='macro'),
    "Recall (weighted)": recall_score(y_test_filtered, y_pred, average='weighted'),
    "F1 Score (macro)": f1_score(y_test_filtered, y_pred, average='macro'),
    "F1 Score (weighted)": f1_score(y_test_filtered, y_pred, average='weighted'),
    "MCC": matthews_corrcoef(y_test_filtered, y_pred),
    "Top-5 Accuracy": top_5_accuracy(y_test_filtered, y_pred_prob),
    "NDCG@5": ndcg_at_k(y_test_filtered, y_pred_prob),
    "MRR": mrr(y_test_filtered, y_pred_prob),
}


metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)


# Print each metric to console
print("\n=== Metrics ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")




# === SHAP ===
try:
    # Initialize SHAP explainer using the trained logistic regression model
    explainer = shap.Explainer(pipeline.named_steps['logreg'], X_train_final)
    shap_values = explainer(X_test_final[:100])  # Using only the first 100 samples for analysis


    # Fix class names shown in SHAP legend using the friendly names
    friendly_class_names = [friendly_drug_names.get(label_to_drug[i], label_to_drug[i]) for i in range(len(label_to_drug))]
    shap_values._names = friendly_class_names  # Override internal names for better legends


    # Aggregate the SHAP values across classes and calculate importance
    shap_importance = np.abs(shap_values.values).mean(axis=2).mean(axis=0)


    # Ensure the number of features in shap_importance matches the columns in X_train_final
    if shap_importance.shape[0] != X_train_final.shape[1]:
        raise ValueError(f"The number of SHAP importances ({shap_importance.shape[0]}) does not match the number of features ({X_train_final.shape[1]}) in X_train_final.")


    # Create a DataFrame to store the feature importances
    shap_importance_df = pd.DataFrame({
        'Feature': X_train_final.columns,  # This should now contain the correct feature names
        'SHAP Importance': shap_importance
    })


    # Sort the importance values in descending order
    shap_importance_df = shap_importance_df.sort_values(by='SHAP Importance', ascending=False)


    # Save SHAP feature importances to CSV
    shap_importance_df.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)


    # Optionally, print the top 10 features to the console
    print(shap_importance_df.head(10))


    # === SHAP Summary Plot ===
    # Generate the summary plot for SHAP feature importance with the friendly class names
    shap.summary_plot(shap_values, X_test_final[:100], class_names=friendly_class_names, show=False)


    # Set title and save the plot
    plt.title("SHAP Summary Plot", fontsize=18)
    plt.tight_layout()
    shap_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(shap_path)
    plt.show()


except Exception as e:
    print("⚠️ SHAP analysis skipped due to:", e)


# === Save Model ===
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
# LightGBM
import json
import numpy as np
import shap
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix,
    top_k_accuracy_score, roc_auc_score, roc_curve
)
from scipy.stats import rankdata


plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})


# === Load Drug Label Mapping ===
with open("drug_label_mapping.json", "r") as f:
    label_mapping = json.load(f)
id_to_label = {int(v): k for k, v in label_mapping.items()}  # integer class label to drug name


# === Model Training (Random Forest) ===
X_train_final = X_train_final.loc[:, ~X_train_final.columns.duplicated()]
X_test_final = X_test_final.loc[:, ~X_test_final.columns.duplicated()]


print("🔍 Training LightGBM...")
model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train_filtered)


# === Predictions ===
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)
class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}


# === Metrics ===
accuracy = accuracy_score(y_test_filtered, y_pred)
precision_macro = precision_score(y_test_filtered, y_pred, average='macro', zero_division=1)
precision_weighted = precision_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
recall_macro = recall_score(y_test_filtered, y_pred, average='macro', zero_division=1)
recall_weighted = recall_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
f1_macro = f1_score(y_test_filtered, y_pred, average='macro', zero_division=1)
f1_weighted = f1_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
mcc = matthews_corrcoef(y_test_filtered, y_pred)


# Binarize y_test for ROC-AUC
y_test_bin = label_binarize(y_test_filtered, classes=np.unique(y_train_filtered))
try:
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba[:, :y_test_bin.shape[1]], average='macro', multi_class='ovr')
except ValueError:
    roc_auc = None


def mean_reciprocal_rank(y_true, y_scores, class_indices):
    ranks = []
    for true, score in zip(y_true, y_scores):
        ranking = rankdata(-score, method='ordinal')
        true_idx = class_indices[true]
        rank = ranking[true_idx]
        ranks.append(1.0 / rank)
    return np.mean(ranks)


mrr = mean_reciprocal_rank(y_test_filtered.to_numpy(), y_pred_proba, class_indices)


def ndcg_score(y_true, y_score, k=5):
    dcg = 0
    for true, scores in zip(y_true, y_score):
        order = np.argsort(scores)[::-1][:k]
        if true in order:
            index = list(order).index(true)
            dcg += 1 / np.log2(index + 2)
    idcg = 1.0
    return dcg / len(y_true) / idcg


ndcg = ndcg_score(y_test_filtered.to_numpy(), y_pred_proba)


try:
    top_k_acc = top_k_accuracy_score(y_test_filtered, y_pred_proba, k=5, labels=np.unique(y_train_filtered))
except ValueError:
    top_k_acc = None


# === Classification Report ===
classification_rep = classification_report(y_test_filtered, y_pred, zero_division=1, target_names=[id_to_label[int(cls)] for cls in model.classes_])


# === Output Directory Setup ===
model_name = "lightgbm"
eval_dir = f"/content/drive/MyDrive/Thesis/evaluation/{model_name}_output/"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs("/content/drive/MyDrive/Thesis/predictor/model/", exist_ok=True)


# === Save Metrics ===
metrics = {
    "Accuracy": accuracy,
    "Precision (Macro)": precision_macro,
    "Precision (Weighted)": precision_weighted,
    "Recall (Macro)": recall_macro,
    "Recall (Weighted)": recall_weighted,
    "F1 Score (Macro)": f1_macro,
    "F1 Score (Weighted)": f1_weighted,
    "MCC": mcc,
    "ROC-AUC": roc_auc if roc_auc is not None else 'N/A',
    "MRR": mrr,
    "NDCG@5": ndcg,
    "Top-5 Accuracy": top_k_acc if top_k_acc is not None else 'Skipped'
}
pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, f"{model_name}_metrics.csv"), index=False)


# === Display Metrics ===
print("🔎 Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision (Macro): {precision_macro}")
print(f"Precision (Weighted): {precision_weighted}")
print(f"Recall (Macro): {recall_macro}")
print(f"Recall (Weighted): {recall_weighted}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Weighted): {f1_weighted}")
print(f"MCC: {mcc}")
print(f"ROC-AUC: {roc_auc if roc_auc is not None else 'N/A'}")
print(f"MRR: {mrr}")
print(f"NDCG@5: {ndcg}")
print(f"Top-5 Accuracy: {top_k_acc if top_k_acc is not None else 'Skipped'}")




# === Display Classification Report ===
print("🔎 Classification Report:")
print(classification_rep)


# === Save Classification Report ===
with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
    f.write(classification_rep)


# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test_filtered, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[id_to_label[int(cls)] for cls in model.classes_],
            yticklabels=[id_to_label[int(cls)] for cls in model.classes_])
plt.title("Confusion Matrix")
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("True", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
plt.show()


# === ROC Curve (Multiclass, One-vs-Rest) ===
plt.figure(figsize=(10, 7))
for i, class_label in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"{id_to_label[int(class_label)]}")
plt.title("ROC Curve (OvR)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "roc_curve.png"))
plt.show()


# === SHAP ===
try:
    # Initialize SHAP explainer using the trained LightGBM model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_final[:100])  # Using only the first 100 samples for analysis


    # Fix class names shown in SHAP legend using the friendly names
    friendly_class_names = [friendly_drug_names.get(label_to_drug[i], label_to_drug[i]) for i in range(len(label_to_drug))]
    shap_values._names = friendly_class_names  # Override internal names for better legends


    # Aggregate the SHAP values across classes and calculate importance
    shap_importance = np.abs(shap_values.values).mean(axis=2).mean(axis=0)


    # Ensure the number of features in shap_importance matches the columns in X_train_final
    if shap_importance.shape[0] != X_train_final.shape[1]:
        raise ValueError(f"The number of SHAP importances ({shap_importance.shape[0]}) does not match the number of features ({X_train_final.shape[1]}) in X_train_final.")


    # Create a DataFrame to store the feature importances
    shap_importance_df = pd.DataFrame({
        'Feature': X_train_final.columns,  # This should now contain the correct feature names
        'SHAP Importance': shap_importance
    })


    # Sort the importance values in descending order
    shap_importance_df = shap_importance_df.sort_values(by='SHAP Importance', ascending=False)


    # Save SHAP feature importances to CSV
    shap_importance_df.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)


    # Optionally, print the top 10 features to the console
    print(shap_importance_df.head(10))


    # === SHAP Summary Plot ===
    # Generate the summary plot for SHAP feature importance with the friendly class names
    shap.summary_plot(shap_values, X_test_final[:100], class_names=friendly_class_names, show=False)


    # Set title and save the plot
    plt.title("SHAP Summary Plot", fontsize=18)
    plt.tight_layout()
    shap_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(shap_path)
    plt.show()


except Exception as e:
    print("⚠️ SHAP analysis skipped due to:", e)


# === Save Model ===
joblib.dump(model, "/content/drive/MyDrive/Thesis/predictor/model/lightgbm_model.pkl")
# Support Vector Machine
import json
import numpy as np
import shap
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score,
   matthews_corrcoef, classification_report, confusion_matrix,
   top_k_accuracy_score, roc_auc_score, roc_curve
)
from scipy.stats import rankdata


plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})


# === Load Drug Label Mapping ===
with open("drug_label_mapping.json", "r") as f:
   label_mapping = json.load(f)
id_to_label = {int(v): k for k, v in label_mapping.items()}


# === Data Preparation ===
X_train_final = X_train_final.loc[:, ~X_train_final.columns.duplicated()]
X_test_final = X_test_final.loc[:, ~X_test_final.columns.duplicated()]


# === Model Training ===
print("Training SVM without probability...")
base_model = SVC(kernel='linear', random_state=42)
base_model.fit(X_train_final, y_train_filtered)


# === Calibrate Model ===
print("Calibrating model for probabilities...")
model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)  # (cv=3 is safe, speeds up)
model.fit(X_train_final, y_train_filtered)


# === Predictions ===
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)
class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}


# === Metrics ===
accuracy = accuracy_score(y_test_filtered, y_pred)
precision_macro = precision_score(y_test_filtered, y_pred, average='macro', zero_division=1)
precision_weighted = precision_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
recall_macro = recall_score(y_test_filtered, y_pred, average='macro', zero_division=1)
recall_weighted = recall_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
f1_macro = f1_score(y_test_filtered, y_pred, average='macro', zero_division=1)
f1_weighted = f1_score(y_test_filtered, y_pred, average='weighted', zero_division=1)
mcc = matthews_corrcoef(y_test_filtered, y_pred)


# ROC-AUC
y_test_bin = label_binarize(y_test_filtered, classes=np.unique(y_train_filtered))
try:
   roc_auc = roc_auc_score(y_test_bin, y_pred_proba[:, :y_test_bin.shape[1]], average='macro', multi_class='ovr')
except ValueError:
   roc_auc = None


# MRR
def mean_reciprocal_rank(y_true, y_scores, class_indices):
   ranks = []
   for true, score in zip(y_true, y_scores):
       ranking = rankdata(-score, method='ordinal')
       true_idx = class_indices[true]
       rank = ranking[true_idx]
       ranks.append(1.0 / rank)
   return np.mean(ranks)


mrr = mean_reciprocal_rank(y_test_filtered.to_numpy(), y_pred_proba, class_indices)


# NDCG@5
def ndcg_score(y_true, y_score, k=5):
   dcg = 0
   for true, scores in zip(y_true, y_score):
       order = np.argsort(scores)[::-1][:k]
       if true in order:
           index = list(order).index(true)
           dcg += 1 / np.log2(index + 2)
   idcg = 1.0
   return dcg / len(y_true) / idcg


ndcg = ndcg_score(y_test_filtered.to_numpy(), y_pred_proba)


# Top-5 Accuracy
try:
   top_k_acc = top_k_accuracy_score(y_test_filtered, y_pred_proba, k=5, labels=np.unique(y_train_filtered))
except ValueError:
   top_k_acc = None


# === Classification Report ===
classification_rep = classification_report(y_test_filtered, y_pred, zero_division=1, target_names=[id_to_label[int(cls)] for cls in model.classes_])


# === Output Directory Setup ===
model_name = "svm"
eval_dir = f"/content/drive/MyDrive/Thesis/evaluation/{model_name}_output/"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs("/content/drive/MyDrive/Thesis/predictor/model/", exist_ok=True)


# === Save Metrics ===
metrics = {
   "Accuracy": accuracy,
   "Precision (Macro)": precision_macro,
   "Precision (Weighted)": precision_weighted,
   "Recall (Macro)": recall_macro,
   "Recall (Weighted)": recall_weighted,
   "F1 Score (Macro)": f1_macro,
   "F1 Score (Weighted)": f1_weighted,
   "MCC": mcc,
   "ROC-AUC": roc_auc if roc_auc is not None else 'N/A',
   "MRR": mrr,
   "NDCG@5": ndcg,
   "Top-5 Accuracy": top_k_acc if top_k_acc is not None else 'Skipped'
}
pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, f"{model_name}_metrics.csv"), index=False)


# === Save Classification Report ===
with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
   f.write(classification_rep)


# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test_filtered, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=[id_to_label[int(cls)] for cls in model.classes_],
           yticklabels=[id_to_label[int(cls)] for cls in model.classes_])
plt.title("Confusion Matrix")
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("True", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
plt.show()


# === ROC Curve ===
plt.figure(figsize=(10, 7))
for i, class_label in enumerate(model.classes_):
   if np.isnan(y_pred_proba[:, i]).any():
       print(f"⚠️ Skipping class {class_label} due to NaN probabilities.")
       continue
   fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
   plt.plot(fpr, tpr, label=f"{id_to_label[int(class_label)]}")
plt.title("ROC Curve (OvR)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "roc_curve.png"))
plt.show()


# === SHAP Analysis ===
print("Starting SHAP analysis...")
sample_X = X_test_final.sample(n=min(100, len(X_test_final)), random_state=42)  # safer and faster
explainer = shap.LinearExplainer(base_model, X_train_final, feature_perturbation="interventional")
shap_values = explainer.shap_values(sample_X)


shap.summary_plot(shap_values, sample_X, show=False)
plt.title("SHAP Summary Plot", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "shap_summary_plot.png"))
plt.show()


# === Save Model ===
joblib.dump(model, "/content/drive/MyDrive/Thesis/predictor/model/svm_model.pkl")


# === Notify Completion ===
os.system('say \"SVM model training and evaluation completed\"')  # Mac notification
print("✅ SVM model training and evaluation completed.")

