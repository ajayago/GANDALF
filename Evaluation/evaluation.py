import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
import pandas as pd

def evaluate_patient(input_file):
    res_df = pd.read_csv(input_file)
    y_true = res_df["y_true"].values
    y_pred = res_df["y_pred"].values
    print(f"AUROC = {round(roc_auc_score(y_true, y_pred), 4)}")
    print(f"AUPRC = {round(average_precision_score(y_true, y_pred), 4)}")

def evaluate_cell_lines(input_file):
    res_df = pd.read_csv(input_file)
    y_true = res_df["y_true"].values
    y_pred = res_df["y_pred"].values
    print(f"MSE = {round(mean_squared_error(y_true, y_pred), 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluation")
    parser.add_argument('--save_dir', dest="save_dir", default="/data//papers_data/DiffDRP/run_files")
    parser.add_argument("--checkpoint", dest="checkpoint", default="saved_model")
    parser.add_argument("--model_save_criteria", dest="model_save_criteria", default="val_corr")
    parser.add_argument("--experiment_id", dest="experiment_id", default="1A", choices=["1A", "1B", "2A", "2B"])
    parser.add_argument("--experiment_settings", dest="experiment_settings", default="CISPLATIN")
    parser.add_argument("--fold", dest="fold", default="0", choices=["0", "1", "2"]) 

    args = parser.parse_args()
    print(args)

    # Patient prediction evaluation
    input_file_patients = f"{args.save_dir}/{args.checkpoint}/prediction_patients_{args.model_save_criteria}_{args.experiment_id}_{args.experiment_settings}_fold{args.fold}.csv"
    evaluate_patient(input_file_patients)
