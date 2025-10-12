
Deployment package for MoE + LGBM residual model.
Location: /content/gdrive/MyDrive/saudi_supply_demand/deployment_package

Files included:
- moe_entropy_final.pth  # PyTorch weights (original trained checkpoint)
- lgbm_resid.txt   # LightGBM residual model
- feat_scaler_moe_plus.pkl  # feature scaler (joblib)
- sku_stats_moe_plus.pkl    # per-SKU train stats (joblib)
- deployment_metadata.json         # metadata describing seq_len, features, etc.
- moe_model_torchscript.pt         # TorchScript export (optional, may not exist)
- moe_model.onnx                   # ONNX export (optional)
- this cell creates a helper function `predict_for_requests(raw_df)`

Inference notes:
- Input DataFrame must contain the feature columns listed in deployment_metadata.json plus 'sku_id' and 'time_idx'.
- You must provide at least SEQ_LEN rows per SKU to create a sequence.
- This predict function produces a row per sliding window (all windows) for each sku in the input.
- For low-latency online deployment you should:
    1) maintain the most recent SEQ_LEN features per SKU (in memory or KV store),
    2) call predict_for_requests with that small batch,
    3) keep LGBM and scaler loaded in the service process for speed.

Example usage (in Colab / server):
from joblib import load
feat_scaler = load("/content/gdrive/MyDrive/saudi_supply_demand/deep_learning_models/feat_scaler_moe_plus.pkl")
sku_stats = load("/content/gdrive/MyDrive/saudi_supply_demand/deep_learning_models/sku_stats_moe_plus.pkl")
# build df_recent with required columns -> call predict_for_requests(df_recent)

