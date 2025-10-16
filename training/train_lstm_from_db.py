import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from urllib.parse import urlparse
from datetime import datetime, timezone

# ========= 隨機種子（可重現） =========
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ========= 先載入 .env，再取環境變數 =========
load_dotenv()

DB_URL      = os.getenv("DB_URL")
STEP_MIN    = int(os.getenv("STEP_MINUTES", "1"))
WINDOW      = int(os.getenv("WINDOW", "72"))
EPOCHS      = int(os.getenv("EPOCHS", "20"))
BATCH       = int(os.getenv("BATCH", "128"))

# 可選訓練期間參數
TRAIN_DAYS      = os.getenv("TRAIN_DAYS")
TRAIN_START_UTC = os.getenv("TRAIN_START_UTC")
TRAIN_END_UTC   = os.getenv("TRAIN_END_UTC")

# 防呆：印出 DB_URL（遮擋密碼）
u = urlparse(DB_URL) if DB_URL else None
if not DB_URL:
    raise RuntimeError("DB_URL 未設定，請在 training/.env 補上。")
print("Using DB_URL:", f"{u.scheme}://{u.username}:******@{u.hostname}:{u.port}{u.path}")
print(f"STEP_MIN={STEP_MIN}, WINDOW={WINDOW}, EPOCHS={EPOCHS}, BATCH={BATCH}, "
      f"TRAIN_DAYS={TRAIN_DAYS}, TRAIN_START_UTC={TRAIN_START_UTC}, TRAIN_END_UTC={TRAIN_END_UTC}")

# ========= 欄位名稱 =========
TIME_COL = "timestamp"
PWR_COL  = "system_power_watt"
N_FEATURES = 3 # <<< 關鍵修正：特徵數量變為 3 (power_w, hour, dayofweek)

# ========= SQL（timestamp_utc::timestamptz） =========
where_extra, params = [], {}
if TRAIN_DAYS and TRAIN_DAYS.isdigit():
    where_extra.append("(timestamp_utc)::timestamptz >= now() - interval :train_days")
    params["train_days"] = f"{int(TRAIN_DAYS)} days"
if TRAIN_START_UTC:
    where_extra.append("(timestamp_utc)::timestamptz >= :start_utc")
    params["start_utc"] = TRAIN_START_UTC
if TRAIN_END_UTC:
    where_extra.append("(timestamp_utc)::timestamptz <= :end_utc")
    params["end_utc"] = TRAIN_END_UTC

where_clause = (" AND " + " AND ".join(where_extra) + "\n") if where_extra else ""

SQL = f"""
SELECT
  (timestamp_utc)::timestamptz AS {TIME_COL},
  {PWR_COL}
FROM energy_cleaned
WHERE timestamp_utc IS NOT NULL
  AND {PWR_COL} IS NOT NULL
{where_clause}ORDER BY (timestamp_utc)::timestamptz
""".strip()
print("DEBUG SQL:\n", SQL)

# ========= 讀資料 =========
engine = create_engine(DB_URL, pool_pre_ping=True)
df = pd.read_sql(text(SQL), engine, params=params, parse_dates=[TIME_COL])
if df.empty:
    raise RuntimeError("energy_cleaned 查無資料（符合條件為 0）。")

print(f"Loaded rows: {len(df)} | time span: {df[TIME_COL].min()} → {df[TIME_COL].max()}")

# ========= 時間對齊、補值與特徵工程 (關鍵修正) =========
df = df.set_index(TIME_COL).sort_index()
df = df.resample(f"{STEP_MIN}min").mean()
df[PWR_COL] = df[PWR_COL].ffill().bfill()
df = df.dropna(subset=[PWR_COL])

# >>>>>>>>>>> 修正 1：新增時間特徵 (提升準確性) <<<<<<<<<<<<
# 確保索引是 datetime 類型
df.index = pd.to_datetime(df.index)

# 創建 Hour (0-23) 和 DayofWeek (0=Monday, 6=Sunday)
df['hour'] = df.index.hour.astype(float)
df['dayofweek'] = df.index.dayofweek.astype(float)

# 選擇所有特徵：功耗、小時、星期幾
features = df[[PWR_COL, 'hour', 'dayofweek']].astype(float).values
# >>>>>>>>>>> 修正 1 結束 <<<<<<<<<<<<


# ========= 轉 numpy + 正規化與建序列 (關鍵修正) =========
if len(features) < WINDOW + 2:
    # 修正：如果數據量不足，自動調整 WINDOW 大小
    new_window = max(1, int(len(features) * 0.7) - 2)
    if new_window < WINDOW:
        print(f"⚠️ 資料量 ({len(features)}) 不足，自動降低 WINDOW: {WINDOW} → {new_window}")
        WINDOW = new_window
    if len(features) < WINDOW + 2:
        raise RuntimeError(f"資料量不足（{len(features)} < WINDOW+2）請增加資料或降低 WINDOW。")

print(f"Total processed rows: {len(features)}")

# 先切分數據，再進行 Scaling (避免資料洩漏)
split_idx = int(len(features) * 0.8)
features_tr, features_val = features[:split_idx], features[split_idx:]
print(f"Split index: {split_idx} / {len(features)}")

# 修正 2A：訓練 Scaler 時使用所有 N_FEATURES
scaler = MinMaxScaler()
features_tr_scaled  = scaler.fit_transform(features_tr)
features_val_scaled = scaler.transform(features_val)

def make_sequences(arr, window, n_features):
    X, y = [], []
    for i in range(len(arr) - window - 1):
        X.append(arr[i:i+window])
        # 修正 2B：預測目標 y 仍然是功耗 (第 0 列)
        y.append(arr[i+window, 0]) 
    # X 的形狀是 (-1, WINDOW, N_FEATURES)
    return np.array(X).reshape(-1, window, n_features), np.array(y).reshape(-1, 1)

# 修正 2C：呼叫 make_sequences 時傳遞 N_FEATURES=3
X_tr, y_tr   = make_sequences(features_tr_scaled, WINDOW, N_FEATURES)
X_val, y_val = make_sequences(features_val_scaled, WINDOW, N_FEATURES)

if len(X_tr) == 0 or len(X_val) == 0:
    raise RuntimeError("切分後無法形成序列，請調整資料量或 WINDOW。")

print(f"Train/Val shapes: X_tr={X_tr.shape}, X_val={X_val.shape}")

# ========= 建模與訓練 (關鍵修正) =========
# 修正 3：LSTM 輸入形狀必須匹配 N_FEATURES=3
model = Sequential([
    LSTM(64, input_shape=(WINDOW, N_FEATURES), return_sequences=False), # return_sequences=False for single output
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# ========= 訓練（LR 調度 + EarlyStopping） =========
es  = EarlyStopping(patience=8, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=[es, rlr],
    verbose=1
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

# ========= 評估（反標準化）(關鍵修正) =========
y_pred = model.predict(X_val, verbose=0)

# 修正 4：反標準化時必須創建包含所有特徵的臨時陣列
y_temp = np.zeros((len(y_val), N_FEATURES))
y_pred_temp = np.zeros((len(y_pred), N_FEATURES))

# 將原始 y 值和預測 y 值放入第 0 列 (power_w 的位置)
y_temp[:, 0] = y_val.flatten()
y_pred_temp[:, 0] = y_pred.flatten()

# 反向轉換時只取第 0 列 (power_w)
y_val_w = scaler.inverse_transform(y_temp)[:, 0]
y_pred_w = scaler.inverse_transform(y_pred_temp)[:, 0]

rmse = float(np.sqrt(mean_squared_error(y_val_w, y_pred_w)))
mape = float(mean_absolute_percentage_error(y_val_w, y_pred_w) * 100)
best_val_loss = float(np.min(history.history["val_loss"]))
try:
    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
except Exception:
    current_lr = 0.0

print(f"✅ 最佳 Epoch 編號：{best_epoch} / {EPOCHS}")
print(f"RMSE(W): {rmse:.2f} | MAPE(%): {mape:.2f} | best_val_loss: {best_val_loss:.6f} | lr={current_lr:.6g}")

# ========= 輸出模型與 Scaler =========
out_dir = Path("/models")
if not out_dir.exists():
    out_dir = Path(__file__).resolve().parents[1] / "models"
out_dir.mkdir(parents=True, exist_ok=True)

model_path  = out_dir / "lstm_carbon_model.keras"
scaler_path = out_dir / "scaler_power.pkl"
model.save(model_path)
joblib.dump(scaler, scaler_path)
print("✅ Saved:", model_path, "and", scaler_path)

# ========= 訓練日誌寫入 /models/train_log.txt =========
log_path = out_dir / "train_log.txt"
data_start = df.index.min()
data_end   = df.index.max()
n_raw      = len(df)
n_tr       = len(X_tr)
n_val      = len(X_val)

log_line = (
    f"{datetime.now(timezone.utc).isoformat()} "
    f"| best_epoch={best_epoch}/{EPOCHS} best_val_loss={best_val_loss:.6f} "
    f"| RMSE={rmse:.3f}W MAPE={mape:.2f}% lr={current_lr:.6g} "
    f"| window={WINDOW} step_min={STEP_MIN} batch={BATCH} "
    f"| data_span=[{data_start} → {data_end}] n_raw={n_raw} n_tr_seq={n_tr} n_val_seq={n_val} "
    f"| model={model_path.name} scaler={scaler_path.name}\n"
)

try:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"📝 Appended train log → {log_path}")
except Exception as e:
    print(f"⚠️ Failed to write train log: {e}")