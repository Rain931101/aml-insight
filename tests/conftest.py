"""pytest 共用設定：避免 PyTorch 與 XGBoost 的 OpenMP 衝突。

macOS 上 PyTorch、scikit-learn 與 Homebrew 各自帶一份 libomp。
同一個程式先載入 torch、再訓練 XGBoost 時，多份 OpenMP 同時啟動多執行緒
會直接 Segmentation fault（exit 139）。

把 OpenMP 限制為單一執行緒即可避免。必須在任何 torch / xgboost 被 import
之前設定，所以放在 conftest.py（pytest 會最先載入）。
XGBoost 在本專案本來就以 n_jobs=1 訓練，對速度影響很小。
若使用者已自行設定 OMP_NUM_THREADS，則尊重原本的值。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
