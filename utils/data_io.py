import pickle
import json
from pathlib import Path
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
import pandas as pd


def read_data_from_dataset(data_dir_path: str):
    """load train data and test data from the dataset

    Args:
        data_dir_path (str): path for the dataset we want to load (資料集所在的路徑)

    Returns:
        tuple: tuple which contains X_train, y_train, X_test and y_test in order
    """
    data_list = []
    for fname in ['X_train', 'y_train', 'X_test', 'y_test']:
        with open(f'{data_dir_path}/{fname}.pkl', 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    return tuple(data_list)


def read_corrected_r1_profile(
    profile_dir,
    *,
    splits=None,
    scale="normalized",
):
    """Read one existing Corrected R1 profile without changing its contents.

    The return value keeps the manifests and split frames together so the
    separate validation function can enforce the fixed contract.  This reader
    never falls back to Legacy pickle files and never creates a dataset copy.
    """

    from r2_config.corrected_r1 import CORRECTED_R1_SPLITS

    directory = Path(profile_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Corrected R1 profile not found: {directory}")
    if scale not in ("normalized", "original"):
        raise ValueError("scale must be 'normalized' or 'original'")

    requested_splits = tuple(CORRECTED_R1_SPLITS if splits is None else splits)
    if not requested_splits or len(set(requested_splits)) != len(requested_splits):
        raise ValueError("Corrected R1 splits must be non-empty and unique")
    unknown_splits = set(requested_splits) - set(CORRECTED_R1_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown Corrected R1 splits: {sorted(unknown_splits)}")

    def load_json(file_name):
        file_path = directory / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Corrected R1 metadata not found: {file_path}")
        try:
            value = json.loads(file_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Corrected R1 metadata {file_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Corrected R1 metadata root must be an object: {file_path}")
        return value

    split_manifest = load_json("split_manifest.json")
    scaler_manifest = load_json("scaler_manifest.json")
    manifest_splits = split_manifest.get("splits")
    if not isinstance(manifest_splits, dict):
        raise ValueError("Corrected R1 split_manifest.json has no splits object")

    frames = {}
    directory_resolved = directory.resolve()
    for split_name in requested_splits:
        entry = manifest_splits.get(split_name)
        if not isinstance(entry, dict):
            raise ValueError(f"Corrected R1 manifest split is missing: {split_name}")
        file_key = f"{scale}_scale_file"
        expected_file_name = f"{scale}_scale_{split_name}.csv"
        file_name = entry.get(file_key)
        if file_name != expected_file_name:
            raise ValueError(
                f"Unexpected Corrected R1 {split_name} filename: {file_name!r}"
            )
        csv_path = (directory / file_name).resolve()
        if csv_path.parent != directory_resolved or not csv_path.is_file():
            raise FileNotFoundError(f"Corrected R1 split file not found: {csv_path}")
        try:
            frames[split_name] = pd.read_csv(csv_path)
        except Exception as exc:
            raise ValueError(f"Cannot read Corrected R1 split {csv_path}: {exc}") from exc

    return {
        "profile_dir": directory,
        "scale": scale,
        "split_manifest": split_manifest,
        "scaler_manifest": scaler_manifest,
        "splits": frames,
    }


def validate_corrected_r1_profile(
    profile,
    *,
    expected_plant=None,
    expected_profile=None,
    expected_rows=None,
    expected_sequences=None,
    feature_order=None,
    target_column=None,
    timestamp_column=None,
    window=None,
    horizon=None,
):
    """Validate one loaded Corrected R1 profile and return its audit facts.

    ``profile`` may be the mapping returned by
    :func:`read_corrected_r1_profile` or a profile directory.  Validation is
    read-only and requires chronological, unpadded, one-step-ahead sequences.
    """

    from r2_config.corrected_r1 import (
        CORRECTED_R1_EXPECTED_ROWS,
        CORRECTED_R1_EXPECTED_SEQUENCES,
        CORRECTED_R1_FEATURE_ORDER,
        CORRECTED_R1_FREQUENCY,
        CORRECTED_R1_HORIZON,
        CORRECTED_R1_SCALED_FEATURES,
        CORRECTED_R1_SPLITS,
        CORRECTED_R1_TARGET,
        CORRECTED_R1_TIMESTAMP,
        CORRECTED_R1_WINDOW,
    )

    if isinstance(profile, (str, Path)):
        profile = read_corrected_r1_profile(profile)
    if not isinstance(profile, dict):
        raise TypeError("profile must be a Corrected R1 profile mapping or path")

    split_manifest = profile.get("split_manifest")
    scaler_manifest = profile.get("scaler_manifest")
    frames = profile.get("splits")
    if not isinstance(split_manifest, dict) or not isinstance(scaler_manifest, dict):
        raise ValueError("Corrected R1 profile manifests are missing")
    if not isinstance(frames, dict):
        raise ValueError("Corrected R1 profile split frames are missing")
    if profile.get("scale") != "normalized":
        raise ValueError("Aligned Corrected R1 sequences require normalized-scale data")

    manifest_profile = split_manifest.get("profile")
    manifest_plant = split_manifest.get("plant")
    role = expected_profile or manifest_profile
    if role not in CORRECTED_R1_EXPECTED_ROWS:
        raise ValueError(f"Unknown Corrected R1 profile role: {role!r}")
    expected_plant = manifest_plant if expected_plant is None else expected_plant
    expected_rows = (
        CORRECTED_R1_EXPECTED_ROWS[role] if expected_rows is None else expected_rows
    )
    expected_sequences = (
        CORRECTED_R1_EXPECTED_SEQUENCES[role]
        if expected_sequences is None
        else expected_sequences
    )
    feature_order = tuple(
        CORRECTED_R1_FEATURE_ORDER if feature_order is None else feature_order
    )
    target_column = CORRECTED_R1_TARGET if target_column is None else target_column
    timestamp_column = (
        CORRECTED_R1_TIMESTAMP if timestamp_column is None else timestamp_column
    )
    window = CORRECTED_R1_WINDOW if window is None else window
    horizon = CORRECTED_R1_HORIZON if horizon is None else horizon

    if tuple(frames) != tuple(CORRECTED_R1_SPLITS):
        raise ValueError(
            "Corrected R1 preflight requires training, validation, and test in fixed order"
        )
    if manifest_plant != expected_plant:
        raise ValueError(
            f"Corrected R1 plant mismatch: expected {expected_plant}, got {manifest_plant}"
        )
    if manifest_profile != role:
        raise ValueError(
            f"Corrected R1 profile mismatch: expected {role}, got {manifest_profile}"
        )
    if split_manifest.get("frequency") != CORRECTED_R1_FREQUENCY:
        raise ValueError("Corrected R1 frequency must be 15min")
    if tuple(split_manifest.get("feature_columns", ())) != feature_order:
        raise ValueError("Corrected R1 feature order mismatch")
    if split_manifest.get("target_column") != target_column:
        raise ValueError("Corrected R1 target column mismatch")
    if split_manifest.get("date_time_index_saved") is not True:
        raise ValueError("Corrected R1 DATE_TIME index is not saved")
    if split_manifest.get("date_time_index_label") != timestamp_column:
        raise ValueError("Corrected R1 timestamp column mismatch")

    expected_training_rows = expected_rows["training"]
    scaler_requirements = {
        "feature_scaler_fit_split": "training",
        "target_scaler_fit_split": "training",
        "target_scaler_column": target_column,
        "time_columns_scaled": False,
        "clipping_applied": False,
        "feature_scaler_n_samples_seen": expected_training_rows,
        "target_scaler_n_samples_seen": expected_training_rows,
    }
    for key, expected_value in scaler_requirements.items():
        if scaler_manifest.get(key) != expected_value:
            raise ValueError(
                f"Corrected R1 scaler contract mismatch for {key}: "
                f"expected {expected_value!r}, got {scaler_manifest.get(key)!r}"
            )
    if tuple(scaler_manifest.get("feature_scaler_columns", ())) != tuple(
        CORRECTED_R1_SCALED_FEATURES
    ):
        raise ValueError("Corrected R1 scaled feature order mismatch")

    manifest_splits = split_manifest.get("splits", {})
    expected_columns = (timestamp_column,) + feature_order + (target_column,)
    row_counts = {}
    sequence_counts = {}
    sequence_data = {}
    previous_end = None

    for split_name in CORRECTED_R1_SPLITS:
        frame = frames[split_name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Corrected R1 {split_name} split is not a DataFrame")
        if tuple(frame.columns) != expected_columns:
            raise ValueError(f"Corrected R1 {split_name} column/order mismatch")
        if len(frame) != expected_rows[split_name]:
            raise ValueError(
                f"Corrected R1 {split_name} rows: expected "
                f"{expected_rows[split_name]}, got {len(frame)}"
            )
        if frame.isna().any().any():
            raise ValueError(f"Corrected R1 {split_name} contains NaN")

        entry = manifest_splits.get(split_name)
        if not isinstance(entry, dict) or entry.get("rows") != len(frame):
            raise ValueError(f"Corrected R1 {split_name} manifest row mismatch")
        if entry.get("nan_total_original") != 0 or entry.get("nan_total_normalized") != 0:
            raise ValueError(f"Corrected R1 {split_name} manifest reports NaN")

        try:
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(frame[timestamp_column], errors="raise")
            )
            X = frame.loc[:, feature_order].to_numpy(dtype=np.float64, copy=True)
            y = frame.loc[:, target_column].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Corrected R1 {split_name} values: {exc}") from exc
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Corrected R1 {split_name} contains non-finite values")
        if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError(f"Corrected R1 {split_name} timestamps are not unique/ordered")
        if len(timestamps) > 1:
            expected_delta = pd.Timedelta(CORRECTED_R1_FREQUENCY)
            if not bool(((timestamps[1:] - timestamps[:-1]) == expected_delta).all()):
                raise ValueError(f"Corrected R1 {split_name} frequency mismatch")
        if previous_end is not None and timestamps[0] - previous_end != pd.Timedelta(
            CORRECTED_R1_FREQUENCY
        ):
            raise ValueError(f"Corrected R1 boundary before {split_name} is not contiguous")
        previous_end = timestamps[-1]
        if timestamps[0] != pd.Timestamp(entry.get("start_time")):
            raise ValueError(f"Corrected R1 {split_name} start timestamp mismatch")
        if timestamps[-1] != pd.Timestamp(entry.get("end_time")):
            raise ValueError(f"Corrected R1 {split_name} end timestamp mismatch")

        X_seq, y_seq, target_indices = build_aligned_sequences(
            X,
            y,
            window=window,
            horizon=horizon,
            return_target_indices=True,
        )
        if len(X_seq) != expected_sequences[split_name]:
            raise ValueError(
                f"Corrected R1 {split_name} sequences: expected "
                f"{expected_sequences[split_name]}, got {len(X_seq)}"
            )
        row_counts[split_name] = len(frame)
        sequence_counts[split_name] = len(X_seq)
        sequence_data[split_name] = {
            "X": X_seq,
            "y": y_seq,
            "target_indices": target_indices,
            "target_timestamps": timestamps.to_numpy()[target_indices],
        }

    first_X = frames["training"].loc[:, feature_order].to_numpy(dtype=np.float64)[:window]
    first_y = float(frames["training"].loc[:, target_column].iloc[window + horizon - 1])
    alignment_ok = bool(
        np.array_equal(sequence_data["training"]["X"][0], first_X)
        and sequence_data["training"]["y"][0] == first_y
    )
    if not alignment_ok:
        raise ValueError("Corrected R1 first sequence is not X[0:5] -> y[5]")

    validation_indices = sequence_data["validation"]["target_indices"]
    validation_duplicate_count = int(
        len(validation_indices) - len(np.unique(validation_indices))
    )
    validation_padding_count = int(
        len(validation_indices) - expected_sequences["validation"]
    )
    if validation_duplicate_count != 0 or validation_padding_count != 0:
        raise ValueError("Corrected R1 validation contains duplicate/padded sequences")

    return {
        "profile_dir": profile["profile_dir"],
        "plant": manifest_plant,
        "profile": role,
        "rows": row_counts,
        "sequences": sequence_counts,
        "feature_order": feature_order,
        "target": target_column,
        "alignment": "X[0:5] -> y[5]",
        "alignment_ok": alignment_ok,
        "validation_duplicate_count": validation_duplicate_count,
        "validation_padding_count": validation_padding_count,
        "sequence_data": sequence_data,
    }


def build_aligned_sequences(
    X,
    y,
    *,
    window=5,
    horizon=1,
    return_target_indices=False,
):
    """Build exact windows using ``X[i:i+window] -> y[i+window]`` at horizon 1.

    Unlike the Legacy training generator, this function performs no shuffling,
    batching, duplication, or padding.  It returns ``(X_seq, y_seq)`` by
    default; preflight callers may request the aligned target row indices.
    """

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X_array.shape}")
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        y_array = y_array.reshape(-1)
    if y_array.ndim != 1:
        raise ValueError(f"y must be 1-D or (rows, 1), got {y_array.shape}")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if len(X_array) != len(y_array):
        raise ValueError("X and y row counts differ")

    sequence_count = len(X_array) - window - horizon + 1
    if sequence_count <= 0:
        raise ValueError("Not enough rows to build one aligned sequence")
    sample_indices = np.arange(sequence_count, dtype=np.int64)
    target_indices = sample_indices + window + horizon - 1
    X_seq = np.stack(
        [X_array[index : index + window] for index in sample_indices],
        axis=0,
    )
    y_seq = y_array[target_indices].copy()
    if return_target_indices:
        return X_seq, y_seq, target_indices
    return X_seq, y_seq


def generator(X: np.array, y: np.array, time_steps: int):
    """get time-series batch dataset

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        time_steps (int): length of time series to consider during learning

    Returns:
        X_time (np.array): preprocessed data for explanatory variables
        y_time (np.array): preprocessed data for target variable

    """
    n_batches = X.shape[0] - time_steps - 1
    
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y_time = np.zeros((n_batches, 1))
    for i in range(n_batches):
        X_time[i] = X[i:(i + time_steps), :]
        y_time[i] = y[i + time_steps]
    return X_time, y_time


def split_dataset(X: np.array, y: np.array, ratio=0.8):
    """split dataset to train data and valid data in deep learning

    Args:
        X (np.array): data for explanatory variables
        y (np.array): data for target variable
        ratio (float, optional): ratio of train data and valid data. Defaults to 0.8.

    Returns:
        tuple: tuple which contains X_train, y_train, X_valid and y_valid in order
    """
    '''split dataset to train data and valid data'''
    X_train = X[:int(X.shape[0] * ratio)]
    y_train = y[:int(y.shape[0] * ratio)]
    X_valid = X[int(X.shape[0] * ratio):]
    y_valid = y[int(y.shape[0] * ratio):]
    dataset = tuple([X_train, y_train, X_valid, y_valid])

    return dataset


class ReccurentTrainingGenerator(Sequence): # 在訓練LSTM模型時生成批次的時序數據。
    """ Reccurent レイヤーを訓練するためのデータgeneratorクラス (訓練Recurrent層的數據生成器類別) """
    def _resetindices(self): # 隨機打亂數據的索引，以生成不同的批次數據。
        """バッチとして出力するデータのインデックスを乱数で生成する (作為批次輸出的數據索引用隨機數生成) """
        self.num_called = 0  # 同一のエポック内で __getitem__　メソッドが呼び出された回数 (在同一個epoch中調用 __getitem__ 方法的次數) # 用來計數在當前 epoch 中已經調用的批次數。
        
        all_idx = np.random.permutation(np.arange(self.num_samples)) #  生成隨機排列的索引
        remain_idx = np.random.choice(np.arange(self.num_samples),
                                      size=(self.steps_per_epoch * self.batch_size - len(all_idx)),
                                      replace=False)  # 足らない分を重複indexで補う (用重複的索引補足不足的部分) # 並填充不足部分的索引，使得每個epoch內的批次數與steps_per_epoch相符。
        self.indices = np.hstack([all_idx, remain_idx]).reshape(self.steps_per_epoch, self.batch_size) # 最終生成的索引數組，將所有批次的索引組合在一起。
        
    def __init__(self, x_set, y_set, batch_size, timesteps, delay):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列
        y_set     : 目的変数 (データ点数×1)のNumPy配列
        batch_size: バッチサイズ
        timesteps : どの程度過去からデータをReccurent層に与えるか (決定要從多遠的過去數據提供給 Recurrent 層)
        delay     : 目的変数をどの程度遅らせるか (決定目標變數要延遲多長時間)
        """
        self.x = np.array(x_set) # 特徵數據
        self.y = np.array(y_set) # 標籤數據
        self.batch_size = batch_size # 批次大小
        self.steps = timesteps # 時間步數，即RNN模型輸入過去多少步的數據。
        self.delay = delay # 延遲步數，用於決定輸出的目標值相對於輸入的偏移量。
        
        self.num_samples = len(self.x) - timesteps - delay + 1 # 樣本數
        self.steps_per_epoch = int(np.ceil(self.num_samples / float(batch_size)))
        
        self._resetindices()
        
    def __len__(self): # 返回每個epoch中的批次數（步數），即steps_per_epoch。
        """ 1エポックあたりのステップ数を返す (返回每個epoch的步數) """ 
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する (將數據整理成批次輸出) """
        indices_temp = self.indices[idx] # 當前批次的索引。
        
        batch_x = np.array([self.x[i:i+self.steps] for i in indices_temp]) # 根據 timesteps 生成的特徵數據，形狀為 (batch_size, timesteps, 特徵數)。
        batch_y = self.y[indices_temp + self.steps + self.delay - 1] # 延遲後的標籤數據，對應於batch_x最後一個時間步的預測目標。
        
        if self.num_called==(self.steps_per_epoch-1): # 在返回批次數據後，若已遍歷所有批次，則調用 _resetindices 隨機打亂索引，以便在下一 epoch 中生成不同的批次。
            self._resetindices() # 1エポック内の全てのバッチを返すと、データをシャッフルする (返回一個 epoch 中的所有批次後，將數據打亂順序)
        else:
            self.num_called += 1
        
        return batch_x, batch_y
    
    
class ReccurentPredictingGenerator(Sequence): # 生成遞歸神經網路（如 LSTM、GRU）模型的預測數據。
    """ 
    Reccurent レイヤーで予測するためのデータgeneratorクラス (用於Recurrent層預測的數據生成器類別) 
    每次會送出一組【過去連續數據片段】，讓模型根據這段歷史來預測未來。
    """ 
    def __init__(self, x_set, batch_size, timesteps):
        """
        x_set     : 説明変数 (データ点数×特徴量数)のNumPy配列 (特徵數據的NumPy陣列，形狀為 (樣本數, 特徵數))
        batch_size: バッチサイズ (批次大小)
        timesteps : どの程度過去からデータをReccurent層に与えるか (時間步數，即每次預測所需的過去時間步數，每個樣本要抓「過去幾個時間點的資料」來預測（例如過去 5 筆）。)
        """
        self.x = np.array(x_set) # 將輸入的數據轉換為 NumPy 陣列
        self.batch_size = batch_size # 批次大小
        self.steps = timesteps # 時間步數
        
        self.num_samples = len(self.x)-timesteps+1 # 計算可用的樣本數。每個樣本需要有timesteps的歷史數據。
        self.steps_per_epoch = int(np.floor(self.num_samples / float(batch_size))) # 根據總樣本數與 batch_size，算出每一個 epoch（完整一輪）能分幾批送出資料。
        
        self.idx_list = [] # 用於記錄批次索引
        
    def __len__(self):
        """ 1エポックあたりのステップ数を返す (返回每個epoch的步數，由樣本數和批次大小決定。) """
        return self.steps_per_epoch # 告訴模型一個epoch有幾步。
        
    def __getitem__(self, idx):
        """ データをバッチにまとめて出力する (將數據整理成批次輸出) """
        start_idx = idx*self.batch_size # 計算當前批次的起始索引 start_idx。
        batch_x = [self.x[start_idx+i : start_idx+i+self.steps] for i in range(self.batch_size)] # 根據timesteps生成當前批次的數據，每個樣本包含self.steps個時間步。
        self.idx_list.append(start_idx) # 記錄當前批次的起始索引
        return np.array(batch_x) # 返回批次數據，形狀為(batch_size, timesteps, 特徵數)。


def decompose_time_series(x):
    '''
    時間序列分解 (Time Series Decomposition), 把原始序列 x 拆解成 趨勢 (Trend)、季節性 (Seasonal / Period)、殘差 (Residual)。
    並且自動找到最適合的週期(period)
    '''
    from statsmodels import api as sm

    step = len(x) // 10 # 設定最大週期範圍，最多要測試的 period 是總長度的十分之一，避免 period 設太大 → 不穩定。
    best_score = np.inf # 用正無限大初始化，用來存放目前找到的最佳 score (愈小愈好)
    print('decomposing time series data ・・・・・')
    for period in tqdm(range(1, step + 1)):
        # 使用 seasonal_decompose 分解時間序列
        decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=period, model='additive', extrapolate_trend='freq')
        # print(len(np.where(decompose_result.resid < 0)[0]))
        score = np.sum(np.abs(decompose_result.resid)) # 用殘差的絕對值總和作為「分解好不好」的指標
                                                       # 越小越好 → 表示殘差越小，趨勢和季節性越能解釋原始數據。

        if score < best_score: # 更新最佳 period
            best_period = period
            best_score = score
    print(f'best period : {best_period}')
    #  最後再用最佳 period 分解一次，確保輸出的分解結果是最佳 period。
    decompose_result = sm.tsa.seasonal_decompose(pd.Series(x), period=best_period, model='additive', extrapolate_trend='freq')

    x = {'trend': decompose_result.trend, 'period': decompose_result.seasonal, 'resid': decompose_result.resid}
    return x, best_period
