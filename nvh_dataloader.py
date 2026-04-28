"""
NVH DataLoader — 支持 .mat / .csv / .tdms 三种工程数据格式,
自动读取并转化为 torch.Tensor, 可直接挂载到 NVHSignalProcessingExpert。

依赖安装:
    pip install scipy pandas numpy nptdms torch

典型用法:
    from nvh_dataloader import NVHDataLoader

    loader = NVHDataLoader(sample_rate=25600.0)
    signals, meta = loader.load("test_data.mat", channel="vibration_ch1")
    # signals: torch.Tensor [batch, seq_len]
    # meta:    dict  (采样率/通道/长度等元信息)

    # 批量加载整个目录
    dataset = loader.load_directory("./data/", pattern="*.tdms")
"""
import os
import glob
import logging
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("nvh_dataloader")


class NVHSignalDataset(Dataset):
    """
    PyTorch Dataset 包装器, 供 DataLoader 直接迭代。
    每条样本为 (signal_tensor, metadata_dict)。
    """

    def __init__(self, signals: List[torch.Tensor],
                 metas: List[Dict]):
        self.signals = signals
        self.metas = metas

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        return self.signals[idx], self.metas[idx]


class NVHDataLoader:
    """
    统一的 NVH 工程数据加载器。

    支持格式:
        .mat  — MATLAB 数据文件 (通过 scipy.io)
        .csv  — 逗号分隔文本 (通过 numpy / pandas)
        .tdms — National Instruments TDMS (通过 nptdms)
    """

    def __init__(self, sample_rate: float = 25600.0,
                 device: str = "cpu"):
        self.fs = sample_rate
        self.device = device

    # ======================= 公共接口 =======================

    def load(self, filepath: str,
             channel: Optional[str] = None,
             column: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        加载单个文件, 返回 (signal_tensor, metadata)。
        signal_tensor shape: [batch, seq_len] (batch 通常为 1 或通道数)
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".mat":
            return self._load_mat(filepath, channel)
        elif ext == ".csv":
            return self._load_csv(filepath, column)
        elif ext == ".tdms":
            return self._load_tdms(filepath, channel)
        else:
            raise ValueError(f"Unsupported format: {ext}  (supported: .mat, .csv, .tdms)")

    def load_directory(self, directory: str,
                       pattern: str = "*.*",
                       channel: Optional[str] = None) -> NVHSignalDataset:
        """批量加载目录下所有匹配文件, 返回 NVHSignalDataset。"""
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {directory}"
            )
        all_signals: List[torch.Tensor] = []
        all_metas: List[Dict] = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".mat", ".csv", ".tdms"):
                continue
            try:
                sig, meta = self.load(f, channel=channel)
                all_signals.append(sig)
                all_metas.append(meta)
                logger.info("Loaded %s  shape=%s", f, sig.shape)
            except Exception:
                logger.exception("Failed to load %s, skipping", f)
        return NVHSignalDataset(all_signals, all_metas)

    # ======================= 格式解析器 =======================

    def _load_mat(self, filepath: str,
                  channel: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        加载 .mat 文件。
        如果指定 channel, 读取该变量; 否则自动选取第一个非系统变量。
        """
        from scipy.io import loadmat
        mat = loadmat(filepath)

        # 过滤掉 MATLAB 系统元数据
        data_keys = [k for k in mat if not k.startswith("__")]
        if not data_keys:
            raise ValueError(f"No data variables found in {filepath}")

        key = channel if channel else data_keys[0]
        if key not in mat:
            raise KeyError(
                f"Channel '{key}' not found. Available: {data_keys}"
            )

        arr = np.asarray(mat[key], dtype=np.float32).flatten()
        # 如果是多列 (多通道), shape 变为 [n_channels, seq_len]
        raw = mat[key]
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            arr = np.asarray(raw, dtype=np.float32)
        else:
            arr = np.asarray(raw, dtype=np.float32).flatten()

        tensor = torch.from_numpy(arr).to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # [1, seq_len]

        meta = {
            "source": filepath,
            "format": "mat",
            "channel": key,
            "sample_rate": self.fs,
            "shape": list(tensor.shape),
        }
        return tensor, meta

    def _load_csv(self, filepath: str,
                  column: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        加载 .csv 文件。
        - 单列: 直接读为 1D 信号
        - 多列: column 指定列索引, 不指定则全部加载为多通道
        假设首行为表头, 自动跳过。
        """
        # 尝试用 numpy 读取 (快速路径)
        try:
            arr = np.genfromtxt(filepath, delimiter=",", skip_header=1,
                                dtype=np.float32)
        except ValueError:
            # 回退到 pandas 处理非数值列
            import pandas as pd
            df = pd.read_csv(filepath)
            # 只保留数值列
            df = df.select_dtypes(include=[np.number])
            arr = df.to_numpy(dtype=np.float32)

        if arr.ndim == 1:
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        elif column is not None:
            tensor = torch.from_numpy(
                arr[:, column].astype(np.float32)
            ).unsqueeze(0).to(self.device)
        else:
            # 多通道: [n_channels, seq_len] -> 转置使 dim0=通道
            if arr.shape[1] < arr.shape[0]:
                tensor = torch.from_numpy(arr.T.astype(np.float32)).to(self.device)
            else:
                tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(self.device)

        meta = {
            "source": filepath,
            "format": "csv",
            "column": column,
            "sample_rate": self.fs,
            "shape": list(tensor.shape),
        }
        return tensor, meta

    def _load_tdms(self, filepath: str,
                   channel: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        加载 .tdms 文件 (NI 测试数据标准格式)。
        如果指定 channel 名称则只读取该通道, 否则全部通道拼接为多通道 Tensor。
        """
        from nptdms import TdmsFile

        tdms = TdmsFile.read(filepath)
        groups = tdms.groups()

        if channel:
            # 查找指定通道
            for grp in groups:
                for ch in grp.channels():
                    if ch.name == channel:
                        arr = np.asarray(ch.data, dtype=np.float32)
                        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
                        meta = {
                            "source": filepath,
                            "format": "tdms",
                            "channel": channel,
                            "sample_rate": self._tdms_sample_rate(ch),
                            "shape": list(tensor.shape),
                        }
                        return tensor, meta
            raise KeyError(
                f"Channel '{channel}' not found. "
                f"Available: {[ch.name for g in groups for ch in g.channels()]}"
            )

        # 未指定通道: 读取所有通道
        channel_data = []
        channel_names = []
        for grp in groups:
            for ch in grp.channels():
                channel_data.append(np.asarray(ch.data, dtype=np.float32))
                channel_names.append(ch.name)

        if not channel_data:
            raise ValueError(f"No channels found in {filepath}")

        # 统一长度 (取最短通道截断)
        min_len = min(len(d) for d in channel_data)
        stacked = np.stack([d[:min_len] for d in channel_data])
        tensor = torch.from_numpy(stacked).to(self.device)

        meta = {
            "source": filepath,
            "format": "tdms",
            "channels": channel_names,
            "sample_rate": self._tdms_sample_rate(groups[0].channels()[0]),
            "shape": list(tensor.shape),
        }
        return tensor, meta

    @staticmethod
    def _tdms_sample_rate(channel) -> float:
        """尝试从 TDMS 通道属性中读取采样率, 默认 25600 Hz。"""
        try:
            props = channel.properties
            if "SampleRate" in props:
                return float(props["SampleRate"])
            if "wf_samples" in props and "wf_increment" in props:
                return 1.0 / float(props["wf_increment"])
        except Exception:
            pass
        return 25600.0


# ======================= 直接运行测试 =======================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = NVHDataLoader(sample_rate=25600.0)

    # 生成一个测试 .csv 文件
    import os
    test_csv = "test_signal.csv"
    if not os.path.exists(test_csv):
        t = np.linspace(0, 1, 25600)
        sig = 0.5 * np.sin(2 * np.pi * 250 * t) + \
              0.2 * np.sin(2 * np.pi * 2000 * t)
        np.savetxt(test_csv, sig, delimiter=",", header="vibration", comments="")

    tensor, meta = loader.load(test_csv)
    print(f"Loaded: shape={tensor.shape}, meta={meta}")

    # 用 Expert 验证
    from expert import NVHSignalProcessingExpert
    expert = NVHSignalProcessingExpert(sample_rate=25600.0, device="cpu")
    result = expert.execute_routing_instruction(tensor[0], "FIND_RESONANCE")
    print(f"FIND_RESONANCE => {result}")
