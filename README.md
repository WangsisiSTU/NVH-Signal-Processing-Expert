# NVH-Signal-Processing-Expert

一个**声振信号处理专家模块**，作为独立物理计算节点，处理 LLM 路由过来的高频一维时序信号，提取声学和振动特征，并与结构动力学模块联动完成共振风险评估。

## 项目结构

```
NVH/
├── expert.py              # 核心模块：NVH 信号处理专家
├── nvh_service.proto      # gRPC 服务定义（同步 + 双向流）
├── nvh_grpc_server.py     # gRPC 服务端
├── nvh_grpc_client.py     # gRPC 客户端（LLM Router 侧调用示例）
├── generate_pb2.py        # 一键生成 protobuf Python 代码
├── nvh_dataloader.py      # DataLoader：.mat / .csv / .tdms → torch.Tensor
├── pipeline.py            # 联合管线：NVH → 结构动力学
└── .gitignore
```

## 功能模块

### 1. NVH 信号处理专家 (`expert.py`)

基于 PyTorch 的信号分析核心，支持 GPU 加速：

| 指令 | 功能 | 输出 |
|------|------|------|
| `FIND_RESONANCE` | FFT 峰值检测，定位主导频率 | Top-K 频率 + 幅值 |
| `ANALYZE_NOISE_TYPE` | 频带声压级分析（轰鸣/啸叫/风噪） | 各频带 SPL (dB) |
| `TRANSIENT_SHOCK` | STFT 时频分析，定位瞬态冲击时刻 | 冲击峰值时间点 (s) |

预设 NVH 频带：

| 频带 | 范围 (Hz) | 典型声源 |
|------|-----------|----------|
| rumble | 20–50 | 低频轰鸣 |
| boom | 50–150 | 车厢共振轰鸣 |
| whine | 150–1000 | 齿轮/电机啸叫 |
| hiss | 1000–5000 | 高频风噪/气流声 |

### 2. gRPC 服务封装

将 `execute_routing_instruction` 封装为标准 gRPC 服务，供 LLM Router 异步调用：

- **同步 RPC** `Execute`：单次请求-响应，低延迟场景
- **双向流 RPC** `ExecuteStream`：批量下发多条信号分析指令

```bash
# 生成 protobuf 代码
pip install grpcio grpcio-tools
python generate_pb2.py

# 启动服务
python nvh_grpc_server.py --port 50051
```

### 3. DataLoader (`nvh_dataloader.py`)

统一读取工程数据格式，自动转化为 `torch.Tensor`：

| 格式 | 说明 | 依赖 |
|------|------|------|
| `.mat` | MATLAB 数据文件 | scipy |
| `.csv` | 逗号分隔文本 | numpy / pandas |
| `.tdms` | NI 测试数据格式 | nptdms |

```python
from nvh_dataloader import NVHDataLoader

loader = NVHDataLoader(sample_rate=25600.0)

# 单文件加载
signal, meta = loader.load("test_data.mat", channel="vibration_ch1")

# 批量加载目录
dataset = loader.load_directory("./data/", pattern="*.tdms")
```

### 4. 联合管线 (`pipeline.py`)

NVH 模块定位异常频率后，自动传递给结构动力学模块评估共振风险：

```
原始传感器信号
  → NVH 模块（定位异常频率）
    → NVHToDynPayload（标准化数据包）
      → 结构动力学模块（共振风险评估）
        → 综合诊断结果（注入 LLM 思维链）
```

动力学模块功能：

- **频响函数 (FRF) 计算**：模态叠加法
- **共振风险评估**：频率比 → 动态放大因子 → 风险等级（LOW / MEDIUM / HIGH / CRITICAL）

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.2+（推荐 CUDA 支持）
- 依赖：scipy, numpy, pandas, grpcio, grpcio-tools, nptdms

```bash
conda create -n nvh python=3.11
conda activate nvh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scipy numpy pandas grpcio grpcio-tools nptdms
```

### 运行测试

```bash
# 测试联合管线
python pipeline.py

# 测试 DataLoader（自动生成合成信号 CSV）
python nvh_dataloader.py
```

输出示例：

```json
{
  "pipeline": "NVH_to_StructDyn",
  "nvh_analysis": {
    "primary_frequencies_hz": [[250.0, 2000.0, 2001.0]],
    "amplitudes": [[0.250, 0.099, 0.008]]
  },
  "struct_dyn_analysis": {
    "anomalies": [
      {
        "excitation_freq_hz": 250.0,
        "closest_natural_freq_hz": 248.0,
        "freq_ratio": 1.0081,
        "dynamic_amplification_factor": 33.33,
        "risk_level": "CRITICAL"
      }
    ],
    "overall_risk": "CRITICAL"
  }
}
```

## 架构定位

1. LLM 在推理过程中遇到物理计算需求时，由 Router 将指令路由到本模块
2. 本模块完成数值计算后，结果以 JSON 字典返回
3. 结果无缝拼接回 LLM 的思维链，继续生成诊断文本
4. 异常频率可自动传递给下游结构动力学模块进行深层次分析

## License

MIT
