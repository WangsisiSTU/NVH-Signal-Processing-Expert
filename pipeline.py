"""
PiERN Pipeline — 信号分析 → 结构动力学 联合管线。

数据流:
    原始传感器数据
        → NVHSignalProcessingExpert (定位异常频率/频带)
        → StructDynExpertModule     (接收异常频率, 计算结构响应)
        → LLM 思维链 (综合两个专家的数值结果生成诊断文本)

本文件定义:
    1. NVHToDynPayload  — 模块间传递的标准数据包
    2. StructDynExpertModule (接口预留) — 结构动力学专家模块
    3. PiERNPipeline     — 编排两个模块的管线控制器
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from expert import NVHSignalProcessingExpert


# ============================================================
# 模块间标准数据包
# ============================================================

@dataclass
class NVHToDynPayload:
    """
    NVH 模块 → 动力学模块 的传递载荷。

    NVH 模块定位到异常频率后, 将以下信息打包传递给下游动力学模块,
    动力学模块据此选择对应的模态/频响函数进行计算。
    """
    # NVH 检测到的异常频率 (Hz)
    anomaly_frequencies_hz: List[float]

    # 对应幅值 (用于判断严重程度)
    amplitudes: List[float]

    # 频带声压级 (如果有)
    band_spl_db: Dict[str, List[float]] = field(default_factory=dict)

    # 原始信号元信息
    sample_rate: float = 25600.0
    signal_length: int = 0

    # NVH 侧的诊断标签 (供 LLM 参考)
    nvh_diagnosis_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_frequencies_hz": self.anomaly_frequencies_hz,
            "amplitudes": self.amplitudes,
            "band_spl_db": self.band_spl_db,
            "sample_rate": self.sample_rate,
            "signal_length": self.signal_length,
            "nvh_diagnosis_tags": self.nvh_diagnosis_tags,
        }


# ============================================================
# 结构动力学专家模块 (接口预留)
# ============================================================

class StructDynExpertModule(nn.Module):
    """
    结构动力学专家模块 — 接口预留。

    职责:
        接收 NVH 传递的异常频率, 结合有限元模型或实验模态数据,
        计算结构在对应频率激励下的动力学响应:
        - 模态叠加 (Modal Superposition)
        - 频响函数 FRF (Frequency Response Function)
        - 应力/应变场重构
        - 共振风险评估

    注意: 本文件仅定义接口和示例实现, 实际计算逻辑
          需要与 AIStructDynSolve / PhysicsNeMo 等求解器对接。
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        # 预设: 结构固有频率 (Hz) — 实际应从 FEM 模型 / 实验模态分析导入
        self.natural_frequencies_hz: List[float] = []
        # 预设: 对应模态阻尼比
        self.damping_ratios: List[float] = []
        # 预设: 模态质量 (kg)
        self.modal_masses: List[float] = []

    def load_modal_parameters(self,
                              natural_freqs: List[float],
                              damping_ratios: List[float],
                              modal_masses: Optional[List[float]] = None):
        """
        从外部导入模态参数 (可由 FEM 求解器或 EMA 实验提供)。
        """
        self.natural_frequencies_hz = natural_freqs
        self.damping_ratios = damping_ratios
        self.modal_masses = modal_masses or [1.0] * len(natural_freqs)

    def compute_frf(self,
                    freq_range_hz: List[float],
                    excitation_point: int = 0,
                    response_point: int = 0) -> torch.Tensor:
        """
        计算频响函数 (FRF)。

        H(ω) = Σ φ_i(x_exc) * φ_i(x_resp) /
                    (m_i * (ω_i² - ω² + j*2*ζ_i*ω_i*ω))

        参数:
            freq_range_hz: 关注的频率点列表
            excitation_point: 激励点编号
            response_point: 响应点编号

        返回:
            frf: Complex Tensor [len(freq_range_hz)]
        """
        if not self.natural_frequencies_hz:
            raise RuntimeError(
                "Modal parameters not loaded. "
                "Call load_modal_parameters() first."
            )

        omega = torch.tensor(freq_range_hz, dtype=torch.float32) * 2 * torch.pi
        omega = omega.to(self.device)

        frf = torch.zeros(len(freq_range_hz), dtype=torch.complex64,
                          device=self.device)

        for i, (fn, zeta, mi) in enumerate(zip(
            self.natural_frequencies_hz,
            self.damping_ratios,
            self.modal_masses,
        )):
            wn = 2 * torch.pi * fn
            # 简化: 假设模态振型系数为 1.0 (实际应查表)
            phi_exc = 1.0
            phi_resp = 1.0
            denominator = mi * (wn**2 - omega**2 + 2j * zeta * wn * omega)
            frf = frf + (phi_exc * phi_resp) / denominator

        return frf

    def evaluate_resonance_risk(self,
                                payload: NVHToDynPayload) -> Dict[str, Any]:
        """
        核心接口: 接收 NVH 传递的异常频率, 评估结构共振风险。

        返回:
            risk_report: dict, 包含:
                - 各异常频率与最近固有频率的接近度
                - 放大因子 (动态放大倍数)
                - 风险等级 (LOW / MEDIUM / HIGH / CRITICAL)
        """
        if not self.natural_frequencies_hz:
            return {
                "status": "NO_MODAL_DATA",
                "message": "Modal parameters not loaded. "
                           "Cannot evaluate resonance risk.",
            }

        risk_report: Dict[str, Any] = {
            "anomalies": [],
            "overall_risk": "LOW",
        }
        max_risk_level = 0  # 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL

        for freq, amp in zip(payload.anomaly_frequencies_hz,
                             payload.amplitudes):
            # 找最近的固有频率
            dists = [abs(freq - fn) for fn in self.natural_frequencies_hz]
            min_dist = min(dists)
            closest_fn = self.natural_frequencies_hz[dists.index(min_dist)]
            ratio = freq / closest_fn if closest_fn != 0 else float("inf")

            # 频率比在 0.9~1.1 范围内视为潜在共振
            if 0.9 <= ratio <= 1.1:
                # 动态放大因子近似: 1 / (2*zeta)
                closest_idx = dists.index(min_dist)
                zeta = self.damping_ratios[closest_idx] if closest_idx < len(self.damping_ratios) else 0.02
                daf = 1.0 / (2 * zeta) if zeta > 0 else float("inf")

                if ratio >= 0.98 and ratio <= 1.02:
                    risk_level = "CRITICAL"
                    risk_val = 3
                elif ratio >= 0.95 and ratio <= 1.05:
                    risk_level = "HIGH"
                    risk_val = 2
                else:
                    risk_level = "MEDIUM"
                    risk_val = 1
            else:
                risk_level = "LOW"
                risk_val = 0
                daf = 1.0

            max_risk_level = max(max_risk_level, risk_val)

            risk_report["anomalies"].append({
                "excitation_freq_hz": freq,
                "closest_natural_freq_hz": closest_fn,
                "freq_ratio": round(ratio, 4),
                "dynamic_amplification_factor": round(daf, 2),
                "risk_level": risk_level,
                "amplitude": amp,
            })

        risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        risk_report["overall_risk"] = risk_labels[max_risk_level]

        return risk_report

    def execute_routing_instruction(self,
                                    payload: NVHToDynPayload,
                                    instruction: str) -> Dict:
        """
        PiERN 路由接口 — 与 NVH 模块的 execute_routing_instruction 对齐。
        LLM 路由器根据上下文调用不同计算指令。

        支持指令:
            EVALUATE_RESONANCE  — 评估共振风险
            COMPUTE_FRF         — 计算频响函数
        """
        if instruction == "EVALUATE_RESONANCE":
            return self.evaluate_resonance_risk(payload)

        elif instruction == "COMPUTE_FRF":
            frf = self.compute_frf(payload.anomaly_frequencies_hz)
            return {
                "frf_frequencies_hz": payload.anomaly_frequencies_hz,
                "frf_magnitude": torch.abs(frf).cpu().tolist(),
                "frf_phase_rad": torch.angle(frf).cpu().tolist(),
            }

        else:
            raise ValueError(
                f"Unknown StructDyn instruction: {instruction}"
            )


# ============================================================
# PiERN 联合管线
# ============================================================

class PiERNPipeline:
    """
    PiERN 多专家联合管线控制器。

    编排流程:
        1. 接收原始传感器信号 + LLM 指令
        2. 调用 NVH 模块完成信号分析
        3. 将 NVH 结果打包为 NVHToDynPayload
        4. (可选) 自动路由到结构动力学模块
        5. 汇总所有数值结果, 返回给 LLM 思维链
    """

    def __init__(self,
                 nvh_sample_rate: float = 25600.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.nvh_expert = NVHSignalProcessingExpert(
            sample_rate=nvh_sample_rate, device=device,
        )
        self.dyn_expert = StructDynExpertModule(device=device)

    def run(self,
            signal: torch.Tensor,
            nvh_instruction: str = "FIND_RESONANCE",
            dyn_instruction: Optional[str] = "EVALUATE_RESONANCE",
            ) -> Dict[str, Any]:
        """
        执行完整的 NVH → 动力学分析管线。

        参数:
            signal: 原始传感器信号 [seq_len] 或 [batch, seq_len]
            nvh_instruction: NVH 模块指令
            dyn_instruction: 动力学模块指令, None 则跳过动力学分析

        返回:
            联合分析结果字典 (可直接序列化为 JSON 注入 LLM 思维链)
        """
        # ---- Stage 1: NVH 信号分析 ----
        nvh_result = self.nvh_expert.execute_routing_instruction(
            signal, nvh_instruction,
        )

        pipeline_result = {
            "pipeline": "NVH_to_StructDyn",
            "nvh_analysis": nvh_result,
        }

        # ---- Stage 2: 构建传递载荷 ----
        if nvh_instruction == "FIND_RESONANCE" and dyn_instruction:
            payload = NVHToDynPayload(
                anomaly_frequencies_hz=nvh_result.get(
                    "primary_frequencies_hz", []
                )[0] if nvh_result.get("primary_frequencies_hz") else [],
                amplitudes=nvh_result.get("amplitudes", [])[0]
                if nvh_result.get("amplitudes") else [],
                sample_rate=self.nvh_expert.fs,
                signal_length=signal.shape[-1],
            )

            # ---- Stage 3: 结构动力学分析 ----
            dyn_result = self.dyn_expert.execute_routing_instruction(
                payload, dyn_instruction,
            )
            pipeline_result["struct_dyn_analysis"] = dyn_result
            pipeline_result["handoff_payload"] = payload.to_dict()

        return pipeline_result


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    pipeline = PiERNPipeline(nvh_sample_rate=25600.0, device="cpu")

    # 模拟 NVH 信号: 250 Hz 轰鸣 + 2000 Hz 风噪
    t = torch.linspace(0, 1, 25600)
    signal = (0.5 * torch.sin(2 * torch.pi * 250 * t)
              + 0.2 * torch.sin(2 * torch.pi * 2000 * t))

    # 加载示例模态参数 (假设结构前三阶固有频率)
    pipeline.dyn_expert.load_modal_parameters(
        natural_freqs=[120.0, 248.0, 510.0],  # 第二阶 248Hz 非常接近 250Hz!
        damping_ratios=[0.02, 0.015, 0.01],
        modal_masses=[50.0, 35.0, 20.0],
    )

    # 运行联合管线
    import json
    result = pipeline.run(signal, nvh_instruction="FIND_RESONANCE",
                          dyn_instruction="EVALUATE_RESONANCE")

    print(json.dumps(result, indent=2, ensure_ascii=False))
