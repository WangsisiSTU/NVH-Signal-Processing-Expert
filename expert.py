"""
NVHSignalProcessingExpert — 声振信号处理专家模块
作为 PiERN 架构中的独立物理计算节点。
专门处理 LLM 路由过来的高频一维时序信号，提取声学和振动特征。
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class NVHSignalProcessingExpert(nn.Module):

    def __init__(self, sample_rate: float = 25600.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.fs = sample_rate
        self.device = torch.device(device)
        self.freq_bands = {
            "rumble": (20, 50),
            "boom": (50, 150),
            "whine": (150, 1000),
            "hiss": (1000, 5000),
        }

    # ---- 核心计算原语 ----

    def compute_fft_and_peaks(self, signal: torch.Tensor,
                              top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        N = signal.shape[-1]
        fft_complex = torch.fft.rfft(signal, norm="forward")
        amps = torch.abs(fft_complex)
        freqs = torch.fft.rfftfreq(N, 1 / self.fs).to(self.device)
        amps[:, 0] = 0.0
        top_amps, top_indices = torch.topk(amps, top_k, dim=-1)
        top_freqs = freqs[top_indices]
        return top_freqs, top_amps

    def compute_spl_in_bands(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        N = signal.shape[-1]
        fft_complex = torch.fft.rfft(signal, norm="forward")
        power_spectrum = torch.abs(fft_complex) ** 2
        freqs = torch.fft.rfftfreq(N, 1 / self.fs).to(self.device)
        band_energy: Dict[str, torch.Tensor] = {}
        for band_name, (f_min, f_max) in self.freq_bands.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            energy = torch.sum(power_spectrum[:, mask], dim=-1)
            ref_value = 2e-5
            spl_db = 10 * torch.log10(energy / (ref_value ** 2) + 1e-12)
            band_energy[band_name] = spl_db
        return band_energy

    def compute_stft_spectrogram(self, signal: torch.Tensor,
                                 n_fft: int = 1024,
                                 hop_length: int = 256) -> torch.Tensor:
        window = torch.hann_window(n_fft).to(self.device)
        stft_matrix = torch.stft(
            signal, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True,
        )
        return torch.abs(stft_matrix)

    # ---- PiERN 路由入口 ----

    def execute_routing_instruction(self, signal: torch.Tensor,
                                    instruction: str) -> Dict:
        signal = signal.to(self.device)
        result_payload: Dict = {}

        if instruction == "FIND_RESONANCE":
            freqs, amps = self.compute_fft_and_peaks(signal, top_k=3)
            result_payload["primary_frequencies_hz"] = freqs.cpu().tolist()
            result_payload["amplitudes"] = amps.cpu().tolist()

        elif instruction == "ANALYZE_NOISE_TYPE":
            band_dbs = self.compute_spl_in_bands(signal)
            result_payload["band_spl_db"] = {
                k: v.cpu().tolist() for k, v in band_dbs.items()
            }

        elif instruction == "TRANSIENT_SHOCK":
            spectrogram = self.compute_stft_spectrogram(signal)
            energy_over_time = torch.sum(spectrogram, dim=1)
            max_shock_idx = torch.argmax(energy_over_time, dim=-1)
            max_shock_time = (max_shock_idx * 256) / self.fs
            result_payload["max_shock_time_sec"] = max_shock_time.cpu().tolist()

        else:
            raise ValueError(f"Unknown routing instruction: {instruction}")

        return result_payload
