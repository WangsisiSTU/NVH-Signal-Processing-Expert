"""
NVH gRPC Server — 将 NVHSignalProcessingExpert 封装为异步 gRPC 服务。

启动方式:
    python nvh_grpc_server.py --port 50051

依赖安装:
    pip install grpcio grpcio-tools
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. nvh_service.proto
"""
import argparse
import asyncio
import json
import logging
from concurrent import futures

import grpc
import torch

import nvh_service_pb2 as pb2
import nvh_service_pb2_grpc as pb2_grpc
from expert import NVHSignalProcessingExpert

logger = logging.getLogger("nvh_grpc_server")


class NVHExpertServicer(pb2_grpc.NVHExpertServiceServicer):
    """gRPC 服务端实现, 内部持有单例 Expert 实例以复用 GPU 显存。"""

    def __init__(self, default_sample_rate: float = 25600.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.default_fs = default_sample_rate
        self.device = device
        # 预实例化 Expert (权重/常驻 GPU)
        self._experts: dict[float, NVHSignalProcessingExpert] = {}

    def _get_expert(self, sample_rate: float) -> NVHSignalProcessingExpert:
        if sample_rate not in self._experts:
            self._experts[sample_rate] = NVHSignalProcessingExpert(
                sample_rate=sample_rate, device=self.device,
            )
        return self._experts[sample_rate]

    # ---- 同步 RPC ----

    def Execute(self, request: pb2.NVHRequest,
                context: grpc.ServicerContext) -> pb2.NVHResponse:
        try:
            signal = torch.tensor(request.signal_data, dtype=torch.float32)
            fs = request.sample_rate or self.default_fs
            expert = self._get_expert(fs)
            top_k = request.top_k or 3

            # 对于 FIND_RESONANCE 传递 top_k
            if request.instruction == "FIND_RESONANCE":
                result = expert.compute_fft_and_peaks(signal, top_k=top_k)
                payload = {
                    "primary_frequencies_hz": result[0].cpu().tolist(),
                    "amplitudes": result[1].cpu().tolist(),
                }
            else:
                result = expert.execute_routing_instruction(
                    signal, request.instruction,
                )
                payload = result

            return pb2.NVHResponse(
                instruction=request.instruction,
                result_json=json.dumps(payload, ensure_ascii=False),
                status_code=0,
            )
        except Exception as exc:
            logger.exception("Execute failed")
            return pb2.NVHResponse(
                instruction=request.instruction,
                status_code=1,
                error_message=str(exc),
            )

    # ---- 双向流式 RPC ----

    def ExecuteStream(self, request_iter,
                      context: grpc.ServicerContext):
        for request in request_iter:
            yield self.Execute(request, context)


def serve(port: int, max_workers: int = 8):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
    )
    pb2_grpc.add_NVHExpertServiceServicer_to_server(
        NVHExpertServicer(), server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("NVH Expert gRPC server listening on port %d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    serve(args.port, args.workers)
