"""
NVH gRPC Client — LLM 路由器侧的调用示例。

典型用法:
    # 同步单次调用
    python nvh_grpc_client.py --instruction FIND_RESONANCE --file signal.csv

    # 流式批量调用
    python nvh_grpc_client.py --stream --files signal1.csv signal2.csv
"""
import argparse
import json
import sys

import grpc

import nvh_service_pb2 as pb2
import nvh_service_pb2_grpc as pb2_grpc


def run_single(stub: pb2_grpc.NVHExpertServiceStub,
               signal_data: list[float],
               instruction: str,
               sample_rate: float = 25600.0,
               top_k: int = 3):
    request = pb2.NVHRequest(
        signal_data=signal_data,
        sample_rate=sample_rate,
        instruction=instruction,
        top_k=top_k,
    )
    response = stub.Execute(request)
    if response.status_code != 0:
        print(f"[ERROR] {response.error_message}", file=sys.stderr)
        return None
    return json.loads(response.result_json)


def run_stream(stub: pb2_grpc.NVHExpertServiceStub,
               signals: list[tuple[list[float], str]],
               sample_rate: float = 25600.0):
    def request_iter():
        for sig_data, instruction in signals:
            yield pb2.NVHRequest(
                signal_data=sig_data,
                sample_rate=sample_rate,
                instruction=instruction,
            )

    for response in stub.ExecuteStream(request_iter()):
        if response.status_code != 0:
            print(f"[ERROR] {response.error_message}", file=sys.stderr)
            continue
        result = json.loads(response.result_json)
        print(f"[{response.instruction}] => {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--instruction", default="FIND_RESONANCE")
    parser.add_argument("--sample_rate", type=float, default=25600.0)
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = pb2_grpc.NVHExpertServiceStub(channel)

    # 简易测试: 用合成信号做一次调用
    import torch
    t = torch.linspace(0, 1, int(args.sample_rate))
    sig = 0.5 * torch.sin(2 * torch.pi * 250 * t) + \
          0.2 * torch.sin(2 * torch.pi * 2000 * t)

    result = run_single(
        stub, sig.tolist(),
        instruction=args.instruction,
        sample_rate=args.sample_rate,
    )
    print(json.dumps(result, indent=2))
