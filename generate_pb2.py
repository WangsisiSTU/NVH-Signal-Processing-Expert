"""
一键生成 gRPC Python 代码。
用法: python generate_pb2.py
"""
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "grpc_tools.protoc",
    "-I", ".",
    "--python_out", ".",
    "--grpc_python_out", ".",
    "nvh_service.proto",
])
print("Generated: nvh_service_pb2.py, nvh_service_pb2_grpc.py")
