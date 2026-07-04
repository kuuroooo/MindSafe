import yaml

from .hf_client import HFClient, HFModelConfig
from .vllm_http_client import VLLMHTTPClient, VLLMServer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_mas_model(config: dict) -> HFClient:
    hf_cfg = HFModelConfig(
        model_id=config["model_id"],
        device=config.get("device"),
        device_map=config.get("device_map"),
        torch_dtype=config.get("torch_dtype", "bfloat16"),
        load_in_8bit=config.get("load_in_8bit", False),
        load_in_4bit=config.get("load_in_4bit", False),
        use_flash_attention=config.get("use_flash_attention", True),
        trust_remote_code=config.get("trust_remote_code", False),
        max_new_tokens=config.get("max_new_tokens", 1024),
    )
    return HFClient(hf_cfg)


def start_judge_server(config: dict) -> VLLMServer:
    server_cfg = config["server"]
    server = VLLMServer(
        model_id=config["model_id"],
        gpu_ids=server_cfg["gpu_ids"],
        port=server_cfg.get("port", 8000),
        max_model_len=server_cfg.get("max_model_len", 8192),
        dtype=server_cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=server_cfg.get("gpu_memory_utilization", 0.9),
        quantization=server_cfg.get("quantization"),
        startup_timeout=server_cfg.get("startup_timeout", 900),
        log_path=server_cfg.get("log_path", "logs/vllm_server.log"),
        extra_args=server_cfg.get("extra_args", []),
    )
    server.start()
    return server


def judge_client_from_config(config: dict) -> VLLMHTTPClient:
    server_cfg = config["server"]
    return VLLMHTTPClient(
        model_id=config["model_id"],
        host=server_cfg.get("host", "localhost"),
        port=server_cfg.get("port", 8000),
        timeout=server_cfg.get("request_timeout", 300.0),
    )
