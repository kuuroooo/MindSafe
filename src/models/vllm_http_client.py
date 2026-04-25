"""vLLM OpenAI-compatible API server managed as a subprocess, plus HTTP client."""

import asyncio
import os
import subprocess
import time
from typing import Optional, List, Dict

import requests


class VLLMServer:
    """Launches `python -m vllm.entrypoints.openai.api_server` as a subprocess.

    CUDA_VISIBLE_DEVICES is set in the subprocess environment only, so the
    parent process (which hosts the MAS model on physical GPU 0) is unaffected.
    """

    def __init__(
        self,
        model_id: str,
        gpu_ids: List[int],
        port: int = 8000,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        quantization: Optional[str] = None,
        startup_timeout: int = 900,
        log_path: str = "logs/vllm_server.log",
        extra_args: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.gpu_ids = gpu_ids
        self.port = port
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self._log_file = None

    def _build_cmd(self) -> List[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id,
            "--tensor-parallel-size", str(len(self.gpu_ids)),
            "--port", str(self.port),
            "--host", "0.0.0.0",
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            # Newer vLLM defaults request logging OFF; the old
            # `--disable-log-requests` flag was removed. To re-enable, add
            # `--enable-log-requests` to extra_args.
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        cmd += list(self.extra_args)
        return cmd

    def start(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        cmd = self._build_cmd()
        print(f"[vLLM] Starting server on GPUs {self.gpu_ids}, port {self.port}")
        print(f"[vLLM] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        print(f"[vLLM] cmd: {' '.join(cmd)}")

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._log_file = open(self.log_path, "w")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()
        print(f"[vLLM] Server ready at http://localhost:{self.port}")

    def _wait_ready(self):
        health_url = f"http://localhost:{self.port}/health"
        start = time.time()
        while time.time() - start < self.startup_timeout:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (code={self.process.returncode}). "
                    f"See logs at {self.log_path}."
                )
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(10)
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {self.startup_timeout}s. "
            f"See logs at {self.log_path}."
        )

    def stop(self):
        if self.process and self.process.poll() is None:
            print("[vLLM] Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


class VLLMHTTPClient:
    """Client for a running vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_id: str,
        host: str = "localhost",
        port: int = 8000,
        timeout: float = 300.0,
    ):
        self.model_id = model_id
        self.base_url = f"http://{host}:{port}/v1"
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9 if temperature > 0 else 1.0,
        }

        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def generate_async(self, *args, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(*args, **kwargs))
