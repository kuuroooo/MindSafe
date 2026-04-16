from typing import List

import numpy as np


class ConsensusMetrics:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model, device=device)

    def compute_agreement(self, texts: List[str]) -> float:
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 2:
            return 1.0
        embs = self.model.encode(texts, normalize_embeddings=True)
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(float(np.dot(embs[i], embs[j])))
        return float(np.mean(sims)) if sims else 1.0

    @staticmethod
    def detect_unsafe_consensus(
        agreement: float,
        external_safety: float,
        agreement_threshold: float = 0.8,
        safety_threshold: float = 0.5,
    ) -> bool:
        return agreement >= agreement_threshold and external_safety < safety_threshold
