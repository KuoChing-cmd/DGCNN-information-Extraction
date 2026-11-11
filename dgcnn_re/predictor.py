"""Inference and SPO triplet extraction pipeline."""

from typing import List, Dict, Tuple, Set, Optional, Any
import torch
import numpy as np

from dgcnn_re.model import DGCNNForRelationExtraction
from dgcnn_re.dataset import Vocab, Schema
from dgcnn_re.config import ModelConfig


class RelationExtractor:
    """High-level predictor for extracting Subject-Predicate-Object triplets from text."""

    def __init__(
        self,
        model: DGCNNForRelationExtraction,
        vocab: Vocab,
        schema: Schema,
        device: Optional[torch.device] = None,
        sub_start_thresh: float = 0.5,
        sub_end_thresh: float = 0.4,
        obj_start_thresh: float = 0.5,
        obj_end_thresh: float = 0.4,
        max_len: int = 512,
    ):
        self.model = model
        self.vocab = vocab
        self.schema = schema
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.sub_start_thresh = sub_start_thresh
        self.sub_end_thresh = sub_end_thresh
        self.obj_start_thresh = obj_start_thresh
        self.obj_end_thresh = obj_end_thresh
        self.max_len = max_len

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: str,
        schema_path: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "RelationExtractor":
        """Factory method to load predictor directly from saved artifacts."""
        vocab = Vocab.load(vocab_path)
        schema = Schema.load(schema_path)

        state = torch.load(checkpoint_path, map_location="cpu")
        raw_config = state["config"]
        if isinstance(raw_config, dict):
            model_config = ModelConfig(**raw_config)
        else:
            model_config = raw_config

        # Ensure vocab and class counts match
        model_config.vocab_size = len(vocab)
        model_config.num_classes = len(schema)

        model = DGCNNForRelationExtraction(model_config)
        model.load_state_dict(state["model_state_dict"])

        return cls(model=model, vocab=vocab, schema=schema, device=device, **kwargs)

    @torch.no_grad()
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extracts unique (subject, predicate, object) triplets from text."""
        detailed = self.extract(text)
        return detailed["triplets"]

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, Any]:
        """Extracts triplets with detailed character offset and entity spans."""
        if not text:
            return {"text": text, "triplets": [], "details": []}

        truncated_text = text[: self.max_len]
        char_ids = self.vocab.encode(truncated_text)
        seq_len = len(char_ids)
        if seq_len == 0:
            return {"text": text, "triplets": [], "details": []}

        char_ids_t = (
            torch.tensor([char_ids], dtype=torch.long).to(self.device)
        )  # (1, L)
        mask_t = torch.ones((1, seq_len), dtype=torch.float32).to(
            self.device
        )  # (1, L)

        # 1. Encode text and decode candidate subjects
        t, mask = self.model.encoder(char_ids_t, mask=mask_t)
        ps1, ps2, pn1, pn2 = self.model.subject_decoder(t, mask)

        ps1_np = ps1[0, :, 0].cpu().numpy()
        ps2_np = ps2[0, :, 0].cpu().numpy()

        sub_starts = np.where(ps1_np > self.sub_start_thresh)[0]
        sub_ends = np.where(ps2_np > self.sub_end_thresh)[0]

        candidates = []
        for s in sub_starts:
            e_candidates = sub_ends[sub_ends >= s]
            if len(e_candidates) > 0:
                e = e_candidates[0]
                sub_text = truncated_text[s : e + 1]
                candidates.append((sub_text, int(s), int(e)))

        if not candidates:
            return {"text": text, "triplets": [], "details": []}

        # 2. Extract objects conditioned on detected subjects
        num_subs = len(candidates)
        t_rep = t.repeat(num_subs, 1, 1)
        mask_rep = mask.repeat(num_subs, 1)
        pn1_rep = pn1.repeat(num_subs, 1, 1)
        pn2_rep = pn2.repeat(num_subs, 1, 1)

        k1_t = torch.tensor([[c[1]] for c in candidates], dtype=torch.long, device=self.device)
        k2_t = torch.tensor([[c[2]] for c in candidates], dtype=torch.long, device=self.device)

        po1, po2, _ = self.model.object_decoder(
            t_rep, k1_t, k2_t, mask_rep, pn1_rep, pn2_rep
        )
        po1_np = po1.cpu().numpy()  # (num_subs, L, num_classes)
        po2_np = po2.cpu().numpy()  # (num_subs, L, num_classes)

        triplets_set: Set[Tuple[str, str, str]] = set()
        details: List[Dict[str, Any]] = []

        for idx, (sub_text, s_start, s_end) in enumerate(candidates):
            o1_cand = np.where(po1_np[idx] > self.obj_start_thresh)
            o2_cand = np.where(po2_np[idx] > self.obj_end_thresh)

            # Match start and end positions by relation class
            for o_start, pred_id1 in zip(*o1_cand):
                for o_end, pred_id2 in zip(*o2_cand):
                    if o_start <= o_end and pred_id1 == pred_id2:
                        obj_text = truncated_text[o_start : o_end + 1]
                        predicate = self.schema.id2predicate.get(int(pred_id1), "Unknown")
                        triplet = (sub_text, predicate, obj_text)
                        if triplet not in triplets_set:
                            triplets_set.add(triplet)
                            details.append(
                                {
                                    "subject": sub_text,
                                    "predicate": predicate,
                                    "object": obj_text,
                                    "subject_span": (s_start, s_end),
                                    "object_span": (int(o_start), int(o_end)),
                                    "subject_prob": float(ps1_np[s_start] * ps2_np[s_end]),
                                    "object_prob": float(po1_np[idx, o_start, pred_id1] * po2_np[idx, o_end, pred_id2]),
                                }
                            )
                        break

        return {
            "text": text,
            "triplets": list(triplets_set),
            "details": details,
        }
