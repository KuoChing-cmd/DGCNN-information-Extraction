"""Dataset, vocabulary management, and batch collation for DGCNN-RE."""

import json
import os
import random
from typing import List, Dict, Tuple, Optional, Any, Set
import jieba
import numpy as np
import torch
from torch.utils.data import Dataset


class Vocab:
    """Manages character-level vocabulary mapping with special tokens."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(
        self,
        char2id: Optional[Dict[str, int]] = None,
        id2char: Optional[Dict[int, str]] = None,
    ):
        if char2id is not None and id2char is not None:
            self.char2id = char2id
            self.id2char = {int(k): v for k, v in id2char.items()}
        else:
            self.char2id = {self.PAD: 0, self.UNK: 1}
            self.id2char = {0: self.PAD, 1: self.UNK}

    def __len__(self) -> int:
        return len(self.char2id)

    def encode(self, text: str) -> List[int]:
        unk_id = self.char2id[self.UNK]
        return [self.char2id.get(c, unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id2char.get(i, self.UNK) for i in ids])

    @classmethod
    def build_from_texts(
        cls, texts: List[str], min_count: int = 2
    ) -> "Vocab":
        counts: Dict[str, int] = {}
        for text in texts:
            for char in text:
                counts[char] = counts.get(char, 0) + 1

        char2id = {cls.PAD: 0, cls.UNK: 1}
        id2char = {0: cls.PAD, 1: cls.UNK}

        idx = 2
        for char, count in counts.items():
            if count >= min_count and char not in char2id:
                char2id[char] = idx
                id2char[idx] = char
                idx += 1

        return cls(char2id=char2id, id2char=id2char)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"char2id": self.char2id, "id2char": self.id2char},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filepath: str) -> "Vocab":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 2:
            id2char, char2id = data
            return cls(char2id=char2id, id2char=id2char)
        return cls(char2id=data["char2id"], id2char=data["id2char"])


class Schema:
    """Manages relation schema definitions and integer ID mappings."""

    def __init__(
        self,
        predicate2id: Optional[Dict[str, int]] = None,
        id2predicate: Optional[Dict[int, str]] = None,
    ):
        self.predicate2id = predicate2id or {}
        self.id2predicate = {int(k): v for k, v in (id2predicate or {}).items()}

    def __len__(self) -> int:
        return len(self.predicate2id)

    @classmethod
    def from_predicates(cls, predicates: List[str]) -> "Schema":
        predicates = sorted(list(set(predicates)))
        predicate2id = {p: i for i, p in enumerate(predicates)}
        id2predicate = {i: p for i, p in enumerate(predicates)}
        return cls(predicate2id=predicate2id, id2predicate=id2predicate)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "predicate2id": self.predicate2id,
                    "id2predicate": self.id2predicate,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filepath: str) -> "Schema":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 2:
            id2pred, pred2id = data
            return cls(predicate2id=pred2id, id2predicate=id2pred)
        return cls(
            predicate2id=data["predicate2id"], id2predicate=data["id2predicate"]
        )


class RelationExtractionDataset(Dataset):
    """PyTorch Dataset for Subject-Predicate-Object relation extraction."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        vocab: Vocab,
        schema: Schema,
        max_len: int = 512,
        is_training: bool = True,
        augment: bool = False,
    ):
        self.data = data
        self.vocab = vocab
        self.schema = schema
        self.max_len = max_len
        self.is_training = is_training
        self.augment = augment

        # Index known predicates for data augmentation if enabled
        self.predicate_examples: Dict[str, List[Tuple[str, str, str]]] = {}
        if self.augment:
            for item in self.data:
                for s, p, o in item.get("spo_list", []):
                    if p not in self.predicate_examples:
                        self.predicate_examples[p] = []
                    self.predicate_examples[p].append((s, p, o))

    def __len__(self) -> int:
        return len(self.data)

    def _augment_sample(
        self, text: str, spo_list: List[Tuple[str, str, str]]
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        if not spo_list or random.random() < 0.5:
            return text, spo_list

        target_spo = random.choice(spo_list)
        p = target_spo[1]
        candidates = self.predicate_examples.get(p, [])
        if not candidates:
            return text, spo_list

        donor_spo = random.choice(candidates)
        # Safely replace entities
        new_text = text.replace(target_spo[0], donor_spo[0]).replace(
            target_spo[2], donor_spo[2]
        )
        new_spo_list = [
            (
                s.replace(target_spo[0], donor_spo[0]),
                pred,
                o.replace(target_spo[2], donor_spo[2]),
            )
            for s, pred, o in spo_list
        ]
        return new_text, new_spo_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        text = item["text"]
        spo_list = [
            (sp[0], sp[1], sp[2]) if isinstance(sp, (list, tuple)) else (sp["subject"], sp["predicate"], sp["object"])
            for sp in item.get("spo_list", [])
        ]

        if self.is_training and self.augment:
            text, spo_list = self._augment_sample(text, spo_list)

        # Truncate text
        text = text[: self.max_len]
        char_ids = self.vocab.encode(text)
        seq_len = len(char_ids)

        if not self.is_training:
            return {
                "text": text,
                "char_ids": char_ids,
                "spo_list": spo_list,
            }

        # Build items map: (sub_start, sub_end) -> [(obj_start, obj_end, pred_id)]
        items: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for s, p, o in spo_list:
            if p not in self.schema.predicate2id:
                continue
            pred_id = self.schema.predicate2id[p]
            s_start = text.find(s)
            o_start = text.find(o)
            if s_start != -1 and o_start != -1:
                s_end = s_start + len(s) - 1  # inclusive
                o_end = o_start + len(o) - 1  # inclusive
                key = (s_start, s_end)
                if key not in items:
                    items[key] = []
                items[key].append((o_start, o_end, pred_id))

        s1 = np.zeros(seq_len, dtype=np.float32)
        s2 = np.zeros(seq_len, dtype=np.float32)
        k1 = 0
        k2 = 0
        num_classes = len(self.schema)
        o1 = np.zeros((seq_len, num_classes), dtype=np.float32)
        o2 = np.zeros((seq_len, num_classes), dtype=np.float32)

        if items:
            for s_start, s_end in items.keys():
                s1[s_start] = 1.0
                s2[s_end] = 1.0

            # Randomly pick one subject as teacher conditioning
            chosen_subject = random.choice(list(items.keys()))
            k1, k2 = chosen_subject

            for o_start, o_end, pred_id in items[chosen_subject]:
                o1[o_start, pred_id] = 1.0
                o2[o_end, pred_id] = 1.0

        return {
            "text": text,
            "char_ids": char_ids,
            "s1": s1,
            "s2": s2,
            "k1": k1,
            "k2": k2,
            "o1": o1,
            "o2": o2,
            "spo_list": spo_list,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a batch of variable-length samples with dynamic padding."""
    is_training = "s1" in batch[0]
    lengths = [len(item["char_ids"]) for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Pad char_ids
    char_ids_padded = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, item in enumerate(batch):
        l = len(item["char_ids"])
        char_ids_padded[i, :l] = item["char_ids"]
        mask[i, :l] = 1.0

    output = {
        "char_ids": torch.tensor(char_ids_padded, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "texts": [item["text"] for item in batch],
        "spo_lists": [item.get("spo_list", []) for item in batch],
    }

    if is_training:
        num_classes = batch[0]["o1"].shape[1]
        s1_padded = np.zeros((batch_size, max_len), dtype=np.float32)
        s2_padded = np.zeros((batch_size, max_len), dtype=np.float32)
        o1_padded = np.zeros((batch_size, max_len, num_classes), dtype=np.float32)
        o2_padded = np.zeros((batch_size, max_len, num_classes), dtype=np.float32)
        k1_list = []
        k2_list = []

        for i, item in enumerate(batch):
            l = len(item["char_ids"])
            s1_padded[i, :l] = item["s1"]
            s2_padded[i, :l] = item["s2"]
            o1_padded[i, :l, :] = item["o1"]
            o2_padded[i, :l, :] = item["o2"]
            k1_list.append([item["k1"]])
            k2_list.append([item["k2"]])

        output.update(
            {
                "s1": torch.tensor(s1_padded, dtype=torch.float32),
                "s2": torch.tensor(s2_padded, dtype=torch.float32),
                "o1": torch.tensor(o1_padded, dtype=torch.float32),
                "o2": torch.tensor(o2_padded, dtype=torch.float32),
                "k1": torch.tensor(k1_list, dtype=torch.long),
                "k2": torch.tensor(k2_list, dtype=torch.long),
            }
        )

    return output

