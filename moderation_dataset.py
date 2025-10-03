from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utility import TTTDatasetBase
import numpy as np
import math

def convert(prob, return_logits: bool = False, temperature: float = 1.0, epsilon: float = 1e-8):
    """Convert probability(s) to logit(s) with temperature scaling and optionally back to probability.

    Works with scalars or NumPy arrays.
    """
    prob = np.clip(prob, epsilon, 1 - epsilon)
    logits = np.log(prob / (1 - prob)) / temperature
    if return_logits:
        return logits
    else:
        return 1 / (1 + np.exp(-logits))

class OpenAIModerationDatasetBase(TTTDatasetBase):
    """Base class that encapsulates common __init__ logic for moderation datasets.

    Handles:
      - core configuration (tokenizer, logits/prob settings)
      - CSV loading and text column validation
      - pre-tokenization of static prompt fragments
      - helpers for rule variant encoding and comment tokenization
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        text_name: str,
        max_length: int,
        COLUMN_VARIANTS: Dict[str, List[str]],
        threshold: float,
    ) -> None:
        super().__init__(max_length)
        self.tokenizer = tokenizer
        self._text_name = text_name

        # Load data
        self.df = pd.read_csv(csv_path)
        if text_name not in self.df.columns:
            raise ValueError(f"Expected '{text_name}' column in the CSV")

        # Pre-tokenize static prompt fragments
        # self._header_ids = self._enc("Does the comment violate the reddit rule?\n")
        # self._comment_prefix_ids = self._enc("Comment: ")
        # self._violation_prompt_ids = self._enc("Violation:")
        self._newline_ids = self._enc("\n")

        # Randomized prompt templates for single-comment prompts
        # Each template provides (header, comment_prefix, violation_prompt)
        self._create_prompt_templates_ids()

        # Common initialization across datasets
        self.score_columns = self._get_usable_score_columns(COLUMN_VARIANTS, threshold)
        print("train score columns:")
        print("\n".join(self.score_columns))
        self._build_rule_variants_ids(COLUMN_VARIANTS)
        self._pretokenize_comments(text_name)

    def _get_usable_score_columns(
        self,
        COLUMN_VARIANTS: Dict[str, List[str]],
        threshold: float,
    ) -> List[str]:
        """Compute usable score columns from the CSV given rule variants and a threshold.

        If ``threshold`` is None, returns all columns present from ``COLUMN_VARIANTS``.
        Otherwise, returns only those present columns whose max value exceeds ``threshold``.
        """
        return [c for c in COLUMN_VARIANTS.keys() if self.df[c].max() > threshold]


    def _build_rule_variants_ids(
        self,
        COLUMN_VARIANTS: Dict[str, List[str]],
    ) -> Dict[str, List[torch.Tensor]]:
        rule_variants: Dict[str, List[torch.Tensor]] = {}
        for col_name, variants in COLUMN_VARIANTS.items():
            encoded_variants: List[torch.Tensor] = []
            for v in variants:
                encoded_variants.append(self._enc(f"Rule: {v}\n"))
            rule_variants[col_name] = encoded_variants
        self._rule_variants_ids = rule_variants
        return rule_variants

    def _pretokenize_comments(self, text_name: Optional[str] = None) -> None:
        name = text_name if text_name is not None else self._text_name
        texts: List[str] = self.df[name].astype(str).tolist()
        self._comment_ids: List[torch.Tensor] = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False)[
            "input_ids"
        ]


class TrainingDataset(OpenAIModerationDatasetBase, Dataset):
    """Map-style dataset for OpenAI moderation scores.

    Each item:
      - randomly samples one moderation category (score column)
      - builds a single-comment prompt similar to ``assemble_prompt_single``
      - returns the input token ids and the selected category's score as target
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        COLUMN_VARIANTS: Dict[str, List[str]],
        return_logits: bool = True,
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        text_name: str = "text",
        threshold: float = 0.3,
        max_length: int = 2048,
        sample_uniform_mix: float = 0.03,
        # for sampling rule, < 1 to sharpen the distribution for focus on max value as offense are rare
        sample_temperature: float = 0.5,

    ) -> None:
        """Create a map-style dataset over a CSV of OpenAI moderation scores.

        Parameters
        ----------
        csv_path : str
            Path to a CSV file that includes a text column and one column per
            moderation category with scores in [0, 1].
        tokenizer
            HuggingFace-compatible tokenizer used to encode prompt fragments and
            comments. Must provide ``encode`` and ``batch_encode_plus``.
        return_logits : bool, optional
            If True, convert probabilities to logits and sample categories in
            logit space; otherwise operate in probability space. Defaults to False.
        temperature : float, optional
            Temperature applied during prob→logit transform in ``convert``.
            Only relevant when ``return_logits`` is False. Defaults to 1.0.
        epsilon : float, optional
            Numerical stability epsilon used in conversions. Defaults to 1e-8.
        text_name : str, optional
            Name of the text column in the CSV. Defaults to "text".
        threshold : float, optional
            Minimum max-score required for a category column to be included.
            Evaluated on raw CSV probabilities before any conversion. Defaults to 0.3.
        max_length : int, optional
            Maximum token length of the assembled prompt. Defaults to 2048.
        sample_uniform_mix : float, optional
            If the maximum category value for a row is below this threshold,
            sample a category uniformly instead of using tempered softmax.
            Interpreted in probability space, automatically converted to logit
            space when ``return_logits`` is True. Defaults to 0.05.
        sample_temperature : float, optional
            Temperature for the category sampling softmax. Lower is peakier.
            Defaults to 0.5.
        """
        super().__init__(
            csv_path=csv_path,
            tokenizer=tokenizer,
            text_name=text_name,
            max_length=max_length,
            COLUMN_VARIANTS=COLUMN_VARIANTS,
            threshold=threshold,
        )
        self._return_logits = return_logits
        self._epsilon = epsilon
        self._threshold = threshold
        self._sample_temperature = sample_temperature
        self._sample_uniform_mix = (
            math.log(sample_uniform_mix / (1 - sample_uniform_mix))/temperature if return_logits else sample_uniform_mix
        )

        # convert scores to logits, scale by temperature, optionally convert back to probabilities
        scores_np = self.df[self.score_columns].to_numpy()
        scores_np = convert(
            scores_np,
            return_logits=return_logits,
            temperature=temperature,
            epsilon=epsilon,
        )
        self.df[self.score_columns] = scores_np

    def __len__(self) -> int:
        """Return the number of rows in the underlying DataFrame."""
        return len(self.df)

    def _choose_category_index(self, values: np.ndarray) -> int:
        """Sample a category index via tempered softmax over rule scores.

        Handles both probability inputs and logit inputs based on configuration.
        """
        # Ensure numeric dtype; object-dtype can break numpy ufuncs like exp
        values = np.asarray(values, dtype=np.float64)
        max_value = values.max()
        if max_value < self._sample_uniform_mix:
            return int(np.random.choice(len(values)))
        
        if not self._return_logits:
            # Convert probabilities to logits for stable softmax shaping
            probs = np.clip(values, self._epsilon, 1 - self._epsilon)
            logits = np.log(probs/(1-probs))
        else:
            logits = values

        # Apply sampling temperature and stabilize
        temp = self._sample_temperature
        scaled = logits / temp
        scaled = scaled - np.max(scaled)
        weights = np.exp(scaled)
        denom = np.sum(weights)

        probs = weights / denom

        return int(np.random.choice(len(probs), p=probs))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single prompt tensor and the selected category score.

        The category is chosen per-row by applying a tempered softmax over all
        rule scores (in prob or logit space depending on configuration). A
        uniform fallback is used when all scores are very small.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(input_ids, target_score)`` where ``input_ids`` has shape
            ``[1, seq_len]`` and ``target_score`` is a scalar float tensor.
        """
        row = self.df.iloc[idx]

        # Tempered softmax over rule scores to select category (multi-label friendly)
        values = row[self.score_columns].to_numpy(dtype=np.float64)
        chosen_idx = self._choose_category_index(values)
        col_name = self.score_columns[chosen_idx]
        score = torch.tensor(float(row[col_name]), dtype=torch.float)

        # Choose a random paraphrase variant for the selected rule
        rule_variant_ids = random.choice(self._rule_variants_ids[col_name])

        comment_ids = torch.tensor(self._comment_ids[idx])
        input_ids, _ = self.assemble_prompt_single(
            rule_variant_ids=rule_variant_ids,
            comment=comment_ids,
        )

        # Return: category name, input ids (as batch dim 1), and target score
        return input_ids, score, col_name

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]], pad_token_id: int = 1):
    input_id_list, score_list, col_name_list = zip(*batch)
    lengths = [t.numel() - 1 for t in input_id_list]
    max_len = max(lengths) + 1
    batch_size = len(input_id_list)
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(input_id_list):
        padded[i, : ids.numel()] = ids
    last_indices = torch.tensor(lengths, dtype=torch.long)
    scores = torch.tensor(score_list, dtype=torch.float)
    return padded, scores, last_indices, list(col_name_list)

def build_training_dataloader(
    csv_path: str,
    tokenizer,
    column_variants: Dict[str, List[str]],
    batch_size: int = 2,
    return_logits: bool = True,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
    text_name: str = "text",
    threshold: float = 0.3,
    max_length: int = 2048,
    sample_uniform_mix: float = 0.05,
    sample_temperature: float = 0.5,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for ``OpenAIModerationMapDataset``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV with one text column and multiple moderation score columns.
    tokenizer
        Tokenizer used by the dataset to encode prompts.
    column_variants : Dict[str, List[str]], optional
        Dictionary mapping column names to lists of rule variants.
    return_logits : bool, optional
        If True, convert probabilities to logits and sample categories in
        logit space; otherwise operate in probability space. Defaults to True.
    temperature : float, optional
        Temperature applied during prob→logit transform in ``convert``.
        Only relevant when ``return_logits`` is False. Defaults to 1.0.
    epsilon : float, optional
        Numerical stability epsilon used in conversions. Defaults to 1e-8.
    text_name : str, optional
        Name of the text column. Defaults to "text".
    threshold : float, optional
        Drop score columns whose max value is ≤ threshold (on raw CSV probs).
        Defaults to 0.3.
    max_length : int, optional
        Maximum prompt length. Defaults to 2048.
    sample_uniform_mix : float, optional
        If the maximum category value for a row is below this threshold,
        sample a category uniformly instead of using tempered softmax.
        Interpreted in probability space, automatically converted to logit
        space when ``return_logits`` is True. Defaults to 0.05.
    sample_temperature : float, optional
        Temperature for the category sampling softmax. Lower is peakier.
        Defaults to 0.5.
    shuffle : bool, optional
        Whether to shuffle rows each epoch. Defaults to True.
    pin_memory : bool, optional
        Passed to the PyTorch DataLoader. Defaults to True.
    """
    
    dataset = TrainingDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        COLUMN_VARIANTS=column_variants,
        return_logits=return_logits,
        temperature=temperature,
        epsilon=epsilon,
        text_name=text_name,
        threshold=threshold,
        max_length=max_length,
        sample_uniform_mix=sample_uniform_mix,
        sample_temperature=sample_temperature,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )



class EvaluationDataset(OpenAIModerationDatasetBase, Dataset):
    """Exhaustive evaluation dataset over all (row, category) combinations.

    For each row in the CSV and for each moderation category column present,
    this dataset yields a single-comment prompt (using the first rule variant
    by default) and the corresponding target score (raw values from CSV).
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        COLUMN_VARIANTS: Dict[str, List[str]],
        text_name: str = "text",
        threshold: float = 0.3,
        max_length: int = 2048,
    ) -> None:
        super().__init__(
            csv_path=csv_path,
            tokenizer=tokenizer,
            text_name=text_name,
            max_length=max_length,
            COLUMN_VARIANTS=COLUMN_VARIANTS,
            threshold=threshold,
        )

        # Evaluation specific: Build exhaustive (row, col) pairs
        num_rows = len(self.df)
        num_cols = len(self.score_columns)
        self._pairs: List[Tuple[int, int]] = [(i, j) for i in range(num_rows) for j in range(num_cols)]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row_idx, col_idx = self._pairs[idx]
        col_name = self.score_columns[col_idx]
        rule_variant_ids = random.choice(self._rule_variants_ids[col_name])

        comment_ids = torch.tensor(self._comment_ids[row_idx])
        input_ids, _ = self.assemble_prompt_single(
            rule_variant_ids=rule_variant_ids,
            comment=comment_ids,
        )

        score = torch.tensor(self.df.iloc[row_idx][col_name], dtype=torch.float)
        # Return 1D input_ids; batching and padding handled by collate_fn
        return input_ids, score, col_name


def build_evaluation_dataloader(
    csv_path: str,
    tokenizer,
    column_variants: Dict[str, List[str]],
    text_name: str = "text",
    threshold: float = 0.3,
    max_length: int = 2048,
    pin_memory: bool = True,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader that iterates every (row, category) pair once.

    Collate function pads sequences and returns the last valid index per sample.
    """
    dataset = EvaluationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        COLUMN_VARIANTS=column_variants,
        text_name=text_name,
        threshold=threshold,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )



class FixedSizePerColumnDataset(OpenAIModerationDatasetBase, Dataset):
    """Dataset that yields a fixed total number of samples balanced across columns.

    Given a desired total ``size``, this dataset distributes samples evenly across
    the usable moderation columns (filtered by ``threshold``). For each column,
    it takes roughly half of its quota from the highest-scoring rows ("positives")
    and the other half from either random rows or the lowest-scoring rows
    ("negatives"), as configured by ``neg_strategy``.

    Returned targets are the continuous scores from the CSV for the selected
    column. Optionally, set ``return_logits=True`` to convert probabilities to
    logits on-the-fly when fetching items.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        COLUMN_VARIANTS: Dict[str, List[str]],
        size: int,
        *,
        neg_strategy: str = "bottom",  # "bottom" or "random"
        return_logits: bool = False,
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        text_name: str = "text",
        threshold: float = 0.1,
        max_length: int = 2048,
    ) -> None:
        super().__init__(
            csv_path=csv_path,
            tokenizer=tokenizer,
            text_name=text_name,
            max_length=max_length,
            COLUMN_VARIANTS=COLUMN_VARIANTS,
            threshold=threshold,
        )

        self._return_logits = return_logits
        self._temperature = temperature
        self._epsilon = epsilon
        self._neg_strategy = neg_strategy
        self._return_logits = return_logits

        # Build (row_idx, col_name) sampling plan
        self._pairs: List[Tuple[int, str]] = []
        num_cols = len(self.score_columns)
        base_half = (size // num_cols) // 2
        for _, col_name in enumerate(self.score_columns):
            # Use raw probabilities for ranking
            values = self.df[col_name].to_numpy(dtype=np.float64)

            # Top-K for positives (descending sort without slicing reversal)
            order_desc = np.argsort(-values)
            pos_indices = order_desc[:base_half].tolist()

            # Negatives: either bottom-K (ascending) excluding positives, or random excluding positives
            if self._neg_strategy == "bottom":
                neg_indices = order_desc[-base_half:].tolist()
            else:  # random
                neg_indices = np.random.choice(order_desc[base_half:], size=base_half, replace=False).tolist()

            for ridx in pos_indices:
                self._pairs.append((int(ridx), col_name))
            for ridx in neg_indices:
                self._pairs.append((int(ridx), col_name))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row_idx, col_name = self._pairs[idx]

        rule_variant_ids = random.choice(self._rule_variants_ids[col_name])
        comment_ids = torch.tensor(self._comment_ids[row_idx])
        input_ids, _ = self.assemble_prompt_single(
            rule_variant_ids=rule_variant_ids,
            comment=comment_ids,
        )

        value = float(self.df.iloc[row_idx][col_name])
        value = convert(value, return_logits=self._return_logits, temperature=self._temperature, epsilon=self._epsilon)
        score = torch.tensor(value, dtype=torch.float)
        return input_ids, score, col_name

    def get_average_score(self) -> float:
        if not self._pairs:
            return 0.0
        total = 0.0
        for row_idx, col_name in self._pairs:
            value = float(self.df.iloc[row_idx][col_name])
            value = convert(value, return_logits=self._return_logits, temperature=self._temperature, epsilon=self._epsilon)
            total += value
        if self._return_logits:
            return total / len(self._pairs)
        else:
            return math.log(total / (len(self._pairs) - total))


def build_fixed_size_per_column_dataloader(
    csv_path: str,
    tokenizer,
    column_variants: Dict[str, List[str]],
    size: int,
    *,
    neg_strategy: str = "bottom",
    batch_size: int = 2,
    return_logits: bool = False,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
    text_name: str = "text",
    threshold: float = 0.1,
    max_length: int = 2048,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for ``FixedSizePerColumnDataset``.

    Parameters
    ----------
    size : int
        Total number of samples to draw across all usable columns.
    neg_strategy : {"bottom", "random"}
        How to draw the negative half within each column's share.
    return_logits : bool
        If True, convert targets from probabilities to logits on-the-fly.
    """
    dataset = FixedSizePerColumnDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        COLUMN_VARIANTS=column_variants,
        size=size,
        neg_strategy=neg_strategy,
        return_logits=return_logits,
        temperature=temperature,
        epsilon=epsilon,
        text_name=text_name,
        threshold=threshold,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
