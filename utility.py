"""
Utility functions for text normalization used in the AI_Moderation project.

Currently includes:
1. normalize_urls        – Replace URLs with placeholder URL_TOKEN.
2. normalize_usernames   – Replace Reddit and @-style user mentions with USER_TOKEN.
3. normalize_emails      – Replace email addresses with EMAIL_TOKEN.
4. normalize_subreddits  – Replace subreddit mentions (r/…) with SUB_TOKEN.
5. normalize_phone_numbers – Replace phone numbers with PHONE_TOKEN.
6. normalize_money       – Replace dollar amounts with MONEY_TOKEN.
7. normalize_random_strings – Replace random strings with RANDOM_TOKEN.
8. normalize_ip_addresses – Remove IP addresses from text.
9. normalize_timestamps  – Remove various timestamp formats from text.
10. normalize_quotes     – Remove leading and trailing quotes but keep internal quotes.
11. reduce_repeated_characters – Reduce repeated chars (letters to max 2, others to max 1).
12. reduce_repeated_words – Reduce repeated words to at most 1 occurrence.
13. reduce_repeated_whitespace – Reduce repeated spaces/tabs to single space.
14. normalize_text       – Convenience wrapper that applies all of the above.
15. apply_incremental_normalization – Apply only new normalization steps to pre-normalized text.

Rationale
---------
• Exact URLs, user names, emails, phone numbers, and specific dollar amounts rarely matter for rule-violation classification.
• Replacing personal identifiers and random strings removes nearly-unique tokens that otherwise bloat the tokenizer's sub-word vocabulary.
• Removing IP addresses and timestamps eliminates metadata that is not relevant for content classification.
• Removing leading/trailing quotes helps normalize text format without affecting content.
• Reducing repeated characters, words, and whitespace helps normalize informal text patterns common in social media.
• This normalization helps models generalize better by focusing on content patterns rather than specific identifiers or stylistic variations.
"""
from __future__ import annotations

# Simple normalization tokens for nlpaug compatibility
SIMPLE_NORMALIZATION_TOKENS = {
    'URL': 'URL_TOKEN',
    'USER': 'USER_TOKEN', 
    'EMAIL': 'EMAIL_TOKEN',
    'SUB': 'SUB_TOKEN',
    'PHONE': 'PHONE_TOKEN',
    'MONEY': 'MONEY_TOKEN',
    'RANDOM': 'RANDOM_TOKEN'
}

# Export for use in inference code
NLPAUG_STOPWORDS = list(SIMPLE_NORMALIZATION_TOKENS.values())

import re
import os
from urllib.parse import urlparse
import random
import math
from typing import Dict, List, Sequence, Tuple, Any

import pandas as pd
import torch
import numpy as np
from collections import defaultdict
# ---------------------------------------------------------------------------
# Rule variants (paraphrased)
# ---------------------------------------------------------------------------
try:
    from rules import RULE_VARIANTS, RULE_NEGATIVE_WHITELIST
except ImportError:
    RULE_VARIANTS = None
    RULE_NEGATIVE_WHITELIST = None
from torch.utils.data import Dataset, IterableDataset, DataLoader


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Generic URL recognizer – looks for http(s)://, ftp://, or bare www.<domain>, or spaced .com/.org/.net etc
_URL_RE: re.Pattern[str] = re.compile(
    r"(?:(?:https?://|ftp://|www\.)[^\s]+)|(?:\w+\s*\.\s*(?:com|org|net|edu|gov|co\.uk|io|ly|me|tv|info|biz|us|ca)(?:\s|$|[^\w]))",  # includes spaced URLs
    flags=re.IGNORECASE,
)

# Random string pattern - matches truly random-looking codes/IDs with mixed digits, letters, and symbols
_RANDOM_STRING_RE = re.compile(r"""
    (?<!\w)                                    # Word boundary start
    (?=\S*[0-9])                              # Must contain digit
    (?=\S*[a-zA-Z])                           # Must contain letter  
    (?=\S*[!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`])  # Must contain special char
    (?:
        [a-zA-Z0-9]*[0-9]+[a-zA-Z]+[!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`][a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`]* |
        [a-zA-Z0-9]*[a-zA-Z]+[0-9]+[!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`][a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`]* |
        [a-zA-Z0-9]*[!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`][a-zA-Z0-9]*[0-9][a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;':\",.<>?~`]*
    )
    (?!\w)                                     # Word boundary end
""", re.VERBOSE | re.IGNORECASE)
# Repeated character patterns
_REPEATED_LETTERS_RE: re.Pattern[str] = re.compile(r"([a-zA-Z])\1{2,}")
_REPEATED_NONLETTERS_RE: re.Pattern[str] = re.compile(r"([^a-zA-Z\s])\1{1,}")

# Repeated word pattern  
_REPEATED_WORDS_RE: re.Pattern[str] = re.compile(r"\b(\w+)(\s+\1){1,}\b", flags=re.IGNORECASE)

# Repeated whitespace pattern (spaces, tabs)
_REPEATED_WHITESPACE_RE: re.Pattern[str] = re.compile(r"[ \t]+", flags=re.MULTILINE)

# Reddit user mention formats: u/username or /u/username (case-insensitive)
_REDDIT_USER_RE: re.Pattern[str] = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+", flags=re.IGNORECASE)

# @username mentions – capped at 30 chars, avoids picking up email addresses
_AT_USER_RE: re.Pattern[str] = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}\b")

# Email addresses
_EMAIL_RE: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)

# Reddit subreddit mentions: r/subreddit or /r/subreddit (case-insensitive)
_SUBREDDIT_RE: re.Pattern[str] = re.compile(r"(?<!\w)/?r/[A-Za-z0-9_-]+", flags=re.IGNORECASE)

# Phone numbers - various formats
_PHONE_RE: re.Pattern[str] = re.compile(
    r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|"  # US phone numbers
    r"1-800-[A-Z0-9-]+",  # 1-800 numbers with letters
    flags=re.IGNORECASE,
)

# Money amounts - currency symbols (e.g., "$1,000", "£50", "€3.2M") OR numeric/placeholder amounts followed by currency words
_MONEY_RE: re.Pattern[str] = re.compile(
    r'''(
        [\$£€]                              # Currency symbols
        [0-9]+(?:[,.][0-9]+)*                # Amount with optional separators
        (?:\s*(?:million|billion|k|M|B))?    # Optional scale/abbreviation
        |                                     # OR
        (?:[0-9]+|[Xx]{2,})\s*              # Number or placeholder
        (?:dollars?|pounds?|euros?)          # Currency words
    )''',
    flags=re.IGNORECASE | re.VERBOSE,
)

# IP addresses (pattern: xxx.xxx.xxx.xxx)
_IP_ADDRESS_RE: re.Pattern[str] = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

# Timestamps with time - Original format: HH:MM, Month DD, YYYY (UTC)
_TIMESTAMP_TIME_ORIG_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}:[0-9]{1,2},\s+[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}\s+\(UTC\)')

# Timestamps with time - NEW format: HH:MM, DD Mon YYYY (UTC)
_TIMESTAMP_TIME_NEW_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}:[0-9]{1,2},\s+[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}\s+\(UTC\)')

# Date-only timestamps - Original format: Month DD, YYYY (UTC)
_DATE_ORIG_RE: re.Pattern[str] = re.compile(r'\b[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}\s+\(UTC\)')

# Date-only timestamps - NEW format: DD Mon YYYY (UTC)
_DATE_NEW_UTC_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}\s+\(UTC\)')

# Date-only timestamps - NEW format: DD Mon YYYY (without UTC)
_DATE_NEW_NO_UTC_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}(?!\s+\(UTC\)|\w)')

# Timestamps without UTC (pattern: HH:MM, Month DD, YYYY)
_TIMESTAMP_NO_UTC_ORIG_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}:[0-9]{1,2},\s+[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}(?!\s+\(UTC\))')

# Timestamps without UTC (pattern: HH:MM, DD Mon YYYY)
_TIMESTAMP_NO_UTC_NEW_RE: re.Pattern[str] = re.compile(r'\b[0-9]{1,2}:[0-9]{1,2},\s+[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}(?!\s+\(UTC\))')

# ISO format timestamps (pattern: YYYY-MM-DD HH:MM:SS+TZ)
_ISO_TIMESTAMP_RE: re.Pattern[str] = re.compile(r'\b[0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}[.0-9]*[+-][0-9]{2}:[0-9]{2}\b')


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_urls(text: str) -> str:
    """Replace every URL in *text* with ``URL_TOKEN``.

    Examples
    --------
    >>> normalize_urls("See https://sub.example.com/path?a=1 and http://example.org")
    'See URL_TOKEN and URL_TOKEN'
    """
    return _URL_RE.sub("URL_TOKEN", text)


def normalize_usernames(text: str) -> str:
    """Replace Reddit-style and @ mentions with the fixed token ``USER_TOKEN``."""
    text = _REDDIT_USER_RE.sub("USER_TOKEN", text)
    text = _AT_USER_RE.sub("USER_TOKEN", text)
    return text


def normalize_emails(text: str) -> str:
    """Replace email addresses with ``EMAIL_TOKEN``."""
    return _EMAIL_RE.sub("EMAIL_TOKEN", text)


def normalize_subreddits(text: str) -> str:
    """Replace subreddit mentions (r/subreddit) with ``SUB_TOKEN``."""
    return _SUBREDDIT_RE.sub("SUB_TOKEN", text)


def normalize_phone_numbers(text: str) -> str:
    """Replace phone numbers with ``PHONE_TOKEN``."""
    return _PHONE_RE.sub("PHONE_TOKEN", text)


def normalize_money(text: str) -> str:
    """Replace money amounts (dollar, pound, euro, etc.) with ``MONEY_TOKEN``."""
    return _MONEY_RE.sub("MONEY_TOKEN", text)


def normalize_random_strings(text: str) -> str:
    """Replace random strings like '44XtPDKtcnDY30!' with ``RANDOM_TOKEN``."""
    return _RANDOM_STRING_RE.sub("RANDOM_TOKEN", text)


def normalize_ip_addresses(text: str) -> str:
    """Remove IP addresses from text."""
    return _IP_ADDRESS_RE.sub("", text)


def normalize_timestamps(text: str) -> str:
    """Remove various timestamp formats from text."""
    # Remove timestamps with time - Original format: HH:MM, Month DD, YYYY (UTC)
    text = _TIMESTAMP_TIME_ORIG_RE.sub("", text)
    
    # Remove timestamps with time - NEW format: HH:MM, DD Mon YYYY (UTC)
    text = _TIMESTAMP_TIME_NEW_RE.sub("", text)
    
    # Remove date-only timestamps - Original format: Month DD, YYYY (UTC)
    text = _DATE_ORIG_RE.sub("", text)
    
    # Remove date-only timestamps - NEW format: DD Mon YYYY (UTC)
    text = _DATE_NEW_UTC_RE.sub("", text)
    
    # Remove date-only timestamps - NEW format: DD Mon YYYY (without UTC)
    text = _DATE_NEW_NO_UTC_RE.sub("", text)
    
    # Remove timestamps without UTC (pattern: HH:MM, Month DD, YYYY)
    text = _TIMESTAMP_NO_UTC_ORIG_RE.sub("", text)
    
    # Remove timestamps without UTC (pattern: HH:MM, DD Mon YYYY)
    text = _TIMESTAMP_NO_UTC_NEW_RE.sub("", text)
    
    # Remove ISO format timestamps (pattern: YYYY-MM-DD HH:MM:SS+TZ)
    text = _ISO_TIMESTAMP_RE.sub("", text)
    
    return text


def normalize_quotes(text: str) -> str:
    """Remove leading and trailing quotes but keep internal quotes."""
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def reduce_repeated_characters(text: str) -> str:
    """Reduce repeated characters: letters to max 2, non-letters to max 1."""
    # Reduce repeated letters to max 2
    text = _REPEATED_LETTERS_RE.sub(lambda m: m.group(1) + m.group(1), text)
    # Reduce repeated non-letters to max 1  
    text = _REPEATED_NONLETTERS_RE.sub(lambda m: m.group(1), text)
    return text


def reduce_repeated_words(text: str) -> str:
    """Reduce repeated words to at most 1 occurrence (e.g., 'but but but' -> 'but')."""
    return _REPEATED_WORDS_RE.sub(lambda m: m.group(1), text)


def reduce_repeated_whitespace(text: str) -> str:
    """Reduce repeated spaces and tabs to single space."""
    return _REPEATED_WHITESPACE_RE.sub(" ", text)


def convert_old_tokens_to_new(text: str) -> str:
    """Convert old angle-bracket tokens to new underscore tokens.
    
    This function helps clean up existing datasets that were processed with
    the old token format (<URL>, <USER>, etc.) by converting them to the new
    nlpaug-compatible format (URL_TOKEN, USER_TOKEN, etc.).
    
    Parameters
    ----------
    text : str
        Text that may contain old-format special tokens
        
    Returns
    -------
    str
        Text with old tokens replaced by new underscore tokens
        
    Examples
    --------
    >>> convert_old_tokens_to_new("Visit <URL> and contact <USER>")
    'Visit URL_TOKEN and contact USER_TOKEN'
    >>> convert_old_tokens_to_new("Email <EMAIL> about <MONEY> payment")
    'Email EMAIL_TOKEN about MONEY_TOKEN payment'
    """
    # Mapping of old tokens to new tokens
    old_to_new_mapping = {
        '<URL>': 'URL_TOKEN',
        '<USER>': 'USER_TOKEN', 
        '<EMAIL>': 'EMAIL_TOKEN',
        '<SUB>': 'SUB_TOKEN',
        '<PHONE>': 'PHONE_TOKEN',
        '<MONEY>': 'MONEY_TOKEN',
        '<RANDOM>': 'RANDOM_TOKEN'
    }
    
    # Replace each old token with its new equivalent
    for old_token, new_token in old_to_new_mapping.items():
        text = text.replace(old_token, new_token)
    
    return text


def normalize_text(text: str) -> str:
    """Apply all normalization functions in sequence."""
    text = normalize_urls(text)
    text = normalize_usernames(text)
    text = normalize_emails(text)
    # text = normalize_subreddits(text)
    text = normalize_phone_numbers(text)
    text = normalize_money(text)
    text = normalize_random_strings(text)
    text = normalize_ip_addresses(text)
    text = normalize_timestamps(text)
    text = normalize_quotes(text)
    text = reduce_repeated_characters(text)
    text = reduce_repeated_words(text)
    text = reduce_repeated_whitespace(text)
    return text


def apply_incremental_normalization(text: str) -> str:
    """Apply only the conservative normalization steps to already normalized text.
    
    Use this for text that already has the original normalize_text applied.
    This applies:
    1. Updated URL normalization (replace <URL_{domain}> with <URL>)
    2. Conservative random string normalization
    3. Character repetition reduction  
    4. Word repetition reduction
    5. Whitespace repetition reduction
    
    This function is designed to only make text shorter, never longer.
    """
    if not isinstance(text, str):
        return text
    
    original_text = text
    
    # Step 1: Replace <URL_{domain}> with <URL>
    text = re.sub(r"<URL_[^>]+>", "<URL>", text)
    
    # Step 2: Apply random string normalization using the improved pattern
    # Temporarily protect existing tokens
    token_pattern = r"<(USER|EMAIL|SUB|PHONE|MONEY|URL|RANDOM)>"
    tokens = re.findall(token_pattern, text)
    protected_text = re.sub(token_pattern, "TOKENPLACEHOLDER", text)
    
    # Apply random string normalization to protected text
    protected_text = normalize_random_strings(protected_text)
    
    # Step 3: Reduce repeated characters
    protected_text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", protected_text)  # Letters to max 2
    protected_text = re.sub(r"([^a-zA-Z\s])\1{1,}", r"\1", protected_text)  # Non-letters to max 1
    
    # Step 4: Reduce repeated words
    protected_text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", protected_text, flags=re.IGNORECASE)
    
    # Step 5: Reduce repeated whitespace
    protected_text = re.sub(r"[ \t]+", " ", protected_text)
    
    # Restore tokens
    for token in tokens:
        protected_text = protected_text.replace("TOKENPLACEHOLDER", f"<{token}>", 1)
    
    # Safety check: if result is longer than original, return original
    if len(protected_text) > len(original_text):
        return original_text
    
    return protected_text

def normalize_text_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ``normalize_text`` to selected text columns of a DataFrame.
    """

    columns = (
        "body",
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    )

    for col in columns:
        df[col] = df[col].map(normalize_text)
    return df


# ---------------------------------------------------------------------------
# Base class shared by TTT datasets
# ---------------------------------------------------------------------------

class TTTDatasetBase:
    """Shared utilities for TTT datasets (map and iter variants)."""
    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
    # -----------------------
    # Sampler helpers
    # -----------------------
    @staticmethod
    def _init_sampler_state(
        rule_to_pools: Dict[str, Dict[str, List[object]]],
        shuffle: bool = True,
    ) -> Dict[str, Dict[str, List[Any]]]:
        state: Dict[str, Dict[str, List[Any]]] = {}
        for rule, pools in rule_to_pools.items():
            pos = pools.get("positives", [])
            neg = pools.get("negatives", [])
            pos_indices = list(range(len(pos)))
            neg_indices = list(range(len(neg)))
            if shuffle:
                random.shuffle(pos_indices)
                random.shuffle(neg_indices)
            state[rule] = {
                "positives": [0, len(pos), pos_indices],
                "negatives": [0, len(neg), neg_indices],
            }
        return state

    @staticmethod
    def _next_from_pool(
        pool_items: List[object],
        pool_state: List[Any],
        *,
        reshuffle_on_cycle: bool = True,
    ) -> tuple[int, object]:
        cursor, total_length, indices = pool_state
        idx = indices[cursor]
        item = pool_items[idx]
        cursor += 1
        if cursor >= total_length:
            cursor = 0
            if reshuffle_on_cycle:
                random.shuffle(indices)
        pool_state[0] = cursor
        return idx, item

    @staticmethod
    def _sample_from_state(
        rule_to_pools: Dict[str, Dict[str, List[object]]],
        rule_state: Dict[str, Dict[str, List[int]]],
        rule: str,
        desired_label: str,
        *,
        reshuffle_on_cycle: bool = True,
        neg_sample_prob: float = 0.0,
    ) -> tuple[int, object, int]:
        pool = rule_to_pools[rule][desired_label]

        # _next_from_pool path, either positives or negatives has examples, 40% of the time we sample from the negative pool
        # 60% we sample from other rules.
        if desired_label == "positives" or (desired_label == "negatives" and len(pool) > 0 and random.random() < (1-neg_sample_prob)):
            state = rule_state[rule][desired_label]
            idx, item = TTTDatasetBase._next_from_pool(
                pool, state, reshuffle_on_cycle=reshuffle_on_cycle
            )
            return idx, item, 1 if desired_label == "positives" else 0

        # ------------------------------------------------------------------
        # Fallback: only happens for negatives.
        # ------------------------------------------------------------------
        if RULE_NEGATIVE_WHITELIST is None: # Kaggle, assume all but the current rule are valid
            rules = list(set(rule_to_pools.keys()) - {rule})
        else:
            rules = RULE_NEGATIVE_WHITELIST[rule]
        while True:
            rand_rule = random.choice(rules)
            rand_polarity = "positives"
            rand_pool = rule_to_pools[rand_rule][rand_polarity]
            if len(rand_pool) > 0:
                item = random.choice(rand_pool)
                return -1, item, 0 # 0 as it is a negative example

    # -----------------------
    # Prompt-token helpers
    # -----------------------
    def _enc(self, text: str, *, add_special_tokens: bool = False) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")[0]
        return ids
    
    def _create_prompt_templates_ids(self) -> None:
        self._prompt_templates_ids = [
            {
                "header": self._enc("Does the comment violate the reddit rule?\n"),
                "comment_prefix": self._enc("Comment: "),
                "violation_prompt": self._enc("Violation:"),
            },
            {
                "header": self._enc("You are a moderator. Check if the comment violates the rule.\n"),
                "comment_prefix": self._enc("User comment: "),
                "violation_prompt": self._enc("Does it violate?"),
            },
            {
                "header": self._enc("Reddit Moderation\n"),
                "comment_prefix": self._enc("post: "),
                "violation_prompt": self._enc("Moderator:"),
            },
            {
                "header": self._enc("Assess the following comment under the rule.\n"),
                "comment_prefix": self._enc("user: "),
                "violation_prompt": self._enc("Violation?"),
            },
            {
                "header": self._enc("Decide if this comment breaks the rule.\n"),
                "comment_prefix": self._enc("User comment: "),
                "violation_prompt": self._enc("Moderator decision:"),
            },
    ]

    def pretokenize_ttt_fragments(
        self,
        rules_to_tokenize: List[str] | None | Dict[str, List[str]],
    ) -> None:

        # Keep BOS/EOS out to maintain simple index math
        self._newline_ids = self._enc("\n")
        # self._header_ids = self._enc(
        #     "Does the comment violate the reddit rule?\n",
        # )
        # self._comment_prefix_ids = self._enc("Comment: ")
        # self._violation_yes_line_ids = self._enc("Violation: Yes\n")
        # self._violation_no_line_ids = self._enc("Violation: No\n")
        # self._violation_prompt_ids = self._enc("Violation:")

        # Randomized prompt templates for single-comment prompts
        # Each template provides (header, comment_prefix, violation_prompt)
        self._create_prompt_templates_ids()

        rule_variants_ids: Dict[str, List[torch.Tensor] | torch.Tensor] = {}
        if rules_to_tokenize is None:
            # default: use paraphrase variants from RULE_VARIANTS
            for rule_text, variants in RULE_VARIANTS.items():
                encoded_variants: List[torch.Tensor] = []
                for v in variants:
                    encoded_variants.append(self._enc(f"Rule: {v}\n"))
                rule_variants_ids[rule_text] = encoded_variants
        elif isinstance(rules_to_tokenize, list):
            # map-style: tokenize only the provided unique rules (single variant)
            for rule_text in rules_to_tokenize:
                rule_variants_ids[rule_text] = [self._enc(f"Rule: {rule_text}\n")]
        elif isinstance(rules_to_tokenize, dict):
            for rule_text, variants in rules_to_tokenize.items():
                encoded_variants: List[torch.Tensor] = []
                for v in variants:
                    encoded_variants.append(self._enc(f"Rule: {v}\n"))
                rule_variants_ids[rule_text] = encoded_variants
        self._rule_variants_ids = rule_variants_ids

    def compute_two_violation_end_indices(
        self,
        *,
        rule_variant_ids: torch.Tensor,
        support_ids: torch.Tensor,
        total_length: int,
    ) -> list[int]:
        prefix_len = (
            self._header_ids.numel()
            + rule_variant_ids.numel()
            + self._comment_prefix_ids.numel()
            + support_ids.numel()
            + self._newline_ids.numel()
        )
        first_violation_end = prefix_len + self._violation_prompt_ids.numel() - 1
        second_violation_end = total_length - 1
        return [first_violation_end, second_violation_end]

    def assemble_prompt_single(
        self,
        rule_variant_ids: torch.Tensor,
        comment: torch.Tensor,
        random_truncate: bool = False,
    ) -> Tuple[torch.Tensor, list[int]]:
        """
        Assemble a single-comment prompt that ends with a single "Violation:".

        This is used when only one comment is available (e.g., inference on a
        bare body or when no holdout split is provided). No label is provided
        in returned input_ids.
        """
        # Select a random template if available; fall back to fixed fragments otherwise
        if hasattr(self, "_prompt_templates_ids") and self._prompt_templates_ids:
            tmpl = random.choice(self._prompt_templates_ids)
            header_ids = tmpl["header"]
            comment_prefix_ids = tmpl["comment_prefix"]
            violation_prompt_ids = tmpl["violation_prompt"]
        else:
            header_ids = self._header_ids
            comment_prefix_ids = self._comment_prefix_ids
            violation_prompt_ids = self._violation_prompt_ids

        placeholder = torch.empty(0)
        pieces = [
            header_ids,
            rule_variant_ids,
            comment_prefix_ids,
            placeholder,  # actual single comment
            self._newline_ids,
            violation_prompt_ids,
        ]
        # Truncate the single comment to fit max_length
        available_for_comment = self.max_length - sum(t.numel() for t in pieces)
        if random_truncate:
            # Random start position truncation
            if comment.numel() > available_for_comment:
                max_start = comment.numel() - available_for_comment
                start_pos = random.randint(0, max_start)
                comment = comment[start_pos:start_pos + available_for_comment]
            # If comment is shorter than available space, use it as is
        else:
            comment = comment[:available_for_comment]
        pieces[3] = comment

        input_ids = torch.cat(pieces, dim=0)
        # Only one Violation:, which is at the end
        vi_index = [input_ids.numel() - 1]
        return input_ids, vi_index

    def assemble_prompt(
        self,
        rule_variant_ids: torch.Tensor,
        comment_train: torch.Tensor,
        label_train: int,
        comment_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[int]]:
        """
        Assemble the full TTT prompt from pre-tokenised fragments.

        Parameters
        ----------
        rule_variant_ids
            Token ids of the chosen rule variant line (``Rule: …``).
        comment_train
            Token ids of the support comment (label known).
        label_train
            1 if the support comment is a violation, else 0.
        comment_test
            Token ids of the target comment (label to predict).

        Returns
        -------
        torch.Tensor
            Concatenated tensor of token ids representing the complete prompt.
        """
        placeholder = torch.empty(0)
        pieces = [
            self._header_ids, # you are given a comment on reddit. Your task is to classify if it violates the given rule.
            rule_variant_ids, # Rule: <rule text>
            self._comment_prefix_ids, # Comment: 
            placeholder, # actual comment for training
            self._newline_ids, # \n
            self._violation_yes_line_ids if label_train == 1 else self._violation_no_line_ids, # Violation: Yes or No
            self._comment_prefix_ids, # Comment: 
            placeholder, # actual comment for testing
            self._newline_ids, # \n 
            self._violation_prompt_ids, # Violation:
        ]
        length = self.max_length - sum(t.numel() for t in pieces)
        if length % 2 == 0:
            comment_train = comment_train[:length//2]
            comment_test = comment_test[:length//2]
        else:
            comment_train = comment_train[:length//2]
            comment_test = comment_test[:length//2+1]
        pieces[3] = comment_train
        pieces[7] = comment_test
        
        # Concatenate all parts to build the final prompt
        input_ids = torch.cat(pieces, dim=0)

        # Compute the end indices for both occurrences of "Violation:" *after* any truncation
        vi_index = self.compute_two_violation_end_indices(
            rule_variant_ids=rule_variant_ids,
            support_ids=comment_train,
            total_length=input_ids.numel(),
        )

        return input_ids, vi_index


# ---------------------------------------------------------------------------
# Data1: Rule-based example aggregation
# ---------------------------------------------------------------------------
"""Dataset & DataLoader for Reddit rule-violation classification - Data1.

For every row in the provided DataFrame we create a prompt that follows the
`ttt_design.md` template:

    You are given a comment on reddit. Your task is to classify if it violates the given rule.
    Rule: <rule text>
    Comment: <support example>
    Violation: <Yes|No>
    Comment: <target comment>
    Violation:

 One labelled *support* example (randomly positive or negative) is sampled for
 the same rule.  The unlabelled *target* comment
(the row's own *body*) is appended last – the model must produce an answer
right after the final "Violation:" token.  To facilitate that, this dataset
returns the *position* (index) of the first token of the **last** "Violation:"
string in the tokenised sequence.

The caller can then gather the model's hidden states / logits at this position
to train a classifier or compute loss directly.
"""

class TTTDataset_map(TTTDatasetBase, Dataset):
    """
    Important Note: only works for num_workers = 1, as _sample_from_state is stateful and each worker start
    from the same cursor.
    Parameters
    ----------
    df
        *Cleaned* DataFrame containing at least the following columns::
            ["rule", "body", "positive_example_1", "negative_example_1"]
        Additional columns (e.g. `positive_example_2`) are ignored but allowed. Any
        `subreddit` column, if present, is ignored in prompt construction.

    grouped_examples
        Mapping produced by ``utility.group_examples_by_rule`` (or an equivalent
        function).  The structure must be::

            {rule_text: {"positives": List[str], "negatives": List[str]}}

        The positive / negative pools are used to randomly sample *support*
        examples for each datum.

    tokenizer
        Any HuggingFace *PreTrainedTokenizer* instance compatible with your
        language model.

    max_length
        Sequence length after padding / truncation.  Defaults to 512.

    Notes
    -----
    •  No heavy preprocessing is performed here – we assume `df` has already
       been cleaned / normalised.
    •  Each row is yielded twice per epoch: once with a positive support and
       once with a negative support. Which one appears first is chosen per-row
       at initialisation time.
    •  Support examples are drawn using a simple per-rule cyclic sampler with
       in-place shuffling on cycle, providing stable coverage across epochs.
    """

    violation_str: str = "Violation:"

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        rule_variants: Dict[str, List[str]],
        return_target: bool = False,
        grouped_examples = None, # deprecated, keep for backward compatibility
        old_to_new: torch.Tensor | None = None,
        Is_DEBUG: bool = False, # mute all randomization if True    
        max_length: int = 2048,
        return_weights: bool = False,
        random_truncate: bool = False,
    ) -> None:
        super().__init__(max_length)
        self.df = df.sort_values('body', key=lambda s: s.str.len()).reset_index(drop=True)
        self.tokenizer = tokenizer
        self._old_to_new = old_to_new
        self.Is_DEBUG = Is_DEBUG
        self.return_target = return_target
        self.return_weights = return_weights
        self.random_truncate = random_truncate

        # Pre-encode static prompt fragments for consistency with iter dataset
        # Note: map dataset still builds full prompt as text below; this keeps API aligned
        # Pre-tokenize static fragments and unique rules present in df (vectorized)
        unique_rules = df["rule"].astype(str).unique().tolist() if rule_variants is None else rule_variants
        self.pretokenize_ttt_fragments(rules_to_tokenize=unique_rules)

        self._sampler_state = None
        self._first_positive_flags = None
        self._expanded_len = len(self.df)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self._expanded_len

    def __getitem__(self, idx: int):  # noqa: D401
        row = self.df.iloc[idx]
        rule = row["rule"]
        rule_variant_ids = random.choice(self._rule_variants_ids[rule.lower()])
        comment_single = self._enc(row["body"])  # target comment only
        input_ids, _ = self.assemble_prompt_single(
            rule_variant_ids=rule_variant_ids,
            comment=comment_single,
            random_truncate=self.random_truncate,
        )
        if self.return_target: # training mode
            if self.return_weights:
                return row["target"], input_ids, row["weight"]
            else:
                return row["target"], input_ids
        else:
            return row["row_id"], input_ids



class FlattenedGroupedDataset(TTTDatasetBase, Dataset):
    """Map-style dataset over flattened grouped examples with single-comment prompts.

    Inputs are grouped by rule and polarity (positives/negatives). This class
    flattens across rules and both polarities, sorts by the example token length,
    and yields single-comment prompts constructed via ``assemble_prompt_single``.

    __getitem__ returns a tuple ``(rule: str, polarity: int, input_ids: Tensor)``.
    """

    def __init__(
        self,
        grouped_examples: Dict[str, Dict[str, List[List[int]]]],
        tokenizer,
        rule_variants: Dict[str, List[str]],
        max_length: int = 2048,
        random_truncate: bool = False,
    ) -> None:
        super().__init__(max_length)
        self.tokenizer = tokenizer
        self.grouped_examples = grouped_examples
        self.random_truncate = random_truncate
        # Pre-tokenize static fragments and a single variant per unique rule
        unique_rules = list(grouped_examples.keys()) if rule_variants is None else rule_variants
        self.pretokenize_ttt_fragments(rules_to_tokenize=unique_rules)
        per_rule_logits = defaultdict(float) # rule -> averaged logit
        
        # Flatten: (rule, polarity, example_token_ids)
        flat_items: List[Tuple[str, int, List[int]]] = []
        for rule, pools in grouped_examples.items():
            pos_count = len(pools["positives"])
            neg_count = len(pools["negatives"])
            per_rule_logits[rule] = math.log(pos_count / neg_count) if pos_count > 0 and neg_count > 0 else 0.0
            for polarity in ("positives", "negatives"):
                examples = pools.get(polarity, [])
                for ex in examples:
                    flat_items.append((rule, 1 if polarity == "positives" else 0, ex))

        # Sort by raw token length (comment only) ascending for efficient batching
        flat_items.sort(key=lambda t: len(t[2]))
        self._flat: List[Tuple[str, int, List[int]]] = flat_items
        self._per_rule_logits = per_rule_logits

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> Tuple[str, int, torch.Tensor]:
        rule, polarity, token_ids_list = self._flat[idx]
        rule_variant_ids = random.choice(self._rule_variants_ids[rule.lower()])
        comment = torch.tensor(token_ids_list)
        input_ids, _ = self.assemble_prompt_single(
            rule_variant_ids=rule_variant_ids,
            comment=comment,
            random_truncate=self.random_truncate,
        )
        return rule, polarity, input_ids

def collate_fn(batch: List[Tuple[str, int, torch.Tensor]], pad_token_id: int = 1):
    rule_list, polarity_list, input_id_list = zip(*batch)
    batch_size = len(input_id_list)
    if batch_size == 1:
        ids0 = input_id_list[0]
        last_indices = torch.tensor([ids0.numel() - 1], dtype=torch.long)
        return ids0.unsqueeze(0), last_indices, list(rule_list), list(polarity_list)
    lengths = [t.numel() - 1 for t in input_id_list]
    max_len = max(lengths) + 1
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(input_id_list):
        padded[i, : ids.numel()] = ids
    last_indices = torch.tensor(lengths, dtype=torch.long)
    return padded, last_indices, list(rule_list), list(polarity_list)

def _collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]], pad_token_id = 1):
    ''' used in Map-style dataset '''
    row_id, input_id_list = zip(*batch)
    batch_size = len(input_id_list)
    if batch_size == 1:
        ids0 = input_id_list[0]
        last_indices = torch.tensor([ids0.numel() - 1], dtype=torch.long)
        return ids0.unsqueeze(0), last_indices, list(row_id)

    lengths = [t.numel() - 1 for t in input_id_list]
    max_len = max(lengths) + 1
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(input_id_list):
        padded[i, : ids.numel()] = ids
    last_indices = torch.tensor(lengths, dtype=torch.long)
    return padded, last_indices, list(row_id)

def group_examples_by_rule(
    df,
    include_body=False,
    tokenizer=None,
    augment_func=None,
    cross_rule_sample_pct: float = 0.0,
    cross_rule_source: str = "neg", # "pos" or "neg"
) -> Dict[str, Dict[str, List[str] | List[List[int]]]]:
    """Return deduplicated positive/negative lists per rule without any I/O or heavy normalisation.

    Parameters
    ----------
    df : pandas.DataFrame
        Pre-cleaned DataFrame containing the Data1 training rows.
        (Assumed to already include the relevant columns and be cleaned.)
    include_body : deprecated, kept for backward compatibility
    cross_rule_sample_pct : float, optional
        Ratio in [0, 1] for how many examples to sample from other rules,
        relative to the current rule's negative count: k = floor(pct * |neg|).
    cross_rule_source : str, optional
        "pos" or "neg". Items sampled from other rules are always added to the
        current rule's negatives. Defaults to "neg".

    Returns
    -------
    dict
        Mapping ``{rule_text: {"positives": [...], "negatives": [...]}}``.
    """

    # Column names for positive and negative example sets
    pos_cols = ["positive_example_1", "positive_example_2"]
    neg_cols = ["negative_example_1", "negative_example_2"]

    def _collect(series_list):
        """Collapse a list of Series into unique values."""
        combined = pd.concat(series_list, ignore_index=True)
        return combined.unique().tolist()

    def _encode(text: List[str]) -> List[List[int]]:
        return tokenizer.batch_encode_plus(text, add_special_tokens=False)["input_ids"]
    
    # First pass: collect base positives/negatives per rule (deduplicated)
    base: Dict[str, Dict[str, List[str]]] = {}

    for rule, group in df.groupby("rule", sort=False):
        rule = str(rule)

        # Build series lists for positive and negative examples
        pos_series_list = [group[c] for c in pos_cols]
        neg_series_list = [group[c] for c in neg_cols]

        # Collect and deduplicate once per group
        pos_examples = _collect(pos_series_list)
        neg_examples = _collect(neg_series_list)

        base[rule] = {"positives": pos_examples, "negatives": neg_examples}

    # Optional: cross-rule sampling to augment negatives of each rule
    if cross_rule_sample_pct > 0:
        ratio = float(cross_rule_sample_pct)
        # normalize source to strict 'pos' or 'neg'
        src = cross_rule_source
        rules = list(base.keys())
        for rule in rules:
            current_neg = base[rule]["negatives"]
            neg_len = len(current_neg)
            k = int(neg_len * ratio)
            if src == "pos":
                pool = [ex for r, d in base.items() if r != rule for ex in d["positives"]]
            else:
                pool = [ex for r, d in base.items() if r != rule for ex in d["negatives"]]
            if k > len(pool):
                k = len(pool)
            sampled = random.sample(pool, k)
            # Extend negatives in place
            current_neg.extend(sampled)

    # Optional augmentation before tokenization (applied after cross-rule sampling)
    if augment_func is not None:
        for rule in base:
            pos_examples = base[rule]["positives"]
            neg_examples = base[rule]["negatives"]
            pos_aug = [augment_func(example) for example in pos_examples]
            neg_aug = [augment_func(example) for example in neg_examples]
            # flatten the list
            pos_aug = [item for sublist in pos_aug for item in sublist]
            neg_aug = [item for sublist in neg_aug for item in sublist]
            pos_examples.extend(pos_aug)
            neg_examples.extend(neg_aug)

    # Tokenize if requested
    if tokenizer is not None:
        for rule in base:
            base[rule]["positives"] = _encode(base[rule]["positives"])
            base[rule]["negatives"] = _encode(base[rule]["negatives"])

    return base

def build_dataloader_map(
    df: pd.DataFrame,
    tokenizer,
    rule_variants: Dict[str, List[str]],
    shuffle: bool = False,
    pin_memory: bool = True,
    include_body: bool = False,
    grouped_examples: Dict[str, Dict[str, List[str]]] | None = None,
    Is_DEBUG: bool = False,
    batch_size: int = 4,
    random_truncate: bool = False,
) -> DataLoader:
    """Return a ready-to-use PyTorch ``DataLoader`` for TTT training.
    
    Parameters
    ----------
    include_body : bool, optional
        If True, include the 'body' column content in the positive/negative lists
        based on the 'rule_violation' values. Defaults to False.
    """
    # Pass grouped_examples through unchanged; if None, dataset will switch to single-comment mode
    dataset = TTTDataset_map(
        df=df,
        rule_variants=rule_variants,
        grouped_examples=grouped_examples,
        tokenizer=tokenizer,
        Is_DEBUG=Is_DEBUG,
        random_truncate=random_truncate,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, # NOTE: num_workers=0 is required due to the stateful sampler
        pin_memory=pin_memory,
        collate_fn=_collate_fn
    )    
# ---------------------------------------------------------------------------
# New iterable TTTDataset implementation for rule-level sampling
# ---------------------------------------------------------------------------
class TTTDataset_iter(TTTDatasetBase, IterableDataset):
    """PyTorch `IterableDataset` that yields tokenised TTT prompts.

    Each sample is constructed on-the-fly from two dictionaries containing
    positive and negative examples per rule – one for training, one for
    hold-out evaluation.

    The expected structure of each dictionary is::

        {rule_text: {"positives": List[str], "negatives": List[str]}}

    Parameters
    ----------
    data_pair
        Tuple ``(train_dict, holdout_dict)`` with the structure described
        above.
    tokenizer
        Any HuggingFace *PreTrainedTokenizer* compatible with your language
        model.
    samples_per_epoch
        Number of prompts that an epoch of this dataset should yield.
    """

    violation_str: str = "Violation:"

    def __init__(
        self,
        train_dict: Dict[str, Dict[str, List[torch.Tensor]]],
        holdout_dict: Dict[str, Dict[str, List[torch.Tensor]]] | None,
        tokenizer,
        old_to_new: torch.Tensor | None = None,
        samples_per_epoch: int = 1000,
        max_length: int = 2048,
        use_rule_variants: bool = True,
        neg_sample_prob: float = 0.0,
    ) -> None:
        super().__init__(max_length)
        self.train_dict = train_dict
        self.holdout_dict = holdout_dict
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch
        self._old_to_new = old_to_new
        self.use_rule_variants = use_rule_variants
        self.neg_sample_prob = neg_sample_prob

        # Rules present in *both* splits – we only sample from these.
        self.rules: List[str] = list(train_dict.keys())

        # Pre-encode static fragments and rule variants once; optionally remap ids
        self.pretokenize_ttt_fragments(None if use_rule_variants else self.rules)
        # For index search, keep bare "Violation:" ids
        # self._violation_ids: torch.Tensor = self._violation_prompt_ids
        # Build simple per-rule per-pool orders and cursors.
        self._train_state = TTTDatasetBase._init_sampler_state(
            self.train_dict,
            shuffle=True
        )
        if self.holdout_dict is not None:
            self._holdout_state = TTTDatasetBase._init_sampler_state(
                self.holdout_dict,
                shuffle=True
            )
        else:
            self._holdout_state = None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self.samples_per_epoch

    def __iter__(self):
        """Yield a *finite* number of samples, each worker getting a distinct slice."""
        k = len(self.rules)
        for i in range(self.samples_per_epoch):
            if i % k == 0:
                random.shuffle(self.rules)
            # to ensure more uniform sampling over rules
            rule = self.rules[i % k]
            # sample support example (always from train)
            idx_train, comment_train, label_train = TTTDatasetBase._sample_from_state(
                self.train_dict,
                self._train_state,
                rule,
                "positives" if random.random() < 0.5 else "negatives",
                reshuffle_on_cycle=True,
                neg_sample_prob=self.neg_sample_prob,
            )

            # Randomly choose a pre-tokenised rule variant line for this rule
            rule_variant_ids = random.choice(self._rule_variants_ids[rule])
            comment_train = comment_train if isinstance(comment_train, torch.Tensor) else torch.tensor(comment_train)

            if self.holdout_dict is not None:
                # sample test example from holdout
                idx_holdout, comment_test, label_test = TTTDatasetBase._sample_from_state(
                    self.holdout_dict,
                    self._holdout_state,
                    rule,
                    "positives" if random.random() < 0.5 else "negatives",
                    reshuffle_on_cycle=True,
                )
                comment_test = comment_test if isinstance(comment_test, torch.Tensor) else torch.tensor(comment_test)

                # Assemble the full prompt and obtain end-indices for both "Violation:" occurrences
                input_ids, vi_index = self.assemble_prompt(
                    rule_variant_ids=rule_variant_ids,
                    comment_train=comment_train,
                    label_train=label_train,
                    comment_test=comment_test,
                )

                labels = torch.tensor([label_train, label_test], dtype=torch.long)
                # (rule, label, idx) is used to track the test example in case of ensemble prediction
                yield (rule, label_test, idx_holdout), input_ids.unsqueeze(0), torch.tensor(vi_index), labels
            else:
                # Single-comment mode: only use the train example and end with one Violation:
                input_ids, vi_index = self.assemble_prompt_single(
                    rule_variant_ids=rule_variant_ids,
                    comment=comment_train,
                )
                labels = torch.tensor([label_train], dtype=torch.long)
                yield (rule, label_train, idx_train), input_ids.unsqueeze(0), torch.tensor(vi_index), labels


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
from pathlib import Path
def load_grouped_data(
    data_dir: str = "Data/grouped",
    load_in_token: bool = True,
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Load pre-generated grouped training data from disk.
    
    This function loads grouped data that was generated by generate_grouped_data.py.
    The loaded data maintains the shared negatives structure for memory efficiency.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing the grouped data files. Defaults to "Data/grouped".
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        A tuple containing (train_grouped, holdout_grouped) where each dict has:
        {rule_text: {"positives": [...], "negatives": shared_list}}
        
    Examples
    --------
    >>> train_data, holdout_data = load_grouped_data()
    >>> # Use with existing TTT pipeline
    >>> dataset = TTTDataset(some_df, train_data, tokenizer)
    """
    import pickle
    import os
    
    if load_in_token:
        train_path = os.path.join(data_dir, "train_grouped_token_ids.pkl")
        holdout_path = os.path.join(data_dir, "holdout_grouped_token_ids.pkl")
    else:
        train_path = os.path.join(data_dir, "train_grouped_final2.pkl")
        holdout_path = os.path.join(data_dir, "holdout_grouped_final2.pkl")
    
    # Load train data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    # Load holdout data
    with open(holdout_path, 'rb') as f:
        holdout_data = pickle.load(f)
    
    return train_data, holdout_data


def _clear_adapter(model, adapter_name: str = "current") -> None:
    if getattr(model, "active_adapter", None):
        model.disable_adapter()
    # remove previous adapter
    try:
        model.remove_adapter(adapter_name)
    except AttributeError:
        model.delete_adapter(adapter_name)

def _resolve_lora_and_head_paths(folder: Path) -> Tuple[Path, Path]:
	"""Return (lora_dir, lm_head_file) inside a given model folder.

	Expected layout:
	- <folder>/LoRA/
	- <folder>/lm_head.pth
	"""
	# Standardized
	std_lora = folder / "LoRA"
	std_head = folder / "lm_head.pth"
	if std_lora.is_dir() and std_head.is_file():
		return std_lora, std_head

	# Not found
	raise FileNotFoundError(
		f"Could not find LoRA+lm_head in '{folder}'. Expected 'LoRA/' and 'lm_head.pth'"
	)

def load_model_and_lm_head_from_folder(
	model,
	folder_path,
	device,
	is_trainable,
	lm_head,
	Is_base_model,
	torch_lib,
	peft_model_class,
	fast_language_model,
	adapter_name: str = "current",
):

	"""Load a single LoRA adapter and lm_head weights from a given folder.

	- Accepts either a base HF model or an already PEFT-wrapped model.
	- Does not create a new lm_head; weights are loaded into the provided module.
	- If given a base model, it is wrapped with PEFT using the provided LoRA.

	Returns (possibly wrapped model, lm_head)
	"""
	folder = Path(folder_path).expanduser().resolve()
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Folder not found: {folder}")

	# Resolve paths
	lora_dir, lm_head_file = _resolve_lora_and_head_paths(folder)

	# Load or attach adapter depending on model type
	if not Is_base_model:
		# Already PEFT-wrapped
		_clear_adapter(model, adapter_name=adapter_name)
		model.load_adapter(str(lora_dir), adapter_name=adapter_name, is_trainable=is_trainable)
		model.set_adapter(adapter_name)
	else:
		# Base model: wrap with PEFT
		model = peft_model_class.from_pretrained(
			model,
			str(lora_dir),
			adapter_name=adapter_name,
			is_trainable=is_trainable,
		)
		model.set_adapter(adapter_name)

	# Load lm_head weights into the provided module
	if lm_head is None:
		raise ValueError("lm_head must be provided; do not create it inside this function.")

	state = torch_lib.load(str(lm_head_file), map_location=device)
	lm_head.load_state_dict(state)
	lm_head.to(device)
	
	if is_trainable:
		fast_language_model.for_training(model)
	else:
		fast_language_model.for_inference(model)
	return model, lm_head


def iter_folder(root_dir: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")
    root_iter = root.iterdir()
    for sub in root_iter:
        if sub.is_dir():
            name = str(sub)
            yield name

def iter_models_and_lm_heads(
	model,
	root_dir: str | List[str],
	device,
	is_trainable,
	lm_head,
	torch_lib,
	peft_model_class,
	fast_language_model,
    is_base_model,
	adapter_name: str = "current",
):
	"""Iterate through a root directory and sequentially load each LoRA+lm_head.

	- Expected structure: subdirs named model1, model2, ... each with LoRA/ and lm_head.pth

	Yields (name, model, lm_head). The same model instance is reused and may be PEFT-wrapped
	on the first successful load.
	"""
	if isinstance(root_dir, str):
		root = Path(root_dir).expanduser().resolve()
		if not root.exists() or not root.is_dir():
			raise FileNotFoundError(f"Root not found: {root}")
		root_iter = root.iterdir()
	else:
		root_iter = root_dir

	# Iterate over all subdirectories
	for sub in root_iter:
		if sub.is_dir():
			name = str(sub)
			model, lm_head = load_model_and_lm_head_from_folder(
				model,
				name,
				adapter_name=adapter_name,
				device=device,
				is_trainable=is_trainable,
				lm_head=lm_head,
				Is_base_model=is_base_model,
				torch_lib=torch_lib,
				peft_model_class=peft_model_class,
				fast_language_model=fast_language_model,
			)
			is_base_model = False
			yield name, model, lm_head

def _sanitize_path_component(name: str) -> str:
    """Return a filesystem-safe single path component derived from ``name``."""
    sanitized = re.sub(r"[^\w\-.]+", "_", str(name))
    return sanitized.strip("_")

def _compose_rule_model_folder(rule: str, model_name: str, output_root: str = ".") -> Path:
    """Compose the destination folder path for saving per-(rule, model) artifacts."""
    import os
    base = os.path.basename(str(model_name).rstrip("/"))
    folder_name = f"{_sanitize_path_component(rule)}__{_sanitize_path_component(base)}"
    return Path(output_root).expanduser().resolve() / folder_name

def save_lora_and_lm_head(
    model,
    lm_head,
    rule: str,
    model_name: str,
    output_root: str = ".",
    gpu_index: int | str | None = None,
):
    """
    Save the active LoRA adapter and ``lm_head`` under a stable layout:

        <output_root>/<sanitized_rule>__<sanitized_model_name>/
            ├─ LoRA/
            └─ lm_head.pth

    Returns the absolute folder path as a string.
    """
    import os
    base_root = Path(output_root).expanduser().resolve()
    if gpu_index is not None:
        base_root = base_root / str(gpu_index)
    folder = _compose_rule_model_folder(rule, model_name, base_root)
    folder.mkdir(parents=True, exist_ok=True)
    # Save adapter weights for the currently active adapter
    model.save_pretrained(str(folder), selected_adapters=["current"])
    # change folder name from current to LoRA
    os.rename(os.path.join(str(folder), "current"), os.path.join(str(folder), "current").replace("current", "LoRA"))
    # Save linear head weights
    torch.save(lm_head.state_dict(), str(folder / "lm_head.pth"))

def load_lora_and_lm_head_from_rule_modelname(
    model,
    lm_head,
    rule: str,
    model_name: str,
    device,
    is_trainable: bool,
    torch_lib,
    peft_model_class,
    fast_language_model,
    output_root: str = ".",
    is_base_model: bool = True,
    adapter_name: str = "current",
    gpu_index: int | str | None = None,
):
    """Wrapper that loads adapter + head saved by ``save_lora_and_lm_head``.

    Returns (model, lm_head) with adapter attached and head weights loaded.
    """
    base_root = Path(output_root).expanduser().resolve()
    if gpu_index is not None:
        base_root = base_root / str(gpu_index)
    folder = _compose_rule_model_folder(rule, model_name, base_root)
    return load_model_and_lm_head_from_folder(
        model,
        folder,
        device,
        is_trainable,
        lm_head,
        is_base_model,
        torch_lib,
        peft_model_class,
        fast_language_model,
        adapter_name=adapter_name,
    )
