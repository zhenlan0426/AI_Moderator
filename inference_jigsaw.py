import gc
import subprocess
import sys

local_dir = "/kaggle/input/unsloth/unsloth"  # Use an absolute path for reliability

def install_unsloth(local_dir):
    # install unsloth
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "unsloth",
            "--no-index",
            f"--find-links={local_dir}"
        ],
        check=True,
        capture_output=True,
        text=True
    )

def fix_unsloth():
    # patch unsloth to avoid TRL errors
    import os, sys, importlib.util
    
    # 1) Try to locate the exact path of unsloth.models.rl via importlib
    unsloth_path = None
    try:
        spec = importlib.util.find_spec("unsloth.models.rl")
        if spec and spec.origin and spec.origin != "built-in":
            if os.path.exists(spec.origin):
                unsloth_path = spec.origin
    except Exception:
        pass

    # 2) Fallback: search common site/dist-packages entries on sys.path
    if not unsloth_path:
        for path in sys.path:
            if ("site-packages" in path) or ("dist-packages" in path):
                test_path = os.path.join(path, "unsloth", "models", "rl.py")
                if os.path.exists(test_path):
                    unsloth_path = test_path
                    break
    
    if not unsloth_path:
        raise ValueError("Could not find unsloth rl.py file")
    
    with open(unsloth_path, 'r') as f:
        original_content = f.read()
    
    # Add early return to PatchFastRL function
    target_string = "def PatchFastRL(algorithm = None, FastLanguageModel = None):"
    replacement_string = "def PatchFastRL(algorithm = None, FastLanguageModel = None):\n    return  # Early exit to prevent TRL errors"
    
    modified_content = original_content.replace(target_string, replacement_string)
    
    # Check if any replacement was made
    if original_content != modified_content:
        with open(unsloth_path, 'w') as f:
            f.write(modified_content)
        print("✓ Patched PatchFastRL with early return")
    else:
        print("⚠ No replacement made - PatchFastRL function signature not found or already patched")
        raise ValueError("PatchFastRL function signature not found")
import random
import os
from typing import Dict, List
import multiprocessing as mp  # use stdlib to avoid importing torch in child processes
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import sys
from utility import normalize_text_columns, build_dataloader_map
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable static KV cache (major VRAM saver for FastModel)
os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
# Optional: if you still see high VRAM, disable Unsloth compile passes
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
sys.path.append('/kaggle/usr/lib/utility')
# Add configurable environment flag: set IS_LOCAL=1 to run local single-GPU inference.
IS_LOCAL = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None
# for nltk data path
os.environ['NLTK_DATA'] = '/kaggle/input/nltk-tagger/nltk_data'
if not IS_LOCAL:
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
Is_DEBUG = False

# hyperparameters
lr = 1.5e-4
clip = 1.0
accumulation_steps = 16
epochs = 2000 # if <10, fixed number of epochs; if >=10, fixed number of iterations
onecycle_warmup_pct = 0.099572
cycle_momentum = False
loss_bce_prob = 1 # prob for bce loss, 1 - prob for huber loss
label_smoothing = 0.072503 # cannot be 0.0 for huber loss!
ss_weight = 0.0 # weight of semantic search logits in the final score
embedding_normalize = True # normalize embeddings before semantic search
ss_activate_fn = "softmax" # "sigmoid" or "softmax" or "tanh" or "none"
temperature = 0.002272 # temperature for semantic search similarity
delta = 2.432148 # for huber loss
NORMALIZE_PER_RULE = True
DIVIDE_BY_STD = True # set True to divide by per-rule std after mean-centering
EPS_STD = 1e-8
weight_decay = 0.174505 # for AdamW
n_aug, nmin, nmax, aug_p = 1, 1, 5, 0.054145 # for data augmentation
max_uncertainty_iters = 4 # 7 hours on Kaggle
uncertain_percent = [1.0, ] # ,0.85,0.5,0.25
aggregate_fn_str = "median" # "mean" or "median" or "trimmed_mean", with max_uncertainty_iters = 1, all are the same.
EARLY_STOP_PATIENCE_RATIO = 0.146658 # check performance every EARLY_STOP_PATIENCE_RATIO * total_steps
DEFAULT_MODEL = "/rule_group_combined"
row_id_to_list_NORMALIZE = True
start_train_layer = 0
pred_entropy_percent = 0
pred_entropy_lr_scale = 0.1
certain_percent = 0
PL_lr_scale = 0.1
normalize_text = False
cross_rule_sample_pct = 0.224249 # 20% of negatives from other rules
cross_rule_source = "neg" # "pos" or "neg"
global_subset = False # for uncertain_df, pred_entropy_df, certain_df, whether to use global sorting or per-rule sorting
autoregressive_weight = 0.1
# hyperparameters
ss_weight *= max_uncertainty_iters # ss only done once while pure model done max_uncertainty_iters times
if not global_subset:
    pred_entropy_percent /= (2 if IS_LOCAL else 6)
    certain_percent /= (2 if IS_LOCAL else 6)

if IS_LOCAL:
    # ---------------------------------------------------------------------
    # Local paths (edit via environment variables if you store models elsewhere)
    # ---------------------------------------------------------------------
    # model_name = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
    model_name = "unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit"
    lora_dir   = "Model/10_11_12"
    # data_path = "combined_datasets_with_weights_formatted.csv"
    data_path = "Data/Data1/combined_train_examples_formatted06.csv"
    aug_src_path="Data/ppdb-2/ppdb-2.0-s-all"
else:
    # ---------------------------------------------------------------------
    # Kaggle competition paths – unchanged
    # ---------------------------------------------------------------------
    # model_name = "/kaggle/input/basemodel/transformers/default/1/base_model"
    model_name = "/kaggle/input/basemodel/transformers/default/2/base_model"
    lora_dir   = "/kaggle/input/lora-jigsaw"
    if Is_DEBUG:
        data_path = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
    else:
        data_path = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
    # rule_gen_path = "/kaggle/input/qwen-3/transformers/32b-awq/1"
    aug_src_path = "/kaggle/input/ppdb-2-0-s-all/ppdb-2.0-s-all"

def normalize_predictions_per_rule(sub: pd.DataFrame, pred_col: str, rule_col: str, divide_by_std: bool = False, eps: float = 1e-8, search: bool = False) -> pd.DataFrame:
    """Return a new DataFrame with pred_col mean-centered per rule and optionally scaled by std.
    Requires rule_col to exist in sub. Does not modify other columns.
    """
    stats = sub.groupby(rule_col)[pred_col].agg(mean='mean', std='std')
    sub = sub.join(stats, on=rule_col)
    sub[pred_col] = sub[pred_col] - sub['mean']
    if search:
        sub["yhat2"] = sub[pred_col] / sub['std']
        return sub
    if divide_by_std:
        denom = sub['std'].replace(0.0, eps)
        sub[pred_col] = sub[pred_col] / denom
    drop_cols = ['mean', 'std']
    if drop_cols:
        sub = sub.drop(columns=drop_cols)
    return sub

def identify_easy_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify easy cases where body text exactly matches positive or negative examples.
    
    Args:
        df: DataFrame with columns ['row_id', 'body', 'rule', 'positive_example_1', 
            'positive_example_2', 'negative_example_1', 'negative_example_2']
    
    Returns:
        tuple of (easy_cases_df, hard_cases_df):
        - easy_cases_df: Cases where body matches examples with predictions
        - hard_cases_df: Cases that need model inference
    """
    pos_cols = ["positive_example_1", "positive_example_2"]
    neg_cols = ["negative_example_1", "negative_example_2"]
    easy_df = [] # [(row_id, prediction), ...] -> df
    easy_ids = set()

    def _collect(series_list):
        """Collapse a list of Series into unique values."""
        combined = pd.concat(series_list, ignore_index=True).unique().tolist()
        return {ex.strip().lower() for ex in combined}

    for rule, group in df.groupby("rule", sort=False):
        rule = str(rule)
        # Build series lists for positive and negative examples
        pos_series_list = [group[c] for c in pos_cols]
        neg_series_list = [group[c] for c in neg_cols]
        # Collect and deduplicate once per group
        pos_examples = _collect(pos_series_list)
        neg_examples = _collect(neg_series_list)

        for row_id, body in zip(group["row_id"], group["body"]):
            body = str(body).strip().lower()
            if body in pos_examples and body not in neg_examples:
                easy_df.append((row_id, 100.0))
                easy_ids.add(row_id)
            elif body in neg_examples and body not in pos_examples:
                easy_df.append((row_id, -100.0))
                easy_ids.add(row_id)
    easy_cases_df = pd.DataFrame(easy_df, columns=["row_id", "rule_violation"])
    if easy_cases_df.shape[0] > 0:
        hard_cases_df = df.loc[~df["row_id"].isin(easy_ids)].reset_index(drop=True)
    else:
        hard_cases_df = df
    return easy_cases_df, hard_cases_df
    
# Heavy libraries are imported *inside* this function after GPU masking so that
# each spawned worker only sees its assigned GPU.
import re
def freeze_early_lora_layers(model, start_train_layer: int, layer_regex= r"\.layers\.(\d+)\."):
    """
    Freeze LoRA params for layers with index < start_train_layer.
    Works with module names like: base_model.model.model.layers.30.self_attn.q_proj.lora_A.weight
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        m = re.search(layer_regex, name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        if layer_idx < start_train_layer:
            param.requires_grad_(False)

def _infer_on_split(
    split_df: pd.DataFrame,
    gpu_index: int,
    model_name: str,
    lora_dir: str,
    grouped_examples: Dict[str, Dict[str, List[List[int]]]],
    rule_variants: Dict[str, List[str]],
    Is_toy: bool,
    loss_type: str,
    pred_entropy_df: pd.DataFrame | None = None,
    need_ss: bool = True,
    certain_df: pd.DataFrame | None = None,
    Is_DEBUG: bool = False,
):
    """Worker process: run inference on one GPU and return per-row aggregates.
    Args:
        split_df: DataFrame with test data split for this GPU.
        gpu_index: Index of the GPU to bind this process to.
        model_name: Path to the base model.
        lora_dir: Path to the LoRA weights directory.
        rule_variants: Dictionary mapping each original rule to its generated variants
        grouped_examples: Dictionary mapping each original rule to its generated variants with tokenization
    Returns:
        Dict[int, List[float]]: Mapping from row_id to list of logits
    """
    # ------------------------------------------------------------------
    # GPU masking *must* happen before importing torch / CUDA libraries.
    # ------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import random
    random.seed(gpu_index)
    # Local (inside-worker) imports: no heavy CUDA initialisation happened yet.
    from unsloth import FastModel, FastLanguageModel
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import OneCycleLR
    from utility import TTTDataset_map, FlattenedGroupedDataset, load_model_and_lm_head_from_folder
    from torch.nn.utils import clip_grad_norm_
    from utility import collate_fn,_collate_fn
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    # Select the (sole) visible GPU as index 0 and prepare dtype
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    AMP_DTYPE = torch.bfloat16 if IS_LOCAL else torch.float16

    # base model
    model, tokenizer = FastModel.from_pretrained(
        model_name,
        load_in_4bit = True,
        device_map={"": 0},            # 0 is correct after CUDA_VISIBLE_DEVICES masking
    )
    lm_head = torch.nn.Linear(model.lm_head.in_features, 1, bias=True)
    ############## model assignment: map each LoRA folder to matching rules ##############
    if IS_LOCAL:
        default_model_dir = lora_dir + "/model2"
    else:
        default_model_dir = lora_dir + DEFAULT_MODEL
    # illegal_activity_model_dir = lora_dir + "/model2"
    model_rule = defaultdict(list)
    for rule in grouped_examples.keys():
        model_rule[default_model_dir].append(rule)
    Is_base_model = True
    ###############################################################################################
    ##################### iterate over models and rules to train and infer ########################
    ###############################################################################################
    def train_step(input_ids, length, labels, model, lm_head, optimizer, scheduler, loss_fn, i, tot_len):
        with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
            input_ids, length, labels = input_ids.to('cuda'), length.to('cuda'), torch.tensor(labels).to('cuda')
            labels = labels * (1 - label_smoothing) + (1 - labels) * label_smoothing
            if loss_type == "huber":
                labels = torch.log(labels / (1 - labels))
            output = model.base_model.model.model(input_ids)            
            # Original classification loss at final token
            logits = lm_head(output.last_hidden_state[torch.arange(input_ids.shape[0]), length]) # (# of Violation, 4096) @ (4096, 2) -> (# of Violation, 2)
            classification_loss = loss_fn(logits.squeeze(1), labels) # first token is used for training
            
            # Autoregressive loss at all tokens
            # Shift targets: predict next token at each position
            shift_logits = model.lm_head(output.last_hidden_state[:, :-1, :])  # (batch, seq_len-1, vocab_size) TODO: check model.lm_head
            shift_labels = input_ids[:, 1:]  # (batch, seq_len-1)
            
            # Flatten for cross-entropy loss
            shift_logits = shift_logits.contiguous().view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.contiguous().view(-1)
            
            # Cross-entropy loss for autoregressive training
            autoregressive_loss = torch.nn.functional.cross_entropy(
                shift_logits, 
                shift_labels, 
                ignore_index=pad_token_id, # TODO: collate_fn
            )
            
            # Combine losses (you can adjust the weight)
            loss = classification_loss + autoregressive_weight * autoregressive_loss

            train_loss = loss / accumulation_steps
            train_loss.backward()
            if ((i + 1) % accumulation_steps == 0) or (i == tot_len):
                clip_grad_norm_(trainable_params,clip)
                optimizer.step()
                try:
                    scheduler.step()
                except:
                    # dont do anything when overstepped because we did not count flushing steps
                    pass
                optimizer.zero_grad()
            return loss.detach().cpu().sum()

    def prediction_entropy_loss(logits: torch.Tensor):
        """
        Computes the prediction entropy loss (binary entropy) given logits.
        
        This is numerically stable and works element-wise on the input tensor.
        For a batch, you can average the result with .mean() if needed.
        
        Args:
            logits: Tensor of logits (any shape).
        
        Returns:
            Tensor of entropy values, same shape as logits.
        """
        abs_logits = torch.abs(logits)
        exp_neg_abs = torch.exp(-abs_logits)
        softplus_neg_abs = F.softplus(-abs_logits)
        entropy = softplus_neg_abs + abs_logits * exp_neg_abs / (1 + exp_neg_abs)
        return entropy.mean()

    row_id_to_list = defaultdict(list) # row_id -> [logit1, logit2, ...]
    loo_auc_rule = dict() # rule -> auc
    unique_rules = set(str(r).strip().lower() for r in split_df["rule"].unique())
    for model_dir, rules in model_rule.items():
        # training #
        for rule in rules:
            if rule.strip().lower() not in unique_rules: # skip if rule not in split_df for inference
                continue
            # load model and lm_head every time for different rules from LoRA folder
            model, lm_head = load_model_and_lm_head_from_folder(model, model_dir, device, is_trainable=True, lm_head=lm_head, torch_lib=torch, peft_model_class=PeftModel, fast_language_model=FastLanguageModel, Is_base_model=Is_base_model)
            freeze_early_lora_layers(model, start_train_layer=start_train_layer)
            Is_base_model = False
            # reset lm_head bias
            pos_count = len(grouped_examples[rule]["positives"])
            neg_count = len(grouped_examples[rule]["negatives"])
            torch.nn.init.constant_(lm_head.bias, math.log(pos_count / neg_count) if pos_count > 0 and neg_count > 0 else 0.0)
            lm_head.bias.requires_grad_(False)
            # training data for each rule
            rule_grouped_examples = {rule: grouped_examples[rule]}
            group_dataset = FlattenedGroupedDataset(rule_grouped_examples, tokenizer, rule_variants, random_truncate=True)
            train_dataloader = DataLoader(group_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
            
            trainable_params = [param for param in model.parameters() if param.requires_grad] + [param for param in lm_head.parameters() if param.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr = lr, weight_decay=weight_decay/len(group_dataset))
            loss_fn = torch.nn.BCEWithLogitsLoss() if loss_type == "bce" else torch.nn.HuberLoss(delta=delta)
            # Scheduler: compute total optimizer steps based on examples seen, not epochs
            micro_bs = train_dataloader.batch_size
            num_batches_per_epoch = len(train_dataloader)
            if epochs < 10:
                # We step optimizer at the end of each accumulation window and flush at end of each epoch
                steps_per_epoch = math.ceil(num_batches_per_epoch / accumulation_steps)
                total_steps = epochs * steps_per_epoch
            else:
                # Target is a fixed number of examples; break early when reached. No final flush.
                target_examples = 6 if Is_toy else epochs
                planned_batches = math.ceil(target_examples / micro_bs)
                total_steps = planned_batches // accumulation_steps
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=max(1, total_steps),
                pct_start=onecycle_warmup_pct,
                anneal_strategy="cos",
                cycle_momentum=cycle_momentum,
                div_factor=10,
                final_div_factor=100,
            )
            count = 0
            best_loss = float("inf")
            cur_loss = 0.0
            cur_examples = 0
            EARLY_STOP = False
            check_freq = max(1, int(total_steps * accumulation_steps * micro_bs * EARLY_STOP_PATIENCE_RATIO)) \
                            if isinstance(EARLY_STOP_PATIENCE_RATIO, float) else EARLY_STOP_PATIENCE_RATIO
            next_check = check_freq
            tot_len = len(train_dataloader) - 1
            if epochs < 10: # fixed number of epochs
                for _ in range(epochs):
                    for i, (input_ids, length, _, labels) in enumerate(train_dataloader):
                        count += len(labels)
                        cur_loss += train_step(input_ids, length, labels, model, lm_head, optimizer, scheduler, loss_fn, i, tot_len)
                        cur_examples += len(labels)
                        if count >= next_check:
                            if cur_examples > 0:
                                avg_loss = cur_loss / cur_examples
                                if avg_loss < best_loss:
                                    best_loss = avg_loss
                                    cur_loss = 0.0
                                    cur_examples = 0
                                    next_check += check_freq
                                else:
                                    EARLY_STOP = True
                                    break
                            else:
                                next_check += check_freq
                    if EARLY_STOP:
                        break
            else: # fixed number of iterations
                while True:
                    for i, (input_ids, length, _, labels) in enumerate(train_dataloader):
                        count += len(labels)
                        cur_loss += train_step(input_ids, length, labels, model, lm_head, optimizer, scheduler, loss_fn, i, tot_len)
                        cur_examples += len(labels)
                        if count >= next_check:
                            if cur_examples > 0:
                                avg_loss = cur_loss / cur_examples
                                if avg_loss < best_loss:
                                    best_loss = avg_loss
                                    cur_loss = 0.0
                                    cur_examples = 0
                                    next_check += check_freq
                                else:
                                    EARLY_STOP = True
                                    break
                            else:
                                next_check += check_freq
                        if count >= (6 if Is_toy else epochs):
                            break
                    if count >= (6 if Is_toy else epochs) or EARLY_STOP:                 
                        break

            # psuedo label training
            if certain_df is not None:
                PL_rule_df = certain_df[certain_df["rule"] == rule]
                if PL_rule_df.shape[0] > 0:
                    torch.nn.init.constant_(lm_head.bias, PL_rule_df['target'].mean())
                    PL_dataset = TTTDataset_map(PL_rule_df, tokenizer, rule_variants, random_truncate=True, return_target=True)
                    PL_dataloader = DataLoader(PL_dataset, batch_size=2, collate_fn=_collate_fn, shuffle=True)
                    trainable_params = [param for param in model.parameters() if param.requires_grad] + [param for param in lm_head.parameters() if param.requires_grad]
                    optimizer = torch.optim.AdamW(trainable_params, lr = lr * PL_lr_scale, weight_decay=weight_decay*PL_lr_scale/len(PL_dataset))
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                        for i, (input_ids, length, labels) in enumerate(PL_dataloader):
                            input_ids, length, labels = input_ids.to(device), length.to(device), torch.tensor(labels).to(device)
                            labels = torch.nn.functional.sigmoid(labels) # logits to sigmoid
                            output = model.base_model.model.model(input_ids)
                            hidden_states = output.last_hidden_state[torch.arange(input_ids.shape[0]), length]
                            logits = lm_head(hidden_states)
                            loss = loss_fn(logits.squeeze(1), labels)
                            train_loss = loss / accumulation_steps
                            train_loss.backward()
                            if (i + 1) % accumulation_steps == 0 or i == len(PL_dataloader) - 1:
                                clip_grad_norm_(trainable_params,clip)
                                optimizer.step()
                                optimizer.zero_grad()
            
            # prediction entropy training
            torch.nn.init.constant_(lm_head.bias, 0.0)
            if pred_entropy_df is not None:
                PE_rule_df = pred_entropy_df[pred_entropy_df["rule"] == rule]
                if PE_rule_df.shape[0] > 0:
                    PE_dataset = TTTDataset_map(PE_rule_df, tokenizer, rule_variants, random_truncate=True)
                    PE_dataloader = DataLoader(PE_dataset, batch_size=2, collate_fn=_collate_fn, shuffle=True)
                    trainable_params = [param for param in model.parameters() if param.requires_grad] + [param for param in lm_head.parameters() if param.requires_grad]
                    optimizer = torch.optim.AdamW(trainable_params, lr = lr * pred_entropy_lr_scale, weight_decay=weight_decay*pred_entropy_lr_scale/len(PE_dataset))
                    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                        for i, (input_ids, length, _) in enumerate(PE_dataloader):
                            input_ids, length = input_ids.to(device), length.to(device)
                            output = model.base_model.model.model(input_ids)
                            hidden_states = output.last_hidden_state[torch.arange(input_ids.shape[0]), length]
                            logits = lm_head(hidden_states)
                            loss = prediction_entropy_loss(logits)
                            train_loss = loss / accumulation_steps
                            train_loss.backward()
                            if (i + 1) % accumulation_steps == 0 or i == len(PE_dataloader) - 1:
                                clip_grad_norm_(trainable_params,clip)
                                optimizer.step()
                                optimizer.zero_grad()


            ############# inference #############
            FastLanguageModel.for_inference(model)
            model.eval()
            if need_ss:
                # semetic search: build source embedding
                eval_dataloader = DataLoader(group_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)
                tgt, src_embed = [], [] # tgt, embedding with shape (n,) (n,d)
                tgt01, _yhat = [], []
                for input_ids, length, _, labels in eval_dataloader:
                    with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE), torch.no_grad():
                        input_ids, length = input_ids.to('cuda'), length.to('cuda')
                        output = model.base_model.model.model(input_ids)
                        embedding = output.last_hidden_state[torch.arange(input_ids.shape[0]), length]
                        logits = lm_head(embedding)
                        _yhat.extend(logits.squeeze(1).detach().cpu().tolist())
                        tgt01.extend(labels)
                        tgt.extend([1.0 if label==1 else -1.0 for label in labels]) # semantic search target needs to be 1 or -1 not 0 or 1
                        src_embed.append(embedding.detach().cpu().float()) # [(4, d), ...]
                tgt = torch.tensor(tgt)
                src_embed = torch.cat(src_embed, dim=0) # (n,d)
                loo_auc_rule[rule] = roc_auc_score(tgt01, _yhat)

            # inference on test dataset
            df_rule = split_df.loc[split_df["rule"] == rule]
            dataloader = build_dataloader_map(
                df=df_rule,
                tokenizer=tokenizer,
                rule_variants=rule_variants, # TODO: test rule_variants,
                shuffle=False,
                pin_memory=True,
                include_body=False,
                grouped_examples=None,
                batch_size=8,
                random_truncate=True,
            )
            
            query_embedding = [] # [(row_id, embedding),...]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                for input_ids, length, row_ids in dataloader:
                    input_ids, length = input_ids.to(device), length.to(device)
                    output = model.base_model.model.model(input_ids)
                    
                    hidden_states = output.last_hidden_state[torch.arange(input_ids.shape[0]), length]
                    logits = lm_head(hidden_states)
                    # for row_id, logit, hidden_state in zip(row_ids, logits, hidden_states):
                    for row_id, logit, hidden_state in zip(row_ids, logits, hidden_states):
                        row_id_to_list[int(row_id)].append(logit[0].item())
                        # embedding for semantic search
                        query_embedding.append((int(row_id), hidden_state.detach().cpu().float()))
            
            if need_ss:
                # semantic search
                query_ids, query_embeddings = zip(*query_embedding) # (m,) (m,d)
                query_embeddings = torch.stack(query_embeddings, dim=0) # (m,d)
                # Normalize embeddings and compute cosine similarity instead of raw dot-product which caused saturation & limited score diversity
                if embedding_normalize:
                    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                    src_embed = torch.nn.functional.normalize(src_embed, p=2, dim=1)
                # Cosine similarity in [-1, 1]; scale with temperature then apply softmax so rows sum to 1
                similarities = (query_embeddings @ src_embed.T) / temperature  # (m, n)
                if ss_activate_fn == "sigmoid":
                    weights = torch.sigmoid(similarities)
                elif ss_activate_fn == "softmax":
                    weights = torch.softmax(similarities, dim=1)
                elif ss_activate_fn == "tanh":
                    weights = torch.tanh(similarities)
                else: # no activation function
                    weights = similarities
                # Weighted vote of +1 / -1 labels from the training set
                yhat = weights @ tgt                                  # (m,)
                # combine semantic search results with llm logits
                for _id, yhat_score in list(zip(query_ids, yhat.tolist())):
                    row_id_to_list[_id].append(ss_weight * yhat_score)
                # return per-example logits dict
                # # save query_embedding, src_embed, tgt for inference to disk
                # import pickle
                # with open(f"query_embedding_{rule}.pkl", "wb") as f:
                #     pickle.dump((query_ids, query_embeddings, src_embed, tgt), f)
                # # load query_embedding, src_embed, tgt for inference from disk
                # with open(f"query_embedding.pkl", "rb") as f:
                #     query_ids, query_embeddings, src_embed, tgt = pickle.load(f)
    if row_id_to_list_NORMALIZE and row_id_to_list:
        if need_ss and all(len(v) == 2 for v in row_id_to_list.values()):
            # Normalize across all rows for both prediction components (z-score per component)
            values_lists = list(row_id_to_list.values())
            comp = np.array(values_lists, dtype=np.float64)
            mean0, std0 = float(comp[:,0].mean()), float(comp[:,0].std())
            mean1, std1 = float(comp[:,1].mean()), float(comp[:,1].std())
            std0 = std0 if std0 > 0 else EPS_STD
            std1 = std1 if std1 > 0 else EPS_STD
            for row_id, vals in row_id_to_list.items():
                vals[0] = (vals[0] - mean0) / std0
                vals[1] = ss_weight * (vals[1] - mean1) / std1
        elif all(len(v) == 1 for v in row_id_to_list.values()):
            values_lists = list(row_id_to_list.values())
            comp = np.array(values_lists, dtype=np.float64)
            mean0, std0 = float(comp[:,0].mean()), float(comp[:,0].std())
            std0 = std0 if std0 > 0 else EPS_STD
            for row_id, vals in row_id_to_list.items():
                vals[0] = (vals[0] - mean0) / std0
    return row_id_to_list, loo_auc_rule

def mean_logits(logits):
    """
    logits: (number of predictions, 2 classes)
    """
    return sum(logits) / len(logits)

def median_logits(logits):
    """
    logits: (number of predictions, 2 classes)
    Median aggregator using sorting - more robust to outliers than mean
    """
    sorted_logits = sorted(logits)
    n = len(sorted_logits)
    
    if n % 2 == 1:
        # Odd number of elements - return middle element
        return sorted_logits[n // 2]
    else:
        # Even number of elements - return average of two middle elements
        mid1 = sorted_logits[n // 2 - 1]
        mid2 = sorted_logits[n // 2]
        return (mid1 + mid2) / 2

def trimmed_mean_logits(logits, k=1):
    """
    logits: (number of predictions, 2 classes)
    Trimmed mean aggregator - skips top k and bottom k elements, then averages
    """
    n = len(logits)
    
    # If we don't have enough elements to trim, fall back to regular mean
    if n <= 2 * k:
        return sum(logits) / len(logits)
    
    sorted_logits = sorted(logits)
    # Skip bottom k and top k elements
    trimmed = sorted_logits[k:-k]
    
    return sum(trimmed) / len(trimmed)
    
def aggregate_predictions(row_id_to_list: Dict[int, List[float]],
                            fn) -> Dict[int, float]:
    """Aggregate predictions for each row_id using the specified function.
    Args:
        row_id_to_list: Mapping from row_id to list of logits of shape (predictions, 2).
        fn: Function to aggregate logits (e.g., mean, max).
    Returns:
        Dict[int, float]: Mapping from row_id to aggregated prediction.
    """
    return [[int(row_id), fn(logits)] for row_id, logits in row_id_to_list.items()]


# def compute_uncertainty(row_id_to_list: Dict[int, List[float]]):
#     """
#     Compute uncertainty for each row_id using the specified function.
#     """
#     return {row_id: -abs(mean_logits(logits)) for row_id, logits in row_id_to_list.items()}
def compute_certainty(sub: pd.DataFrame):
    """
    Compute certainty for each row_id using the specified function.
    """
    return sub['yhat'].abs().values

def clear_cache():
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main entry point – executed only in the parent process (not in workers).  
# Heavy CUDA libraries are imported here because GPU masking is irrelevant for
# the *parent*; only child processes need strict isolation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    _start_time = time.time()
    mp.set_start_method("spawn", force=True)
    import unsloth
    print(f"------------------unsloth version: {unsloth.__version__}------------------")
    # Install here if not local
    if not IS_LOCAL:
        # install_unsloth(local_dir)
        fix_unsloth()
        dtypes = {"row_id": int,
                    "body": str,
                    "rule": str,
                    "subreddit": str,
                    "positive_example_1": str,
                    "positive_example_2": str,
                    "negative_example_1": str,
                    "negative_example_2": str,
                    }
    else:
        dtypes = {"row_id": int,
                    "body": str,
                    "rule": str,
                    "subreddit": str,
                    "positive_example_1": str,
                    "positive_example_2": str,
                    "negative_example_1": str,
                    "negative_example_2": str,
                    "rule_violation": float,
                    }
    test_df = pd.read_csv(
        data_path,
        usecols=list(dtypes.keys()),
        dtype=dtypes,    )
    if normalize_text:
        test_df = normalize_text_columns(test_df)
    
    # Filter easy cases before model inference
    easy_cases_df, hard_cases_df = identify_easy_cases(test_df)
    if easy_cases_df.shape[0] > 0:
        original_test_df = test_df.copy()
        test_df = hard_cases_df
    else:
        original_test_df = test_df

    Is_toy = test_df.shape[0] <= 10
    # generate rule variants and save to rule_variants.json
    rule_variants = {
        # 1. Advertising / spam
        "no advertising: spam, referral links, unsolicited advertising, and promotional content are not allowed.": [
            "no advertising: spam, referral links, unsolicited advertising, and promotional content are not allowed.",
            "Advertising of any kind—spam, referral codes, or promotional posts—is strictly forbidden.",
            "Please keep the feed ad-free; no promo links or unsolicited commercial content.",
            "No ads here, folks. That includes spam, shout-outs, and referral URLs.",
            "The community prohibits promotional material, including unsolicited advertising and spam links.",
            "Kindly avoid marketing content; referral links and promo spam will be removed."
        ],

        # 2. Legal advice
        "no legal advice: do not offer or request legal advice.": [
            "no legal advice: do not offer or request legal advice.",
            "We’re not lawyers—please don’t ask for or hand out legal counsel.",
            "Refrain from providing or seeking legal advice in this space.",
            "Need legal help? Consult a professional; legal recommendations are off-limits here.",
            "Offering or requesting legal guidance violates our rules.",
            "No courtroom tips or legal how-tos, thanks."
        ],

        # 3. Financial advice
        "no financial advice: we do not permit comments that make personal recommendations for investments, taxes, or careers.": [
            "no financial advice: we do not permit comments that make personal recommendations for investments, taxes, or careers.",
            "Financial recommendations (investments, tax moves, career picks) are disallowed.",
            "This isn’t a brokerage—skip the stock tips and personal finance directives.",
            "For money matters, see a pro; posting individualized financial advice breaks the rules.",
            "Please avoid dispensing or soliciting investment or tax guidance here.",
            "No ‘buy this coin’ or ‘quit that job’ advice, keep finances personal."
        ],

        # 4. Medical advice
        "no medical advice: do not offer or request specific medical advice, diagnoses, or treatment recommendations.": [
            "no medical advice: do not offer or request specific medical advice, diagnoses, or treatment recommendations.",
            "We’re not doctors—no asking for or giving medical diagnoses or treatment plans.",
            "Medical guidance, prescriptions, or diagnostic opinions are prohibited content.",
            "Feeling unwell? Consult a healthcare professional; medical advice is not allowed here.",
            "Providing or requesting specific medical recommendations violates community policy.",
            "No ‘take this pill’ or ‘you have X’ posts—medical talk stops here."
        ],

        # 5. Promotion of illegal activity
        "no promotion of illegal activity: do not encourage or promote illegal activities, such as drug-related activity, violence, exploitation, theft, or other criminal behavior.": [
            "no promotion of illegal activity: do not encourage or promote illegal activities, such as drug-related activity, violence, exploitation, theft, or other criminal behavior.",
            "Encouraging crime—drug deals, violence, theft, or any unlawful act—is banned.",
            "Posts that promote or facilitate illegal behavior are strictly disallowed.",
            "No ‘how to break the law’ guides or endorsements of wrongdoing.",
            "Content advocating illicit activities (violence, exploitation, etc.) will be removed.",
            "Urging others to commit unlawful acts contravenes community standards."
        ],

        # 6. Spoilers
        "no spoilers: do not reveal important details that would limit people's ability to enjoy a show or movie.": [
            "no spoilers: do not reveal important details that would limit people's ability to enjoy a show or movie.",
            "Keep spoilers to yourself—don’t ruin the plot for everyone else.",
            "Please hide major story details so others can enjoy the show spoiler-free.",
            "No plot reveals or ending leaks; let fellow fans watch in peace.",
            "Revealing key twists without a spoiler tag violates our guidelines.",
            "Avoid disclosing crucial storyline information that could spoil the experience."
        ],
    }
    # Decide how many GPUs/workers to use
    GPU_COUNT = 1 if IS_LOCAL else 2

    # data augmentation once for both GPUs
    from utility import NLPAUG_STOPWORDS
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.char as nac
    import nlpaug.flow as naf
    aug = naf.Sequential([  naw.SynonymAug(aug_src='ppdb', model_path=aug_src_path, aug_min=nmin, aug_max=nmax, aug_p=aug_p*15, stopwords=NLPAUG_STOPWORDS),
                            naw.SpellingAug(aug_min=nmin, aug_max=nmax, aug_p=aug_p, stopwords=NLPAUG_STOPWORDS),
                            nac.KeyboardAug(aug_char_min=nmin, aug_char_max=nmax, aug_char_p=aug_p, aug_word_p=aug_p, stopwords=NLPAUG_STOPWORDS),
                            nac.OcrAug(aug_char_min=nmin, aug_char_max=nmax, aug_char_p=aug_p, aug_word_p=aug_p, stopwords=NLPAUG_STOPWORDS),
                            ])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from utility import group_examples_by_rule, NLPAUG_STOPWORDS
    from sklearn.metrics import roc_auc_score
    if IS_LOCAL:
        # logging
        from logging_utils import ExperimentLogger, extract_hyperparameters_from_globals
        logger = ExperimentLogger()
        hyperparams = extract_hyperparameters_from_globals()
        logger.log_hyperparameters(hyperparams)
        source_files = [
            "inference_jigsaw.py",
            "utility.py"
        ]
        logger.copy_source_files(source_files)

    row_logits_dict = defaultdict(list)
    iter_loo_auc_rule_accum = defaultdict(list)
    tot_n = test_df.shape[0]
    aggregate_fn = mean_logits if aggregate_fn_str == "mean" else (median_logits if aggregate_fn_str == "median" else trimmed_mean_logits)
    _unique_rules = test_df['rule'].unique()
    for iter_idx in range(max_uncertainty_iters):
        clear_cache()
        # TODO: normalize predictions? per rule uncertainty?
        if iter_idx == 0:
            uncertain_df = test_df
            pred_entropy_df = None
            certain_df = None
        elif global_subset:
            certainties = compute_certainty(sub)
            sorted_idx = np.argsort(certainties) # ascending order
            sorted_ids = sub.iloc[sorted_idx]['row_id'].tolist()
            select_ids = sorted_ids[:int(tot_n * uncertain_percent[min(iter_idx, len(uncertain_percent) - 1)])]
            uncertain_df = test_df[test_df['row_id'].isin(select_ids)]

            if pred_entropy_percent > 0:
                select_ids = sorted_ids[:int(tot_n * pred_entropy_percent)]
                pred_entropy_df = test_df[test_df['row_id'].isin(select_ids)]
                pred_entropy_df.loc[:, 'body'] = pred_entropy_df['body'].apply(lambda x: aug.augment(x, n=1)[0])
            else:
                pred_entropy_df = None
            
            if certain_percent > 0:
                select_ids = sorted_ids[-int(tot_n * certain_percent):]
                certain_df = test_df[test_df['row_id'].isin(select_ids)]
                certain_df.loc[:, 'body'] = certain_df['body'].apply(lambda x: aug.augment(x, n=1)[0])
                # join with sub to get yhat (pseudo label)
                certain_df = certain_df.merge(sub[['row_id', 'yhat']], on='row_id', how='inner')
                certain_df = certain_df.rename(columns={'yhat': 'target'})
            else:
                certain_df = None
        else:
            # Loop over each rule to apply uncertainty selection per rule
            uncertain_dfs = []
            pred_entropy_dfs = []
            certain_dfs = []
            for rule in _unique_rules:
                # Get data for this specific rule
                rule_sub = sub[sub['rule'] == rule]
                rule_test_df = test_df[test_df['rule'] == rule]
                rule_n = len(rule_sub)
                if rule_n == 0:
                    continue
                
                # Compute certainties for this rule
                rule_certainties = compute_certainty(rule_sub)
                rule_sorted_idx = np.argsort(rule_certainties) # ascending order
                rule_sorted_ids = rule_sub.iloc[rule_sorted_idx]['row_id'].tolist()
                
                # Select uncertain samples for this rule
                rule_uncertain_count = int(rule_n * uncertain_percent[min(iter_idx, len(uncertain_percent) - 1)])
                if rule_uncertain_count > 0:
                    rule_uncertain_ids = rule_sorted_ids[:rule_uncertain_count]
                    rule_uncertain_df = rule_test_df[rule_test_df['row_id'].isin(rule_uncertain_ids)]
                    if len(rule_uncertain_df) > 0:
                        uncertain_dfs.append(rule_uncertain_df)
                
                # Select prediction entropy samples for this rule
                if pred_entropy_percent > 0:
                    rule_pred_entropy_count = int(rule_n * pred_entropy_percent)                  
                    rule_pred_entropy_ids = rule_sorted_ids[:rule_pred_entropy_count]
                    rule_pred_entropy_df = rule_test_df[rule_test_df['row_id'].isin(rule_pred_entropy_ids)]
                    if len(rule_pred_entropy_df) > 0:
                        rule_pred_entropy_df = rule_pred_entropy_df.copy()
                        rule_pred_entropy_df.loc[:, 'body'] = rule_pred_entropy_df['body'].apply(lambda x: aug.augment(x, n=1)[0])
                        pred_entropy_dfs.append(rule_pred_entropy_df)
                
                # Select certain samples for this rule
                if certain_percent > 0:
                    rule_certain_count = int(rule_n * certain_percent)
                    rule_certain_ids = rule_sorted_ids[-rule_certain_count:]
                    rule_certain_df = rule_test_df[rule_test_df['row_id'].isin(rule_certain_ids)]
                    if len(rule_certain_df) > 0:
                        rule_certain_df = rule_certain_df.copy()
                        rule_certain_df.loc[:, 'body'] = rule_certain_df['body'].apply(lambda x: aug.augment(x, n=1)[0])
                        # join with sub to get yhat (pseudo label) for this rule
                        rule_certain_df = rule_certain_df.merge(rule_sub[['row_id', 'yhat']], on='row_id', how='inner')
                        rule_certain_df = rule_certain_df.rename(columns={'yhat': 'target'})
                        certain_dfs.append(rule_certain_df)
            
            # Concatenate all per-rule dataframes
            uncertain_df = pd.concat(uncertain_dfs, ignore_index=True) if uncertain_dfs else None
            pred_entropy_df = pd.concat(pred_entropy_dfs, ignore_index=True) if pred_entropy_dfs else None
            certain_df = pd.concat(certain_dfs, ignore_index=True) if certain_dfs else None
        if GPU_COUNT == 1:
            # -------------------------------------------------------------
            # Single-GPU/local execution – no multiprocessing or splitting
            # -------------------------------------------------------------
            grouped_examples = group_examples_by_rule(
                original_test_df,
                include_body=False,
                tokenizer=tokenizer,
                augment_func=lambda x: aug.augment(x, n=n_aug, num_thread=4),
                cross_rule_sample_pct=cross_rule_sample_pct,
                cross_rule_source=cross_rule_source,
            )
            _logits_dict, _loo_auc_rule = _infer_on_split(
                uncertain_df,
                0,
                model_name,
                lora_dir,
                grouped_examples,
                rule_variants,
                Is_toy,
                "bce" if random.random() < loss_bce_prob else "huber",
                pred_entropy_df,
                False,
                certain_df,
            )
            for row_id, logits_list in _logits_dict.items():
                row_logits_dict[row_id].extend(logits_list)
            for rule, auc in _loo_auc_rule.items():
                iter_loo_auc_rule_accum[rule].append(auc)
            results = aggregate_predictions(row_logits_dict, aggregate_fn)
            # Calculate and log results
            sub = pd.DataFrame(results, columns=["row_id", "yhat"])
            sub = sub.merge(test_df[['row_id', 'rule_violation', 'rule']], on='row_id')
            
            print(f"------------------iterations {iter_idx}------------------")
            # Calculate per-rule AUC
            rule_results = {}
            for group, group_sub in sub.groupby('rule'):
                auc_score = roc_auc_score(group_sub['rule_violation'], group_sub['yhat'])
                rule_results[group] = auc_score
                print(f"{group}: {auc_score}")
            
            # Calculate overall AUC
            if NORMALIZE_PER_RULE:
                sub = normalize_predictions_per_rule(sub, pred_col='yhat', rule_col='rule', divide_by_std=DIVIDE_BY_STD, eps=EPS_STD)
            overall_auc = roc_auc_score(sub['rule_violation'], sub['yhat'])
            print(f"Overall: {overall_auc}")
            if overall_auc < 0.785:
                print(f"Overall AUC is too low: {overall_auc}, exiting")
                break
            # log iteration AUCs
            logger.log_iteration_aucs(iter_idx, rule_results, overall_auc)
        else:
            # -------------------------------------------------------------
            # Multi-GPU/Kaggle execution
            # -------------------------------------------------------------
            # train on entire dataset and score on test set twice with different random seed
            df_splits = [uncertain_df, uncertain_df]
            grouped_examples = [
                group_examples_by_rule(
                    original_test_df,
                    include_body=False,
                    tokenizer=tokenizer,
                    augment_func=lambda x: aug.augment(x, n=n_aug, num_thread=4),
                    cross_rule_sample_pct=cross_rule_sample_pct,
                    cross_rule_source=cross_rule_source,
                )
                for _ in range(2)
            ]
            args_list = [
                (
                    df_part,
                    gpu_idx,
                    model_name,
                    lora_dir,
                    grouped_examples[gpu_idx],
                    rule_variants,
                    Is_toy,
                    "bce" if random.random() < loss_bce_prob else "huber",
                    pred_entropy_df,
                    False,
                    certain_df,
                )
                for gpu_idx, df_part in enumerate(df_splits)
            ]

            with mp.Pool(processes=len(args_list)) as pool:
                worker_results = pool.starmap(_infer_on_split, args_list)
            # merge results
            for row_dict, loo_auc_rule_dict in worker_results:
                for row_id, logits_list in row_dict.items():
                    row_logits_dict[row_id].extend(logits_list)
                for rule, auc in loo_auc_rule_dict.items():
                    iter_loo_auc_rule_accum[rule].append(auc)
            results = aggregate_predictions(row_logits_dict, aggregate_fn)
            sub = pd.DataFrame(results, columns=["row_id", "yhat"])
            if NORMALIZE_PER_RULE:
                sub = sub.merge(test_df[['row_id', 'rule']], on='row_id', how='left')
                sub = normalize_predictions_per_rule(sub, pred_col='yhat', rule_col='rule', divide_by_std=DIVIDE_BY_STD, eps=EPS_STD)

    if IS_LOCAL:
        # Copy auc_rule.json artifact to the run directory if present
        from pathlib import Path
        import shutil
        auc_rule_path = Path("auc_rule.json")
        if auc_rule_path.exists():
            shutil.copy2(auc_rule_path, logger.get_run_directory() / "auc_rule.json")
        
        # Calculate and log results
        sub = pd.DataFrame(results, columns=["row_id", "yhat"])
        sub = sub.merge(test_df[['row_id', 'rule_violation', 'rule']], on='row_id')
        
        # Calculate per-rule Spearman correlations
        rule_results = {}
        for group, group_sub in sub.groupby('rule'):
            auc_score = roc_auc_score(group_sub['rule_violation'], group_sub['yhat'])
            rule_results[group] = auc_score
            print(f"{group}: {auc_score}")
        
        # Save raw per-row logits aggregation for analysis
        logger.log_row_logits(row_logits_dict)
        # Save per-iteration LOO AUCs per rule for analysis
        logger.log_iter_loo_auc_rule_accum(iter_loo_auc_rule_accum)
        
        # Calculate overall Spearman correlation
        if NORMALIZE_PER_RULE:
            sub = normalize_predictions_per_rule(sub, pred_col='yhat', rule_col='rule', divide_by_std=DIVIDE_BY_STD, eps=EPS_STD)

        overall_auc = roc_auc_score(sub['rule_violation'], sub['yhat'])
        print(f"Overall: {overall_auc}")
        logger.log_results(rule_results, overall_auc)
        # # search over demean / demean & de-std
        # sub = normalize_predictions_per_rule(sub, pred_col='yhat', rule_col='rule', divide_by_std=DIVIDE_BY_STD, eps=EPS_STD, search=True)
        # demean_auc = roc_auc_score(sub['rule_violation'], sub['yhat'])
        # print(f"Demean: {demean_auc}")
        # demeand_de_std_auc = roc_auc_score(sub['rule_violation'], sub['yhat2'])
        # print(f"Demean & De-std: {demeand_de_std_auc}")
        # # Log normalization comparison results
        # logger.log_normalization_results(demean_auc, demeand_de_std_auc)
        
        
        # Create summary
        # Log total runtime
        _total_seconds = time.time() - _start_time
        logger.log_total_runtime(_total_seconds)

        logger.create_summary(hyperparams, rule_results, overall_auc)
        
        print(f"\nExperiment logged to: {logger.get_run_directory()}")
    else:
        # -------------------------------------------------------------
        # Save submission file
        # -------------------------------------------------------------
        sub_final = sub.rename(columns={'yhat': 'rule_violation'})[['row_id', 'rule_violation']]
        if easy_cases_df.shape[0] > 0:
            sub_final = pd.concat([sub_final, easy_cases_df], axis=0, ignore_index=True)
        sub_final.to_csv("submission.csv", index=False)
