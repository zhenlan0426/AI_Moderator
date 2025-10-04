# AI_Moderator created with help of AI assistant

## Objective
The task is to create a binary classifier that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, with a range of subreddit norms, tones, and community expectations. The main challenge is to build a model that can generalize across different rules and contexts.

## Model Fine-tuning
Pre-trained LLM Qwen3 1.5B. Further fine-tuned on diverse Reddit comment datasets with QLoRA. Model distillation is used to learn subtle difference from soft target given by openAI moderation model via API. 

## Model inference
At inference time, a few dozen comments are available as few shot examples for Kaggle competition. Test-time-training (TTT) is used to further fine-tune the model to better adapt to the specific rules and test distribution. Self-supervised techniques such as Predictive entropy minimization and psuedo labeling are iteratively applied to further improve the model performance. At each iteration, ensembles of prior predictions are used to identify the most confident predictions to be used as pseudo labels and least confident predictions to be used for predictive entropy minimization. This results in a significant performance boost, with 0.917 AUC on public leaderboard (priviate leaderboard not available yet as the competition is ongoing).
