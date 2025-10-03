# AI_Moderator created with help of AI assistant

## Objective
The task is to create a binary classifier that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, with a range of subreddit norms, tones, and community expectations.

## Model
Pre-trained LLM Qwen3 1.5B. Further fine-tuned on diverse Reddit comment datasets with QLoRA. Model distillation is used to learn subtle difference from soft target given by openAI moderation model via API. 