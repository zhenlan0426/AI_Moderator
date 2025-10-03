"""OpenAI Moderation categories: column-keyed paraphrase variants.

This module defines paraphrased rule variants for OpenAI moderation categories,
keyed directly by dataset score column names.

Reference: https://platform.openai.com/docs/guides/moderation
"""
from __future__ import annotations

from typing import Dict, List


# Paraphrase variants: score column name -> list of 5 variants
OPENAI_MODERATION_COLUMN_VARIANTS: Dict[str, List[str]] = {
    # Harassment
    "harassment_score": [
        "No harassment or abusive behavior",
        "Do not harass, bully, or abuse others",
        "Harassment and abusive conduct are prohibited",
        "Avoid insulting, bullying, or targeted abuse",
        "Be respectful — no harassment or abuse",
        "Don't be a jerk — no bullying or pile-ons",
        "Keep it civil; no insults or dogpiling",
        "Be kind, not cruel — skip the harassment",
    ],
    "harassment_threatening_score": [
        "No harassment involving threats or violence",
        "Threatening or menacing harassment is not allowed",
        "Do not issue threats or imply harm toward others",
        "Violent or coercive harassment is forbidden",
        "No intimidation or threats of physical harm",
        "No 'I'll hurt you' vibes or scare tactics",
        "Don't threaten people, even as a joke",
        "No menacing, no 'or else' messages",
    ],

    # Hate
    "hate_score": [
        "No hate content targeting protected groups",
        "Do not attack people based on protected characteristics",
        "Hate against protected classes is prohibited",
        "Avoid slurs or demeaning content toward protected groups",
        "No hateful remarks about protected identities",
        "No hating on protected groups — not cool",
        "Cut the slurs; respect people's identities",
        "No targeting folks for who they are",
    ],
    "hate_threatening_score": [
        "No hateful threats or violence toward protected groups",
        "Advocating violence against protected classes is banned",
        "Do not threaten or incite harm toward protected identities",
        "Violent hate content is strictly prohibited",
        "No calls for harm against protected groups",
        "No threats or violence against protected groups, period",
        "Don't call for harm against anyone's identity group",
        "No 'go hurt them' hate",
    ],

    # Illicit
    # "illicit_score": [
    #     "No content facilitating or praising illegal activities",
    #     "Do not promote or celebrate illegal behavior",
    #     "Assistance or encouragement for unlawful acts is not allowed",
    #     "Avoid content that supports or glamorizes crime",
    #     "No guidance or praise for illegal activity",
    # ],
    # "illicit_violent_score": [
    #     "No instructions or praise for violent wrongdoing",
    #     "Do not assist with planning or committing violent crimes",
    #     "Guidance for violent acts is prohibited",
    #     "No endorsement or facilitation of violent offenses",
    #     "Avoid content that enables or glorifies violent crime",
    # ],

    # Self-harm
    "self_harm_score": [
        "No content encouraging or depicting self-harm or suicide",
        "Do not encourage or normalize self-harm",
        "Content about self-injury or suicide is not allowed",
        "Avoid promoting or describing self-harm",
        "No discussion that endorses self-harm",
        "Don't normalize or encourage self-harm",
        "No posts glorifying self-injury",
        "Keep self-harm talk out",
    ],
    "self_harm_instructions_score": [
        "No instructions that facilitate self-harm or suicide",
        "Do not provide tips or methods for self-harm",
        "Guidance that enables self-injury is prohibited",
        "Avoid step-by-step or how-to self-harm content",
        "No facilitation of suicide or self-harm",
        "No how-to content about self-harm",
        "Don't share methods or tips for self-injury",
        "No step-by-step self-harm content",
    ],
    "self_harm_intent_score": [
        "No admissions of intent or desire to self-harm",
        "Expressions of wanting to self-harm are not permitted",
        "Do not state plans or desire to self-injure",
        "Content indicating self-harm intent is disallowed",
        "No declarations of self-harm intent",
        "No 'I want to hurt myself' statements",
        "Don't share plans or intentions to self-harm",
        "No content expressing desire to self-injure",
    ],

    # Sexual
    "sexual_score": [
        "No sexually explicit content",
        "Do not post pornographic or explicit sexual material",
        "Sexually explicit content is prohibited",
        "Avoid graphic sexual descriptions or content",
        "Keep it SFW — no explicit sexual content",
        "Keep it PG — no NSFW explicit stuff",
        "No porn-y or graphic sex talk",
        "Save the explicit content for elsewhere",
    ],
    "sexual_minors_score": [
        "No sexual content involving minors",
        "Any sexualized content about minors is strictly banned",
        "Do not discuss sexual acts with or involving minors",
        "No depictions or descriptions sexualizing minors",
        "Zero tolerance for sexual content about minors",
        "Absolutely no sexual content involving minors",
        "No sexualizing kids, ever",
        "Zero tolerance: minors + sexual content = ban",
    ],

    # Violence
    "violence_score": [
        "No violent content or threats of violence",
        "Do not threaten or advocate physical harm",
        "Violent content and threats are not allowed",
        "Avoid content suggesting or endorsing violence",
        "No calls for violence against anyone",
        "Don't promote or threaten violence",
        "No 'beat them up' talk",
        "Keep it nonviolent — no threats",
    ],
    # "violence_graphic_score": [
    #     "No graphic depictions of violence or injury",
    #     "Do not post gore or graphic injury content",
    #     "Graphic violence is prohibited",
    #     "Avoid explicit depictions of injuries or blood",
    #     "No gory or gruesome content",
    # ],
}

DATA3_MODERATION_COLUMN_VARIANTS: Dict[str, List[str]] = {
    # Severe toxicity
    "severe_toxicity": [
        "No comments that are severely toxic, highly offensive language",
        "Extremely toxic or highly offensive language is not permitted.",
        "Avoid using abusive or offensive language in your comments.",
        "Comments containing intense toxicity or strong offensive wording are disallowed.",
        "Highly offensive or deeply toxic remarks are forbidden.",
    ],

    # Obscene
    "obscene": [
        "No comments that contain obscene language",
        "Vulgar language is not allowed in comments.",
        "Please refrain from using obscene wording.",
        "Comments with vulgar language are prohibited.",
        "Profanity is strictly disallowed.",
    ],

    # Threats
    "threat": [
        "No comments that contain threats of violence or harm against individuals or groups",
        "Threatening violence or harm toward any person or group is prohibited.",
        "Do not post comments that threaten physical or emotional harm to others.",
        "Comments containing threats, violence, intimidation, or menacing content are not allowed.",
        "Intimidation of any form is strictly forbidden.",
    ],

    # Insults / personal attacks
    "insult": [
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "Comments with personal insults or derogatory language will be removed.",
        "Please avoid insulting others in your comments.",
        "Do not post personal insults or use derogatory language.",
        "Directing derogatory language or personal attacks at someone is forbidden.",
    ],

    # Identity-based attacks (map to hate speech rule variants)
    "identity_attack": [
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "Hate speech aimed at any protected identity group (race, religion, gender, etc.) is prohibited.",
        "Do not post hate speech. This includes attacks on any group based on identity (race, religion, gender, etc.).",
        "We prohibit any content that constitutes hate speech, defined as direct attacks targeted at individuals or groups on the basis of their identity.",
        "Any hate speech directed at people based on race, religion, gender, or similar will not be tolerated.",
    ],

    # Sexually explicit content
    # Note: Some CSVs may name this column "sexual_explicit".
    "sexual_explicit": [
        "No comments that contain sexually explicit content",
        "Keep it SFW (Safe For Work). No sexually explicit or adult content, please.",
        "Please avoid posting comments with explicit sexual material.",
        "No adult content. All posts must be appropriate for a general audience.",
        "Do not include sexually explicit material in your comments.",
    ],
}

# ALL_COLUMN_VARIANTS-style dict for DATA3 moderation columns (no grouping)
DATA3_ALL_COLUMN_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    column_name: {column_name: variants}
    for column_name, variants in DATA3_MODERATION_COLUMN_VARIANTS.items()
}

# ---------------------------------------------------------------------------
# Grouped OpenAI Moderation column variants
#
# Build specialised groups by merging variants from related OpenAI moderation
# columns. This grouping is based ONLY on OPENAI_MODERATION_COLUMN_VARIANTS.
# ---------------------------------------------------------------------------

# Group name -> list of member OpenAI moderation columns
OPENAI_GROUPS: Dict[str, List[str]] = {
    "harassment": [
        "harassment_score",
        "harassment_threatening_score",
    ],
    "hate": [
        "hate_score",
        "hate_threatening_score",
    ],
    "self_harm": [
        "self_harm_score",
        "self_harm_instructions_score",
        "self_harm_intent_score",
    ],
    "sexual": [
        "sexual_score",
        "sexual_minors_score",
    ],
    "violence": [
        "violence_score",
    ],
}


# Per-group COLUMN_VARIANTS dictionaries for specialised fine-tuning
def _subset_variants(columns: List[str]) -> Dict[str, List[str]]:
    return {c: OPENAI_MODERATION_COLUMN_VARIANTS[c] for c in columns if c in OPENAI_MODERATION_COLUMN_VARIANTS}
ALL_COLUMN_VARIANTS = {rule_group: _subset_variants(columns) for rule_group, columns in OPENAI_GROUPS.items()}