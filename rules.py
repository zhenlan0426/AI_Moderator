"""Rule variants used for prompt randomization in AI_Moderation.

This module maps each moderation rule to a list of five different phrasings
(original + 4 paraphrases).  When constructing training prompts, a random
variant can be selected to encourage the model to generalise beyond a single
fixed wording of the rule.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Mapping: canonical rule -> list of 5 variants (original + 4 paraphrases)
# ---------------------------------------------------------------------------

RULE_VARIANTS: Dict[str, List[str]] = {
    # 1 ─────────────────────────────────────────────────────────────────────
    "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "Advertising is prohibited, including spam, discount lsink, and other promotional material.",
        "Please do not post any promotional content, unsolicited ads, or referral links.",
        "Spam, adverts, and promotional posts are not permitted here.",
        "Unsolicited advertising, including promo link, is forbidden.",
        "No shilling — keep ads and referrals out.",
        "No promos, no spamming, no referral codes.",
        "Keep it ad-free — no plugging products.",
    ],
    # 2 ─────────────────────────────────────────────────────────────────────
    "No legal advice: Do not offer or request legal advice.": [
        "No legal advice: Do not offer or request legal advice.",
        "Offering or requesting legal advice is not allowed.",
        "Please refrain from giving or seeking legal counsel.",
        "This platform does not permit any form of legal advice.",
        "Do not ask for or provide legal guidance.",
        "Not a lawyer — don’t ask for legal takes here.",
        "No legal hot takes or advice requests.",
        "Skip the legal counsel — not the place for it.",
    ],
    # 3 ─────────────────────────────────────────────────────────────────────
    "No comments that are severely toxic, highly offensive language": [
        "No comments that are severely toxic, highly offensive language",
        "Extremely toxic or highly offensive language is not permitted.",
        "Avoid using abusive or offensive language in your comments.",
        "Comments containing intense toxicity or strong offensive wording are disallowed.",
        "Highly offensive or deeply toxic remarks are forbidden.",
        "Don't be toxic — keep it respectful.",
        "Tone it down — no over-the-top hostility.",
        "Keep it cool; skip the nasty stuff.",
    ],
    # 4 ─────────────────────────────────────────────────────────────────────
    "No comments that contain obscene language": [
        "No comments that contain obscene language",
        "Vulgar language is not allowed in comments.",
        "Please refrain from using obscene wording.",
        "Comments with vulgar language are prohibited.",
        "Profanity is strictly disallowed.",
        "Watch the language — keep it clean.",
        "Dial back the swearing — no obscene words.",
        "Keep it PG — no vulgarity.",
    ],
    # 5 ─────────────────────────────────────────────────────────────────────
    "No comments that contain threats of violence or harm against individuals or groups": [
        "No comments that contain threats of violence or harm against individuals or groups",
        "Threatening violence or harm toward any person or group is prohibited.",
        "Do not post comments that threaten physical or emotional harm to others.",
        "Comments containing threats, violence, intimidation, or menacing content are not allowed.",
        "Intimidation of any form is strictly forbidden.",
        "No tough-guy threats or 'I'll hurt you' talk.",
        "Skip the menacing tone — no threats.",
        "No 'or else' messages — period.",
    ],
    # 6 ─────────────────────────────────────────────────────────────────────
    "No comments that contain personal attacks, insults, or derogatory language directed at individuals": [
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "Comments with personal insults or derogatory language will be removed.",
        "Please avoid insulting others in your comments.",
        "Do not post personal insults or use derogatory language.",
        "Directing derogatory language or personal attacks at someone is forbidden.",
        "Don't take shots at people — argue ideas, not folks.",
        "No name-calling or cheap shots.",
        "Keep it about the topic, not the person.",
    ],
    # 7 ─────────────────────────────────────────────────────────────────────
    "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)": [
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "Hate speech aimed at any protected identity group (race, religion, gender, etc.) is prohibited.",
        "Do not post hate speech. This includes attacks on any group based on identity (race, religion, gender, etc.).",
        "We prohibit any content that constitutes hate speech, defined as direct attacks targeted at individuals or groups on the basis of their identity.",
        "Any hate speech directed at people based on race, religion, gender, or similar will not be tolerated.",
        "Cut the slurs — don't target people for who they are.",
        "No trashing protected groups — not here.",
        "Respect identities — no hate speech.",
    ],
    # 8 ─────────────────────────────────────────────────────────────────────
    "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)": [
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "Hate speech aimed at any protected identity group (race, religion, gender, etc.) is prohibited.",
        "Do not post hate speech. This includes attacks on any group based on identity (race, religion, gender, etc.).",
        "We prohibit any content that constitutes hate speech, defined as direct attacks targeted at individuals or groups on the basis of their identity.",
        "Any hate speech directed at people based on race, religion, gender, or similar will not be tolerated.",
        "No identity-based attacks — full stop.",
        "Don't go after groups — that's a no-go.",
        "No piling on people for their identity.",
    ],
    # 9 ─────────────────────────────────────────────────────────────────────
    "No comments that contain sexually explicit content": [
        "No comments that contain sexually explicit content",
        "Keep it SFW (Safe For Work). No sexually explicit or adult content, please.",
        "Please avoid posting comments with explicit sexual material.",
        "No adult content. All posts must be appropriate for a general audience.",
        "Do not include sexually explicit material in your comments.",
        "No NSFW — keep it PG here.",
        "Save the explicit stuff for elsewhere.",
        "No porn-y or graphic sex content.",
    ],
    # 10 ─────────────────────────────────────────────────────────────────────
    "No political or partisan content": [
        "No political or partisan content",
        "Politics-related posts and comments are prohibited.",
        "Please refrain from posting political or partisan material.",
        "Content that is political in nature or advocates for partisan positions is not allowed.",
        "Avoid posting any politically-oriented or partisan comments.",
        "No politics talk — not the place for it.",
        "Save the partisanship for elsewhere.",
        "Keep it non-political here.",
    ],
    # 11 ─────────────────────────────────────────────────────────────────────
    "No inflammatory or highly controversial content": [
        "No inflammatory or highly controversial content",
        "Overly provocative or divisive comments are not permitted.",
        "Let’s avoid heated topics or picking fights.",
        "Content designed to provoke or that is highly controversial is prohibited.",
        "Do not post inflammatory or divisive material.",
        "No drama-bait — keep it chill.",
        "Don't stir the pot for the sake of it.",
        "Dial down the outrage — no flame wars.",
    ],
    # 12 ─────────────────────────────────────────────────────────────────────
    "post should not contain financial advice, trading signals, or content that could influence investment decisions": [
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "Comments giving investment tips, stock picks, or trading signals are prohibited.",
        "Do not provide financial advice or trading recommendations that could influence investment choices.",
        "Posts that could sway someone’s money moves are not allowed.",
        "Please refrain from offering financial guidance or trading suggestions.",
        "No investment tips or 'buy/sell' calls here.",
        "Not your broker — skip the market advice.",
        "No trading signals or financial guidance.",
    ],
    # 13 ─────────────────────────────────────────────────────────────────────
    "No sarcasm, snark, or flamebait": [
        "No sarcasm, snark, or flamebait",
        "Sarcastic, snarky, or baiting comments are not allowed.",
        "Please avoid sarcasm, snark, or content designed to provoke others.",
        "Keep it respectful — avoid sarcasm, snark, or baiting others.",
        "No sarcasm, no snark, no picking fights.",
        "No passive-aggressive digs or zingers.",
        "Don't bait people — keep it earnest.",
        "Skip the snide remarks.",
    ],
    # 14 ─────────────────────────────────────────────────────────────────────
    "No self-harm or suicidal ideation": [
        "No self-harm or suicidal ideation",
        "Let’s keep this space free from self-harm or suicide topics.",
        "Please do not post about self-harm or suicidal ideation.",
        "No comments encouraging or describing self-harm or suicide.",
        "Content involving self-injury or suicidal thoughts is not allowed.",
        "No 'I want to hurt myself' posts.",
        "Don't share self-harm feelings here — seek help instead.",
        "Keep self-harm talk out of here.",
    ],
    # 15 ─────────────────────────────────────────────────────────────────────
    "No Content Inducing Anxiety or Stress": [
        "No Content Inducing Anxiety or Stress",
        "Content that induces anxiety or stress is not permitted.",
        "Keep things calm — no stress-inducing content.",
        "Posts designed to create anxiety or stress are prohibited.",
        "Let’s avoid comments that cause unnecessary stress or anxiety.",
        "No panic-inducing or stress-bait content.",
        "Keep it low-stress and supportive.",
        "Don't try to freak people out.",
    ],
    # 16 ─────────────────────────────────────────────────────────────────────
    "No Depressive Mental Health Narratives": [
        "No Depressive Mental Health Narratives",
        "Depressive mental health comments are not allowed.",
        "This space isn’t for sharing depressive or mental health crisis stories.",
        "Let’s avoid content centered on depressive themes.",
        "Do not share depressive mental health stories or narratives.",
        "No heavy depressive monologues here.",
        "This isn't a crisis forum — seek proper support.",
        "Keep deeply depressive narratives out.",
    ],
}

# ---------------------------------------------------------------------------
# Mapping: canonical rule -> list of other rules considered *safe negatives*
# ---------------------------------------------------------------------------
# A “safe negative” means: if a comment is a *positive* example for the listed
# rule, it should almost certainly be a **negative** for the key rule.  This
# helps build training batches where the same comment can appear with both
# label 1 (for its own rule) and label 0 (for other, non-overlapping rules),
# forcing the model to attend to the (rule, comment) interaction rather than
# memorising individual comments.
#
# The heuristic used here is to avoid pairing rules that have substantial
# semantic overlap (e.g. different flavours of toxicity, or closely-related
# mental-health themes).  The mapping can of course be refined later if manual
# inspection reveals edge-cases.

RULE_NEGATIVE_WHITELIST: Dict[str, List[str]] = {
    # 1 ─────────────────────────────────────────────────────────────────────
    "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.": [
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 2 ─────────────────────────────────────────────────────────────────────
    "No legal advice: Do not offer or request legal advice.": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 3 ─────────────────────────────────────────────────────────────────────
    "No comments that are severely toxic, highly offensive language": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 4 ─────────────────────────────────────────────────────────────────────
    "No comments that contain obscene language": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 5 ─────────────────────────────────────────────────────────────────────
    "No comments that contain threats of violence or harm against individuals or groups": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 6 ─────────────────────────────────────────────────────────────────────
    "No comments that contain personal attacks, insults, or derogatory language directed at individuals": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 7 ─────────────────────────────────────────────────────────────────────
    "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 8 ─────────────────────────────────────────────────────────────────────
    "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 9 ─────────────────────────────────────────────────────────────────────
    "No comments that contain sexually explicit content": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No political or partisan content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 10 ────────────────────────────────────────────────────────────────────
    "No political or partisan content": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 11 ────────────────────────────────────────────────────────────────────
    "No inflammatory or highly controversial content": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 12 ────────────────────────────────────────────────────────────────────
    "post should not contain financial advice, trading signals, or content that could influence investment decisions": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 13 ────────────────────────────────────────────────────────────────────
    "No sarcasm, snark, or flamebait": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No self-harm or suicidal ideation",
        "No Content Inducing Anxiety or Stress",
        "No Depressive Mental Health Narratives",
    ],
    # 14 ────────────────────────────────────────────────────────────────────
    "No self-harm or suicidal ideation": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No Content Inducing Anxiety or Stress",
    ],
    # 15 ────────────────────────────────────────────────────────────────────
    "No Content Inducing Anxiety or Stress": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No self-harm or suicidal ideation",
        "No Depressive Mental Health Narratives",
    ],
    # 16 ────────────────────────────────────────────────────────────────────
    "No Depressive Mental Health Narratives": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "No legal advice: Do not offer or request legal advice.",
        "No comments that are severely toxic, highly offensive language",
        "No comments that contain obscene language",
        "No comments that contain threats of violence or harm against individuals or groups",
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)",
        "No comments that contain sexually explicit content",
        "No political or partisan content",
        "No inflammatory or highly controversial content",
        "post should not contain financial advice, trading signals, or content that could influence investment decisions",
        "No sarcasm, snark, or flamebait",
        "No Content Inducing Anxiety or Stress",
    ],
}
