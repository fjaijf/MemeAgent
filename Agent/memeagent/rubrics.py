from __future__ import annotations


MEME_ANALYSIS_RUBRIC = """
Use these project-specific meme analysis standards.

Harmfulness labels:
- Discrimination: discriminates based on race, gender, religion, nationality, LGBTQ+ identity, or comparable protected/social identity.
- Offensive: insults, belittles, or maliciously attacks individuals or groups. Any reference to sensitive events, including disasters, pandemics, historical tragedies, racial issues, or violent incidents, is offensive in any form: humorous, sarcastic, dismissive, factual, or otherwise.
- Violence: directly or indirectly encourages, endorses, threatens, celebrates, or depicts violent behavior.
- Vulgar: sexual implication, obscenity, nudity, vulgar language, or implicit sexual behavior information in any form.
- Antagonism: aggressive discontent, resentment, or negativity toward groups or society through hostile satire, malicious sarcasm, harmful exaggeration, pessimism, or nihilism without constructive intent.

Sentiment labels:
- Joy: genuine happiness, harmless wit, wholesome playfulness, community bonding, positive image and text alignment.
- Sadness: loneliness, pressure, burnout, despair, or smiling-through-tears self-deprecation; smiling visuals with depressive text should still be judged as sadness when the core effect is ironic self-pity.
- Anger: frustration, moral outrage, breakdown, frantic ranting, aggressive gestures, all-caps, excessive exclamation, or high-arousal negative loss of control.
- Disgust: contempt, cynicism, social rejection, disapproval, call-outs, eye-roll/side-eye/cold-face reactions; the meme acts as a verdict.
- Fear: anxiety, deadline panic, social phobia, imposter syndrome, sweating, trembling, wide eyes, hiding, being chased, or imminent-disaster framing.
- Surprise: astonishment, plot twist, mind-blown cognitive shift, jaw-drop/wide-eye/recoil, sharp textual pivot. If surprise mainly mocks predictable failure, treat disgust or sarcasm as primary.

Audience prediction:
- Gemeinschaft-oriented audience: micro-level, high-context, subcultural, domain-specific, ideological, or localized in-group. Requires specialized knowledge, jargon, community history, or external knowledge graph retrieval.
- Gesellschaft-oriented audience: macro-level mass internet society. Uses universal human experience, broad demographic cohorts, modern lifestyle, or globally viral events with low background knowledge threshold.

Intent detection:
- Teleological or strategic action: utility-driven propaganda, marketing, astroturfing, persuasion, disinformation, or weaponized meme use toward political/economic outcomes.
- Normative action: value-regulation intent; moral correction, social critique, satire, accountability, or calling out toxic behavior, authorities, public figures, or systemic flaws.
- Dramaturgical action: self-expression and coping; self-deprecation, existential dread, frantic venting, burnout, loneliness, or projecting the poster's subjective state.
- Communicative action: mutual understanding, bonding, inside jokes, harmless wit, shared vibe, and recognition without attack, exploitation, or complaint.

Evolution analysis:
- Multimodal phylogenetic tracking: identify visual drift, semantic shifting, template lineage, mutations, recomposition, and medium translation.
- Core kernel fidelity: isolate invariant visual anchors, cultural genotype, structural irony, metaphor, and coping mechanism from surface variations.
- Lifecycle and diffusion dynamics: classify incubation, outbreak, saturation, or decay/obsolescence using platform spread, volume signals, overuse, and audience fatigue.
- Semiotic and intertextual splicing: identify historical meme assets, pop culture anchors, platform-specific references, and splicing index when multiple subcultural anchors combine.

Evidence discipline:
- Separate direct image/OCR evidence, user context, retrieved evidence, and inference.
- Do not invent entities, source IDs, platforms, dates, intent, audience, or evolution history.
- If evidence is weak, state uncertainty and ask for the next visual, retrieval, or contextual evidence needed.
""".strip()


CONTROLLER_OUTPUT_SCHEMA = """
Return exactly these sections:
ITERATION_CONFIDENCE:
- A float from 0.00 to 1.00 indicating whether the evidence is sufficient for final output under the rubric.

SHOULD_FINALIZE:
- yes when ITERATION_CONFIDENCE is at or above the configured threshold; otherwise no. The workflow stops by threshold.

CONFIDENCE_REASON:
- One concise sentence explaining the score.

KEY_FINDINGS_SO_FAR:
- 2-6 bullets grounded in available evidence.

FOCUS_QUESTIONS:
- 2-6 concrete questions the next multimodal/retrieval pass should answer. Use "None" only if SHOULD_FINALIZE is yes.

MULTIMODAL_ANALYSIS_REQUESTS:
- 1-5 concrete instructions for the vision model, focused on OCR, visual anchors, harmfulness cues, sentiment, audience, intent, or evolution. Use "None" only if SHOULD_FINALIZE is yes.

SUPPLEMENTAL_WEB_QUERIES:
- 0-5 short concrete web queries. Use exact OCR, names, platform clues, template names, or event anchors. Use "None" if no useful query exists.

SUPPLEMENTAL_NEWS_QUERIES:
- 0-2 short news-friendly queries for public figures, current/recent events, disasters, controversies, organizations, or sensitive incidents. Use "None" if no useful query exists.

FINAL_OUTPUT_NOTES:
- 1-4 bullets telling the final model what must be emphasized or kept uncertain.
""".strip()
