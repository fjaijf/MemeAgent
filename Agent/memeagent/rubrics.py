from __future__ import annotations


MEME_ANALYSIS_RUBRIC = """
Use these project-specific meme analysis standards.

Harmfulness labels:
- Discrimination: discriminates based on race, gender, religion, nationality, LGBTQ+ identity, or comparable protected/social identity.
- Offensive: insults, belittles, humiliates, exploits, or maliciously attacks individuals or groups. Sensitive-event references require contextual assessment: label them Offensive when the meme mocks victims, trivializes harm, targets affected people, invites contempt, or otherwise turns the event into a harmful attack. Neutral, factual, commemorative, critical, or clearly condemnatory references are not Offensive without an additional harm mechanism.
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


HARMFULNESS_DETECTION_RUBRIC = """
Use these project-specific binary harmfulness detection standards.

Harmfulness labels:
- Discrimination: discriminates based on race, gender, religion, nationality, LGBTQ+ identity, or comparable protected/social identity.
- Offensive: insults, belittles, humiliates, exploits, or maliciously attacks individuals or groups. Sensitive-event references require contextual assessment: label them Offensive when the meme mocks victims, trivializes harm, targets affected people, invites contempt, or otherwise turns the event into a harmful attack. Neutral, factual, commemorative, critical, or clearly condemnatory references are not Offensive without an additional harm mechanism.
- Violence: directly or indirectly encourages, endorses, threatens, celebrates, or depicts violent behavior.
- Vulgar: sexual implication, obscenity, nudity, vulgar language, or implicit sexual behavior information in any form.
- Antagonism: aggressive discontent, resentment, or negativity toward groups or society through hostile satire, malicious sarcasm, harmful exaggeration, pessimism, or nihilism without constructive purpose.

Binary decision rule:
- Return harmful when at least one harmfulness label is supported by concrete image, OCR, user-context, or retrieved evidence.
- Apply label-specific evidence requirements instead of requiring target, stance, and harm mechanism for every label:
  - Discrimination and Offensive usually require a target, targeted group, or victim-directed framing.
  - Violence requires endorsement, threat, celebration, instruction, or graphic depiction of violent behavior.
  - Vulgar can be harmful based on explicit or implicit sexual content, sexual coercion, obscenity, nudity, vulgar language, or sexualized framing even when no protected-group target is present.
  - Antagonism requires hostile, resentment-amplifying, maliciously sarcastic, or nihilistic framing toward a person, group, institution, public figure, political bloc, religion, gender category, nationality, or society.
- Return harmless only when no harmfulness label is concretely supported by the available evidence.
- Evidence limitations affect confidence and the explanation, not the binary label; do not use harmless as a fallback when concrete evidence supports a harmfulness label.

Vulgar calibration:
- Sexual innuendo, sexual touching, sexualized body framing, coercive sexual framing, obscene jokes, profanity, crude insults, vulgar gestures such as a middle finger, or implicit sexual behavior should be labeled Vulgar when visible in OCR or image evidence.
- Vulgar does not require hate, harassment, protected identity, an explicitly named target, or proof of speaker intent.
- If sexual content is clearly present but severity or intent is uncertain, choose harmful with lower confidence rather than harmless.

Sensitive-event calibration:
- Public-health crises, disasters, deaths, abuse, war, terrorism, historical trauma, racial violence, and other public tragedies are sensitive-event contexts.
- Label Offensive when the meme turns a sensitive event into a joke, game mechanic, disposable punchline, trivialized inconvenience, victim-blaming frame, or audience-amusement device.
- Do not require explicit victim naming for sensitive-event trivialization; the harm mechanism can be minimization, entertainment framing, or normalization of suffering.
- Keep harmless only for clearly factual, commemorative, corrective, awareness-raising, or condemnatory treatment of the event.

Group and ideology calibration:
- Races, ethnicities, political parties, supporters, religions, nationalities, genders, and policy-affected groups can be targets for Offensive, Discrimination, or Antagonism even when no slur appears or the target is only shown visually.
- Strong negative group attribution, stereotype reinforcement, humiliation, blanket accusation, or ridicule of a racial, ethnic, social, religious, gender, or political group can support harmfulness.
- Do not dismiss racial, ethnic, political, religious, or gendered attacks as harmless satire when the visible framing is hostile, contemptuous, or stereotype-driven.

Controller confidence calibration:
- Do not finalize high-confidence harmless when visible evidence includes profanity, vulgar gestures, sexualized framing, sensitive-event humor, public-health/death/disaster references, religious or gender stereotypes, or hostile group/political attribution.
- If such cues are present but the exact stance is uncertain, choose harmful with moderate confidence when a label is supported, or keep confidence low and request targeted follow-up rather than high-confidence harmless.

Recall calibration:
- Avoid systematic false negatives. When concrete evidence supports any harmfulness label, assign harmful even if confidence is moderate.
- Do not classify edgy, vulgar, hostile, or violent content as harmless solely because speaker identity, source context, or full intent is uncertain.

Evidence discipline:
- Separate direct image/OCR evidence, user context, retrieved evidence, and inference.
- Do not invent entities, targets, source IDs, platforms, dates, events, or speaker stance.
- Cite the specific evidence supporting each assigned label and state important missing evidence.
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
- 1-5 concrete instructions for the vision model, focused on OCR, visual anchors, targets, stance, and harmfulness cues. Use "None" only if SHOULD_FINALIZE is yes.

SUPPLEMENTAL_WEB_QUERIES:
- 0-5 short concrete web queries. Use exact OCR, names, platform clues, template names, or event anchors. Use "None" if no useful query exists.

SUPPLEMENTAL_NEWS_QUERIES:
- 0-2 short news-friendly queries for public figures, current/recent events, disasters, controversies, organizations, or sensitive incidents. Use "None" if no useful query exists.

FINAL_OUTPUT_NOTES:
- 1-4 bullets telling the final model what must be emphasized or kept uncertain.
""".strip()
