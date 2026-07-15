from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import _normalize_content
from .harmfulness_ensemble import HarmfulnessEnsembleAgent
from .rubrics import MEME_ANALYSIS_RUBRIC


@dataclass(frozen=True)
class AnalysisHead:
    name: str
    title: str
    description: str
    prompt: str


@dataclass(frozen=True)
class HeadResult:
    name: str
    title: str
    output: str


HEADS: dict[str, AnalysisHead] = {
    "harmfulness": AnalysisHead(
        name="harmfulness",
        title="Harmfulness Analysis",
        description="判断有害性、目标对象、伤害类型、严重度与证据强度。",
        prompt="""
You are the harmfulness analysis head in a multi-agent meme analysis system.
Analyze only harmfulness and safety-relevant risk. Do not produce a general meme report.

Return Chinese output with exactly these sections:
1. Label
   - harmfulness_decision: exactly one of harmful, harmless
   - harmfulness_labels: any of Discrimination, Offensive, Violence, Vulgar, Antagonism; use an empty list for harmless
   - severity: one of high, medium, low, none, unknown
   - confidence: high, medium, low
2. Target and harm type
   - target:
   - harm_types:
3. Evidence
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
4. Rationale
5. Uncertainty and missing evidence

Apply the project harmfulness rubric carefully. Sensitive-event references require stance and harm-mechanism analysis; do not label them Offensive solely because they are mentioned.
The final harmfulness decision is strictly binary: harmful or harmless. Do not output a third category; express evidence limitations through confidence and missing evidence.
Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
    "sentiment": AnalysisHead(
        name="sentiment",
        title="Sentiment Analysis",
        description="判断情绪极性、强度、讽刺/反讽以及情绪对象。",
        prompt="""
You are the sentiment analysis head in a multi-agent meme analysis system.
Analyze only emotion and sentiment. Do not produce a general meme report.

Return Chinese output with exactly these sections:
1. Label
   - primary_sentiment: one of Joy, Sadness, Anger, Disgust, Fear, Surprise, unclear
   - secondary_sentiments:
   - intensity: high, medium, low, none, unknown
   - confidence: high, medium, low
2. Emotion profile
   - primary_emotions:
   - secondary_emotions:
   - sentiment_target:
3. Irony and ambiguity
   - irony_or_sarcasm:
   - ambiguity:
4. Evidence
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
5. Uncertainty and missing evidence

Use the project sentiment rubric, including multimodal mismatch rules.
Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
    "intent": AnalysisHead(
        name="intent",
        title="Intent Recognition",
        description="判断娱乐、讽刺、羞辱、动员、误导等传播意图。",
        prompt="""
You are the intent recognition head in a multi-agent meme analysis system.
Analyze only the likely communicative intent. Do not produce a general meme report.

Return Chinese output with exactly these sections:
1. Label
   - primary_intent: one of Teleological, Normative, Dramaturgical, Communicative, mixed, unclear
   - secondary_intents:
   - confidence: high, medium, low
2. Actor and audience hypothesis
   - likely_speaker_position:
   - intended_audience:
3. Evidence
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
4. Alternative interpretations
5. Uncertainty and missing evidence

Use the project Habermas-based intent rubric.
Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
    "evolution": AnalysisHead(
        name="evolution",
        title="Meme Evolution Analysis",
        description="分析模板来源、变体、语义漂移、传播场景和演化方向。",
        prompt="""
You are the meme evolution analysis head in a multi-agent meme analysis system.
Analyze only meme evolution, template lineage, mutation, circulation, and cultural references.
Do not produce a general meme report.

Return Chinese output with exactly these sections:
1. Template and lineage
   - likely_template:
   - known_reference_or_origin:
   - confidence: high, medium, low
2. Mutation and phylogenetic tracking
   - visual_mutations:
   - textual_mutations:
   - semantic_shift:
3. Core kernel fidelity
   - invariant_core:
   - variable_elements:
   - metaphor_or_structural_irony:
4. Lifecycle and diffusion
   - lifecycle_phase: incubation, outbreak, saturation, decay/obsolescence, unclear
   - likely_platform_or_community:
   - cross_platform_potential:
5. Intertextual splicing
   - referenced_assets:
   - splicing_index: low, medium, high, unclear
6. Evidence
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
7. Uncertainty and next retrieval targets

Use the project evolution rubric.
Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
    "audience": AnalysisHead(
        name="audience",
        title="Audience Reception Prediction",
        description="预测不同受众的理解、接受、抵触和误读风险。",
        prompt="""
You are the audience reception head in a multi-agent meme analysis system.
Analyze only audience interpretation and likely reception. Do not produce a general meme report.

Return Chinese output with exactly these sections:
1. Likely audiences
   - primary_audience_type: Gemeinschaft-oriented, Gesellschaft-oriented, mixed, unclear
   - primary_audience:
   - secondary_audiences:
   - knowledge_threshold: high, medium, low
   - confidence: high, medium, low
2. Reception prediction
   - supportive_reading:
   - oppositional_reading:
   - confused_or_misread_reading:
3. Risk of misinterpretation
4. Evidence
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
5. Uncertainty and missing evidence

Use the project Tonnies-based audience rubric.
Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
    "evidence-audit": AnalysisHead(
        name="evidence-audit",
        title="Evidence Audit",
        description="检查证据充分性、推断边界、缺口和需要补充检索的问题。",
        prompt="""
You are the evidence audit head in a multi-agent meme analysis system.
Analyze only evidence quality. Do not produce a general meme report or any final synthesis.

Return Chinese output with exactly these sections:
1. Evidence coverage
   - image_evidence:
   - user_context_evidence:
   - retrieved_evidence:
2. Unsupported or weakly supported claims to avoid
3. Important uncertainties
4. Recommended next searches
5. Overall evidence confidence
   - confidence: high, medium, low

Use source tags strictly: [Image], [User Context], [W#], [N#], [Inference].
If iterative retrieval is used, cite round-specific labels exactly as shown, such as [R2-W1] or [R2-N1].
Do not invent source IDs.
""".strip(),
    ),
}


DEFAULT_HEAD_NAMES = ["harmfulness"]


class MemeAnalysisHeadRunner:
    """Runs independent task heads against a shared evidence pack."""

    def __init__(self, llm: Any, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def run_head(
        self,
        head: AnalysisHead,
        topic: str,
        evidence_context: str,
        input_mode: str,
    ) -> HeadResult:
        if head.name == "harmfulness":
            ensemble = HarmfulnessEnsembleAgent(
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            return HeadResult(
                name=head.name,
                title=head.title,
                output=ensemble.run(
                    topic=topic,
                    evidence_context=evidence_context,
                    input_mode=input_mode,
                ),
            )

        user_prompt = f"""
Topic: {topic or "None"}

Input mode: {input_mode}

Shared evidence pack:
{evidence_context or "None"}

Project rubric:
{MEME_ANALYSIS_RUBRIC}

Task instruction:
{head.prompt}
""".strip()
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return HeadResult(
            name=head.name,
            title=head.title,
            output=_normalize_content(getattr(response, "content", response)),
        )

    def run_heads(
        self,
        head_names: list[str],
        topic: str,
        evidence_context: str,
        input_mode: str,
    ) -> list[HeadResult]:
        return [
            self.run_head(
                head=HEADS[name],
                topic=topic,
                evidence_context=evidence_context,
                input_mode=input_mode,
            )
            for name in head_names
        ]


def normalize_head_names(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_HEAD_NAMES.copy()

    normalized: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name == "all":
                for head_name in HEADS:
                    if head_name not in normalized:
                        normalized.append(head_name)
                continue
            if name not in HEADS:
                choices = ", ".join([*HEADS.keys(), "all"])
                raise ValueError(f"Unknown task head '{name}'. Available heads: {choices}")
            if name not in normalized:
                normalized.append(name)

    return normalized or DEFAULT_HEAD_NAMES.copy()


def format_head_results(results: list[HeadResult]) -> str:
    return "\n\n".join(
        f"## {result.title}\n\n{result.output}".strip()
        for result in results
    )
