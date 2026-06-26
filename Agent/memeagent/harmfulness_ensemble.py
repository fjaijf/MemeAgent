from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import _normalize_content
from .rubrics import MEME_ANALYSIS_RUBRIC


HARMFUL_LABELS = [
    "Discrimination",
    "Offensive",
    "Violence",
    "Vulgar",
    "Antagonism",
]
OUTPUT_LABELS = [*HARMFUL_LABELS, "Not harmful", "Unclear"]


@dataclass(frozen=True)
class PerspectiveAgentSpec:
    name: str
    title: str
    background: str
    weight: float


@dataclass(frozen=True)
class PerspectiveAgentResult:
    agent_name: str
    title: str
    weight: float
    harmful_probability: float
    not_harmful_probability: float
    unclear_probability: float
    label_probabilities: dict[str, float]
    primary_label: str
    severity: str
    confidence: float
    key_evidence: list[str]
    reasoning_summary: str
    uncertainties: list[str]
    raw_output: str


@dataclass(frozen=True)
class CounterfactualResult:
    tests: list[dict[str, Any]]
    robustness: str
    harmful_probability_delta: float
    confidence_delta: float
    label_probability_deltas: dict[str, float]
    summary: str
    raw_output: str


@dataclass(frozen=True)
class EnsembleDecision:
    harmful_probability: float
    not_harmful_probability: float
    unclear_probability: float
    label_probabilities: dict[str, float]
    decision: str
    labels: list[str]
    severity: str
    confidence: float


PERSPECTIVE_AGENTS = [
    PerspectiveAgentSpec(
        name="mainstream_safety",
        title="Mainstream Safety Reviewer",
        background=(
            "You are a mainstream platform safety reviewer. You are sensitive to "
            "clear harassment, hate, graphic violence, vulgar sexual content, and "
            "hostile ridicule, while still requiring evidence tags for claims."
        ),
        weight=1.0,
    ),
    PerspectiveAgentSpec(
        name="cross_cultural",
        title="Cross-Cultural Reviewer",
        background=(
            "You are a cross-cultural and multilingual reviewer. You focus on "
            "culture-specific insults, symbols, historical trauma, regional politics, "
            "translation ambiguity, and meanings that may be missed by a single-culture reviewer."
        ),
        weight=1.2,
    ),
    PerspectiveAgentSpec(
        name="ingroup_context",
        title="In-Group Context Reviewer",
        background=(
            "You are an in-group and subculture context reviewer. You consider whether "
            "the meme may be self-deprecation, community bonding, reclaimed language, "
            "or an inside joke, and you flag when broader public audiences may read it differently."
        ),
        weight=0.9,
    ),
    PerspectiveAgentSpec(
        name="vulnerable_group_advocate",
        title="Vulnerable-Group Advocate",
        background=(
            "You review from the perspective of people who may be targeted or harmed. "
            "You are attentive to stereotypes, dehumanization, sensitive-event joking, "
            "social stigma, and cumulative harms to marginalized or vulnerable groups."
        ),
        weight=1.2,
    ),
    PerspectiveAgentSpec(
        name="pragmatic_intent",
        title="Pragmatic Intent Reviewer",
        background=(
            "You are a pragmatics and communicative intent reviewer. You focus on "
            "speaker stance, target direction, irony, quotation, satire, whether the meme "
            "attacks a target or criticizes harmful behavior, and likely uptake."
        ),
        weight=1.0,
    ),
    PerspectiveAgentSpec(
        name="conservative_evidence",
        title="Conservative Evidence Reviewer",
        background=(
            "You are a conservative evidence reviewer. You avoid over-calling harmfulness "
            "when image, user context, or retrieved evidence is weak. You should raise "
            "Unclear when the target, event, or intent is unsupported."
        ),
        weight=1.1,
    ),
]


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def _coerce_str_list(value: Any, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        items = [value]
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()][:limit]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    lookup = {label.lower(): label for label in OUTPUT_LABELS}
    return lookup.get(text.lower(), "Unclear")


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low", "none", "unknown"}:
        return text
    return "unknown"


def _severity_score(value: str) -> float | None:
    return {
        "none": 0.0,
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0,
    }.get(value)


def _score_to_severity(value: float) -> str:
    if value >= 2.5:
        return "high"
    if value >= 1.5:
        return "medium"
    if value >= 0.5:
        return "low"
    return "none"


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _format_probability(value: float) -> str:
    return f"{value:.2f}"


class HarmfulnessEnsembleAgent:
    """Runs perspective-based harmfulness reviewers and soft-votes their labels."""

    def __init__(self, llm: Any, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def run(
        self,
        topic: str,
        evidence_context: str,
        input_mode: str,
    ) -> str:
        perspective_results = [
            self._run_perspective_agent(
                spec=spec,
                topic=topic,
                evidence_context=evidence_context,
                input_mode=input_mode,
            )
            for spec in PERSPECTIVE_AGENTS
        ]
        preliminary = self._aggregate(
            perspective_results,
            evidence_context=evidence_context,
        )
        counterfactual = self._run_counterfactual_agent(
            topic=topic,
            evidence_context=evidence_context,
            input_mode=input_mode,
            perspective_results=perspective_results,
            preliminary=preliminary,
        )
        final_decision = self._aggregate(
            perspective_results,
            counterfactual=counterfactual,
            evidence_context=evidence_context,
        )
        return self._format_report(
            perspective_results=perspective_results,
            counterfactual=counterfactual,
            preliminary=preliminary,
            final_decision=final_decision,
            evidence_context=evidence_context,
        )

    def _run_perspective_agent(
        self,
        spec: PerspectiveAgentSpec,
        topic: str,
        evidence_context: str,
        input_mode: str,
    ) -> PerspectiveAgentResult:
        user_prompt = f"""
Topic: {topic or "None"}

Input mode: {input_mode}

Shared evidence pack:
{evidence_context or "None"}

Project harmfulness rubric:
{MEME_ANALYSIS_RUBRIC}

You are one perspective reviewer in a harmful meme detection ensemble.
Return only JSON. Do not include chain-of-thought. Use concise evidence-grounded summaries.

JSON schema:
{{
  "harmful_probability": 0.0,
  "not_harmful_probability": 0.0,
  "unclear_probability": 0.0,
  "label_probabilities": {{
    "Discrimination": 0.0,
    "Offensive": 0.0,
    "Violence": 0.0,
    "Vulgar": 0.0,
    "Antagonism": 0.0
  }},
  "primary_label": "Discrimination|Offensive|Violence|Vulgar|Antagonism|Not harmful|Unclear",
  "severity": "high|medium|low|none|unknown",
  "confidence": 0.0,
  "key_evidence": ["Use exact source tags such as [Image], [User Context], [W1], [N1], [R2-W1], or [Inference]."],
  "reasoning_summary": "Short Chinese summary of this perspective's judgment.",
  "uncertainties": ["Short uncertainty notes."]
}}

Rules:
- Treat harmful labels as multi-label probabilities; they do not need to sum to 1.
- harmful_probability, not_harmful_probability, and unclear_probability should reflect the overall decision.
- Apply the project rule that any reference to sensitive events is Offensive.
- Do not invent source IDs, targets, events, platforms, or dates.
- If evidence is weak, increase unclear_probability and lower confidence.
""".strip()
        messages = [
            SystemMessage(content=f"{self.system_prompt}\n\n{spec.background}"),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        raw_output = _normalize_content(getattr(response, "content", response))
        return self._parse_perspective_result(spec, raw_output)

    def _parse_perspective_result(
        self,
        spec: PerspectiveAgentSpec,
        raw_output: str,
    ) -> PerspectiveAgentResult:
        parsed = _extract_json_object(raw_output) or {}
        raw_label_probs = parsed.get("label_probabilities")
        label_probabilities: dict[str, float] = {}
        if isinstance(raw_label_probs, dict):
            for label in HARMFUL_LABELS:
                label_probabilities[label] = _clamp(raw_label_probs.get(label))
        else:
            for label in HARMFUL_LABELS:
                label_probabilities[label] = 0.0

        harmful_probability = _clamp(parsed.get("harmful_probability"))
        not_harmful_probability = _clamp(parsed.get("not_harmful_probability"))
        unclear_probability = _clamp(parsed.get("unclear_probability"))
        if not parsed:
            harmful_probability = 0.0
            not_harmful_probability = 0.0
            unclear_probability = 1.0
        return PerspectiveAgentResult(
            agent_name=spec.name,
            title=spec.title,
            weight=spec.weight,
            harmful_probability=harmful_probability,
            not_harmful_probability=not_harmful_probability,
            unclear_probability=unclear_probability,
            label_probabilities=label_probabilities,
            primary_label=_normalize_label(parsed.get("primary_label")),
            severity=_normalize_severity(parsed.get("severity")),
            confidence=_clamp(parsed.get("confidence"), minimum=0.0, maximum=1.0),
            key_evidence=_coerce_str_list(parsed.get("key_evidence")),
            reasoning_summary=str(parsed.get("reasoning_summary") or "").strip(),
            uncertainties=_coerce_str_list(parsed.get("uncertainties")),
            raw_output=raw_output,
        )

    def _run_counterfactual_agent(
        self,
        topic: str,
        evidence_context: str,
        input_mode: str,
        perspective_results: list[PerspectiveAgentResult],
        preliminary: EnsembleDecision,
    ) -> CounterfactualResult:
        vote_summary = self._format_vote_summary(perspective_results)
        label_summary = ", ".join(
            f"{label}={_format_probability(probability)}"
            for label, probability in preliminary.label_probabilities.items()
        )
        user_prompt = f"""
Topic: {topic or "None"}

Input mode: {input_mode}

Shared evidence pack:
{evidence_context or "None"}

Perspective vote summary:
{vote_summary}

Preliminary soft vote:
- harmful_probability: {_format_probability(preliminary.harmful_probability)}
- not_harmful_probability: {_format_probability(preliminary.not_harmful_probability)}
- unclear_probability: {_format_probability(preliminary.unclear_probability)}
- labels: {label_summary}
- severity: {preliminary.severity}
- confidence: {_format_probability(preliminary.confidence)}

You are the counterfactual reasoning reviewer. Test whether the preliminary
harmfulness judgment depends on fragile assumptions. Return only JSON.

JSON schema:
{{
  "counterfactual_tests": [
    {{
      "condition": "If ...",
      "expected_change": "How the harmfulness judgment would change.",
      "affected_labels": ["Offensive"],
      "reason": "Short evidence-grounded explanation.",
      "evidence_needed": "What evidence would resolve this counterfactual."
    }}
  ],
  "robustness": "high|medium|low",
  "recommended_adjustment": {{
    "harmful_probability_delta": 0.0,
    "confidence_delta": 0.0,
    "label_probability_deltas": {{
      "Discrimination": 0.0,
      "Offensive": 0.0,
      "Violence": 0.0,
      "Vulgar": 0.0,
      "Antagonism": 0.0
    }}
  }},
  "summary": "Short Chinese summary of whether the decision is robust."
}}

Rules:
- Deltas must be small and conservative, usually between -0.15 and 0.15.
- Do not reverse the decision unless a counterfactual is strongly supported by evidence.
- Use counterfactuals such as removing OCR, changing target identity, in-group vs public context, satire direction, or missing source context.
""".strip()
        messages = [
            SystemMessage(
                content=(
                    f"{self.system_prompt}\n\n"
                    "You specialize in counterfactual reasoning for harmful meme detection."
                )
            ),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        raw_output = _normalize_content(getattr(response, "content", response))
        return self._parse_counterfactual_result(raw_output)

    def _parse_counterfactual_result(self, raw_output: str) -> CounterfactualResult:
        parsed = _extract_json_object(raw_output) or {}
        adjustment = parsed.get("recommended_adjustment")
        if not isinstance(adjustment, dict):
            adjustment = {}
        raw_label_deltas = adjustment.get("label_probability_deltas")
        label_deltas: dict[str, float] = {}
        if isinstance(raw_label_deltas, dict):
            for label in HARMFUL_LABELS:
                label_deltas[label] = _clamp(
                    raw_label_deltas.get(label),
                    minimum=-0.25,
                    maximum=0.25,
                )
        else:
            for label in HARMFUL_LABELS:
                label_deltas[label] = 0.0
        raw_tests = parsed.get("counterfactual_tests")
        tests = raw_tests if isinstance(raw_tests, list) else []
        robustness = str(parsed.get("robustness") or "medium").strip().lower()
        if robustness not in {"high", "medium", "low"}:
            robustness = "medium"
        return CounterfactualResult(
            tests=[item for item in tests if isinstance(item, dict)][:5],
            robustness=robustness,
            harmful_probability_delta=_clamp(
                adjustment.get("harmful_probability_delta"),
                minimum=-0.25,
                maximum=0.25,
            ),
            confidence_delta=_clamp(
                adjustment.get("confidence_delta"),
                minimum=-0.30,
                maximum=0.20,
            ),
            label_probability_deltas=label_deltas,
            summary=str(parsed.get("summary") or "").strip(),
            raw_output=raw_output,
        )

    def _dynamic_weight(
        self,
        result: PerspectiveAgentResult,
        evidence_context: str = "",
    ) -> float:
        text = evidence_context.lower()
        weight = result.weight
        if result.agent_name == "cross_cultural" and re.search(r"[\u4e00-\u9fff]", evidence_context):
            weight += 0.2
        if result.agent_name == "ingroup_context" and any(
            marker in text
            for marker in ("subreddit", "reddit", "weibo", "bilibili", "zhihu", "thread", "community")
        ):
            weight += 0.2
        if result.agent_name == "vulnerable_group_advocate" and any(
            marker in text
            for marker in (
                "disaster",
                "pandemic",
                "tragedy",
                "racial",
                "violence",
                "sensitive",
                "灾难",
                "疫情",
                "悲剧",
                "种族",
                "暴力",
                "敏感",
            )
        ):
            weight += 0.2
        if result.agent_name == "conservative_evidence" and (
            "no web or news results found" in text or len(evidence_context.strip()) < 300
        ):
            weight += 0.2
        confidence_factor = 0.5 + 0.5 * result.confidence
        return max(0.1, weight * confidence_factor)

    def _aggregate(
        self,
        results: list[PerspectiveAgentResult],
        counterfactual: CounterfactualResult | None = None,
        evidence_context: str = "",
    ) -> EnsembleDecision:
        if not results:
            return EnsembleDecision(
                harmful_probability=0.0,
                not_harmful_probability=0.0,
                unclear_probability=1.0,
                label_probabilities={label: 0.0 for label in HARMFUL_LABELS},
                decision="unclear",
                labels=["Unclear"],
                severity="unknown",
                confidence=0.0,
            )

        weights = [
            self._dynamic_weight(result, evidence_context=evidence_context)
            for result in results
        ]
        total_weight = sum(weights) or 1.0
        harmful_probability = sum(
            result.harmful_probability * weight
            for result, weight in zip(results, weights)
        ) / total_weight
        not_harmful_probability = sum(
            result.not_harmful_probability * weight
            for result, weight in zip(results, weights)
        ) / total_weight
        unclear_probability = sum(
            result.unclear_probability * weight
            for result, weight in zip(results, weights)
        ) / total_weight
        label_probabilities = {
            label: sum(
                result.label_probabilities.get(label, 0.0) * weight
                for result, weight in zip(results, weights)
            )
            / total_weight
            for label in HARMFUL_LABELS
        }
        confidence = sum(
            result.confidence * weight for result, weight in zip(results, weights)
        ) / total_weight

        severity_values = [
            (_severity_score(result.severity), weight)
            for result, weight in zip(results, weights)
            if _severity_score(result.severity) is not None
        ]
        if severity_values:
            severity_score = sum(score * weight for score, weight in severity_values if score is not None) / sum(
                weight for score, weight in severity_values if score is not None
            )
        else:
            severity_score = 0.0

        if counterfactual is not None:
            harmful_probability = _clamp(
                harmful_probability + counterfactual.harmful_probability_delta
            )
            not_harmful_probability = _clamp(
                not_harmful_probability - counterfactual.harmful_probability_delta
            )
            confidence = _clamp(confidence + counterfactual.confidence_delta)
            for label in HARMFUL_LABELS:
                label_probabilities[label] = _clamp(
                    label_probabilities[label]
                    + counterfactual.label_probability_deltas.get(label, 0.0)
                )

        selected_labels = [
            label for label, probability in label_probabilities.items() if probability >= 0.35
        ]
        if harmful_probability >= 0.65:
            decision = "harmful"
            if not selected_labels:
                selected_labels = [max(label_probabilities, key=label_probabilities.get)]
        elif harmful_probability >= 0.45 or unclear_probability >= 0.35:
            decision = "unclear"
            if not selected_labels:
                selected_labels = ["Unclear"]
        else:
            decision = "not harmful"
            selected_labels = ["Not harmful"]

        if decision == "not harmful":
            severity = "none"
        elif decision == "unclear" and not selected_labels:
            severity = "unknown"
        else:
            severity = _score_to_severity(severity_score)
            if decision == "harmful" and severity == "none":
                severity = "low"

        return EnsembleDecision(
            harmful_probability=harmful_probability,
            not_harmful_probability=not_harmful_probability,
            unclear_probability=unclear_probability,
            label_probabilities=label_probabilities,
            decision=decision,
            labels=selected_labels,
            severity=severity,
            confidence=confidence,
        )

    def _format_vote_summary(self, results: list[PerspectiveAgentResult]) -> str:
        lines = []
        for result in results:
            labels = ", ".join(
                f"{label}={_format_probability(probability)}"
                for label, probability in result.label_probabilities.items()
            )
            lines.append(
                f"- {result.title}: harmful={_format_probability(result.harmful_probability)}, "
                f"not_harmful={_format_probability(result.not_harmful_probability)}, "
                f"unclear={_format_probability(result.unclear_probability)}, "
                f"primary={result.primary_label}, severity={result.severity}, "
                f"confidence={_format_probability(result.confidence)}, labels=({labels})"
            )
        return "\n".join(lines)

    def _format_report(
        self,
        perspective_results: list[PerspectiveAgentResult],
        counterfactual: CounterfactualResult,
        preliminary: EnsembleDecision,
        final_decision: EnsembleDecision,
        evidence_context: str,
    ) -> str:
        label_lines = [
            f"- {label}: {_format_probability(probability)}"
            for label, probability in final_decision.label_probabilities.items()
        ]
        vote_lines = []
        for result in perspective_results:
            vote_lines.append(
                f"- {result.title}: harmful={_format_probability(result.harmful_probability)}, "
                f"primary={result.primary_label}, severity={result.severity}, "
                f"confidence={_format_probability(result.confidence)}, "
                f"weight={_format_probability(self._dynamic_weight(result, evidence_context=evidence_context))}"
            )

        evidence_lines = []
        for result in perspective_results:
            evidence = "; ".join(result.key_evidence) if result.key_evidence else "未提供明确证据摘录"
            summary = result.reasoning_summary or "未提供摘要"
            evidence_lines.append(f"- {result.title}: {summary} Evidence: {evidence}")

        uncertainty_items: list[str] = []
        for result in perspective_results:
            uncertainty_items.extend(result.uncertainties)
        deduped_uncertainties = []
        seen_uncertainties = set()
        for item in uncertainty_items:
            key = item.lower()
            if key in seen_uncertainties:
                continue
            seen_uncertainties.add(key)
            deduped_uncertainties.append(item)
            if len(deduped_uncertainties) >= 6:
                break

        counterfactual_lines = []
        for item in counterfactual.tests:
            condition = str(item.get("condition") or "未说明条件").strip()
            expected_change = str(item.get("expected_change") or "未说明变化").strip()
            reason = str(item.get("reason") or "未说明原因").strip()
            counterfactual_lines.append(
                f"- {condition}: {expected_change} Reason: {reason}"
            )
        if not counterfactual_lines:
            counterfactual_lines.append("- 未获得有效反事实测试；未应用额外语境翻转。")

        preliminary_delta = (
            final_decision.harmful_probability - preliminary.harmful_probability
        )
        final_labels = ", ".join(final_decision.labels)
        uncertainties = (
            "\n".join(f"- {item}" for item in deduped_uncertainties)
            if deduped_uncertainties
            else "- 当前子智能体没有报告额外不确定性。"
        )

        return f"""
1. Ensemble label
   - final_decision: {final_decision.decision}
   - harmfulness_labels: {final_labels}
   - harmful_probability: {_format_probability(final_decision.harmful_probability)}
   - not_harmful_probability: {_format_probability(final_decision.not_harmful_probability)}
   - unclear_probability: {_format_probability(final_decision.unclear_probability)}
   - severity: {final_decision.severity}
   - confidence: {_confidence_label(final_decision.confidence)} ({_format_probability(final_decision.confidence)})

2. Label probabilities
{chr(10).join(label_lines)}

3. Soft-vote breakdown
{chr(10).join(vote_lines)}

4. Multi-perspective evidence summaries
{chr(10).join(evidence_lines)}

5. Counterfactual reasoning
   - robustness: {counterfactual.robustness}
   - harmful_probability_delta: {preliminary_delta:+.2f}
   - confidence_delta: {counterfactual.confidence_delta:+.2f}
   - summary: {counterfactual.summary or "反事实智能体未提供摘要。"}
{chr(10).join(counterfactual_lines)}

6. Final rationale
   - 软投票先聚合不同背景子智能体的有害性概率，再用反事实推理检查结论是否依赖脆弱假设。
   - 最终标签来自概率阈值和多标签概率：harmful >= 0.65 判为 harmful；0.45-0.65 或不确定性较高判为 unclear；低于 0.45 判为 not harmful。
   - 所有关键判断应回到 [Image]、[User Context]、[W#]、[N#]、[R#-W#] 或 [Inference] 证据标签。

7. Uncertainty and missing evidence
{uncertainties}
""".strip()
