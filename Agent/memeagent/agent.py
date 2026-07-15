from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .rubrics import CONTROLLER_OUTPUT_SCHEMA, HARMFULNESS_DETECTION_RUBRIC


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item.strip())
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in text_parts if part)
    return str(content).strip()


class MemeAgent:
    """A minimal agent that directly calls an LLM with a prompt."""

    def __init__(self, llm: Any, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def _image_path_to_data_url(self, image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "image/png"

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _build_user_content_with_images(
        self,
        text: str,
        image_paths: list[str],
        image_urls: list[str],
    ) -> str | list[dict[str, Any]]:
        if not image_paths and not image_urls:
            return text

        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image_path in image_paths:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_path_to_data_url(image_path)},
                }
            )
        for image_url in image_urls:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return content_blocks

    def describe_images_for_search(
        self,
        topic: str = "",
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        image_urls = image_urls or []
        if not image_paths and not image_urls:
            return ""

        user_prompt = build_image_search_description_prompt(topic=topic, context=context)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self._build_user_content_with_images(
                    user_prompt,
                    image_paths,
                    image_urls,
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def plan_retrieval(
        self,
        topic: str = "",
        context: str = "",
        visual_report: str = "",
        input_mode: str = "text_only",
    ) -> str:
        user_prompt = build_retrieval_plan_prompt(
            topic=topic,
            context=context,
            visual_report=visual_report,
            input_mode=input_mode,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def plan_analysis_iteration(
        self,
        topic: str = "",
        context: str = "",
        visual_report: str = "",
        search_report: str = "",
        iteration_history: str = "",
        input_mode: str = "text_only",
        round_index: int = 1,
        max_rounds: int = 3,
        confidence_threshold: float = 0.8,
        retrieval_enabled: bool = True,
    ) -> str:
        user_prompt = build_analysis_iteration_prompt(
            topic=topic,
            context=context,
            visual_report=visual_report,
            search_report=search_report,
            iteration_history=iteration_history,
            input_mode=input_mode,
            round_index=round_index,
            max_rounds=max_rounds,
            confidence_threshold=confidence_threshold,
            retrieval_enabled=retrieval_enabled,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def analyze_images_for_plan(
        self,
        topic: str = "",
        context: str = "",
        controller_plan: str = "",
        previous_visual_report: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        image_urls = image_urls or []
        if not image_paths and not image_urls:
            return ""

        user_prompt = build_image_followup_analysis_prompt(
            topic=topic,
            context=context,
            controller_plan=controller_plan,
            previous_visual_report=previous_visual_report,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self._build_user_content_with_images(
                    user_prompt,
                    image_paths,
                    image_urls,
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def reflect_retrieval(
        self,
        topic: str = "",
        context: str = "",
        visual_report: str = "",
        retrieval_plan: str = "",
        search_report: str = "",
        input_mode: str = "text_only",
        round_index: int = 1,
        max_rounds: int = 3,
    ) -> str:
        user_prompt = f"""
Topic hint: {topic or "None"}

Input mode: {input_mode}

Current retrieval round: {round_index} of {max_rounds}

User-provided context:
{context or "None"}

Image-derived visual report:
{visual_report or "None"}

Initial retrieval plan:
{retrieval_plan or "None"}

Current cumulative search report:
{search_report or "None"}

You are the retrieval reflection planner for MemeAgent.
Your task is NOT to analyze harmfulness. Your task is only to decide whether more retrieval is needed,
identify evidence gaps, and propose the next small set of concrete searches.

Evaluate whether the current results are enough to support later meme analysis, especially:
- original post or repost candidates
- comment, reply, thread, or discussion context
- platform context
- exact OCR/text matches
- event/news/background evidence
- harmfulness-relevant context such as target, speaker stance, and harm mechanism

Rules:
- Do not invent unsupported people, platforms, events, usernames, dates, or URLs.
- Prefer exact OCR phrases, usernames, watermarks, visible platform clues, source URLs, named events, and candidate result titles.
- Use site-specific queries when the current results suggest a platform, such as site:zhihu.com, site:reddit.com, site:x.com, site:weibo.com.
- Keep queries short and concrete.
- Continue only if the next searches are likely to add useful evidence.
- Stop if the current evidence is already sufficient, if results are irrelevant, or if only broad generic searches remain.

Return exactly these sections:
RETRIEVAL_SCORE:
- Integer 0-10, where 10 means strong context for analysis.

SHOULD_CONTINUE:
- yes or no

STOP_REASON:
- One concise sentence.

EVIDENCE_GAPS:
- 1-5 bullets. Use "None" if there are no important gaps.

SUPPLEMENTAL_WEB_QUERIES:
- 0-5 short queries for the next round. Use "None" if no useful query exists.

SUPPLEMENTAL_NEWS_QUERIES:
- 0-2 short news-friendly queries. Use "None" if no useful query exists.

QUERY_CAUTIONS:
- 1-3 cautions about what should not be assumed without evidence.
""".strip()

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def run(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        messages = self._build_analysis_messages(
            topic=topic,
            context=context,
            image_paths=image_paths,
            image_urls=image_urls,
        )

        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

    def stream(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ):
        messages = self._build_analysis_messages(
            topic=topic,
            context=context,
            image_paths=image_paths,
            image_urls=image_urls,
        )

        if not hasattr(self.llm, "stream"):
            yield self.run(
                topic=topic,
                context=context,
                image_paths=image_paths,
                image_urls=image_urls,
            )
            return

        yield from self.llm.stream(messages)

    def _build_analysis_messages(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> list[Any]:
        image_paths = image_paths or []
        image_urls = image_urls or []
        image_count = len(image_paths) + len(image_urls)

        user_prompt = build_final_analysis_prompt(
            topic=topic,
            context=context,
            image_count=image_count,
        )

        user_content: str | list[dict[str, Any]]
        user_content = self._build_user_content_with_images(
            user_prompt,
            image_paths,
            image_urls,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]
        return messages


def build_image_search_description_prompt(topic: str = "", context: str = "") -> str:
    return f"""

You are doing the first multimodal pass for MemeAgent. Produce a 
research-oriented visual intelligence report that can support 
controller planning, harmfulness judgement, and final meme analysis. This is
not the final report. Be concrete, evidence-first, and exhaustive about visible
details that may matter later.

Core rules:
- Separate visible evidence from inference and hypotheses.
- Do not invent identities, platforms, dates, places, events, source URLs, or
  original-post context. Mark uncertain identities as uncertain.
- Preserve exact OCR text, spelling, punctuation, casing, emoji, hashtags,
  handles, watermarks, logos, UI labels, and language when visible.
- Pay special attention to potentially harmful cues.


Return exactly these sections:
1. OCR and textual evidence
   - Transcribe all visible text. Preserve original language and spelling.
   - Note placement, font emphasis, captions, speech bubbles, UI text, usernames,
     hashtags, URLs, watermarks, and uncertain characters.
2. Visual inventory
   - Describe people, characters, objects, symbols, gestures, facial expressions,
     background, layout, composition, editing, cropping, screenshots, colors,
     and any platform or media interface.
3. Image-text relation
   - Explain how text and image interact: captioning, contradiction, irony,
     reaction image, comparison, punchline, accusation, threat, ridicule, or
     ambiguity.
4. Harmfulness cues
   - Inspect all potential harmful features and cues within the meme.
8. Evidence gaps for the controller
   - List unresolved visual ambiguities, likely retrieval needs, and claims that
     must not be made without more evidence for the harmfulness decision.
""".strip()


def build_retrieval_plan_prompt(
    topic: str = "",
    context: str = "",
    visual_report: str = "",
    input_mode: str = "text_only",
) -> str:
    return f"""
Topic hint: {topic or "None"}

Input mode: {input_mode}

User-provided context:
{context or "None"}

Image-derived visual report:
{visual_report or "None"}

You are planning retrieval for a MemeAgent analysis. Convert the image-derived
visual report and user context into a small, high-precision retrieval plan. The
    goal is to fill evidence gaps for source context, event background,
    target/stance interpretation, and harmfulness-relevant context.

Rules:
- Do not invent people, events, places, organizations, dates, platforms, source
  IDs, or URLs not supported by the input.
- Prefer exact OCR phrases, visible names, public figures, specific incidents,
  template names, locations, screenshots, UI labels, watermarks, or symbols.
- Use news queries only for public figures, current/recent events, disasters,
  controversies, organizations, conflicts, or sensitive incidents.
- Keep each query short and concrete. Avoid broad abstract phrases such as
  "meme harmfulness discourse analysis".
- If there is no strong concrete anchor for a query, write "None" rather than
  inventing a weak query.
- Include cautions for any tempting but unsupported interpretation.

Return exactly these sections:
EVIDENCE_QUESTIONS:
- 3-6 concise questions retrieval should help answer. Cover source/event,
  target/stance, and harmfulness context when relevant.

HIGH_CONFIDENCE_ANCHORS:
- Exact visible OCR phrases, names, handles, symbols, watermarks, locations,
  UI/platform clues, or template terms that are safe to search.

UNCERTAIN_ANCHORS:
- Plausible but uncertain identities, events, or platform clues.
- State why each is uncertain.

SUPPLEMENTAL_WEB_QUERIES:
- 0-5 short queries. Use "None" if no useful query exists.

SUPPLEMENTAL_NEWS_QUERIES:
- 0-3 short news-friendly queries. Use "None" if no useful news query exists.

SEARCH_PRIORITY:
- Rank the top 1-3 queries and explain what each should resolve.

QUERY_CAUTIONS:
- 2-5 cautions about what should not be assumed without evidence.
""".strip()


def build_analysis_iteration_prompt(
    topic: str = "",
    context: str = "",
    visual_report: str = "",
    search_report: str = "",
    iteration_history: str = "",
    input_mode: str = "text_only",
    round_index: int = 1,
    max_rounds: int = 3,
    confidence_threshold: float = 0.8,
    retrieval_enabled: bool = True,
) -> str:
    return f"""
Topic hint: {topic or "None"}

Input mode: {input_mode}

Current controller round: {round_index} of {max_rounds}
Finalization threshold: {confidence_threshold:.2f}
External retrieval enabled: {"yes" if retrieval_enabled else "no"}


User/context evidence:
{context or "None"}

Latest multimodal image analysis:
{visual_report or "None"}


Previous controller and multimodal iteration history:
{iteration_history or "None"}

You are the controller model for MemeAgent. Your job is to audit whether the
current evidence is sufficient for a final meme harmfulness analysis
and, if not, plan the next targeted evidence-gathering step. You are not the
final writer unless the evidence is sufficient. Be strict about evidence quality
and concrete about what the multimodal model or retrieval should do next.

Coverage to audit:
- Visual/OCR evidence: exact text, visual entities, layout, UI/platform clues,
  symbols, gestures, editing, and image-text relation.
- Harmfulness cues.
- Evidence discipline: every important claim must map to image/OCR, user
  context, retrieved evidence, or explicit inference.

Rules:
- Use the harmfulness criteria carefully with label-specific evidence
  requirements. Lower confidence when important evidence is
  missing.
- Ask for image re-analysis only when pixels/OCR/layout/symbols can answer the
  question. Make each request specific and non-duplicative.
- Do not invent entities, events, platforms, dates, or source IDs.
- The confidence score must reflect evidence sufficiency, not how plausible the
  current interpretation feels.
- Penalize confidence when OCR is uncertain, target identity is unsupported,
  or harmfulness depends on external context that lacks evidence.
- Raise confidence only when remaining gaps would not materially change the
  final harmfulness conclusion.
- When external retrieval is disabled, only ask follow-up multimodal questions
  that can be answered by re-inspecting pixels, OCR, layout, symbols, visible UI,
  watermarks, or visual ambiguity. Do not repeat focus questions already answered
  in Previous controller and multimodal iteration history.
- When external retrieval is disabled and no new image-answerable question
  remains, set SHOULD_FINALIZE to yes, set multimodal/retrieval requests to
  None, and make the confidence score reflect best available offline certainty
  rather than missing web evidence.
- When asking follow-up questions, include what evidence would change and which
  section of the final analysis it affects.

{CONTROLLER_OUTPUT_SCHEMA}

Additional output guidance:
- KEY_FINDINGS_SO_FAR should distinguish direct evidence from inference.
- FOCUS_QUESTIONS should be answerable and prioritized; avoid generic requests
  like "analyze more carefully".
- MULTIMODAL_ANALYSIS_REQUESTS should point to exact visual tasks: OCR check,
  symbol identification, face/gesture/layout inspection, UI/watermark reading,
  or image-text relation.
- FINAL_OUTPUT_NOTES should tell the final model what harmfulness evidence to
  emphasize, what labels are supported, what remains uncertain, and what not to claim.
""".strip()


def build_image_followup_analysis_prompt(
    topic: str = "",
    context: str = "",
    controller_plan: str = "",
    previous_visual_report: str = "",
) -> str:
    return f"""
Topic hint: {topic or "None"}

User/context evidence:
{context or "None"}

Previous image analysis:
{previous_visual_report or "None"}

Controller plan and focus questions:
{controller_plan or "None"}


Re-examine the attached meme image(s) according to the controller's focus
questions. This is a targeted visual follow-up, not the final report. Your job
is to answer exactly what can be answered from the image pixels and to state
when something is not visible or remains uncertain.

Rules:
- Start from the controller plan; do not repeat the previous report unless you
  are correcting, confirming, or adding evidence.
- Answer each focus question directly. If an answer is unsupported, write
  "not visible", "uncertain", or "requires retrieval/context" and explain why.
- Separate confirmed visible evidence from inference.
- Do not invent identities, source events, platforms, dates, usernames, URLs,
  or speaker stance beyond what the image/context supports.
- Re-check exact OCR, including small text, UI labels, watermarks, captions,
  signs, usernames, emoji, punctuation, casing, and language.
- For harmfulness, inspect whether visible content supports Discrimination,
  Offensive, Violence, Vulgar, or Antagonism; assess whether sensitive-event
  framing mocks, trivializes, targets, or exploits harm.

Return these sections:
1. Direct answers to controller questions
   - Quote or paraphrase each question, then answer with visible evidence.
2. OCR and text verification
   - Exact visible text, language, spelling, placement, uncertain characters,
     watermarks, handles, hashtags, UI labels, and small text.
3. Corrected or newly observed visual evidence
   - New details or corrections versus the previous image analysis.
4. Harmfulness follow-up
   - For each relevant label, state supported / unsupported / uncertain and
     cite visible evidence. Include sensitive-event assessment when relevant,
     and separate mere mention from harmful framing.
5. Updated search anchors
   - Exact OCR phrases, names, symbols, platforms, templates, events, and 3-8
     concrete follow-up queries if useful.
6. Remaining uncertainty
   - Separate image-answerable gaps from retrieval/context-only gaps.
""".strip()


def build_final_analysis_prompt(
    topic: str,
    context: str = "",
    image_count: int = 0,
) -> str:
    image_instruction = ""
    if image_count:
        image_instruction = (
            f"\nAttached images: {image_count}\n"
            "Please incorporate the visual evidence into the analysis. "
            "If the images contain memes, symbols, screenshots, comments, "
            "captions, edits, or social posts, explain what is visible and "
            "how it affects the harmfulness decision, target, stance, and harm "
            "mechanism.\n"
        )

    return f"""
Topic: {topic}

Extra context:
{context or "None"}
{image_instruction}

Please produce a rigorous, researcher-oriented meme analysis. The final answer should synthesize visual evidence,
user context, retrieval evidence, controller notes, and explicit inference
without overstating unsupported claims. Do not frame the topic as crypto,
finance, or market speculation unless the topic or evidence explicitly requires it.


Analytical requirements:
- Harmfulness must explicitly judge whether the meme is harmful or harmless.
- Keep each section concise but evidence-dense. Avoid generic commentary.

Required sections:
1. Meme object and context
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
2. Visual/OCR evidence
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
3. Harmfulness analysis
   - Apply the harmful or harmless labels .
   - Lables:
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
""".strip()
