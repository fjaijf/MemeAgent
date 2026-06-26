from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .rubrics import CONTROLLER_OUTPUT_SCHEMA, MEME_ANALYSIS_RUBRIC


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

        user_prompt = f"""
Topic hint: {topic or "None"}

User-provided context:
{context or "None"}

Look at the attached meme image(s) and produce a search-oriented visual description in Chinese.
Focus on details that can help retrieve related background information before the final analysis.
Pay special attention to searchable anchors: specific people, public figures, fictional characters,
organizations, locations, flags, uniforms, buildings, events, disasters, protests, elections,
platform UI, screenshots, watermarks, slogans, exact OCR text, and distinctive background scenes.

Return these sections:
1. OCR/text visible in the image: transcribe all visible text, preserving language and notable spelling.
2. Searchable visual anchors: list identifiable people, characters, organizations, places, background settings, events, platforms, logos, symbols, and objects. Mark uncertain identities as uncertain.
3. Event/background hypotheses: describe whether the image may refer to a specific news event, political moment, social incident, campaign, conflict, trend, or public controversy. Separate visual evidence from speculation.
4. Visual description: describe layout, expressions, gestures, colors, edits, screenshots, and meme composition.
5. Meme template or cultural references: identify possible meme templates, named figures, events, slogans, fandoms, communities, or platform conventions.
6. Harm/sentiment cues: list visible cues related to hostility, stereotyping, misinformation, persuasion, mobilization, irony, satire, fear, anger, ridicule, or ambiguity.
7. Search keywords: provide 8-20 concise keywords or phrases in both Chinese and English when useful. Put each high-value search phrase in double quotes. Include exact OCR phrases, named people, background/event terms, locations, organizations, fictional characters, media titles, and visual entities.
8. Suggested retrieval queries: provide 5-8 short web/news queries. Each query should contain 1-3 concrete searchable elements from the image, such as an OCR phrase, character name, show/movie title, public figure, event, location, or visible object. Do not add generic words like "meme", "memes", "meme template", "template", "Chinese text", "表情包", or "梗图" unless they are part of an official title or exact visible OCR.

Be descriptive rather than interpretive. If something is uncertain, mark it as uncertain.
""".strip()

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
        user_prompt = f"""
Topic hint: {topic or "None"}

Input mode: {input_mode}

User-provided context:
{context or "None"}

Image-derived visual report:
{visual_report or "None"}

Plan a small set of supplemental searches before meme analysis.
This plan must NOT replace the visual/OCR anchor queries. It should only add a few useful searches
when the topic, OCR, visual report, or user context clearly supports them.

Rules:
- Do not invent people, events, places, organizations, dates, or platforms not supported by the input.
- Prefer exact OCR phrases, named people, specific events, meme templates, locations, screenshots, or platform clues.
- Keep queries short. Avoid broad abstract phrases such as "meme harmfulness discourse analysis".
- If there is no strong concrete anchor for a supplemental query, write "None".
- News queries should be used only for concrete public figures, current/recent events, controversies, organizations, or incidents.

Return exactly these sections:
EVIDENCE_QUESTIONS:
- 2-4 concise questions that retrieval should help answer.

SUPPLEMENTAL_WEB_QUERIES:
- 0-3 short queries. Use "None" if no useful query exists.

SUPPLEMENTAL_NEWS_QUERIES:
- 0-2 short news-friendly queries. Use "None" if no useful news query exists.

QUERY_CAUTIONS:
- 1-3 cautions about what should not be assumed without evidence.
""".strip()

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
    ) -> str:
        user_prompt = f"""
Topic hint: {topic or "None"}

Input mode: {input_mode}

Current controller round: {round_index} of {max_rounds}
Finalization threshold: {confidence_threshold:.2f}

Project rubric:
{MEME_ANALYSIS_RUBRIC}

User/context evidence:
{context or "None"}

Latest multimodal image analysis:
{visual_report or "None"}

Latest/cumulative retrieval evidence:
{search_report or "None"}

Previous controller and multimodal iteration history:
{iteration_history or "None"}

You are the controller model for MemeAgent. Your role is to plan the next
analysis step under the project rubric, not to write the final answer unless
the evidence is already sufficient. Evaluate harmfulness, sentiment, audience,
intent, and evolution coverage. If confidence is below threshold, produce
concrete questions/instructions for the multimodal model and concrete retrieval
queries. If confidence is at or above threshold, set SHOULD_FINALIZE to yes and
keep follow-up requests as None.
The workflow will stop when ITERATION_CONFIDENCE is at or above the threshold.

Rules:
- Use the user's harmfulness criteria strictly, especially that any reference to sensitive events is offensive.
- Ask for image re-analysis when visual/OCR evidence is ambiguous or when harm/sentiment/intent/evolution cues need closer inspection.
- Ask for retrieval when source event, template, platform context, audience boundary, or evolution lineage needs external evidence.
- Do not invent entities, events, platforms, dates, or source IDs.
- The confidence score must reflect evidence sufficiency, not how plausible your current guess feels.

{CONTROLLER_OUTPUT_SCHEMA}
""".strip()

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

        user_prompt = f"""
Topic hint: {topic or "None"}

User/context evidence:
{context or "None"}

Previous image analysis:
{previous_visual_report or "None"}

Controller plan and focus questions:
{controller_plan or "None"}

Project rubric:
{MEME_ANALYSIS_RUBRIC}

Re-examine the attached meme image(s) in detail according to the controller's
focus questions. This is not the final report. Provide image-grounded evidence
that can be fed back to the controller and retrieval system.

Return these sections:
1. OCR and text verification:
   - exact visible text, language, spelling, placement, and uncertain characters.
2. Focus-question answers:
   - answer each controller question using visible evidence; say "not visible" when unsupported.
3. Harmfulness cues:
   - discrimination, offensive sensitive-event references, violence, vulgarity, antagonism.
4. Sentiment cues:
   - Joy, Sadness, Anger, Disgust, Fear, Surprise, including multimodal mismatch.
5. Audience and intent cues:
   - Gemeinschaft/Gesellschaft signals and Teleological/Normative/Dramaturgical/Communicative intent evidence.
6. Evolution/template cues:
   - template, visual drift, kernel fidelity, lifecycle hints, intertextual splicing.
7. Search anchors:
   - exact OCR phrases, names, symbols, platforms, template names, events, and 3-8 concrete search queries.

Separate visible evidence from inference. Do not invent unsupported identities,
events, platforms, dates, or source URLs.
""".strip()

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
- meme template, origin, or variant evidence
- event/news/background evidence
- harmfulness-relevant context such as target, audience, intent, and reception

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
        image_instruction = ""
        if image_count:
            image_instruction = (
                f"\nAttached images: {image_count}\n"
                "Please incorporate the visual evidence into the analysis. "
                "If the images contain memes, symbols, screenshots, comments, "
                "captions, edits, or social posts, explain what is visible and "
                "how it affects harmfulness, sentiment, audience reception, "
                "intent, and meme evolution.\n"
            )

        user_prompt = f"""
Topic: {topic}

Extra context:
{context or "None"}
{image_instruction}

Please produce a researcher-oriented meme analysis in Chinese unless the user asks otherwise.
Do not frame the topic as crypto, finance, or market speculation unless the topic or evidence explicitly requires it.

Project-specific analysis rubric:
{MEME_ANALYSIS_RUBRIC}

Evidence citation rules:
- Cite evidence for every important claim using source tags.
- Use [Image] for visible image/OCR evidence.
- Use [User Context] for information provided directly by the user.
- Use [W1], [W2], ... only for web search results with those exact IDs.
- Use [N1], [N2], ... only for news results with those exact IDs.
- If iterative retrieval is used, cite round-specific search labels exactly as shown, such as [R2-W1] or [R2-N1].
- Use [Inference] for reasoning that is not directly stated by a source.
- Do not invent source IDs. If no source supports a claim, say it is an inference or uncertain.
- Distinguish direct evidence from speculation.

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
3. Sentiment analysis
   - Use exactly one primary label from Joy, Sadness, Anger, Disgust, Fear, Surprise, plus optional secondary labels.
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
4. Harmfulness analysis
   - Apply the harmfulness labels Discrimination, Offensive, Violence, Vulgar, Antagonism; multiple labels are allowed.
   - Treat any reference to sensitive events as Offensive according to the project rubric.
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
5. Audience and reception prediction
   - Classify audience as Gemeinschaft-oriented, Gesellschaft-oriented, or mixed/uncertain.
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
6. Intent recognition
   - Classify intent as Teleological, Normative, Dramaturgical, Communicative, or mixed/uncertain.
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
7. Evolution tracking
   - Address phylogenetic/mutation cues, invariant kernel, lifecycle phase, and intertextual splicing when evidence allows.
   - Claim:
   - Evidence:
   - Confidence:
   - Uncertainty:
8. Evidence map
   - Explain what [Image], [User Context], [W#], [N#], and [Inference] were used for.
9. Evidence gaps and overall confidence
   - State what remains unsupported, what should be searched next, and give an overall confidence level.
""".strip()


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
