from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from batch_agent_evaluator import (
    SampleState,
    _apply_controller_decision,
    _call_controller,
    _call_final,
    _call_main,
    _controller_prompt,
    _final_record,
    _has_genuinely_new_evidence,
    _main_prompt,
    _normalize_final_analysis,
    _normalize_main_analysis,
    _normalize_controller_decision,
    _normalize_prediction_label,
    _prediction_binary_from_values,
    _redacted_argv,
)


class FakeEndpoint:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return next(self.outputs)


class BatchAgentEvaluatorTests(unittest.TestCase):
    def test_command_metadata_redacts_api_key(self) -> None:
        self.assertEqual(
            [
                "batch_agent_evaluator.py",
                "--api-key",
                "<redacted>",
                "--dataset=data.jsonl",
                "--api-key=<redacted>",
            ],
            _redacted_argv(
                [
                    "batch_agent_evaluator.py",
                    "--api-key",
                    "secret-one",
                    "--dataset=data.jsonl",
                    "--api-key=secret-two",
                ]
            ),
        )

    def test_final_record_excludes_duplicate_intermediate_outputs(self) -> None:
        sample = SampleState(
            sample_id="sample-1",
            image_path=Path("meme.png"),
            final_answer="Final analysis.",
            main_outputs=[{"output": "Main analysis."}],
            controller_outputs=[{"decision": {"reason": "Ready."}}],
            final_outputs=[
                {
                    "output": "Final analysis.",
                    "decision": {"final_answer": "Final analysis."},
                }
            ],
        )

        record = _final_record(sample)

        self.assertEqual("Final analysis.", record["final_answer"])
        self.assertNotIn("last_main_output", record)
        self.assertNotIn("last_controller_decision", record)
        self.assertNotIn("last_final_output", record)
        self.assertNotIn("last_final_decision", record)
        self.assertEqual("controller", record["prediction_source"])

    def test_controller_prompt_allows_implicit_image_text_targeting(self) -> None:
        sample = SampleState(
            sample_id="sample-1",
            image_path=Path("meme.png"),
            context="ambiguous caption",
        )

        prompt = _controller_prompt(
            sample,
            confidence_threshold=0.8,
            max_rounds=3,
        )

        self.assertIn(
            "Image-text\ncombinations may provide an implicit target",
            prompt,
        )
        self.assertIn("do not default to harmless", prompt)
        self.assertNotIn(
            "Choose harmful only when the\nevidence supports a target, stance, and concrete harm mechanism",
            prompt,
        )

    def test_uncertain_harmful_wording_falls_back_to_harmless(self) -> None:
        self.assertEqual(
            "harmless",
            _normalize_prediction_label("The meme is not clearly harmful."),
        )
        self.assertEqual(
            0,
            _prediction_binary_from_values(
                None,
                "",
                "",
                "There is insufficient evidence to classify the meme as harmful.",
            )
        )

    def test_controller_decision_tolerates_missing_harmfulness_field(self) -> None:
        decision = _normalize_controller_decision(
            output='{"should_finalize": true, "confidence": 0.9}',
            parsed={"should_finalize": True, "confidence": 0.9},
            confidence_threshold=0.8,
            is_last_round=False,
        )

        self.assertTrue(decision["should_finalize"])
        self.assertEqual("", decision["harmfulness_analysis"])

    def test_main_prompt_requires_structured_evidence_without_visible_thinking(self) -> None:
        sample = SampleState(sample_id="sample-1", image_path=Path("meme.png"))

        prompt = _main_prompt(sample, round_index=0)

        self.assertIn('"direct_answers"', prompt)
        self.assertIn('"new_evidence"', prompt)
        self.assertIn("requires_retrieval", prompt)
        self.assertIn("Do not output planning, a thinking process", prompt)

    def test_main_analysis_normalizes_answerability_and_new_evidence(self) -> None:
        result = _normalize_main_analysis(
            {
                "direct_answers": [
                    {
                        "question": "What is the source?",
                        "answer": "Not visible.",
                        "answerability": "requires_retrieval",
                        "confidence": 0.9,
                    }
                ],
                "new_evidence": [],
                "retrieval_required": True,
            }
        )

        self.assertEqual("requires_retrieval", result["direct_answers"][0]["answerability"])
        self.assertEqual([], result["new_evidence"])
        self.assertTrue(result["retrieval_required"])

    def test_repeated_evidence_does_not_count_as_progress(self) -> None:
        sample = SampleState(
            sample_id="sample-1",
            image_path=Path("meme.png"),
            main_outputs=[
                {"parsed": {"visible_evidence": ["No watermark is visible."]}}
            ],
        )

        self.assertFalse(
            _has_genuinely_new_evidence(
                sample,
                {"new_evidence": ["  no WATERMARK is visible. "]},
            )
        )

    def test_retrieval_only_controller_questions_finalize_image_only_sample(self) -> None:
        decision = _normalize_controller_decision(
            output="",
            parsed={
                "should_finalize": False,
                "confidence": 0.6,
                "prediction_label": "harmless",
                "prediction_binary": 0,
                "next_questions": [],
                "retrieval_questions": ["Verify the statistic source."],
            },
            confidence_threshold=0.8,
            is_last_round=False,
        )

        self.assertTrue(decision["should_finalize"])
        self.assertEqual(["Verify the statistic source."], decision["retrieval_questions"])

    def test_no_new_followup_evidence_stops_visual_iteration(self) -> None:
        sample = SampleState(
            sample_id="sample-1",
            image_path=Path("meme.png"),
            no_progress_rounds=1,
        )

        _apply_controller_decision(
            sample,
            {
                "should_finalize": False,
                "prediction_label": "harmless",
                "prediction_binary": 0,
                "next_questions": ["Inspect the same OCR again."],
                "retrieval_questions": [],
                "reason": "Still uncertain.",
            },
        )

        self.assertEqual("needs_final", sample.status)
        self.assertIn("No new image-grounded evidence", sample.final_reason)

    def test_final_analysis_cannot_override_controller_prediction(self) -> None:
        sample = SampleState(
            sample_id="sample-1",
            image_path=Path("meme.png"),
            prediction_label="harmful",
            prediction_binary=1,
            final_reason="controller evidence",
        )

        decision = _normalize_final_analysis(
            sample=sample,
            output='{"prediction_label":"harmless","prediction_binary":0}',
            parsed={"prediction_label": "harmless", "prediction_binary": 0},
        )

        self.assertEqual("harmful", decision["prediction_label"])
        self.assertEqual(1, decision["prediction_binary"])
        self.assertEqual("controller", decision["prediction_source"])

    def test_default_flow_uses_main_controller_and_text_only_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "meme.png"
            image_path.write_bytes(b"test-image")
            sample = SampleState(sample_id="sample-1", image_path=image_path)
            endpoint = FakeEndpoint(
                [
                    "Visible OCR and image evidence.",
                    '{"should_finalize":true,"confidence":0.9,'
                    '"prediction_label":"harmful","prediction_binary":1,'
                    '"harmfulness_labels":["Offensive"],'
                    '"harmfulness_analysis":"Direct ridicule is supported by visible evidence.",'
                    '"reason":"Evidence is sufficient","next_questions":[]}',
                    "4. Harmfulness analysis\nThe meme is harmful because it "
                    "uses visible ridicule against a target. [Image]\n\n"
                    "5. Audience and reception prediction\nMixed reception.",
                ]
            )

            with patch(
                "batch_agent_evaluator._main_system_prompt",
                return_value="system",
            ):
                _call_main(
                    input_index=0,
                    sample=sample,
                    endpoint=endpoint,  # type: ignore[arg-type]
                    model="main-model",
                    round_index=0,
                    temperature=0,
                    max_tokens=100,
                    save_prompts=False,
                )
                self.assertEqual("needs_controller", sample.status)

                _call_controller(
                    input_index=0,
                    sample=sample,
                    endpoint=endpoint,  # type: ignore[arg-type]
                    model="controller-model",
                    round_index=0,
                    temperature=0,
                    max_tokens=100,
                    confidence_threshold=0.8,
                    max_rounds=3,
                    is_last_round=False,
                    save_prompts=False,
                )
                self.assertEqual("needs_final", sample.status)

                _call_final(
                    input_index=0,
                    sample=sample,
                    endpoint=endpoint,  # type: ignore[arg-type]
                    model="main-model",
                    temperature=0,
                    max_tokens=200,
                    save_prompts=False,
                )

            self.assertEqual("final", sample.status)
            self.assertEqual("harmful", sample.prediction_label)
            self.assertEqual(1, sample.prediction_binary)
            self.assertIn("The meme is harmful", sample.final_answer)
            self.assertEqual("main-model", endpoint.calls[0]["model"])
            self.assertEqual("controller-model", endpoint.calls[1]["model"])
            self.assertEqual("main-model", endpoint.calls[2]["model"])

            initial_content = endpoint.calls[0]["messages"]  # type: ignore[index]
            final_content = endpoint.calls[2]["messages"]  # type: ignore[index]
            self.assertIsInstance(initial_content[1]["content"], list)
            self.assertIsInstance(final_content[1]["content"], str)
            self.assertNotIn("image_url", str(final_content))


if __name__ == "__main__":
    unittest.main()
