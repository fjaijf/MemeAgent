from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_harmful_memes import (
    EvalItem,
    _prediction_from_text,
    load_dataset,
    predict_workflow,
)


class FakeWorkflowResult:
    def __init__(self, formatted_output: str) -> None:
        self.formatted_output = formatted_output


class FakeWorkflow:
    def __init__(self) -> None:
        self.kwargs = None

    def run_heads(self, **kwargs):
        self.kwargs = kwargs
        return FakeWorkflowResult(
            """
## Harmfulness Analysis

1. Ensemble label
   - final_decision: harmful
   - harmfulness_labels: Offensive, Antagonism
   - harmful_probability: 0.91
   - confidence: high (0.88)
""".strip()
        )


class EvaluateHarmfulMemesTests(unittest.TestCase):
    def test_prediction_from_harmfulness_head_output(self) -> None:
        prediction = _prediction_from_text(
            """
## Harmfulness Analysis

1. Ensemble label
   - final_decision: not harmful
   - harmfulness_labels: Not harmful
   - harmful_probability: 0.21
   - confidence: high (0.82)
""".strip()
        )

        self.assertEqual(0, prediction.binary)
        self.assertEqual("Not harmful", prediction.label)

    def test_unclear_harmfulness_decision_stays_unknown(self) -> None:
        prediction = _prediction_from_text(
            """
1. Ensemble label
   - final_decision: unclear
   - harmfulness_labels: Offensive
   - harmful_probability: 0.51
""".strip()
        )

        self.assertIsNone(prediction.binary)

    def test_conversation_schema_extracts_path_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "Dataset" / "HarmP" / "Img" / "memes_48.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake")
            dataset_path = root / "Dataset" / "HarmP" / "CoT.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "image_48",
                            "conversations": [
                                {
                                    "from": "user",
                                    "value": (
                                        "Please judge:<|vision_start|>"
                                        f"{image_path}"
                                        "<|vision_start|>"
                                    ),
                                },
                                {
                                    "from": "assistant",
                                    "value": (
                                        "<JUDGEMENT>\n"
                                        "The image's label is harmful.\n"
                                        "</JUDGEMENT>"
                                    ),
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            items = load_dataset(
                dataset_path=dataset_path,
                image_root=None,
                id_field=None,
                image_field=None,
                label_field=None,
                text_field=None,
                schema="auto",
            )

        self.assertEqual(1, len(items))
        self.assertEqual("image_48", items[0].sample_id)
        self.assertEqual(1, items[0].label)
        self.assertTrue(items[0].image_path.endswith("memes_48.png"))

    def test_predict_workflow_passes_main_task_head_options(self) -> None:
        workflow = FakeWorkflow()
        item = EvalItem(
            sample_id="sample",
            image_path="/tmp/sample.png",
            label=1,
            context="ctx",
            raw={},
        )

        prediction = predict_workflow(
            workflow,  # type: ignore[arg-type]
            item,
            task_heads=["harmfulness"],
            use_search=False,
            force_search=False,
            iterative_search=False,
            search_max_rounds=5,
            controller_max_rounds=3,
            controller_confidence_threshold=0.8,
        )

        self.assertEqual(1, prediction.binary)
        assert workflow.kwargs is not None
        self.assertEqual(["harmfulness"], workflow.kwargs["task_heads"])
        self.assertFalse(workflow.kwargs["use_search"])
        self.assertEqual(["/tmp/sample.png"], workflow.kwargs["image_paths"])
        self.assertEqual("sample", workflow.kwargs["topic"])


if __name__ == "__main__":
    unittest.main()
