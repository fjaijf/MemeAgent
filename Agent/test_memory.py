from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memeagent.memory import MemeMemoryStore


class MemeMemoryStoreTest(unittest.TestCase):
    def test_recalls_previous_analysis_by_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemeMemoryStore(Path(tmp) / "memory.sqlite3")
            store.remember(
                topic="Distracted Boyfriend",
                image_paths=[],
                image_urls=[],
                input_mode="text_only",
                analysis="Prior analysis for the meme.",
            )

            records = store.recall(
                topic="  distracted   boyfriend ",
                image_paths=[],
                image_urls=[],
            )

            self.assertEqual(1, len(records))
            self.assertEqual("Prior analysis for the meme.", records[0].analysis)

    def test_recalls_previous_analysis_by_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "meme.png"
            image_path.write_bytes(b"fake image bytes")
            store = MemeMemoryStore(Path(tmp) / "memory.sqlite3")
            store.remember(
                topic="",
                image_paths=[str(image_path)],
                image_urls=[],
                input_mode="image_only",
                analysis="Prior image analysis.",
            )

            records = store.recall(
                topic="different topic",
                image_paths=[str(image_path)],
                image_urls=[],
            )

            self.assertEqual(1, len(records))
            self.assertEqual("Prior image analysis.", records[0].analysis)

    def test_formats_memory_as_prior_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemeMemoryStore(Path(tmp) / "memory.sqlite3")
            store.remember(
                topic="Example",
                image_paths=[],
                image_urls=[],
                input_mode="text_only",
                analysis="A remembered analysis.",
            )

            report = store.format_records(
                store.recall(topic="example", image_paths=[], image_urls=[])
            )

            self.assertIn("Local MemeAgent memory", report)
            self.assertIn("A remembered analysis.", report)

    def test_updates_topic_memory_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemeMemoryStore(Path(tmp) / "memory.sqlite3")
            analysis = """
1. Meme object and context
   - Claim: PEPE is used as a reaction meme.
2. Visual/OCR evidence
   - Claim: Green frog face.
3. Sentiment analysis
   - Claim: Ironic amusement.
4. Harmfulness analysis
   - Claim: Low harmfulness in this sample.
6. Intent recognition
   - Claim: Satirical reaction.
7. Evolution tracking
   - Claim: Variant reuse across platforms.
9. Evidence gaps and overall confidence
   - Claim: Source context remains limited.
""".strip()

            store.remember(
                topic="PEPE",
                image_paths=[],
                image_urls=[],
                input_mode="text_only",
                analysis=analysis,
            )
            store.remember(
                topic="pepe",
                image_paths=[],
                image_urls=[],
                input_mode="text_only",
                analysis=analysis.replace("Low harmfulness", "Ambiguous harmfulness"),
            )

            card = store.recall_card(" PEPE ")

            self.assertIsNotNone(card)
            self.assertEqual(2, card.artifact_count)
            self.assertIn("reaction meme", card.object_context)
            self.assertIn("Ambiguous harmfulness", card.harmfulness_notes)
            self.assertIn("Variant reuse", card.evolution_notes)


if __name__ == "__main__":
    unittest.main()
