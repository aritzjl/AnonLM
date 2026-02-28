from __future__ import annotations

from pathlib import Path

from anonlm import cli
from anonlm.benchmarking.runner import BenchmarkRunResult
from anonlm.benchmarking.splits import BenchmarkSplit
from anonlm.config import AnonLMConfig
from anonlm.engine import AnonymizationResult, ChunkingMetadata


class FakeEngine:
    def anonymize(self, text: str) -> AnonymizationResult:
        return AnonymizationResult(
            anonymized_text=text.replace("Jane Doe", "[[PERSON_1]]"),
            mapping_forward={"Jane Doe": "[[PERSON_1]]"},
            mapping_reverse={"[[PERSON_1]]": "Jane Doe"},
            all_entities=[{"type": "PERSON", "text": "Jane Doe", "canonical": "Jane Doe", "token": "[[PERSON_1]]"}],
            type_counters={"PERSON": 1},
            chunking=ChunkingMetadata(
                chunk_count=1,
                chunks=[text],
                max_chunk_chars=1000,
                chunk_overlap_chars=100,
            ),
        )


def test_cli_anonymize_text(monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr(cli, "create_engine", lambda config: FakeEngine())

    code = cli.main(["anonymize", "--text", "Jane Doe", "--api-key", "test-key"])
    captured = capsys.readouterr()

    assert code == 0
    assert "[[PERSON_1]]" in captured.out
    assert '"chunking"' in captured.out


def test_cli_anonymize_file_output(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.json"
    input_path.write_text("Jane Doe", encoding="utf-8")

    monkeypatch.setattr(cli, "create_engine", lambda config: FakeEngine())

    code = cli.main(
        [
            "anonymize",
            "--file",
            str(input_path),
            "--output",
            str(output_path),
            "--api-key",
            "test-key",
        ]
    )

    assert code == 0
    output = output_path.read_text(encoding="utf-8")
    assert "[[PERSON_1]]" in output
    assert '"chunking"' in output


def test_cli_benchmark_run(monkeypatch) -> None:  # noqa: ANN001
    config = AnonLMConfig(api_key="test-key")
    expected_result = BenchmarkRunResult(
        split=BenchmarkSplit.DEV,
        rows=[],
        overall={"tp": 0.0, "fp": 0.0, "fn": 0.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        by_type={},
        exit_code=0,
        threshold_f1=0.8,
        config=config,
        artifact_path=None,
    )

    monkeypatch.setattr(cli, "create_engine", lambda cfg: object())
    monkeypatch.setattr(cli, "run_benchmark", lambda **kwargs: expected_result)

    code = cli.main(["benchmark", "run", "--dataset", "datasets/pii_mvp_dataset.csv", "--api-key", "test-key"])
    assert code == 0
