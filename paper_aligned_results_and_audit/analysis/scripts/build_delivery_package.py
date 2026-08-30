"""Build the final round-2 delivery from an explicit, fail-closed whitelist.

The builder never packages a workspace by recursion.  It validates final
documents and the canonical result manifest, stages selected files in a
same-volume temporary directory, writes a SHA-256 manifest, creates a temporary
ZIP, and only then atomically renames the two deliverables.  Existing targets
are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid
import zipfile


PACKAGE_NAME = "sensors_round2_revision_final"
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "VERIFIED_RESULT",
    "MANUSCRIPT_LOCATION",
)

# These phrases require contextual human review.  They are deliberately not
# hard failures: for example, a limitation may correctly say that a method is
# *not* universally superior.  The generated audit records literal hits and
# keeps the decision with the manuscript author.
MANUAL_TEXT_AUDIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("extreme_p_value", re.compile(r"\bp\s*<\s*0\.0001\b", re.IGNORECASE)),
    ("strict_markov", re.compile(r"\bstrict\s+Markov\b", re.IGNORECASE)),
    (
        "markov_sufficient",
        re.compile(r"\bMarkov[-\s]+sufficient\b", re.IGNORECASE),
    ),
    (
        "universal_superiority",
        re.compile(r"\buniversally\s+superior\b", re.IGNORECASE),
    ),
)

EXPERIMENT_ROOT_FILES: tuple[str, ...] = (
    "README.md",
    "PROTOCOL.md",
    "PROTOCOL_AMENDMENT_2026-08-10.md",
    "profile_model_costs.py",
    "requirements.txt",
    "run_experiment.py",
)
DATA_AUDIT_REQUIRED_FILES: tuple[str, ...] = (
    "data_audit.json",
    "source_file_manifest.csv",
)
FORMAL_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "FROZEN_BEFORE_EVALUATION.json",
    "PREDECLARED_EVALUATION_SCHEDULE.json",
    "evaluation_episodes.csv",
    "seed_summary.csv",
    "run_summary.json",
)
FORMAL_OPTIONAL_FILES: tuple[str, ...] = (
    "software.json",
    "normalizer.json",
    "evaluation_conditions.json",
    "evaluation_schedule.csv",
    "training_history.csv",
    "primary_comparisons.csv",
    "primary_comparisons.json",
    "model_costs.csv",
)
ANALYSIS_REQUIRED_BASENAMES: tuple[str, ...] = (
    "isolated_model_costs.csv",
    "isolated_model_costs.json",
    "polynomial_branch_weight_audit_seed_level.csv",
    "polynomial_branch_weight_audit_summary.json",
    "polynomial_branch_weight_plot_source.csv",
    "main_performance.pdf",
    "main_performance_source_data.csv",
    "primary_forest.pdf",
    "primary_forest_source_data.csv",
    "ablation_forest.pdf",
    "ablation_forest_source_data.csv",
    "captions.md",
)
ANALYSIS_OPTIONAL_BASENAMES: tuple[str, ...] = (
    "main_performance.png",
    "main_performance.svg",
    "primary_forest.png",
    "primary_forest.svg",
    "ablation_forest.png",
    "ablation_forest.svg",
    "hnp_architecture_48d.pdf",
    "hnp_architecture_48d.png",
    "hnp_architecture_48d.svg",
    "training_snr_direction.pdf",
    "training_snr_direction.png",
    "training_snr_direction.svg",
    "training_snr_direction_source_data.csv",
    "INDEPENDENT_QA.md",
)
ANALYSIS_SCRIPT_BASENAMES: tuple[str, ...] = (
    "audit_polynomial_branch_weights.py",
    "build_delivery_package.py",
    "build_revision_artifacts.py",
    "make_clean_tex.py",
    "map_manuscript_locations.py",
    "plot_hnp_architecture.py",
    "plot_revision_results.py",
    "plot_training_snr_direction.py",
    "summarize_revision_results.py",
)
DEFINITION_ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".tex",
        ".bib",
        ".bst",
        ".cls",
        ".sty",
        ".def",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".eps",
        ".txt",
    }
)
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".tex", ".md", ".txt", ".csv", ".json", ".bib", ".cls", ".sty", ".bst"}
)
FORBIDDEN_PATH_PARTS: frozenset[str] = frozenset(
    {
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "raw",
        "smoke",
        "logs",
        "log",
        "preview",
        "pdf_preview",
    }
)


class DeliveryValidationError(RuntimeError):
    """Raised before publication when a delivery invariant is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliveryValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeliveryValidationError(f"JSON root must be an object: {path}")
    return payload


def _assert_regular_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DeliveryValidationError(f"missing {label}: {path}")
    if path.is_symlink():
        raise DeliveryValidationError(f"symlinks are not allowed: {path}")
    if path.stat().st_size <= 0:
        raise DeliveryValidationError(f"empty {label}: {path}")
    return path


def _validate_pdf(path: Path, label: str) -> None:
    _assert_regular_file(path, label)
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise DeliveryValidationError(f"invalid PDF header for {label}: {path}")


def _docx_visible_text(path: Path) -> str:
    _assert_regular_file(path, "response DOCX")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise DeliveryValidationError(f"invalid DOCX structure: {path}")
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise DeliveryValidationError(f"invalid DOCX zip: {path}") from exc
    # Join visible XML text even when a marker was split across Word runs.
    return "".join(re.findall(r">([^<>]*)<", xml)).replace("&lt;", "<").replace("&gt;", ">")


def _forbidden_source(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    for part in lowered_parts:
        if part in FORBIDDEN_PATH_PARTS or part.startswith("smoke"):
            return part
        if "aborted" in part or "formal_primary_v1" in part or "formal_primary_v2" in part:
            return part
    if path.suffix.lower() in {".log", ".aux", ".out", ".pyc", ".tmp"}:
        return path.suffix.lower()
    return None


def _check_placeholder_text(path: Path) -> list[str]:
    # Python builders intentionally contain the sentinel literals that enforce
    # this gate; final manuscript/response and data artifacts must not.
    if path.suffix.lower() == ".docx":
        text = _docx_visible_text(path)
    elif path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        return []
    return [marker for marker in PLACEHOLDER_MARKERS if marker in text]


def _auditable_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _docx_visible_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _manual_text_audit_rows(selected: Mapping[Path, Path]) -> list[dict[str, str]]:
    """Return literal-risk hits without deciding whether their context is valid."""

    scoped_destinations = (
        Path("manuscript/template.tex"),
        Path("response/response_final.md"),
        Path("response/response_final.docx"),
    )
    rows: list[dict[str, str]] = []
    detected_ids: set[str] = set()
    for destination in scoped_destinations:
        source = selected[destination]
        text = _auditable_text(source)
        for risk_id, pattern in MANUAL_TEXT_AUDIT_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                context_start = max(0, match.start() - 90)
                context_end = min(len(text), match.end() + 90)
                context = re.sub(r"\s+", " ", text[context_start:context_end]).strip()
                rows.append(
                    {
                        "risk_id": risk_id,
                        "document": destination.as_posix(),
                        "line_or_run": str(line),
                        "matched_text": match.group(0),
                        "context": context,
                        "automated_status": "literal_hit_requires_context_review",
                        "required_manual_action": (
                            "Confirm this is not an unsupported affirmative claim; "
                            "negative/limitation wording may be retained."
                        ),
                    }
                )
                detected_ids.add(risk_id)
    for risk_id, pattern in MANUAL_TEXT_AUDIT_PATTERNS:
        if risk_id not in detected_ids:
            rows.append(
                {
                    "risk_id": risk_id,
                    "document": "manuscript/template.tex; response/response_final.md; "
                    "response/response_final.docx",
                    "line_or_run": "",
                    "matched_text": "",
                    "context": f"No literal hit for automated pattern: {pattern.pattern}",
                    "automated_status": "not_detected_by_literal_scan",
                    "required_manual_action": (
                        "Still review equivalent paraphrases before submission."
                    ),
                }
            )
    return rows


def _write_manual_text_audit(stage: Path, rows: Sequence[Mapping[str, str]]) -> Path:
    path = stage / "MANUAL_TEXT_AUDIT.csv"
    fieldnames = (
        "risk_id",
        "document",
        "line_or_run",
        "matched_text",
        "context",
        "automated_status",
        "required_manual_action",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _resolve_manifest_run_path(
    raw_path: str, *, workspace_root: Path, manifest_path: Path
) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    alternatives = (
        manifest_path.parent / candidate,
        workspace_root / candidate,
    )
    for alternative in alternatives:
        if alternative.exists():
            return alternative.resolve()
    return alternatives[0].resolve()


def _find_unique_by_basename(root: Path, basename: str, *, required: bool) -> Path | None:
    matches = [
        path.resolve()
        for path in root.rglob(basename)
        if path.is_file() and not path.is_symlink()
    ]
    if len(matches) > 1:
        raise DeliveryValidationError(
            f"ambiguous analysis artifact {basename!r}: {matches}"
        )
    if not matches:
        if required:
            raise DeliveryValidationError(f"missing analysis artifact {basename!r}")
        return None
    return matches[0]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not label:
        raise DeliveryValidationError(f"invalid manifest run label: {value!r}")
    return label


def collect_whitelisted_files(
    *,
    workspace_root: str | Path,
    manuscript_tex: str | Path,
    manuscript_pdf: str | Path,
    response_md: str | Path,
    response_docx: str | Path,
    response_pdf: str | Path,
    canonical_manifest: str | Path,
) -> dict[Path, Path]:
    """Return ``destination-relative -> source`` after all source validation."""

    workspace_root = Path(workspace_root).expanduser().resolve()
    if not workspace_root.is_dir():
        raise DeliveryValidationError(f"workspace root does not exist: {workspace_root}")
    manuscript_tex = _assert_regular_file(Path(manuscript_tex), "manuscript TEX")
    manuscript_pdf = _assert_regular_file(Path(manuscript_pdf), "manuscript PDF")
    response_md = _assert_regular_file(Path(response_md), "final response Markdown")
    response_docx = _assert_regular_file(Path(response_docx), "final response DOCX")
    response_pdf = _assert_regular_file(Path(response_pdf), "final response PDF")
    canonical_manifest = _assert_regular_file(
        Path(canonical_manifest), "canonical result manifest"
    )
    _validate_pdf(manuscript_pdf, "manuscript PDF")
    _validate_pdf(response_pdf, "response PDF")
    _docx_visible_text(response_docx)

    manifest = _read_json(canonical_manifest)
    if str(manifest.get("status", "")).lower() != "validated":
        raise DeliveryValidationError(
            "canonical result manifest status must be exactly 'validated'"
        )
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise DeliveryValidationError("canonical manifest has no validated runs")

    selected: dict[Path, Path] = {}

    def add(destination: str | Path, source: str | Path) -> None:
        destination = Path(destination)
        if destination.is_absolute() or ".." in destination.parts:
            raise DeliveryValidationError(f"unsafe delivery destination: {destination}")
        source = _assert_regular_file(Path(source), f"source for {destination}")
        forbidden = _forbidden_source(source)
        if forbidden:
            raise DeliveryValidationError(
                f"forbidden source path component/type {forbidden!r}: {source}"
            )
        previous = selected.get(destination)
        if previous is not None and previous != source:
            raise DeliveryValidationError(
                f"delivery collision at {destination}: {previous} vs {source}"
            )
        selected[destination] = source

    add("manuscript/template.tex", manuscript_tex)
    add("manuscript/template.pdf", manuscript_pdf)
    clean_tex = manuscript_tex.parent / "template_clean.tex"
    clean_pdf = manuscript_tex.parent / "template_clean.pdf"
    if clean_tex.exists() != clean_pdf.exists():
        missing = clean_pdf if clean_tex.exists() else clean_tex
        raise DeliveryValidationError(
            "clean manuscript must be supplied as a complete TEX/PDF pair; "
            f"missing: {missing}"
        )
    if clean_tex.exists():
        _validate_pdf(clean_pdf, "clean manuscript PDF")
        add("manuscript/clean/template.tex", clean_tex)
        add("manuscript/clean/template.pdf", clean_pdf)
    definitions = manuscript_tex.parent / "Definitions"
    if not definitions.is_dir():
        raise DeliveryValidationError(f"missing manuscript Definitions: {definitions}")
    definition_files = [path for path in definitions.rglob("*") if path.is_file()]
    if not definition_files:
        raise DeliveryValidationError("manuscript Definitions is empty")
    for source in definition_files:
        if source.suffix.lower() not in DEFINITION_ALLOWED_SUFFIXES:
            continue
        add(Path("manuscript/Definitions") / source.relative_to(definitions), source)

    add("response/response_final.md", response_md)
    add("response/response_final.docx", response_docx)
    add("response/response_final.pdf", response_pdf)
    add(
        "response/build_response_docx.py",
        workspace_root / "response" / "build_response_docx.py",
    )

    experiment_root = workspace_root / "experiment_v2"
    for subtree in ("src", "tests"):
        directory = experiment_root / subtree
        if not directory.is_dir():
            raise DeliveryValidationError(f"missing experiment subtree: {directory}")
        python_files = [path for path in directory.rglob("*.py") if path.is_file()]
        if not python_files:
            raise DeliveryValidationError(f"no Python files in {directory}")
        for source in python_files:
            add(Path("experiment_v2") / source.relative_to(experiment_root), source)
    for filename in EXPERIMENT_ROOT_FILES:
        add(Path("experiment_v2") / filename, experiment_root / filename)
    data_audit_root = experiment_root / "results" / "data_audit"
    for filename in DATA_AUDIT_REQUIRED_FILES:
        add(
            Path("experiment_v2/results/data_audit") / filename,
            data_audit_root / filename,
        )

    add("analysis/result_manifest.json", canonical_manifest)
    analysis_root = workspace_root / "analysis"
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict) or not outputs:
        raise DeliveryValidationError("canonical manifest has no validated output artifacts")
    for output_label, output_info in outputs.items():
        if not isinstance(output_info, dict) or "path" not in output_info:
            raise DeliveryValidationError(f"invalid manifest output {output_label!r}")
        source = (canonical_manifest.parent / str(output_info["path"])).resolve()
        source = _assert_regular_file(source, f"manifest output {output_label}")
        expected_hash = output_info.get("sha256")
        if expected_hash and _sha256(source) != str(expected_hash):
            raise DeliveryValidationError(f"manifest output hash mismatch: {source}")
        add(Path("analysis") / source.name, source)

    for basename in ANALYSIS_REQUIRED_BASENAMES:
        source = _find_unique_by_basename(analysis_root, basename, required=True)
        assert source is not None
        add(Path("analysis") / source.relative_to(analysis_root), source)
    for basename in ANALYSIS_OPTIONAL_BASENAMES:
        source = _find_unique_by_basename(analysis_root, basename, required=False)
        if source is not None:
            add(Path("analysis") / source.relative_to(analysis_root), source)
    for basename in ANALYSIS_SCRIPT_BASENAMES:
        source = analysis_root / basename
        if source.is_file():
            add(Path("analysis/scripts") / basename, source)
    root_source_manifest = workspace_root / "SOURCE_MANIFEST.md"
    if root_source_manifest.is_file():
        add("analysis/SOURCE_MANIFEST.md", root_source_manifest)

    for run_label, run_info in runs.items():
        if not isinstance(run_info, dict) or "path" not in run_info:
            raise DeliveryValidationError(f"invalid canonical run {run_label!r}")
        run_dir = _resolve_manifest_run_path(
            str(run_info["path"]),
            workspace_root=workspace_root,
            manifest_path=canonical_manifest,
        )
        forbidden = _forbidden_source(run_dir)
        if forbidden:
            raise DeliveryValidationError(
                f"canonical run points to forbidden/obsolete path {run_dir}"
            )
        if not run_dir.is_dir():
            raise DeliveryValidationError(f"canonical run directory missing: {run_dir}")
        destination_root = Path("formal_artifacts") / _safe_label(str(run_label))
        for filename in FORMAL_REQUIRED_FILES:
            add(destination_root / filename, run_dir / filename)
        for filename in FORMAL_OPTIONAL_FILES:
            source = run_dir / filename
            if source.is_file():
                add(destination_root / filename, source)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
        if not checkpoints:
            raise DeliveryValidationError(f"no checkpoints in canonical run: {run_dir}")
        for checkpoint in checkpoints:
            add(destination_root / "checkpoints" / checkpoint.name, checkpoint)

    placeholder_failures: list[str] = []
    for destination, source in selected.items():
        for marker in _check_placeholder_text(source):
            placeholder_failures.append(f"{destination}: {marker}")
    if placeholder_failures:
        raise DeliveryValidationError(
            "unresolved manuscript/response placeholders: "
            + "; ".join(sorted(placeholder_failures))
        )
    return selected


def _write_delivery_manifest(stage: Path) -> Path:
    manifest_path = stage / "DELIVERY_MANIFEST.csv"
    rows: list[tuple[str, int, str]] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            (
                path.relative_to(stage).as_posix(),
                path.stat().st_size,
                _sha256(path),
            )
        )
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        writer.writerows(rows)
    return manifest_path


def build_delivery_package(
    *,
    workspace_root: str | Path,
    manuscript_tex: str | Path,
    manuscript_pdf: str | Path,
    response_md: str | Path,
    response_docx: str | Path,
    response_pdf: str | Path,
    canonical_manifest: str | Path,
    output_parent: str | Path,
) -> tuple[Path, Path]:
    """Validate, stage and atomically publish the directory and same-name ZIP."""

    selected = collect_whitelisted_files(
        workspace_root=workspace_root,
        manuscript_tex=manuscript_tex,
        manuscript_pdf=manuscript_pdf,
        response_md=response_md,
        response_docx=response_docx,
        response_pdf=response_pdf,
        canonical_manifest=canonical_manifest,
    )
    manual_audit_rows = _manual_text_audit_rows(selected)
    output_parent = Path(output_parent).expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    target = output_parent / PACKAGE_NAME
    zip_target = output_parent / f"{PACKAGE_NAME}.zip"
    if target.exists() or zip_target.exists():
        raise FileExistsError(
            f"refusing to overwrite delivery target: {target} or {zip_target}"
        )

    stage = Path(
        tempfile.mkdtemp(prefix=f".{PACKAGE_NAME}.stage-", dir=output_parent)
    )
    zip_stage = output_parent / f".{PACKAGE_NAME}.zip-stage-{uuid.uuid4().hex}"
    target_published = False
    zip_published = False
    try:
        for destination, source in sorted(selected.items(), key=lambda item: item[0].as_posix()):
            destination_path = stage / destination
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_path)
            if destination_path.stat().st_size != source.stat().st_size:
                raise DeliveryValidationError(f"copy size mismatch: {source}")
            if _sha256(destination_path) != _sha256(source):
                raise DeliveryValidationError(f"copy hash mismatch: {source}")
        _write_manual_text_audit(stage, manual_audit_rows)
        _write_delivery_manifest(stage)

        with zipfile.ZipFile(
            zip_stage, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=(Path(PACKAGE_NAME) / path.relative_to(stage)).as_posix(),
                    )
        with zipfile.ZipFile(zip_stage) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise DeliveryValidationError(f"ZIP CRC failure: {bad_member}")

        os.replace(stage, target)
        target_published = True
        os.replace(zip_stage, zip_target)
        zip_published = True
        hit_count = sum(
            row["automated_status"] == "literal_hit_requires_context_review"
            for row in manual_audit_rows
        )
        print(
            "Manual text audit reminder: "
            f"{hit_count} literal high-risk phrase hit(s); review "
            f"{target / 'MANUAL_TEXT_AUDIT.csv'} in context before submission.",
            file=sys.stderr,
        )
        return target, zip_target
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if zip_stage.exists():
            zip_stage.unlink()
        # Roll back only outputs created by this invocation.
        if target_published and target.exists() and not zip_published:
            shutil.rmtree(target)
        if zip_published and zip_target.exists() and not target_published:
            zip_target.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and build the final round-2 delivery package."
    )
    workspace_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument(
        "--manuscript-tex",
        type=Path,
        default=workspace_default / "manuscript" / "template.tex",
    )
    parser.add_argument(
        "--manuscript-pdf",
        type=Path,
        default=workspace_default / "manuscript" / "template.pdf",
    )
    parser.add_argument("--response-md", type=Path, required=True)
    parser.add_argument("--response-docx", type=Path, required=True)
    parser.add_argument("--response-pdf", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=workspace_default / "delivery",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target, zip_target = build_delivery_package(
        workspace_root=args.workspace_root,
        manuscript_tex=args.manuscript_tex,
        manuscript_pdf=args.manuscript_pdf,
        response_md=args.response_md,
        response_docx=args.response_docx,
        response_pdf=args.response_pdf,
        canonical_manifest=args.canonical_manifest,
        output_parent=args.output_parent,
    )
    print(target)
    print(zip_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
