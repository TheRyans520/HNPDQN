#!/usr/bin/env python3
"""Resolve response-letter manuscript-location markers against a final PDF.

This utility is intentionally read-only with respect to the manuscript and the
response letter.  It extracts every ``[[MANUSCRIPT_LOCATION:key]]`` marker,
looks up an explicit, version-controlled anchor for that key, and checks that
the anchor occurs exactly once in both the final PDF and the TeX source.

Two review artifacts are written:

* ``manuscript_locations.json`` -- machine-readable locations and diagnostics.
* ``response_location_suggestions.md`` -- human-reviewable replacement hints.

The response letter is never edited.  Missing or non-unique anchors are still
recorded in both artifacts, after which the command exits with status 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
MARKER_RE = re.compile(
    r"\[\[(?:MANUSCRIPT|PENDING)_LOCATION:([a-z][a-z0-9_]*)\]\]"
)


@dataclass(frozen=True)
class AnchorSpec:
    """One explicit response-marker anchor specification."""

    section_label: str
    pdf_anchor: str
    tex_anchor: str


# Keep this mapping explicit.  Anchors are deliberately substantive prose
# rather than page numbers, which remain stable when final typesetting moves a
# section.  PDF and TeX anchors differ only where TeX markup is unavoidable.
LOCATION_ANCHORS: dict[str, AnchorSpec] = {
    "r1_c1_state_transition": AnchorSpec(
        "Section 2.4, Agent Observation Representation",
        "The adaptive spectrum-selection task is represented as a switching-aware sequential decision process.",
        "The adaptive spectrum-selection task is represented as a switching-aware sequential decision process.",
    ),
    "r1_c1_architecture": AnchorSpec(
        "Section 3.1, Proposed Architecture",
        "HNP-DQN combines a neural feature extractor with an explicit element-wise second-order expansion and LayerNorm.",
        "HNP-DQN combines a neural feature extractor with an explicit element-wise second-order expansion and LayerNorm.",
    ),
    "r1_c1_experiment": AnchorSpec(
        "Section 4.3, Comparative Experiments",
        "The primary learned comparison includes HNP-DQN and the capacity-matched MLP-DDQN.",
        "The primary learned comparison includes HNP-DQN and the capacity-matched MLP-DDQN.",
    ),
    "r1_c1_claims": AnchorSpec(
        "Section 5.1, Summary",
        "The revised formulation makes the switching state explicit: the observation contains 40 RF summaries and an eight-dimensional previous-channel encoding, while the RF sequence itself is exogenous to the selected action.",
        "The revised formulation makes the switching state explicit: the observation contains 40 RF summaries and an eight-dimensional previous-channel encoding, while the RF sequence itself is exogenous to the selected action.",
    ),
    "r1_c2_baselines_results": AnchorSpec(
        "Section 4.3, Comparative Experiments",
        "The following non-learning policies test whether a deep value function is needed:",
        "The following non-learning policies test whether a deep value function is needed:",
    ),
    "r1_c2_collision_switch": AnchorSpec(
        "Section 4.3, Comparative Experiments",
        "Core-method performance under the frozen file-level protocol.",
        r"\caption{Core-method performance under the frozen file-level protocol.",
    ),
    "r1_c2_claims": AnchorSpec(
        "Section 5.1, Summary",
        "Accordingly, the defensible contribution is a reproducible evaluation of a specific representation choice under switching-dependent utility",
        "Accordingly, the defensible contribution is a reproducible evaluation of a specific representation choice under switching-dependent utility",
    ),
    "r1_c3_dataset_split": AnchorSpec(
        "Section 2.1, RF Jamming Dataset",
        "Measurements are split at the raw CSV scan-file level before window construction.",
        "Measurements are split at the raw CSV scan-file level before window construction.",
    ),
    "r1_c3_windows": AnchorSpec(
        "Section 2.1, RF Jamming Dataset",
        "Within each raw file, a 32-sample window is generated with stride 1.",
        "Within each raw file, a 32-sample window is generated with stride~1.",
    ),
    "r1_c3_evaluation": AnchorSpec(
        "Section 4.1, Experimental Setup and Parameter Configuration",
        "For each evaluation-only physical condition and jammer mode, the formal endpoint uses a scan-stratified set of 20 fixed trajectories:",
        "For each evaluation-only physical condition and jammer mode, the formal endpoint uses a scan-stratified set of 20 fixed trajectories:",
    ),
    "r1_c3_claims": AnchorSpec(
        "Section 5.2, Limitations and Future Work",
        "The protocol separates raw scan files before window generation, but it does not establish independent confirmation.",
        "The protocol separates raw scan files before window generation, but it does not establish independent confirmation.",
    ),
    "r1_c4_endpoints": AnchorSpec(
        "Section 2.6, Reward Design and Performance Objectives",
        "Supporting endpoints are collision count/rate and switch count/rate.",
        "Supporting endpoints are collision count/rate and switch count/rate.",
    ),
    "r1_c4_analysis_plan": AnchorSpec(
        "Section 4.2, Endpoints and Statistical Analysis",
        "Every seed is reported. The independent experimental unit is the trained seed, not an episode or time step.",
        "Every seed is reported. The independent experimental unit is the trained seed, not an episode or time step.",
    ),
    "r1_c4_main_statistics": AnchorSpec(
        "Section 4.3, Comparative Experiments",
        "Primary paired seed-level comparisons.",
        r"\caption{Primary paired seed-level comparisons.",
    ),
    "r1_c4_ablation_statistics": AnchorSpec(
        "Section 4.5, Ablation Study",
        "Ablations form a separate exploratory Holm family and are not mixed with the primary comparison family.",
        "Ablations form a separate exploratory Holm family and are not mixed with the primary comparison family.",
    ),
    "r1_c5_architecture": AnchorSpec(
        "Section 3.2, Capacity-Matched Conventional Control",
        "To separate the explicit expansion from total model capacity, the revised evaluation includes a conventional MLP-DDQN selected using only the training and validation files.",
        "To separate the explicit expansion from total model capacity, the revised evaluation includes a conventional MLP-DDQN selected using only the training and validation files.",
    ),
    "r1_c5_profiling": AnchorSpec(
        "Section 4.1, Experimental Setup and Parameter Configuration",
        "The final software and hardware environment is recorded directly from the frozen run rather than reconstructed from memory.",
        "The final software and hardware environment is recorded directly from the frozen run rather than reconstructed from memory.",
    ),
    "r1_c5_results_cost": AnchorSpec(
        "Section 4.4, Model Capacity and Computational Cost",
        "The cost analysis reports trainable parameters, persistent parameter-and-buffer tensor bytes, serialized state/checkpoint size, training time, and batch-1 CPU and GPU inference latency under a stated warm-up and repetition protocol.",
        "The cost analysis reports trainable parameters, persistent parameter-and-buffer tensor bytes, serialized state/checkpoint size, training time, and batch-1 CPU and GPU inference latency under a stated warm-up and repetition protocol.",
    ),
    "r1_c5_claims": AnchorSpec(
        "Section 5.2, Limitations and Future Work",
        "reported parameter, timing, and memory measurements describe only the frozen software/hardware stack and do not establish embedded-device suitability",
        "reported parameter, timing, and memory measurements describe only the frozen software/hardware stack and do not establish embedded-device suitability",
    ),
    "r1_c6_method": AnchorSpec(
        "Section 3.3, Training Strategy and Stabilization Mechanism",
        "HNP-DQN is trained with uniform experience replay, a Double-DQN bootstrap target, a softly updated target network, and gradient clipping.",
        "HNP-DQN is trained with uniform experience replay, a Double-DQN bootstrap target, a softly updated target network, and gradient clipping.",
    ),
    "r1_c6_results": AnchorSpec(
        "Section 4.3, Comparative Experiments",
        "Cross-model TD-loss magnitude is not used as evidence that the polynomial mapping or LayerNorm is more stable.",
        "Cross-model TD-loss magnitude is not used as evidence that the polynomial mapping or LayerNorm is more stable.",
    ),
    "r1_c6_conclusion": AnchorSpec(
        "Section 5.1, Summary",
        "These results do not rely on cross-model TD-loss comparisons or on pooled episode-level tests.",
        "These results do not rely on cross-model TD-loss comparisons or on pooled episode-level tests.",
    ),
    "r1_c7_language": AnchorSpec(
        "Section 1.2, Objectives and Contributions",
        "This work evaluates HNP-DQN as a structured value-function approximator for a controlled, measurement-driven spectrum-selection layer.",
        "This work evaluates HNP-DQN as a structured value-function approximator for a controlled, measurement-driven spectrum-selection layer.",
    ),
    "r3_c1_scope": AnchorSpec(
        "Section 5.2, Limitations and Future Work",
        "The measurements form a controlled communication-system RF testbed rather than an end-to-end radar experiment.",
        "The measurements form a controlled communication-system RF testbed rather than an end-to-end radar experiment.",
    ),
    "r3_c2_interpretability": AnchorSpec(
        "Section 4.6, Post-Hoc Analysis of Expanded-Feature Weights",
        "They are therefore not causal feature importance, branch-contribution, robustness, or interpretability evidence",
        "They are therefore not causal feature importance, branch-contribution, robustness, or interpretability evidence",
    ),
    "r3_c3_implementation": AnchorSpec(
        "Section 3.4, Agent Training Procedure",
        "The 100-step boundary is implemented as a time-limit truncation rather than an absorbing environmental terminal state.",
        "The 100-step boundary is implemented as a time-limit truncation rather than an absorbing environmental terminal state.",
    ),
    "r3_c4_reward": AnchorSpec(
        "Section 2.6, Reward Design and Performance Objectives",
        "The reward prioritizes interference avoidance and then uses the switching indicator to distinguish between two successful decisions.",
        r"\textcolor{red}{The reward prioritizes interference avoidance and then uses the switching indicator to distinguish between two successful decisions.",
    ),
    "r3_c4_statistics_ablation": AnchorSpec(
        "Section 4.5, Ablation Study",
        "Ablations form a separate exploratory Holm family and are not mixed with the primary comparison family.",
        "Ablations form a separate exploratory Holm family and are not mixed with the primary comparison family.",
    ),
    "r3_c4_model_cost": AnchorSpec(
        "Section 4.4, Model Capacity and Computational Cost",
        "The cost analysis reports trainable parameters, persistent parameter-and-buffer tensor bytes, serialized state/checkpoint size, training time, and batch-1 CPU and GPU inference latency under a stated warm-up and repetition protocol.",
        "The cost analysis reports trainable parameters, persistent parameter-and-buffer tensor bytes, serialized state/checkpoint size, training time, and batch-1 CPU and GPU inference latency under a stated warm-up and repetition protocol.",
    ),
    "r3_c4_scope": AnchorSpec(
        "Section 5.2, Limitations and Future Work",
        "is fixed rather than calibrated to retuning latency, energy, synchronization delay, or sensing loss.",
        "is fixed rather than calibrated to retuning latency, energy, synchronization delay, or sensing loss.",
    ),
}


@dataclass
class LocationRecord:
    key: str
    section_label: str
    pdf_page: int | None
    tex_line: int | None
    anchor: str
    tex_anchor: str
    pdf_occurrences: list[dict[str, int]]
    tex_occurrence_lines: list[int]
    is_unique: bool
    status: str
    diagnostic: str
    suggested_text: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_marker_keys(response_text: str) -> list[str]:
    """Return markers in first-occurrence order, rejecting duplicated keys."""

    keys = MARKER_RE.findall(response_text)
    seen: set[str] = set()
    duplicates: list[str] = []
    ordered: list[str] = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
            ordered.append(key)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate MANUSCRIPT_LOCATION key(s): {duplicate_text}")
    return ordered


def normalize_pdf_text(value: str) -> str:
    """Normalize extraction differences without discarding meaningful words."""

    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "")
    # Join words split by PDF line-end hyphenation before collapsing whitespace.
    value = re.sub(r"(?<=\w)-\s*[\r\n]+\s*(?=\w)", "", value)
    value = value.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )
    # Ignore discretionary intra-word hyphenation consistently in both PDF
    # text and anchors (for example, ``second-order`` split at a line end).
    value = re.sub(r"(?<=\w)-(?=\w)", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _extract_with_pypdf(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader  # type: ignore[import-not-found]

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_with_pdfplumber(pdf_path: Path) -> list[str]:
    import pdfplumber  # type: ignore[import-not-found]

    with pdfplumber.open(str(pdf_path)) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def extract_pdf_pages(pdf_path: Path) -> tuple[list[str], str]:
    """Extract page text with pypdf, falling back to pdfplumber."""

    failures: list[str] = []
    for name, extractor in (
        ("pypdf", _extract_with_pypdf),
        ("pdfplumber", _extract_with_pdfplumber),
    ):
        try:
            pages = extractor(pdf_path)
        except (ImportError, ModuleNotFoundError) as exc:
            failures.append(f"{name}: unavailable ({exc})")
            continue
        except Exception as exc:  # pragma: no cover - library/file specific
            failures.append(f"{name}: extraction failed ({type(exc).__name__}: {exc})")
            continue
        if not pages:
            failures.append(f"{name}: PDF contained no pages")
            continue
        if not any(page.strip() for page in pages):
            failures.append(f"{name}: no extractable text (possibly scanned PDF)")
            continue
        return pages, name
    detail = "; ".join(failures)
    raise RuntimeError(
        "Could not extract manuscript PDF text. Install pypdf or pdfplumber, "
        f"or provide a text-bearing PDF. Details: {detail}"
    )


def find_pdf_occurrences(page_texts: Sequence[str], anchor: str) -> list[dict[str, int]]:
    normalized_anchor = normalize_pdf_text(anchor)
    if not normalized_anchor:
        return []
    occurrences: list[dict[str, int]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        count = normalize_pdf_text(page_text).count(normalized_anchor)
        if count:
            occurrences.append({"page": page_number, "count": count})
    return occurrences


def _whitespace_flexible_pattern(anchor: str) -> re.Pattern[str]:
    parts = re.split(r"\s+", anchor.strip())
    return re.compile(r"\s+".join(re.escape(part) for part in parts), re.MULTILINE)


def find_tex_occurrence_lines(tex_text: str, anchor: str) -> list[int]:
    if not anchor.strip():
        return []
    pattern = _whitespace_flexible_pattern(anchor)
    return [tex_text.count("\n", 0, match.start()) + 1 for match in pattern.finditer(tex_text)]


def locate_records(
    marker_keys: Sequence[str],
    page_texts: Sequence[str],
    tex_text: str,
    anchors: Mapping[str, AnchorSpec] = LOCATION_ANCHORS,
) -> list[LocationRecord]:
    records: list[LocationRecord] = []
    for key in marker_keys:
        spec = anchors.get(key)
        if spec is None:
            records.append(
                LocationRecord(
                    key=key,
                    section_label="UNMAPPED",
                    pdf_page=None,
                    tex_line=None,
                    anchor="",
                    tex_anchor="",
                    pdf_occurrences=[],
                    tex_occurrence_lines=[],
                    is_unique=False,
                    status="error",
                    diagnostic="No explicit anchor mapping exists for this marker key.",
                    suggested_text=None,
                )
            )
            continue

        pdf_occurrences = find_pdf_occurrences(page_texts, spec.pdf_anchor)
        tex_lines = find_tex_occurrence_lines(tex_text, spec.tex_anchor)
        pdf_total = sum(item["count"] for item in pdf_occurrences)
        is_unique = pdf_total == 1 and len(tex_lines) == 1
        pdf_page = pdf_occurrences[0]["page"] if pdf_total == 1 else None
        tex_line = tex_lines[0] if len(tex_lines) == 1 else None

        diagnostics: list[str] = []
        if pdf_total == 0:
            diagnostics.append("PDF anchor not found")
        elif pdf_total != 1:
            diagnostics.append(f"PDF anchor occurs {pdf_total} times")
        if len(tex_lines) == 0:
            diagnostics.append("TeX anchor not found")
        elif len(tex_lines) != 1:
            diagnostics.append(f"TeX anchor occurs {len(tex_lines)} times")
        diagnostic = "; ".join(diagnostics) if diagnostics else "Unique in both PDF and TeX."
        suggested = None
        if is_unique:
            suggested = f"{spec.section_label} (manuscript p. {pdf_page}; TeX line {tex_line})"

        records.append(
            LocationRecord(
                key=key,
                section_label=spec.section_label,
                pdf_page=pdf_page,
                tex_line=tex_line,
                anchor=spec.pdf_anchor,
                tex_anchor=spec.tex_anchor,
                pdf_occurrences=pdf_occurrences,
                tex_occurrence_lines=tex_lines,
                is_unique=is_unique,
                status="ok" if is_unique else "error",
                diagnostic=diagnostic,
                suggested_text=suggested,
            )
        )
    return records


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_suggestions(records: Sequence[LocationRecord], inputs: Mapping[str, object]) -> str:
    errors = [record for record in records if not record.is_unique]
    overall = "PASS" if not errors else "FAIL"
    status_html = (
        "<span style=\"color:#008000\"><strong>PASS</strong></span>"
        if not errors
        else "<span style=\"color:#C00000\"><strong>FAIL</strong></span>"
    )
    lines = [
        "# Response Location Suggestions",
        "",
        "> This is a read-only review artifact. The response letter was **not** modified.",
        "> Copy a suggestion only after checking it against the final rendered manuscript.",
        "",
        f"Overall anchor validation: {status_html} (`{overall}`; {len(records) - len(errors)}/{len(records)} unique).",
        "",
        f"- Final PDF: `{inputs.get('pdf_path', '')}`",
        f"- TeX source: `{inputs.get('tex_path', '')}`",
        f"- Response source: `{inputs.get('response_path', '')}`",
        f"- PDF extractor: `{inputs.get('pdf_extractor', '')}`",
        "",
    ]
    if errors:
        lines.extend(
            [
                "## Blocking anchor errors",
                "",
                "| Marker | Status | Diagnostic | PDF matches | TeX lines |",
                "|---|---|---|---|---|",
            ]
        )
        for record in errors:
            pdf_matches = ", ".join(
                f"p.{item['page']} x{item['count']}" for item in record.pdf_occurrences
            ) or "none"
            tex_matches = ", ".join(str(line) for line in record.tex_occurrence_lines) or "none"
            lines.append(
                "| `[[MANUSCRIPT_LOCATION:{key}]]` | "
                "<span style=\"color:#C00000\"><strong>ERROR</strong></span> | {diagnostic} | {pdf} | {tex} |".format(
                    key=_markdown_escape(record.key),
                    diagnostic=_markdown_escape(record.diagnostic),
                    pdf=_markdown_escape(pdf_matches),
                    tex=_markdown_escape(tex_matches),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Suggested manual replacements",
            "",
            "| Marker | Suggested location text | Anchor | Unique? |",
            "|---|---|---|---|",
        ]
    )
    for record in records:
        suggested = record.suggested_text or "DO NOT REPLACE — resolve anchor error"
        unique = "yes" if record.is_unique else '<span style="color:#C00000">no</span>'
        lines.append(
            "| `[[MANUSCRIPT_LOCATION:{key}]]` | {suggested} | “{anchor}” | {unique} |".format(
                key=_markdown_escape(record.key),
                suggested=_markdown_escape(suggested),
                anchor=_markdown_escape(record.anchor),
                unique=unique,
            )
        )
    lines.extend(
        [
            "",
            "## Validation rule",
            "",
            "A marker passes only when its explicit anchor occurs exactly once across the entire PDF and exactly once in the TeX source. Page numbers are 1-based; TeX lines identify the first source line of the matched anchor.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_input_file(path: Path, label: str, suffix: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    if suffix and path.suffix.casefold() != suffix.casefold():
        raise ValueError(f"{label} must have extension {suffix}: {path}")


def write_outputs(
    output_dir: Path,
    records: Sequence[LocationRecord],
    inputs: Mapping[str, object],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "manuscript_locations.json"
    markdown_path = output_dir / "response_location_suggestions.md"
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}; pass --overwrite")

    errors = [record for record in records if not record.is_unique]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "pass" if not errors else "fail",
        "read_only": True,
        "response_modified": False,
        "marker_count": len(records),
        "unique_marker_count": len(records) - len(errors),
        "error_count": len(errors),
        "inputs": dict(inputs),
        "locations": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_suggestions(records, inputs), encoding="utf-8")
    return json_path, markdown_path


def run_locator(
    *,
    pdf_path: Path,
    tex_path: Path,
    response_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    anchors: Mapping[str, AnchorSpec] = LOCATION_ANCHORS,
) -> tuple[list[LocationRecord], Path, Path]:
    _validate_input_file(pdf_path, "Final manuscript PDF", ".pdf")
    _validate_input_file(tex_path, "TeX source", ".tex")
    _validate_input_file(response_path, "Response source", ".md")

    response_text = response_path.read_text(encoding="utf-8")
    tex_text = tex_path.read_text(encoding="utf-8")
    marker_keys = extract_marker_keys(response_text)
    if not marker_keys:
        raise ValueError(
            "No concrete [[MANUSCRIPT_LOCATION:key]] or "
            "[[PENDING_LOCATION:key]] markers found in response source"
        )
    page_texts, extractor_name = extract_pdf_pages(pdf_path)
    records = locate_records(marker_keys, page_texts, tex_text, anchors)

    inputs: dict[str, object] = {
        "pdf_path": str(pdf_path.resolve()),
        "pdf_sha256": sha256_file(pdf_path),
        "pdf_page_count": len(page_texts),
        "pdf_extractor": extractor_name,
        "tex_path": str(tex_path.resolve()),
        "tex_sha256": sha256_file(tex_path),
        "response_path": str(response_path.resolve()),
        "response_sha256": sha256_file(response_path),
    }
    json_path, markdown_path = write_outputs(
        output_dir, records, inputs, overwrite=overwrite
    )
    return records, json_path, markdown_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map response MANUSCRIPT_LOCATION markers to unique final-PDF pages "
            "and TeX source lines without modifying the response."
        )
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Final manuscript PDF")
    parser.add_argument("--tex", required=True, type=Path, help="Final template.tex")
    parser.add_argument("--response", required=True, type=Path, help="Response Markdown")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for manuscript_locations.json and response_location_suggestions.md",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing locator outputs (never modifies the response source)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        records, json_path, markdown_path = run_locator(
            pdf_path=args.pdf,
            tex_path=args.tex,
            response_path=args.response,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    failures = [record for record in records if not record.is_unique]
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")
    if failures:
        print(
            f"ERROR: {len(failures)} marker(s) have missing or non-unique anchors; "
            "see the red diagnostics before editing the response.",
            file=sys.stderr,
        )
        return 2
    print(f"Validated {len(records)} unique manuscript locations; response was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
