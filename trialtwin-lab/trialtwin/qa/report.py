"""QA report generator.

Produces a markdown report summarizing all clinical data validation checks.
"""

from pathlib import Path

from trialtwin.qa.checks import QAResult


def generate_qa_report(results: list[QAResult], output_path: Path) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    lines = [
        "# Clinical Data QA Report",
        "",
        f"**{passed}/{total} checks passed**",
        "",
        "| Check | Status | Message |",
        "|-------|--------|---------|",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| {r.name} | {status} | {r.message} |")

    # Detailed findings for failures
    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("")
        lines.append("## Failures")
        for r in failures:
            lines.append(f"\n### {r.name}")
            lines.append(r.message)
            if r.details:
                lines.append(f"\n```\n{r.details}\n```")

    # Details section
    lines.append("")
    lines.append("## Details")
    for r in results:
        if r.details:
            lines.append(f"\n**{r.name}**: {r.details}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
