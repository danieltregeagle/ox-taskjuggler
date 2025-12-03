#!/usr/bin/env python3
"""
tj2pgfgantt: Convert TaskJuggler CSV export to pgfgantt LaTeX code.

Usage: python tj2pgfgantt.py input.csv [options]

Options:
    --unit=day|month|auto    Time unit for chart (default: auto)
    --links                  Include dependency links (default: on)
    --no-links               Omit dependency links
    --progress               Show progress bars (default: on)
    --no-progress            Hide progress bars
    --standalone             Generate complete LaTeX document
    --output=FILE            Output file (default: stdout)
"""

import csv
import sys
import re
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# Threshold: if project has more than this many days, switch to month units
DAY_UNIT_THRESHOLD = 90  # ~3 months


@dataclass
class Task:
    id: str
    name: str
    start: datetime
    end: datetime
    duration: float
    complete: int  # 0-100
    precursors: list[str]  # list of task IDs
    bsi: str  # WBS number like "1.2.3"
    depth: int  # hierarchy depth
    is_milestone: bool
    is_group: bool = False  # set later based on children


def parse_precursors(precursor_str: str) -> list[str]:
    """
    Parse TJ3 precursor string format.
    Example: "IRB Approval (sweetarmor.objective__5_2_2.sht_svy_prep.irb) ]->[ 2025-12-15"
    Can have multiple comma-separated entries.
    Returns list of task IDs.
    """
    if not precursor_str or precursor_str.strip() == '':
        return []

    # Pattern: name (task_id) ]->[ date
    # Extract task IDs from parentheses
    pattern = r'\(([^)]+)\)\s*\]->\['
    matches = re.findall(pattern, precursor_str)
    return matches


def parse_completion(comp_str: str) -> int:
    """Parse completion percentage string like '0%' or '50%'."""
    if not comp_str:
        return 0
    match = re.search(r'(\d+)', comp_str)
    return int(match.group(1)) if match else 0


def parse_tj_csv(filepath: str) -> list[Task]:
    """Parse TJ3 CSV export, return list of tasks."""
    tasks = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')

        for row in reader:
            # Parse dates
            start = datetime.strptime(row['Start'], '%Y-%m-%d')
            end = datetime.strptime(row['End'], '%Y-%m-%d')

            # Parse duration
            duration = float(row.get('Duration', 0) or 0)

            # Strip leading whitespace from name (TJ3 indentation)
            name = row['Name'].strip()

            # Hierarchy depth from BSI (WBS number)
            bsi = row.get('BSI', '')
            depth = bsi.count('.') if bsi else 0

            # Parse precursors
            precursors = parse_precursors(row.get('Precursors', ''))

            # Milestone detection: duration is 0 or very small
            # (will refine after identifying groups)
            is_milestone = duration < 0.5

            task = Task(
                id=row['Id'],
                name=name,
                start=start,
                end=end,
                duration=duration,
                complete=parse_completion(row.get('Completion', '')),
                precursors=precursors,
                bsi=bsi,
                depth=depth,
                is_milestone=is_milestone,
            )
            tasks.append(task)

    # Second pass: identify groups (tasks with children)
    # and refine milestone detection (groups can't be milestones)
    task_ids = {t.id for t in tasks}
    for task in tasks:
        prefix = task.id + '.'
        task.is_group = any(tid.startswith(prefix) for tid in task_ids)
        # Groups are never milestones, even if duration is 0
        if task.is_group:
            task.is_milestone = False

    return tasks


def get_project_bounds(tasks: list[Task]) -> tuple[datetime, datetime]:
    """Get earliest start and latest end dates."""
    starts = [t.start for t in tasks]
    ends = [t.end for t in tasks]
    return min(starts), max(ends)


def determine_time_unit(project_start: datetime, project_end: datetime,
                        requested_unit: str = 'auto') -> str:
    """Determine whether to use day or month time slots."""
    if requested_unit in ('day', 'month'):
        return requested_unit

    # Auto-detect based on project duration
    total_days = (project_end - project_start).days + 1
    if total_days > DAY_UNIT_THRESHOLD:
        return 'month'
    return 'day'


def date_to_slot_day(date: datetime, project_start: datetime) -> int:
    """Convert date to day-based time slot number (1-indexed)."""
    delta = date - project_start
    return delta.days + 1


def date_to_slot_month(date: datetime, project_start: datetime) -> int:
    """Convert date to month-based time slot number (1-indexed)."""
    # Calculate months between project_start and date
    months = (date.year - project_start.year) * 12 + (date.month - project_start.month)
    return months + 1


def count_months(project_start: datetime, project_end: datetime) -> int:
    """Count number of months spanned by the project."""
    months = (project_end.year - project_start.year) * 12 + (project_end.month - project_start.month)
    return months + 1


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def sanitize_node_name(task_id: str) -> str:
    """
    Convert task ID to valid TikZ node name.
    TikZ node names can't have underscores or other special chars.
    """
    # Replace problematic characters with safe alternatives
    name = task_id.replace('_', '-')
    name = name.replace('.', '-')
    name = name.replace(' ', '-')
    return name


def format_iso_date(date: datetime) -> str:
    """Format date as ISO for pgfgantt time slot format."""
    return date.strftime('%Y-%m-%d')


def format_yearmonth(date: datetime) -> str:
    """Format date as YYYY-MM for pgfgantt isodate-yearmonth format."""
    return date.strftime('%Y-%m')


def generate_pgfgantt(tasks: list[Task],
                      include_links: bool = True,
                      show_progress: bool = True,
                      time_unit: str = 'auto') -> str:
    """Generate complete pgfgantt LaTeX code."""

    if not tasks:
        return "% No tasks found"

    project_start, project_end = get_project_bounds(tasks)

    # Determine time unit
    unit = determine_time_unit(project_start, project_end, time_unit)

    lines = []
    lines.append(r"% Generated by tj2pgfgantt")
    lines.append(f"% Time unit: {unit}")
    lines.append(r"% Requires: \usepackage{pgfgantt}")
    lines.append("")

    if unit == 'month':
        # Month-based chart
        total_slots = count_months(project_start, project_end)
        start_tss = format_yearmonth(project_start)
        end_tss = format_yearmonth(project_end)

        lines.append(r"\begin{ganttchart}[")
        lines.append(r"  expand chart=\textwidth,")
        lines.append(r"  time slot format=isodate-yearmonth,")
        lines.append(r"  time slot unit=month,")
        lines.append(r"  y unit chart=0.5cm,")
        lines.append(r"  y unit title=0.6cm,")
        lines.append(r"  hgrid,")
        lines.append(r"  vgrid,")
        lines.append(r"  title height=1,")
        lines.append(r"  bar height=0.6,")
        lines.append(r"  bar top shift=0.2,")
        lines.append(r"  group height=0.3,")
        lines.append(r"  group top shift=0.35,")
        lines.append(r"  group peaks tip position=0,")
        if show_progress:
            lines.append(r"  bar incomplete/.append style={fill=black!25},")
        lines.append(r"  milestone height=0.6,")
        lines.append(r"  milestone top shift=0.2,")
        # Narrower milestones for month view (each slot is wider)
        lines.append(r"  milestone left shift=0.9,")
        lines.append(r"  milestone right shift=0.1,")
        lines.append(f"]{{{start_tss}}}{{{end_tss}}}")
        lines.append("")

        # Title calendar for months
        lines.append(r"\gantttitlecalendar{year, month=shortname} \\")
        lines.append("")

        # Convert dates to month slots
        def date_to_slot(date):
            return format_yearmonth(date)

        # Tasks
        for task in tasks:
            start_slot = date_to_slot(task.start)
            end_slot = date_to_slot(task.end)

            safe_name = escape_latex(task.name)
            node_name = sanitize_node_name(task.id)

            # Truncate very long names
            max_name_len = 40
            if len(safe_name) > max_name_len:
                safe_name = safe_name[:max_name_len-3] + "..."

            if task.is_milestone:
                cmd = f"\\ganttmilestone[name={node_name}]{{{safe_name}}}{{{start_slot}}}"
            elif task.is_group:
                cmd = f"\\ganttgroup[name={node_name}]{{{safe_name}}}{{{start_slot}}}{{{end_slot}}}"
            else:
                opts = [f"name={node_name}"]
                if show_progress and task.complete > 0:
                    opts.append(f"progress={task.complete}")
                opts_str = ", ".join(opts)
                cmd = f"\\ganttbar[{opts_str}]{{{safe_name}}}{{{start_slot}}}{{{end_slot}}}"

            lines.append(cmd + r" \\")

    else:
        # Day-based chart
        total_days = (project_end - project_start).days + 1
        start_tss = format_iso_date(project_start)
        end_tss = format_iso_date(project_end)

        lines.append(r"\begin{ganttchart}[")
        lines.append(r"  expand chart=\textwidth,")
        lines.append(r"  time slot format=isodate,")
        lines.append(r"  y unit chart=0.5cm,")
        lines.append(r"  y unit title=0.6cm,")
        lines.append(r"  hgrid,")
        # Weekly vertical grid (every 7 days)
        lines.append(r"  vgrid={*6{draw=none}, dotted},")
        lines.append(r"  title height=1,")
        lines.append(r"  bar height=0.6,")
        lines.append(r"  bar top shift=0.2,")
        lines.append(r"  group height=0.3,")
        lines.append(r"  group top shift=0.35,")
        lines.append(r"  group peaks tip position=0,")
        if show_progress:
            lines.append(r"  bar incomplete/.append style={fill=black!25},")
        lines.append(r"  milestone height=0.6,")
        lines.append(r"  milestone top shift=0.2,")
        lines.append(f"]{{{start_tss}}}{{{end_tss}}}")
        lines.append("")

        # Title calendar for days
        lines.append(r"\gantttitlecalendar{year, month=shortname, week} \\")
        lines.append("")

        # Convert dates to ISO format for time slot specifier
        def date_to_slot(date):
            return format_iso_date(date)

        # Tasks
        for task in tasks:
            start_slot = date_to_slot(task.start)
            end_slot = date_to_slot(task.end)

            safe_name = escape_latex(task.name)
            node_name = sanitize_node_name(task.id)

            # Truncate very long names
            max_name_len = 40
            if len(safe_name) > max_name_len:
                safe_name = safe_name[:max_name_len-3] + "..."

            if task.is_milestone:
                cmd = f"\\ganttmilestone[name={node_name}]{{{safe_name}}}{{{start_slot}}}"
            elif task.is_group:
                cmd = f"\\ganttgroup[name={node_name}]{{{safe_name}}}{{{start_slot}}}{{{end_slot}}}"
            else:
                opts = [f"name={node_name}"]
                if show_progress and task.complete > 0:
                    opts.append(f"progress={task.complete}")
                opts_str = ", ".join(opts)
                cmd = f"\\ganttbar[{opts_str}]{{{safe_name}}}{{{start_slot}}}{{{end_slot}}}"

            lines.append(cmd + r" \\")

    lines.append("")

    # Dependency links
    if include_links:
        valid_ids = {t.id for t in tasks}

        link_count = 0
        for task in tasks:
            for precursor_id in task.precursors:
                if precursor_id in valid_ids:
                    from_node = sanitize_node_name(precursor_id)
                    to_node = sanitize_node_name(task.id)
                    lines.append(f"\\ganttlink{{{from_node}}}{{{to_node}}}")
                    link_count += 1

        if link_count > 0:
            lines.append("")

    lines.append(r"\end{ganttchart}")

    return '\n'.join(lines)


def generate_standalone_document(gantt_code: str) -> str:
    """Wrap gantt code in a complete LaTeX document."""
    doc = []
    doc.append(r"\documentclass[landscape]{article}")
    doc.append(r"\usepackage[margin=0.5in, paperwidth=20in, paperheight=11in]{geometry}")
    doc.append(r"\usepackage{pgfgantt}")
    doc.append(r"\usepackage[utf8]{inputenc}")
    doc.append(r"\pagestyle{empty}")
    doc.append("")
    doc.append(r"\begin{document}")
    doc.append(gantt_code)
    doc.append(r"\end{document}")
    return '\n'.join(doc)


def main():
    parser = argparse.ArgumentParser(
        description='Convert TaskJuggler CSV to pgfgantt LaTeX code'
    )
    parser.add_argument('input', help='Input CSV file from TJ3')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--links', action='store_true', default=True,
                        help='Include dependency links (default: on)')
    parser.add_argument('--no-links', action='store_false', dest='links',
                        help='Omit dependency links')
    parser.add_argument('--progress', action='store_true', default=True,
                        help='Show progress bars (default: on)')
    parser.add_argument('--no-progress', action='store_false', dest='progress',
                        help='Hide progress bars')
    parser.add_argument('--standalone', action='store_true',
                        help='Generate complete LaTeX document')
    parser.add_argument('--unit', choices=['day', 'month', 'auto'],
                        default='auto', help='Time unit (default: auto)')

    args = parser.parse_args()

    # Parse input
    tasks = parse_tj_csv(args.input)

    # Generate gantt code
    gantt_code = generate_pgfgantt(
        tasks,
        include_links=args.links,
        show_progress=args.progress,
        time_unit=args.unit
    )

    # Optionally wrap in document
    if args.standalone:
        output = generate_standalone_document(gantt_code)
    else:
        output = gantt_code

    # Write output
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
