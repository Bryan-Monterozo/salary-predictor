from __future__ import annotations

import re

from salary_predictor.taxonomy import Taxonomy, normalize_experience

from .profile_flow import EMPTY

LABELS = {
    "job_title": re.compile(r"^(?:job title|position|current role|role)\s*:\s*(.+)$", re.I),
    "occupation": re.compile(r"^(?:occupation|psoc occupation)\s*:\s*(.+)$", re.I),
    "region": re.compile(r"^(?:region|location)\s*:\s*(.+)$", re.I),
    "experience": re.compile(r"^(?:years? of experience|experience)\s*:\s*(.+)$", re.I),
    "education": re.compile(r"^(?:education|highest education|degree)\s*:\s*(.+)$", re.I),
    "skills": re.compile(r"^(?:skills?|technical skills?)\s*:\s*(.+)$", re.I),
}

CONTACT = re.compile(r"@|https?://|www\.|\b(?:\+?63|0)9\d{9}\b", re.I)
DATE_RANGE = re.compile(
    r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present)\b", re.I
)
ROLE_WORD = re.compile(
    r"\b(?:developer|engineer|analyst|specialist|technician|support|"
    r"waiter|attendant|installer|assistant|administrator|officer|"
    r"manager|teacher|nurse|accountant|designer|clerk|operator|"
    r"electrician|encoder|programmer|consultant|supervisor)\b", re.I
)
HEADINGS = {
    "SKILLS", "TECHNICAL SKILLS", "PROFESSIONAL SKILLS",
    "WORK EXPERIENCE", "EMPLOYMENT HISTORY", "PROFESSIONAL EXPERIENCE", "EXPERIENCE",
    "EDUCATION", "EDUCATIONAL BACKGROUND", "CONTACT",
    "PROJECTS", "SELECTED PROJECT EXPERIENCE", "EARLIER EXPERIENCE",
    "PROFESSIONAL SUMMARY", "CERTIFICATIONS", "CERTIFICATIONS & QUALIFICATIONS", "REFERENCES",
}


def _readable_line(original: str) -> str:
    chunks = []
    for chunk in re.split(r" {2,}", original.strip()):
        glyphs = chunk.split()
        # Some design PDFs store every letter with a space, but words have two.
        if len(glyphs) >= 2 and all(len(glyph.strip("()[]{}.,:;/")) <= 1 for glyph in glyphs):
            chunks.append("".join(glyphs))
        else:
            chunks.append(chunk)
    return " ".join(chunks).strip()


def _lines(text: str) -> list[str]:
    return [
        line for original in text.splitlines()[:300]
        if (line := _readable_line(original))
    ]


def _safe_value(value: str, *, maximum: int = 150) -> str:
    value = " ".join(value.split())
    if not 2 <= len(value) <= maximum or CONTACT.search(value):
        return ""
    return value


def _section(lines: list[str], names: set[str], *, max_lines: int) -> list[str]:
    for index, line in enumerate(lines):
        if line.upper() in names:
            selected = []
            for item in lines[index + 1:index + 1 + max_lines]:
                if item.upper() in HEADINGS:
                    break
                selected.append(item)
            return selected
    return []


def _work_role(lines: list[str]) -> tuple[str, str] | None:
    history = _section(
        lines, {"WORK EXPERIENCE", "EMPLOYMENT HISTORY", "PROFESSIONAL EXPERIENCE"},
        max_lines=100,
    )
    possibilities = []
    for index, entry in enumerate(history):
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", entry)]
        if not years or not ("-" in entry or "–" in entry or "—" in entry or "OJT" in entry):
            continue

        inline = entry.split("|", 1)[0].strip() if "|" in entry else ""
        candidates = [inline] if inline and ROLE_WORD.search(inline) else history[index + 1:index + 5]
        for nearby in candidates:
            if len(nearby) > 75 or not 2 <= len(nearby.split()) <= 7:
                continue
            if CONTACT.search(nearby) or not ROLE_WORD.search(nearby):
                continue
            if nearby.isupper() or re.search(r"\b(?:company|college|university)\b", nearby, re.I):
                continue
            title = _safe_value(nearby, maximum=75)
            if title:
                possibilities.append((
                    "present" in entry.casefold(), max(years), -index,
                    title, "OJT" in entry.upper(),
                ))
                break
    if not possibilities:
        return None
    present, year, _, title, internship = max(possibilities)
    context = f"most recent dated work-history entry ({year})"
    if present:
        context += ", marked as ongoing"
    if internship:
        context += ", marked as OJT"
    return title, context


def _section_skills(lines: list[str]) -> list[str]:
    values = _section(
        lines, {"SKILLS", "TECHNICAL SKILLS", "PROFESSIONAL SKILLS"},
        max_lines=30,
    )
    result = []
    for item in values:
        candidate = _safe_value(item, maximum=260).lstrip("•*- ").strip()
        if ":" in candidate:
            label, _, value = candidate.partition(":")
            if label.casefold() in {
                "languages & frameworks", "backend & databases", "tools & practices",
                "technologies", "frameworks", "languages", "databases", "tools",
            }:
                candidate = value.strip()
        if not candidate or candidate.upper() in HEADINGS:
            continue
        if candidate.casefold() == "troubleshooting" and result and result[-1].casefold() in {
            "basic computer", "basic network"
        }:
            result[-1] += " " + candidate
        elif not re.search(r"\b(?:19|20)\d{2}\b", candidate):
            result.append(candidate)
    return result


def _set_skills(draft: dict[str, str], raw: list[str]) -> bool:
    skills = []
    for item in raw:
        for part in re.split(r"[,;|]", item):
            skill = _safe_value(part.strip().lstrip("•*- ").strip(), maximum=80)
            if skill and skill.casefold() not in {value.casefold() for value in skills}:
                skills.append(skill)
    while skills and (len(", ".join(skills)) > 500 or len(skills) > 30):
        skills.pop()
    if not skills:
        return False
    draft["skills"] = ", ".join(skills)
    return True


def suggest_profile(text: str, catalog: Taxonomy | None) -> tuple[dict[str, str], list[str]]:
    draft = EMPTY.copy()
    notes = []
    lines = _lines(text)
    values: dict[str, str] = {}
    for line in lines:
        for name, pattern in LABELS.items():
            match = pattern.fullmatch(line)
            if match and name not in values:
                values[name] = match.group(1).strip()

    title = _safe_value(values.get("job_title", ""))
    if title:
        draft["job_title"] = title
        notes.append("Job title: extracted from a labeled line; confirm your current role.")
    elif work := _work_role(lines):
        draft["job_title"] = work[0]
        notes.append(
            f"Job title: suggested from {work[1]}; this may not be your current job. "
            "Correct it if needed."
        )
    else:
        notes.append("Job title: not confidently found; enter it yourself.")

    occupation = _safe_value(values.get("occupation", ""), maximum=120)
    if catalog and occupation:
        result = catalog.resolve("occupation", occupation)
        if result.status == "matched" and result.chosen.source == "psoc_2022_workbook":
            draft["occupation_psoc_code"] = result.chosen.id
            notes.append("Occupation: exact PSOC label/code match; confirm the classification.")
        else:
            notes.append("Occupation: ambiguous/unmatched PSOC; select one manually.")
    elif occupation:
        notes.append("Occupation: lookup unavailable; choose a classification manually.")

    region = _safe_value(values.get("region", ""), maximum=80)
    if catalog and region:
        result = catalog.resolve("region", region)
        if result.status == "matched":
            draft["region_psgc_code"] = result.chosen.id
            notes.append("Region: matched a named Philippine region; verify it is your job region.")
        else:
            notes.append("Region: unrecognized location; choose your job region yourself.")
    elif any("QUEZON CITY" in line.upper() for line in lines):
        notes.append(
            "Region: contact-city information is not necessarily your job region; "
            "select the correct region yourself."
        )

    experience = _safe_value(values.get("experience", ""), maximum=60)
    if experience:
        years = normalize_experience(experience)
        if years.status == "matched":
            draft["years_experience"] = str(years.years)
            notes.append("Experience: explicit duration converted to years; confirm it.")
        else:
            notes.append("Experience: duration ambiguous; enter verified total experience.")
    elif DATE_RANGE.search("\n".join(lines)) or re.search(
        r"\b(?:19|20)\d{2}\b.*\b(?:OJT|part.?time)\b", "\n".join(lines), re.I
    ) or re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+(?:19|20)\d{2}\s*[-–—]\s*(?:Present|[A-Za-z]*\s*(?:19|20)\d{2})",
        "\n".join(lines), re.I,
    ):
        periods = DATE_RANGE.findall("\n".join(lines))
        if len(periods) == 1 and not _work_role(lines):
            notes.append(f"Experience: detected {periods[0]}; enter confirmed total years.")
        else:
            notes.append(
                "Experience: work dates appear in the résumé, but internships, part-time "
                "work and overlaps need review. Enter your actual total years."
            )

    education = _safe_value(values.get("education", ""), maximum=80)
    if catalog and education:
        result = catalog.resolve("education", education)
        if result.status == "matched":
            draft["education_level"] = result.chosen.id
            notes.append("Education: matched an explicit education label; confirm attainment.")
        else:
            notes.append("Education: attainment not uniquely matched; choose it yourself.")
    elif re.search(r"\bbachelor(?:'s| of)\b", " ".join(lines), re.I):
        notes.append(
            "Education: a bachelor's program is mentioned, but completion is unclear. "
            "Choose your highest completed level."
        )

    skills = values.get("skills")
    if skills and _set_skills(draft, [skills]):
        notes.append("Skills: taken from a labeled line; remove anything inaccurate.")
    elif _set_skills(draft, _section_skills(lines)):
        notes.append("Skills: taken from the skills section; review each suggested skill.")

    if not draft["occupation_psoc_code"]:
        notes.append("No unreviewed job title was automatically mapped to a PSOC code.")
    return draft, notes
