"""Conservatively identify title candidates in a Netflix viewing export."""
import re
import unicodedata

UI_VERIFIED_EPISODES = {
    f"The Hunting Party: {episode}": "The Hunting Party"
    for episode in (
        "Jenna Wells", "Tom Beecher", "Denise Flynn", "Mark Marsden",
        "Arlo Brandt", "Roy Barber", "Dr. Ezekiel Malak", "Lowe",
        "Clayton Jessup", "Richard Harris",
    )
}

SEASON = re.compile(
    r"^(.+?):\s*(?:Season\s+\d+|Series\s+\d+|Part\s+\d+|Limited Series|Miniseries)\s*:\s*(.+)$",
    re.IGNORECASE,
)

CHAPTER = re.compile(
    r"^(.+?):\s*Chapter\s+(?:\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*:\s*(.+)$",
    re.IGNORECASE,
)

PROMO = re.compile(r"\b(?:trailer|teaser|preview)\b", re.IGNORECASE)

def parse_title(raw: str) -> dict:
    title = unicodedata.normalize("NFC", raw).strip()
    if not title or title.startswith(":"):
        return dict(canonical_candidate="", media_kind="unknown", parser_rule="missing_title_prefix",
                    review_required=True, suggested_title="", reason="Series/title prefix missing; episode number cannot identify a work.")
    if title in UI_VERIFIED_EPISODES:
        return dict(canonical_candidate=UI_VERIFIED_EPISODES[title], media_kind="series", parser_rule="manual_netflix_ui_verification",
                    review_required=False, suggested_title="", reason="Manually verified this exact variant as a The Hunting Party episode in Netflix activity UI on 2026-10-17; finite evidence-backed mapping.")
    match = SEASON.match(title)
    if match:
        return dict(canonical_candidate=match[1].strip(), media_kind="series", parser_rule="explicit_season_series_part",
                    review_required=False, suggested_title="", reason="Explicit season/series/part marker followed by episode label.")
    # Netflix sometimes repeats a series name as its season label, e.g.
    # 'Stranger Things: Stranger Things 5: Chapter Eight: The Rightside Up'.
    chunks = title.split(": ", 2)
    if len(chunks) == 3 and re.fullmatch(re.escape(chunks[0]) + r"\s+\d+", chunks[1], re.IGNORECASE):
        return dict(canonical_candidate=chunks[0], media_kind="series", parser_rule="repeated_name_number_season",
                    review_required=False, suggested_title="", reason="Repeated series name plus numeric season, followed by an episode label.")
    match = CHAPTER.match(title)
    if match:
        return dict(canonical_candidate=match[1].strip(), media_kind="series", parser_rule="explicit_numbered_chapter",
                    review_required=False, suggested_title="", reason="Numbered chapter followed by an episode label.")
    if PROMO.search(title):
        return dict(canonical_candidate=title, media_kind="unknown", parser_rule="possible_promotional_item",
                    review_required=True, suggested_title="", reason="Trailer/teaser/preview wording requires review; no exclusion by guesswork.")
    if ":" in title:
        # Candidate suggestion only: repeated prefix is a clue, not proof that a
        # colon means episode boundary. This never merges the title automatically.
        return dict(canonical_candidate=title, media_kind="unknown", parser_rule="unresolved_colon",
                    review_required=True, suggested_title="", reason="Colon can separate a film subtitle or an unmarked episode name; retain intact pending review.")
    return dict(canonical_candidate=title, media_kind="unknown", parser_rule="intact_title",
                review_required=False, suggested_title="", reason="No explicit episode marker; intact candidate still requires metadata identity matching.")
