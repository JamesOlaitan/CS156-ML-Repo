"""Name normalization and observed media-type compatibility for title matching."""
import unicodedata

def normalize(value: str) -> str:
    value = unicodedata.normalize('NFKD', value.casefold())
    return ''.join(c for c in value if c.isalnum() and not unicodedata.combining(c))

def compatible_type(hint, title_type):
    """Only reject contradictions when source type was actually observed."""
    hint = hint.casefold().strip()
    if hint in {'show', 'series'}:
        return title_type in {'TV Series', 'TV Mini Series'}
    if hint == 'movie':
        return title_type in {'Movie', 'TV Movie'}
    return True
