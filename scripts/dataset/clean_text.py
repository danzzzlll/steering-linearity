import re

hieroglyph_pattern = re.compile(
    r'[\u3000-\u303F]|'  # CJK Symbols and Punctuation
    r'[\u3040-\u309F]|'  # Hiragana
    r'[\u30A0-\u30FF]|'  # Katakana
    r'[\u4E00-\u9FFF]|'  # CJK Unified Ideographs
    r'[\uAC00-\uD7AF]|'  # Hangul Syllables
    r'[\uFF00-\uFFEF]|'  # Halfwidth and Fullwidth Forms
    r'[\uF900-\uFAFF]|'  # CJK Compatibility Ideographs
    r'[\u2E80-\u2EFF]|'  # CJK Radicals Supplement
    r'[\u2F00-\u2FDF]|'  # Kangxi Radicals
    r'[\u3130-\u318F]|'  # Hangul Compatibility Jamo
    r'[\u31F0-\u31FF]|'  # Katakana Phonetic Extensions
    r'[\u2FF0-\u2FFF]'   # Ideographic Description Characters
)

def contains_hieroglyphs(text_to_check):
    """
    Проверяет, содержит ли строка какие-либо иероглифы.
    Возвращает True, если найдены иероглифы, иначе False.
    """
    return bool(hieroglyph_pattern.search(text_to_check))