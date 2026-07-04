"""
Script file parser for script.txt.

Parses the tab-separated sentence file format:
    1\tsentence text
    ...
    类别N：title — N条
    ...

Category header lines and blank lines are filtered out.
Categories are determined by header line position.
"""

import os
import re
from typing import List, Optional, Dict


class SentenceInfo:
    """Single sentence entry."""
    def __init__(self, idx: int, text: str, category: str = ""):
        self.idx = idx
        self.text = text
        self.category = category

    def __repr__(self):
        return f"SentenceInfo({self.idx}: {self.text[:20]}...)"


class CategoryInfo:
    """Category group info."""
    def __init__(self, name: str, start_idx: int, end_idx: int, total: int):
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.total = total

    def __repr__(self):
        return f"Category({self.name}: {self.start_idx}-{self.end_idx}, {self.total}条)"


class PageData:
    """Data for a single page (10 sentences)."""
    def __init__(self, page: int, sentences: List[SentenceInfo],
                 category: str, total_pages: int, total_sentences: int):
        self.page = page
        self.sentences = sentences
        self.category = category
        self.total_pages = total_pages
        self.total_sentences = total_sentences


class ScriptData:
    """Full parsed script data."""
    def __init__(self, sentences: List[SentenceInfo], categories: List[CategoryInfo]):
        self.sentences = sentences
        self.categories = categories


class ScriptReader:
    """Reads and parses script.txt with pagination support."""

    CATEGORY_PATTERN = re.compile(r'^类别\d+[:：]')

    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self._data = None  # cache parsed result

    def read(self, filepath: str = None) -> ScriptData:
        """Read and parse script file. Returns ScriptData."""
        path = filepath or self.filepath
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Script file not found: {path}")

        sentences = []
        categories = []
        current_category = "通用"
        category_start = 1
        category_sentences_expected = 0
        category_line_count = 0

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if it's a category header
            if self.CATEGORY_PATTERN.match(line):
                # Save previous category
                if category_line_count > 0 and categories:
                    categories[-1].total = category_line_count
                    categories[-1].end_idx = category_start + category_line_count - 1

                # Start new category
                current_category = line
                category_start = len(sentences) + 1
                category_line_count = 0
                # Extract just the name (remove "类别N：" prefix)
                name = re.sub(r'^\d+\t', '', line) if '\t' in line else line
                categories.append(CategoryInfo(
                    name=current_category,
                    start_idx=category_start,
                    end_idx=category_start,  # will be updated
                    total=0
                ))
                continue

            # Parse sentence line: idx\ttext
            parts = line.split('\t', 1)
            if len(parts) >= 2:
                try:
                    idx = int(parts[0].strip())
                    text = parts[1].strip()
                    if not text:
                        # Skip empty lines
                        continue
                    # Check if text starts with "类别" (category header)
                    if re.match(r'^类别\d+[:：]', text):
                        # Save previous category before starting new one
                        if category_line_count > 0 and categories:
                            categories[-1].total = category_line_count
                            categories[-1].end_idx = category_start + category_line_count - 1
                        # Start new category
                        current_category = text
                        category_start = len(sentences) + 1
                        category_line_count = 0
                        categories.append(CategoryInfo(
                            name=current_category,
                            start_idx=category_start,
                            end_idx=category_start,
                            total=0
                        ))
                        continue
                    # Regular sentence
                    sentences.append(SentenceInfo(idx, text, current_category))
                    category_line_count += 1
                except ValueError:
                    # Line doesn't start with number, skip
                    continue
            elif parts[0].strip():
                # Line without tab-separated number - treat as plain sentence
                idx = len(sentences) + 1
                if re.match(r'^类别\d+[:：]', parts[0].strip()):
                    # Category header without number prefix
                    if category_line_count > 0 and categories:
                        categories[-1].total = category_line_count
                        categories[-1].end_idx = category_start + category_line_count - 1
                    current_category = parts[0].strip()
                    category_start = len(sentences) + 1
                    category_line_count = 0
                    categories.append(CategoryInfo(
                        name=current_category,
                        start_idx=category_start,
                        end_idx=category_start,
                        total=0
                    ))
                    continue
                sentences.append(SentenceInfo(idx, parts[0].strip(), current_category))
                category_line_count += 1

        # Update last category count
        if categories:
            categories[-1].total = category_line_count
            categories[-1].end_idx = category_start + category_line_count - 1

        # For sentences before first category, assign to first category
        if categories and sentences:
            first_cat = categories[0]
            prefix_sentences = [s for s in sentences if s.idx < first_cat.start_idx]
            for s in prefix_sentences:
                s.category = first_cat.name
            first_cat.start_idx = 1
            first_cat.total += len(prefix_sentences)

        self._data = ScriptData(sentences, categories)
        return self._data

    @property
    def data(self) -> Optional[ScriptData]:
        return self._data

    @property
    def total_sentences(self) -> int:
        if not self._data:
            return 0
        return len(self._data.sentences)

    @property
    def total_pages(self) -> int:
        return max(1, (self.total_sentences + 9) // 10)

    def get_page(self, page: int, page_size: int = 10) -> Optional[PageData]:
        """Get sentences for given page (1-based)."""
        if not self._data:
            return None

        total = len(self._data.sentences)
        total_pages = self.total_pages

        if page < 1 or page > total_pages:
            return None

        start = (page - 1) * page_size
        end = min(start + page_size, total)
        page_sentences = self._data.sentences[start:end]

        # Determine current category (most common category on this page)
        category_counts = {}
        for s in page_sentences:
            category_counts[s.category] = category_counts.get(s.category, 0) + 1
        current_category = max(category_counts, key=category_counts.get) if category_counts else ""

        return PageData(
            page=page,
            sentences=page_sentences,
            category=current_category,
            total_pages=total_pages,
            total_sentences=total
        )

    def get_categories(self) -> List[CategoryInfo]:
        if not self._data:
            return []
        return self._data.categories

    def get_sentence_category(self, idx: int) -> str:
        if not self._data:
            return ""
        for s in self._data.sentences:
            if s.idx == idx:
                return s.category
        return ""
