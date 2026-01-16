"""Textbook data structures and loader for education process.

This module provides the data structures and loader for textbook-based
agent training as described in architecture.ja.md section 4.3.

YAML Schema:
    textbook:
      title: "Textbook Title"
      perspective: "Perspective Name"

      chapters:
        - title: "Chapter Title"
          content: "Chapter content (chunk target)"
          quiz:
            - question: "Test question"
              expected_keywords: ["keyword1", "keyword2"]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Quiz:
    """Quiz question for testing comprehension."""

    question: str
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class Chapter:
    """Chapter of a textbook."""

    title: str
    content: str
    quiz: list[Quiz] = field(default_factory=list)


@dataclass
class Textbook:
    """Textbook for agent education."""

    title: str
    perspective: str
    chapters: list[Chapter] = field(default_factory=list)


class TextbookLoader:
    """Loader for textbook YAML files."""

    def load(self, path: str) -> Textbook:
        """Load a textbook from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Textbook instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML structure is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Textbook file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty YAML file: {path}")

        textbook_data = data.get("textbook")
        if textbook_data is None:
            raise ValueError("YAML must have 'textbook' root key")

        return self._parse_textbook(textbook_data)

    def _parse_textbook(self, data: dict) -> Textbook:
        """Parse textbook data from dict."""
        title = data.get("title", "")
        perspective = data.get("perspective", "")

        chapters = []
        for chapter_data in data.get("chapters", []):
            chapter = self._parse_chapter(chapter_data)
            chapters.append(chapter)

        return Textbook(
            title=title,
            perspective=perspective,
            chapters=chapters,
        )

    def _parse_chapter(self, data: dict) -> Chapter:
        """Parse chapter data from dict."""
        title = data.get("title", "")
        content = data.get("content", "")

        quiz_list = []
        for quiz_data in data.get("quiz", []):
            quiz = self._parse_quiz(quiz_data)
            quiz_list.append(quiz)

        return Chapter(
            title=title,
            content=content,
            quiz=quiz_list,
        )

    def _parse_quiz(self, data: dict) -> Quiz:
        """Parse quiz data from dict."""
        question = data.get("question", "")
        expected_keywords = data.get("expected_keywords", [])

        return Quiz(
            question=question,
            expected_keywords=expected_keywords,
        )

    def validate(self, textbook: Textbook) -> bool:
        """Validate a textbook structure.

        Args:
            textbook: Textbook to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not textbook.title:
            return False

        if not textbook.perspective:
            return False

        if not textbook.chapters:
            return False

        for chapter in textbook.chapters:
            if not chapter.title:
                return False
            if not chapter.content:
                return False

            for quiz in chapter.quiz:
                if not quiz.question:
                    return False
                if not quiz.expected_keywords:
                    return False

        return True
