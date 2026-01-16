"""Tests for textbook module."""

import tempfile
from pathlib import Path

import pytest

from src.education.textbook import Chapter, Quiz, Textbook, TextbookLoader


class TestQuiz:
    """Tests for Quiz dataclass."""

    def test_create_quiz(self):
        """Test creating a Quiz instance."""
        quiz = Quiz(
            question="What is TCO?",
            expected_keywords=["Total Cost", "Ownership"],
        )
        assert quiz.question == "What is TCO?"
        assert quiz.expected_keywords == ["Total Cost", "Ownership"]

    def test_quiz_default_keywords(self):
        """Test Quiz with default empty keywords."""
        quiz = Quiz(question="Simple question")
        assert quiz.question == "Simple question"
        assert quiz.expected_keywords == []


class TestChapter:
    """Tests for Chapter dataclass."""

    def test_create_chapter(self):
        """Test creating a Chapter instance."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(
            title="Chapter 1",
            content="Chapter content here.",
            quiz=[quiz],
        )
        assert chapter.title == "Chapter 1"
        assert chapter.content == "Chapter content here."
        assert len(chapter.quiz) == 1
        assert chapter.quiz[0].question == "Q1"

    def test_chapter_default_quiz(self):
        """Test Chapter with default empty quiz."""
        chapter = Chapter(title="Title", content="Content")
        assert chapter.title == "Title"
        assert chapter.content == "Content"
        assert chapter.quiz == []


class TestTextbook:
    """Tests for Textbook dataclass."""

    def test_create_textbook(self):
        """Test creating a Textbook instance."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(
            title="Test Textbook",
            perspective="testing",
            chapters=[chapter],
        )
        assert textbook.title == "Test Textbook"
        assert textbook.perspective == "testing"
        assert len(textbook.chapters) == 1

    def test_textbook_default_chapters(self):
        """Test Textbook with default empty chapters."""
        textbook = Textbook(title="Title", perspective="perspective")
        assert textbook.chapters == []


class TestTextbookLoader:
    """Tests for TextbookLoader class."""

    def test_load_valid_yaml(self, tmp_path: Path):
        """Test loading a valid YAML file."""
        yaml_content = """
textbook:
  title: "Test Textbook"
  perspective: "testing"
  chapters:
    - title: "Chapter 1"
      content: "This is the content."
      quiz:
        - question: "What is this?"
          expected_keywords: ["keyword1", "keyword2"]
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = TextbookLoader()
        textbook = loader.load(str(yaml_file))

        assert textbook.title == "Test Textbook"
        assert textbook.perspective == "testing"
        assert len(textbook.chapters) == 1
        assert textbook.chapters[0].title == "Chapter 1"
        assert textbook.chapters[0].content == "This is the content."
        assert len(textbook.chapters[0].quiz) == 1
        assert textbook.chapters[0].quiz[0].question == "What is this?"
        assert textbook.chapters[0].quiz[0].expected_keywords == ["keyword1", "keyword2"]

    def test_load_file_not_found(self):
        """Test loading a non-existent file."""
        loader = TextbookLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/non/existent/path.yaml")

    def test_load_empty_yaml(self, tmp_path: Path):
        """Test loading an empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        loader = TextbookLoader()
        with pytest.raises(ValueError, match="Empty YAML file"):
            loader.load(str(yaml_file))

    def test_load_missing_textbook_key(self, tmp_path: Path):
        """Test loading YAML without 'textbook' root key."""
        yaml_content = """
title: "Wrong structure"
"""
        yaml_file = tmp_path / "wrong.yaml"
        yaml_file.write_text(yaml_content)

        loader = TextbookLoader()
        with pytest.raises(ValueError, match="textbook"):
            loader.load(str(yaml_file))

    def test_load_multiple_chapters(self, tmp_path: Path):
        """Test loading textbook with multiple chapters."""
        yaml_content = """
textbook:
  title: "Multi-Chapter Book"
  perspective: "learning"
  chapters:
    - title: "Chapter 1"
      content: "Content 1"
      quiz:
        - question: "Q1"
          expected_keywords: ["k1"]
    - title: "Chapter 2"
      content: "Content 2"
      quiz:
        - question: "Q2a"
          expected_keywords: ["k2a"]
        - question: "Q2b"
          expected_keywords: ["k2b", "k2c"]
"""
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(yaml_content)

        loader = TextbookLoader()
        textbook = loader.load(str(yaml_file))

        assert len(textbook.chapters) == 2
        assert textbook.chapters[0].title == "Chapter 1"
        assert textbook.chapters[1].title == "Chapter 2"
        assert len(textbook.chapters[1].quiz) == 2

    def test_load_sample_textbook(self):
        """Test loading the sample textbook."""
        sample_path = Path(__file__).parent.parent.parent / "examples" / "sample_textbook.yaml"
        if not sample_path.exists():
            pytest.skip("Sample textbook not found")

        loader = TextbookLoader()
        textbook = loader.load(str(sample_path))

        assert textbook.title == "調達基礎"
        assert textbook.perspective == "調達"
        assert len(textbook.chapters) == 2

    def test_validate_valid_textbook(self):
        """Test validating a valid textbook."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(
            title="Valid Textbook",
            perspective="testing",
            chapters=[chapter],
        )

        loader = TextbookLoader()
        assert loader.validate(textbook) is True

    def test_validate_missing_title(self):
        """Test validating textbook without title."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(title="", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_missing_perspective(self):
        """Test validating textbook without perspective."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(title="Title", perspective="", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_no_chapters(self):
        """Test validating textbook without chapters."""
        textbook = Textbook(title="Title", perspective="perspective", chapters=[])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_chapter_missing_title(self):
        """Test validating chapter without title."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="", content="Content", quiz=[quiz])
        textbook = Textbook(title="Title", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_chapter_missing_content(self):
        """Test validating chapter without content."""
        quiz = Quiz(question="Q1", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="", quiz=[quiz])
        textbook = Textbook(title="Title", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_quiz_missing_question(self):
        """Test validating quiz without question."""
        quiz = Quiz(question="", expected_keywords=["k1"])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(title="Title", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_quiz_missing_keywords(self):
        """Test validating quiz without expected_keywords."""
        quiz = Quiz(question="Q1", expected_keywords=[])
        chapter = Chapter(title="Ch1", content="Content", quiz=[quiz])
        textbook = Textbook(title="Title", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        assert loader.validate(textbook) is False

    def test_validate_chapter_without_quiz(self):
        """Test validating chapter without quiz (valid case)."""
        chapter = Chapter(title="Ch1", content="Content", quiz=[])
        textbook = Textbook(title="Title", perspective="testing", chapters=[chapter])

        loader = TextbookLoader()
        # Chapter without quiz is valid
        assert loader.validate(textbook) is True
