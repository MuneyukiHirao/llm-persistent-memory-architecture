"""Tests for education_process module."""

from typing import List, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.education.education_process import EducationProcess, EducationResult
from src.education.textbook import Chapter, Quiz, Textbook
from src.models.memory import AgentMemory


class TestEducationResult:
    """Tests for EducationResult dataclass."""

    def test_create_result(self):
        """Test creating an EducationResult instance."""
        result = EducationResult(
            chapters_completed=3,
            memories_created=10,
            tests_passed=5,
            tests_total=6,
        )
        assert result.chapters_completed == 3
        assert result.memories_created == 10
        assert result.tests_passed == 5
        assert result.tests_total == 6

    def test_pass_rate_calculation(self):
        """Test pass_rate property calculation."""
        result = EducationResult(
            chapters_completed=1,
            memories_created=5,
            tests_passed=8,
            tests_total=10,
        )
        assert result.pass_rate == 0.8

    def test_pass_rate_no_tests(self):
        """Test pass_rate when no tests exist."""
        result = EducationResult(
            chapters_completed=1,
            memories_created=5,
            tests_passed=0,
            tests_total=0,
        )
        assert result.pass_rate == 1.0

    def test_pass_rate_all_passed(self):
        """Test pass_rate when all tests pass."""
        result = EducationResult(
            chapters_completed=2,
            memories_created=8,
            tests_passed=5,
            tests_total=5,
        )
        assert result.pass_rate == 1.0


class TestEducationProcess:
    """Tests for EducationProcess class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock MemoryRepository."""
        repo = MagicMock()
        # create returns the memory with an ID
        def mock_create(memory: AgentMemory) -> AgentMemory:
            return memory
        repo.create.side_effect = mock_create
        return repo

    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock AzureEmbeddingClient."""
        client = MagicMock()
        # Return a dummy embedding
        client.get_embedding.return_value = [0.1] * 1536
        return client

    @pytest.fixture
    def config(self):
        """Create a Phase1Config for testing."""
        return Phase1Config(chunk_size=100)  # Small chunks for testing

    @pytest.fixture
    def sample_textbook(self):
        """Create a sample textbook for testing."""
        quiz1 = Quiz(question="What is TCO?", expected_keywords=["Total Cost", "Ownership"])
        chapter1 = Chapter(
            title="Chapter 1: Cost Management",
            content="TCO stands for Total Cost of Ownership. It is a financial estimate.",
            quiz=[quiz1],
        )

        quiz2 = Quiz(question="What is ROI?", expected_keywords=["Return", "Investment"])
        chapter2 = Chapter(
            title="Chapter 2: ROI Analysis",
            content="ROI means Return on Investment. It measures profitability.",
            quiz=[quiz2],
        )

        return Textbook(
            title="Business Finance Basics",
            perspective="finance",
            chapters=[chapter1, chapter2],
        )

    def test_init(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test EducationProcess initialization."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        assert process.agent_id == "test_agent"
        assert process.textbook == sample_textbook
        assert process.config.chunk_size == 100

    def test_split_into_chunks(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test _split_into_chunks method."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Test with content shorter than chunk_size
        short_content = "Short content"
        chunks = process._split_into_chunks(short_content)
        assert len(chunks) == 1
        assert chunks[0] == "Short content"

        # Test with content longer than chunk_size
        long_content = "A" * 250  # 250 chars, chunk_size=100
        chunks = process._split_into_chunks(long_content)
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50

        # Test with empty content
        chunks = process._split_into_chunks("")
        assert chunks == []

    def test_check_keywords(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test _check_keywords method."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Test with matching keyword
        assert process._check_keywords("Total Cost of Ownership", ["Total Cost", "Ownership"]) is True

        # Test with no matching keyword
        assert process._check_keywords("Something else", ["Total Cost", "Ownership"]) is False

        # Test case insensitivity
        assert process._check_keywords("total cost analysis", ["Total Cost"]) is True

        # Test with empty keywords
        assert process._check_keywords("Any content", []) is False

    def test_read_chapter(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test read_chapter method."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        memory_ids = process.read_chapter(chapter)

        # Should create memories for each chunk
        assert len(memory_ids) > 0

        # Repository.create should have been called
        assert mock_repository.create.call_count > 0

        # Embedding client should have been called
        assert mock_embedding_client.get_embedding.call_count > 0

        # Verify created memory has correct properties
        created_memory = mock_repository.create.call_args[0][0]
        assert created_memory.agent_id == "test_agent"
        assert created_memory.source == "education"
        assert created_memory.strength == 0.5  # INITIAL_STRENGTH_EDUCATION

    def test_read_chapter_tags(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test that read_chapter sets correct tags."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        process.read_chapter(chapter)

        # Verify tags include chapter title and textbook perspective
        created_memory = mock_repository.create.call_args[0][0]
        assert chapter.title in created_memory.tags
        assert sample_textbook.perspective in created_memory.tags

    def test_run_test_correct_answer(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test run_test with correct answers."""
        # Setup mock to return memory with matching content
        test_memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="TCO stands for Total Cost of Ownership.",
        )
        mock_repository.get_by_id.return_value = test_memory

        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        memory_ids = [str(uuid4())]

        passed = process.run_test(chapter, memory_ids)

        # Should pass because content contains expected_keywords
        assert passed == 1

    def test_run_test_incorrect_answer(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test run_test with incorrect answers."""
        # Setup mock to return memory without matching content
        test_memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="This content has nothing related to the question.",
        )
        mock_repository.get_by_id.return_value = test_memory

        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        memory_ids = [str(uuid4())]

        passed = process.run_test(chapter, memory_ids)

        # Should fail because content doesn't contain expected_keywords
        assert passed == 0

    def test_run_full_process(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test full run() process."""
        # Setup mock to return memory with matching content for all chapters
        def mock_get_by_id(memory_id: UUID) -> AgentMemory:
            return AgentMemory.create_from_education(
                agent_id="test_agent",
                content="TCO Total Cost of Ownership ROI Return on Investment.",
            )
        mock_repository.get_by_id.side_effect = mock_get_by_id

        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        result = process.run()

        # Should complete all chapters
        assert result.chapters_completed == 2

        # Should create memories (at least one per chapter)
        assert result.memories_created >= 2

        # Should have tests
        assert result.tests_total == 2

        # Should pass tests (content matches keywords)
        assert result.tests_passed == 2
        assert result.pass_rate == 1.0

    def test_run_with_no_quiz(self, mock_repository, mock_embedding_client, config):
        """Test run() with chapters that have no quizzes."""
        chapter = Chapter(
            title="No Quiz Chapter",
            content="Just content, no quiz.",
            quiz=[],
        )
        textbook = Textbook(
            title="No Quiz Book",
            perspective="testing",
            chapters=[chapter],
        )

        process = EducationProcess(
            agent_id="test_agent",
            textbook=textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        result = process.run()

        assert result.chapters_completed == 1
        assert result.tests_total == 0
        assert result.tests_passed == 0
        assert result.pass_rate == 1.0  # No tests means 100% pass rate

    def test_run_marks_candidates(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test that run_test marks memories as candidates."""
        test_memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="TCO Total Cost",
        )
        mock_repository.get_by_id.return_value = test_memory
        mock_repository.batch_increment_candidate_count = MagicMock(return_value=1)

        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        memory_ids = [str(uuid4())]

        process.run_test(chapter, memory_ids)

        # Should call batch_increment_candidate_count (via strength_manager.mark_as_candidate)
        assert mock_repository.batch_increment_candidate_count.called

    def test_embedding_generation(self, mock_repository, mock_embedding_client, config, sample_textbook):
        """Test that embeddings are generated for each chunk."""
        process = EducationProcess(
            agent_id="test_agent",
            textbook=sample_textbook,
            repository=mock_repository,
            embedding_client=mock_embedding_client,
            config=config,
        )

        chapter = sample_textbook.chapters[0]
        process.read_chapter(chapter)

        # Check that embedding was generated
        created_memory = mock_repository.create.call_args[0][0]
        assert created_memory.embedding is not None
        assert len(created_memory.embedding) == 1536
