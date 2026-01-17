"""YAML loading and minimal schema validation for CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import yaml

VALID_SCOPE_LEVELS = {"universal", "domain", "project"}


class YamlValidationError(ValueError):
    """YAML schema validation error."""


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file and return data."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise YamlValidationError("YAMLのルートはオブジェクトである必要があります")
    return data


def validate_agent_definition(data: Dict[str, Any]) -> None:
    """Validate agent definition YAML data."""
    _require_fields(data, ["agent_id", "name", "role", "perspectives"])

    if not isinstance(data["agent_id"], str) or not data["agent_id"]:
        raise YamlValidationError("agent_id は文字列で指定してください")
    if not isinstance(data["name"], str) or not data["name"]:
        raise YamlValidationError("name は文字列で指定してください")
    if not isinstance(data["role"], str) or not data["role"]:
        raise YamlValidationError("role は文字列で指定してください")

    perspectives = data["perspectives"]
    if not isinstance(perspectives, list) or not perspectives:
        raise YamlValidationError("perspectives は配列で指定してください")

    # 文字列配列またはオブジェクト配列（{name, description}）をサポート
    normalized_perspectives = []
    for item in perspectives:
        if isinstance(item, str) and item:
            # 文字列の場合はそのまま使用
            normalized_perspectives.append(item)
        elif isinstance(item, dict):
            # オブジェクトの場合は name フィールドを抽出
            if "name" not in item or not isinstance(item["name"], str) or not item["name"]:
                raise YamlValidationError("perspectives のオブジェクトには name フィールド（文字列）が必要です")
            normalized_perspectives.append(item["name"])
        else:
            raise YamlValidationError("perspectives は文字列配列またはオブジェクト配列（{name, description}）で指定してください")

    # 正規化された perspectives を data に反映
    data["perspectives"] = normalized_perspectives

    if "capabilities" in data and data["capabilities"] is not None:
        if not isinstance(data["capabilities"], list):
            raise YamlValidationError("capabilities は配列で指定してください")
        if not all(isinstance(item, str) and item for item in data["capabilities"]):
            raise YamlValidationError("capabilities は文字列配列で指定してください")

    if "system_prompt" in data and data["system_prompt"] is not None:
        if not isinstance(data["system_prompt"], str):
            raise YamlValidationError("system_prompt は文字列で指定してください")

    if "initial_memories" in data and data["initial_memories"] is not None:
        if not isinstance(data["initial_memories"], list):
            raise YamlValidationError("initial_memories は配列で指定してください")
        for mem in data["initial_memories"]:
            if not isinstance(mem, dict):
                raise YamlValidationError("initial_memories の要素はオブジェクトで指定してください")
            if "content" not in mem or not isinstance(mem["content"], str):
                raise YamlValidationError("initial_memories.content は文字列で指定してください")
            scope_level = mem.get("scope_level")
            if scope_level and scope_level not in VALID_SCOPE_LEVELS:
                raise YamlValidationError("scope_level は universal/domain/project のいずれかです")


def validate_textbook(data: Dict[str, Any]) -> None:
    """Validate textbook YAML data."""
    _require_fields(data, ["title", "chapters"])

    if not isinstance(data["title"], str) or not data["title"]:
        raise YamlValidationError("title は文字列で指定してください")

    chapters = data["chapters"]
    if not isinstance(chapters, list) or not chapters:
        raise YamlValidationError("chapters は配列で指定してください")

    scope_level = data.get("scope_level")
    if scope_level and scope_level not in VALID_SCOPE_LEVELS:
        raise YamlValidationError("scope_level は universal/domain/project のいずれかです")

    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise YamlValidationError("chapters の要素はオブジェクトで指定してください")
        _require_fields(chapter, ["title", "content"], prefix="chapters")
        if not isinstance(chapter["title"], str) or not chapter["title"]:
            raise YamlValidationError("chapter.title は文字列で指定してください")
        if not isinstance(chapter["content"], str) or not chapter["content"]:
            raise YamlValidationError("chapter.content は文字列で指定してください")

        quiz = chapter.get("quiz")
        if quiz is None:
            continue
        if not isinstance(quiz, list):
            raise YamlValidationError("chapter.quiz は配列で指定してください")
        for item in quiz:
            if not isinstance(item, dict):
                raise YamlValidationError("quiz の要素はオブジェクトで指定してください")
            _require_fields(item, ["question", "answer"], prefix="quiz")
            if not isinstance(item["question"], str) or not item["question"]:
                raise YamlValidationError("quiz.question は文字列で指定してください")
            if not isinstance(item["answer"], str) or not item["answer"]:
                raise YamlValidationError("quiz.answer は文字列で指定してください")


def _require_fields(data: Dict[str, Any], fields: List[str], prefix: str | None = None) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        label = f"{prefix}." if prefix else ""
        raise YamlValidationError(f"必須フィールドが不足しています: {', '.join(label + f for f in missing)}")

