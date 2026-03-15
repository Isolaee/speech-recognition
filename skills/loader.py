import os

import yaml

from skills.base import Skill


class SkillLoader:
    def __init__(self, definitions_dir: str):
        self._skills: dict[str, Skill] = {}
        for filename in os.listdir(definitions_dir):
            if filename.endswith(".yaml"):
                path = os.path.join(definitions_dir, filename)
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                skill = Skill(
                    name=data["name"],
                    description=data.get("description", ""),
                    system_prompt=data.get("system_prompt", ""),
                    enabled_tools=data.get("enabled_tools", []),
                    tts_voice=data.get("tts_voice"),
                    ollama_model_override=data.get("ollama_model_override"),
                    always_escalate=data.get("always_escalate", False),
                    escalation_threshold=data.get("escalation_threshold", 0.8),
                )
                self._skills[skill.name] = skill

    def load(self, name: str) -> Skill:
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' not found. Available: {list(self._skills)}")
        return self._skills[name]

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())
