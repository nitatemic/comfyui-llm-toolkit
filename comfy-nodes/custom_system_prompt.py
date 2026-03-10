import os
import sys
from typing import Any, Dict, Optional, Tuple
import logging

# Ensure parent directory in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


class CustomSystemPromptNode:
    """ComfyUI node to set a fully custom system prompt written by the user."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant.",
                    },
                ),
                "output_as_text": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, outputs prompt as text only without adding to context",
                    },
                ),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "system_prompt")
    FUNCTION = "set_prompt"
    CATEGORY = "🔗llm_toolkit/prompt"

    def set_prompt(
        self,
        system_prompt: str,
        output_as_text: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Inject or return the user-defined system prompt.

        Args:
            system_prompt: The free-text system prompt written by the user.
            output_as_text: When True, place the prompt in
                ``context["prompt_config"]["text"]`` instead of
                ``context["provider_config"]["system_message"]``.
            context: Optional upstream context dict to update.

        Returns:
            A tuple of (updated_context, system_prompt_string).
        """
        # Prepare context copy / init
        if context is None:
            output_context: Dict[str, Any] = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            output_context = {"passthrough_data": context}

        if output_as_text:
            prompt_config = output_context.get("prompt_config", {})
            if not isinstance(prompt_config, dict):
                prompt_config = {}
            prompt_config["text"] = system_prompt
            output_context["prompt_config"] = prompt_config
            logger.info("CustomSystemPromptNode: Output prompt as text.")
        else:
            provider_config = output_context.get("provider_config", {})
            if not isinstance(provider_config, dict):
                provider_config = {}
            provider_config["system_message"] = system_prompt
            output_context["provider_config"] = provider_config
            logger.info("CustomSystemPromptNode: Set system_message in provider_config.")

        return (output_context, system_prompt)


# Node Mappings
NODE_CLASS_MAPPINGS = {
    "CustomSystemPrompt": CustomSystemPromptNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomSystemPrompt": "Custom System Prompt (🔗LLMToolkit)",
}
