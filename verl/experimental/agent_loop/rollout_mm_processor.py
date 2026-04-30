# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Callable

from verl.experimental.agent_loop.preprocessed_multimodal import (
    build_vllm_preprocessed_multimodal_input,
    refresh_vllm_preprocessed_multimodal_prompt_ids,
)

PreprocessedMultimodalBuilder = Callable[..., Any | None]
PreprocessedMultimodalRefresher = Callable[..., Any | None]

_PREPROCESSED_MULTIMODAL_BUILDERS: dict[str, tuple[str, PreprocessedMultimodalBuilder]] = {
    "vllm": ("use_preprocessed_multimodal_input", build_vllm_preprocessed_multimodal_input),
}

_PREPROCESSED_MULTIMODAL_GENERATE_KWARGS = {
    "vllm": "preprocessed_multimodal_input",
}

_PREPROCESSED_MULTIMODAL_REFRESHERS: dict[str, PreprocessedMultimodalRefresher] = {
    "vllm": refresh_vllm_preprocessed_multimodal_prompt_ids,
}


def build_rollout_preprocessed_multimodal_input(
    *,
    rollout_name: str | None,
    rollout_config: Any,
    processor: Any,
    prompt_ids: list[int],
    model_inputs: Any,
    images: Any = None,
    videos: Any = None,
) -> Any | None:
    builder_entry = _PREPROCESSED_MULTIMODAL_BUILDERS.get(str(rollout_name))
    if builder_entry is None:
        return None

    config_key, builder = builder_entry
    if not rollout_config.get(config_key, False):
        return None

    return builder(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        images=images,
        videos=videos,
    )


def append_preprocessed_multimodal_input_kwargs(
    generate_kwargs: dict[str, Any],
    *,
    rollout_name: str | None,
    preprocessed_multimodal_input: Any | None,
) -> None:
    if preprocessed_multimodal_input is None:
        return

    kwarg_name = _PREPROCESSED_MULTIMODAL_GENERATE_KWARGS.get(str(rollout_name))
    if kwarg_name is None:
        raise ValueError(f"Preprocessed multimodal input is not supported by rollout backend {rollout_name!r}")
    generate_kwargs[kwarg_name] = preprocessed_multimodal_input


def refresh_preprocessed_multimodal_input_prompt_ids(
    *,
    rollout_name: str | None,
    preprocessed_multimodal_input: Any | None,
    prompt_ids: list[int],
) -> Any | None:
    refresher = _PREPROCESSED_MULTIMODAL_REFRESHERS.get(str(rollout_name))
    if refresher is None:
        return preprocessed_multimodal_input
    return refresher(
        preprocessed_multimodal_input,
        prompt_ids=prompt_ids,
    )
