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

from verl.experimental.agent_loop.vllm_mm_processor import build_vllm_mm_processor_data

ProcessorOutputBuilder = Callable[..., dict[str, Any] | None]

_PROCESSOR_OUTPUT_BUILDERS: dict[str, tuple[str, ProcessorOutputBuilder]] = {
    "vllm": ("use_vllm_mm_processor_output", build_vllm_mm_processor_data),
}

_PROCESSOR_OUTPUT_GENERATE_KWARGS = {
    "vllm": "vllm_multi_modal_data",
}


def build_rollout_processor_output_data(
    *,
    rollout_name: str | None,
    rollout_config: Any,
    processor: Any,
    model_inputs: Any,
    images: Any = None,
    videos: Any = None,
) -> dict[str, Any] | None:
    builder_entry = _PROCESSOR_OUTPUT_BUILDERS.get(str(rollout_name))
    if builder_entry is None:
        return None

    config_key, builder = builder_entry
    if not rollout_config.get(config_key, False):
        return None

    return builder(
        processor=processor,
        model_inputs=model_inputs,
        images=images,
        videos=videos,
    )


def append_processor_output_kwargs(
    generate_kwargs: dict[str, Any],
    *,
    rollout_name: str | None,
    processor_output_data: dict[str, Any] | None,
) -> None:
    if processor_output_data is None:
        return

    kwarg_name = _PROCESSOR_OUTPUT_GENERATE_KWARGS.get(str(rollout_name))
    if kwarg_name is None:
        raise ValueError(
            "Processor-output multimodal data is not supported by "
            f"rollout backend {rollout_name!r}"
        )
    generate_kwargs[kwarg_name] = processor_output_data
