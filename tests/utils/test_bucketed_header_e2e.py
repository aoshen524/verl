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
"""End-to-end test for the new ``header`` field on BucketedWeightSender /
BucketedWeightReceiver.

This is the *live* counterpart to ``tests/workers/rollout/
test_sharded_moe_weight_load.py`` (which uses mock zmq sockets). Here sender
and receiver run in separate processes with real ZMQ + real CUDA IPC, exactly
matching the wire format used by ``vLLMColocateWorkerExtension`` for EP-sharded
MoE routed-expert weight sync.

Verifies:
  * Sender's ``header`` dict piggy-backs on the first bucket and arrives at
    the receiver intact (not on later buckets).
  * Receiver's 2-arg callback signature ``f(weights, header=...)`` is invoked
    correctly and sees the header on every bucket of the same call.
  * Legacy single-arg callback ``f(weights)`` continues to work via the
    TypeError fallback.
"""

import asyncio
import multiprocessing as mp
import uuid

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device, is_support_ipc

PROCESS_TIMEOUT = 60

HAS_ACCELERATOR = get_device_name() != "cpu"


def _unique_zmq_handle():
    return f"ipc:///tmp/test-bwt-hdr-{uuid.uuid4().hex}.sock"


def _generate_weights(weight_specs, seed):
    device_name = get_device_name()
    device = torch.device(f"{device_name}:0")
    get_torch_device().manual_seed(seed)
    weights = []
    for name, shape, dtype in weight_specs:
        t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
        weights.append((name, t))
    return weights


def _sender_with_header_fn(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm, header):
    """Sender process: generate weights and send with a header attached."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    weights = _generate_weights(weight_specs, seed)
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
        header=header,
    )
    asyncio.run(sender.async_send_weights(iter(weights)))


def _receiver_2arg_fn(zmq_handle, use_shm, result_queue):
    """Receiver process: 2-arg callback. Records (header, names) per bucket."""
    from verl.utils.device import get_device_name
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)

    per_bucket = []

    def cb(weights, header=None):
        per_bucket.append((header, [name for name, _ in weights]))

    receiver.receive_weights(on_bucket_received=cb)
    result_queue.put(per_bucket)


def _receiver_1arg_legacy_fn(zmq_handle, use_shm, result_queue):
    """Legacy single-arg callback. Should still work via TypeError fallback."""
    from verl.utils.device import get_device_name
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)

    per_bucket = []

    def cb(weights):  # legacy 1-arg
        per_bucket.append([name for name, _ in weights])

    receiver.receive_weights(on_bucket_received=cb)
    result_queue.put(per_bucket)


def _run_pair(sender_fn, sender_args, receiver_fn, receiver_args):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    sender_p = ctx.Process(target=sender_fn, args=sender_args)
    receiver_p = ctx.Process(target=receiver_fn, args=(*receiver_args, result_queue))
    sender_p.start()
    receiver_p.start()
    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)
    assert sender_p.exitcode == 0, f"sender exit {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"receiver exit {receiver_p.exitcode}"
    return result_queue.get(timeout=5)


# ---------------------------------------------------------------------------
# IPC path (CUDA-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_support_ipc(), reason="Requires CUDA IPC support")
class TestHeaderRoundTripIPC:
    def test_header_arrives_on_first_bucket_and_is_cached(self):
        """Header set on sender → receiver's 2-arg callback sees it on every bucket."""
        zmq_handle = _unique_zmq_handle()
        # 20 weights of ~64KB each in a 1MB bucket → 2 buckets total
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        header = {
            "moe_routed_expert_global_ids": {
                "model.layers.0.mlp.experts.gate_up_proj": [0, 4, 8, 12],
                "model.layers.0.mlp.experts.down_proj": [0, 4, 8, 12],
            }
        }

        per_bucket = _run_pair(
            _sender_with_header_fn,
            (zmq_handle, specs, 42, 1, False, header),
            _receiver_2arg_fn,
            (zmq_handle, False),
        )

        # Should have at least one bucket; every bucket should report header.
        assert per_bucket, "no buckets received"
        assert all(h is not None for h, _ in per_bucket), (
            f"some buckets had None header: {[h for h, _ in per_bucket]}"
        )
        # All headers must equal what we sent (cached, same dict every time).
        for h, _ in per_bucket:
            assert h == header

    def test_no_header_yields_none_at_callback(self):
        """If sender does not set a header, receiver's 2-arg callback gets None."""
        zmq_handle = _unique_zmq_handle()
        specs = [("w", (128,), torch.float32)]

        per_bucket = _run_pair(
            _sender_with_header_fn,
            (zmq_handle, specs, 7, 1, False, None),
            _receiver_2arg_fn,
            (zmq_handle, False),
        )

        assert per_bucket, "no buckets received"
        for h, _ in per_bucket:
            assert h is None

    def test_legacy_1arg_callback_still_works_with_header(self):
        """Sender sets a header but receiver uses legacy 1-arg callback;
        TypeError fallback path keeps it working — header is silently dropped."""
        zmq_handle = _unique_zmq_handle()
        specs = [(f"w{i}", (64,), torch.float32) for i in range(5)]
        header = {"moe_routed_expert_global_ids": {"x": [0]}}

        names_per_bucket = _run_pair(
            _sender_with_header_fn,
            (zmq_handle, specs, 11, 1, False, header),
            _receiver_1arg_legacy_fn,
            (zmq_handle, False),
        )

        # Returned shape: list[list[str]] — names from each bucket.
        flat = [n for bucket in names_per_bucket for n in bucket]
        assert sorted(flat) == sorted([n for n, _, _ in specs])


# ---------------------------------------------------------------------------
# SHM path (used on NPU; sanity-check non-CUDA-IPC)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (HAS_ACCELERATOR and not is_support_ipc()), reason="SHM-only path")
class TestHeaderRoundTripSHM:
    def test_header_arrives_via_shm(self):
        zmq_handle = _unique_zmq_handle()
        specs = [("w", (32,), torch.float32)]
        header = {"moe_routed_expert_global_ids": {"a": [3]}}

        per_bucket = _run_pair(
            _sender_with_header_fn,
            (zmq_handle, specs, 13, 1, True, header),
            _receiver_2arg_fn,
            (zmq_handle, True),
        )
        assert per_bucket
        assert all(h == header for h, _ in per_bucket)
