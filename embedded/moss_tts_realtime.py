# /// script
# dependencies = [
#   # See moss_tts.py for the full rationale. Windows-only cu128 pin;
#   # Linux falls through to the bare `torch>=2.1` from @python_requires.
#   "torch @ https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
#   "torchaudio @ https://download.pytorch.org/whl/cu128/torchaudio-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
# ]
# ///

"""
MOSS-TTS-Realtime — streaming, multi-turn, context-aware TTS node.

The realtime variant depends on the OpenMOSS ``mossttsrealtime`` python
package — published only inside the upstream
`OpenMOSS/MOSS-TTS <https://github.com/OpenMOSS/MOSS-TTS>`_ git repo
(``moss_tts_realtime/mossttsrealtime/``), no PyPI release and not in
the HF model snapshot's ``trust_remote_code`` modules.

Solved by **vendoring**: a verbatim copy of the six upstream files
lives at :mod:`remotemedia._vendor.mossttsrealtime` (Apache 2.0,
same license as the rest of this repo). The node imports the vendored
classes directly — no PYTHONPATH gymnastics, no separate pip install.
To re-sync with upstream::

    # From the repo root
    for f in __init__.py configuration_mossttsrealtime.py \\
             modeling_mossttsrealtime.py modeling_mossttsrealtime_local.py \\
             processing_mossttsrealtime.py streaming_mossttsrealtime.py; do
        curl -fsSL \\
          "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/moss_tts_realtime/mossttsrealtime/$f" \\
          -o "clients/python/remotemedia/_vendor/mossttsrealtime/$f"
    done

Built on `OpenMOSS/MOSS-TTS`_'s `MossTTSRealtime` (1.7B). Consumes
*incremental text deltas* and emits playable PCM audio chunks in
realtime — the same pattern as the reference
``example_llm_stream_to_tts.py`` / ``example_multiturn_stream_to_tts.py``
scripts. TTFB is ~180 ms on a single L20 after warm-up.

.. _OpenMOSS/MOSS-TTS:
    https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md

## Architecture (matches the reference inferencer)

The realtime path uses *three* objects layered on top of HF:

    AutoModel("MOSS-Audio-Tokenizer")   ─→ ``codec``     (encode + decode)
    MossTTSRealtime.from_pretrained()   ─→ ``model``
    MossTTSRealtimeProcessor(tokenizer) ─→ ``processor``  (chat template)

Then per session::

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length)
    session    = MossTTSRealtimeStreamingSession(
                     inferencer, processor,
                     codec=codec, codec_sample_rate=24000,
                     codec_encode_kwargs={"chunk_duration": 0.24},
                     prefill_text_len=processor.delay_tokens_len,
                     temperature=0.8, top_p=0.6, top_k=30,
                     do_sample=True, repetition_penalty=1.1,
                     repetition_window=50,
                 )
    session.set_voice_prompt_tokens(prompt_tokens)   # encoded ref audio
    session.reset_turn(user_text=..., user_audio_tokens=...,
                       include_system_prompt=first_turn,
                       reset_cache=first_turn)

And the per-delta inner loop::

    with codec.streaming(batch_size=1):
        for delta in text_deltas:
            frames = session.push_text(delta)
            for chunk in decode(frames): yield chunk
        frames = session.end_text(); yield from decode(frames)
        while frames := session.drain(max_steps=1):
            yield from decode(frames)
            if session.inferencer.is_finished: break
        yield from decoder.flush()

We wrap that loop into the SDK's per-IPC-session model: a single
``MossTTSRealtimeStreamingSession`` per ``session_id``, with the
``codec.streaming(...)`` context held open across deltas and torn down
on session timeout / explicit reset.

## Input shapes

* ``RuntimeData.Text(text)``               — a text delta. If the text
  equals one of ``_FLUSH_MARKERS`` the current turn is finalised
  (end_text + drain + flush).
* ``RuntimeData.Text("<|reset|>")``        — reset the current turn
  (new prompt, KV cache stays unless ``audio.in.reset`` was also sent).
* dict / JSON                              — control envelope. Recognised
  top-level keys:

      ``text``           — same as text input above
      ``user_text``      — set the *user-turn text* for the next turn
                            (logged into chat context).
      ``final``          — bool. When true (with ``text``), pushes the
                            delta then flushes.

## Aux-port surface (control bus)

    audio.in.reference         — set the voice prompt. Payload accepts:
                                 ``{"audio_path": str}`` (local file or
                                 URL — torchaudio loads it),
                                 ``{"samples": [float], "sample_rate":
                                 int}`` (numpy-like float waveform), or
                                 ``{"tokens": [[int]]}`` for already-
                                 encoded codec tokens. Invalidates the
                                 current session so the next push re-
                                 initialises with the new voice.
    audio.in.system_prompt     — informational only on the realtime model
                                 (the system prompt is baked into the
                                 chat template). Logged then ignored.
    audio.in.reset             — clear conversation history. Next push
                                 starts a new turn 0 with ``reset_cache=
                                 True``.
    audio.in.flush             — same as sending one of ``_FLUSH_MARKERS``
                                 on the main channel.
    audio.in.new_turn          — start a new user turn boundary; the
                                 next text delta is treated as the
                                 beginning of an assistant response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

# Heavy deps optional at import time — same rationale as moss_tts.py.
_ML_IMPORT_ERROR: Optional[BaseException] = None
try:
    import numpy as np
    import torch
    import torchaudio
    from transformers import AutoTokenizer, AutoModel
    _ML_DEPS_AVAILABLE = True
except BaseException as _exc:  # noqa: BLE001
    _ML_DEPS_AVAILABLE = False
    _ML_IMPORT_ERROR = _exc
    np = None  # type: ignore
    torch = None  # type: ignore
    torchaudio = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    logging.getLogger(__name__).warning(
        "MOSS-TTS-Realtime ML imports failed (%s): %s",
        type(_exc).__name__, _exc,
    )

try:
    from remotemedia.core.multiprocessing.data import RuntimeData
    _HAS_RUNTIME_DATA = True
except ImportError:
    _HAS_RUNTIME_DATA = False
    RuntimeData = None  # type: ignore

try:
    from remotemedia.core.multiprocessing.data import numpy_to_audio
except ImportError:
    numpy_to_audio = None  # type: ignore

from remotemedia.core.multiprocessing import (
    MultiprocessNode,
    NodeConfig,
    python_requires,
    register_node,
)

logger = logging.getLogger(__name__)


AUX_PORT_KEY = "__aux_port__"

# Markers that, when seen as a *complete* text delta, finalise the current
# turn (call session.end_text + drain). LFM2 uses these tags too — they're
# emitted by some upstream LLM clients on sentence / response boundaries.
_FLUSH_MARKERS = frozenset({"<|text_end|>", "<|audio_end|>", "<|end|>", "<|eot|>", "\x04"})

CODEC_SAMPLE_RATE = 24000
DEFAULT_MAX_LENGTH = 5000

_MOSS_REALTIME_REQUIRES = [
    # The realtime variant ships its own python package inside the model
    # repo (`mossttsrealtime`). HF `trust_remote_code` pulls that package
    # in from the snapshot when loading the model and processor — no
    # separate pip install needed for `mossttsrealtime` itself.
    # OpenMOSS needs transformers>=5.0 (see moss_tts.py for full rationale).
    "transformers>=5.0,<6.0",
    "torch>=2.1",
    "torchaudio>=2.1",
    "accelerate>=0.33",
    # Same Windows safetensors hazard as moss_tts.py — see that module
    # for the full rationale.
    "safetensors<0.5",
    # torchaudio >=2.10 delegates non-PCM container loading (.m4a / .mp3
    # / etc.) to torchcodec. Without it `torchaudio.load(...)` on the
    # reference voice prompt raises `TorchCodec is required for
    # load_with_torchcodec`. The realtime node falls back to running
    # without a voice prompt when this fails, but for the canonical
    # OpenMOSS demo audio (which is `.m4a` / `.mp3`) we want this
    # installed by default.
    "torchcodec",
]


@dataclass
class _RealtimeSession:
    """One in-flight conversation. One per IPC ``session_id``."""

    session_id: str
    session: Any                           # MossTTSRealtimeStreamingSession
    decoder: Any                           # AudioStreamDecoder
    codec_streaming_ctx: Any               # codec.streaming(batch_size=1)
    streaming_ctx_entered: bool = False
    codebook_size: int = 1024
    audio_eos_token: int = 1026
    turn_count: int = 0
    in_turn: bool = False                  # True while we're pushing deltas for a turn
    last_accessed: datetime = field(default_factory=datetime.now)

    def touch(self) -> None:
        self.last_accessed = datetime.now()


@register_node("MossTTSRealtimeNode")
@python_requires([
    # Inline literal — see moss_tts.py for the full rationale (the
    # static AST reader can only resolve literal lists at the
    # decorator site). The bare-name `_MOSS_REALTIME_REQUIRES`
    # constant above is kept for documentation; it's not what the
    # runtime reads.
    # OpenMOSS needs transformers>=5.0 (see moss_tts.py for full rationale).
    "transformers>=5.0,<6.0",
    "torch>=2.1",
    "torchaudio>=2.1",
    "accelerate>=0.33",
    "safetensors<0.5",
    # torchaudio >=2.10 delegates non-PCM decode to torchcodec; without
    # it `.m4a`/`.mp3` voice prompts fail at load. Realtime node falls
    # back gracefully but voice cloning is the headline feature.
    "torchcodec",
])
class MossTTSRealtimeNode(MultiprocessNode):
    """
    Streaming MOSS-TTS-Realtime node.

    Per IPC session we hold a single ``MossTTSRealtimeStreamingSession``
    plus its decoder. Text deltas arrive on the main channel and are
    pushed straight through ``session.push_text`` — the resulting audio
    frames are decoded inside the same call and yielded as
    ``RuntimeData.Audio`` (float32 mono @ 24 kHz).

    Turn boundaries: implicit by default (the session keeps pushing into
    the current turn forever). To finalise a turn so the model can drain
    its tail and the next call starts fresh, either:

    - send a ``<|text_end|>`` marker on the main text channel, OR
    - send a dict input ``{"text": "...", "final": true}``, OR
    - publish to the ``audio.in.flush`` aux port.

    Voice cloning: provide a reference audio at construction time via
    ``voice_prompt_path=`` / ``voice_prompt_url=``, or change it live by
    publishing to ``audio.in.reference``. The reference is encoded once
    by the audio codec and the resulting tokens drive every future
    session created in this node.
    """

    def __init__(
        self,
        config: Union[NodeConfig, Dict[str, Any], None] = None,
        *,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        hf_repo: str = "OpenMOSS-Team/MOSS-TTS-Realtime",
        codec_repo: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        device: Optional[str] = None,
        voice_prompt_path: Optional[str] = None,
        voice_prompt_url: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        sample_rate: int = CODEC_SAMPLE_RATE,
        # Decoder pacing — small chunk_frames = lower TTFA, more overhead.
        decode_chunk_frames: int = 3,
        decode_overlap_frames: int = 0,
        chunk_duration: float = 0.24,
        # Decoding hyperparams (from the model card).
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
        session_timeout_minutes: int = 30,
        auto_flush: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, str):
            raise TypeError(
                "MossTTSRealtimeNode requires NodeConfig or keyword-only params; "
                "bare positional node_id not supported"
            )
        if config is None:
            config = NodeConfig(
                node_id=node_id or name or "moss_tts_realtime",
                node_type="MossTTSRealtimeNode",
                params={},
            )
        elif isinstance(config, dict):
            config = NodeConfig(
                node_id=config.get("node_id", node_id or "moss_tts_realtime"),
                node_type=config.get("node_type", "MossTTSRealtimeNode"),
                params=config.get("params", {}),
            )
        super().__init__(config, **kwargs)

        p = config.params or {}
        self.hf_repo = p.get("hf_repo", hf_repo)
        self.codec_repo = p.get("codec_repo", codec_repo)
        self.device = _resolve_device(p.get("device", device))
        self.voice_prompt_path: Optional[str] = (
            p.get("voice_prompt_path", voice_prompt_path)
        )
        self.voice_prompt_url: Optional[str] = (
            p.get("voice_prompt_url", voice_prompt_url)
        )
        self.max_length = int(p.get("max_length", max_length))
        self.sample_rate = int(p.get("sample_rate", sample_rate))
        self.decode_chunk_frames = int(p.get("decode_chunk_frames", decode_chunk_frames))
        self.decode_overlap_frames = int(p.get("decode_overlap_frames", decode_overlap_frames))
        self.chunk_duration = float(p.get("chunk_duration", chunk_duration))
        self.temperature = float(p.get("temperature", temperature))
        self.top_p = float(p.get("top_p", top_p))
        self.top_k = int(p.get("top_k", top_k))
        self.do_sample = bool(p.get("do_sample", do_sample))
        self.repetition_penalty = float(p.get("repetition_penalty", repetition_penalty))
        rep_win = p.get("repetition_window", repetition_window)
        self.repetition_window = (
            int(rep_win) if rep_win is not None and int(rep_win) > 0 else None
        )
        self.session_timeout_minutes = int(p.get("session_timeout_minutes", session_timeout_minutes))
        # When True, treat every text input as a complete utterance:
        # push_text + end_text + drain in one process() call. This is the
        # right default for S2S where Whisper hands us one transcript per
        # turn and there's no streaming-LLM token cadence to preserve.
        # Set to False (or pass `{text:..., final:false}` dicts) when
        # driving from a streaming LLM and you want push_text to
        # accumulate across multiple calls before flushing.
        self.auto_flush = bool(p.get("auto_flush", auto_flush))

        # Lazily-loaded heavyweights.
        self._tokenizer: Any = None
        self._processor: Any = None      # MossTTSRealtimeProcessor
        self._model: Any = None          # MossTTSRealtime
        self._codec: Any = None          # MossAudioTokenizer (AutoModel)
        self._init_classes: Dict[str, Any] = {}   # imported symbols for fast access
        self._initialized = False

        # Encoded voice prompt — a single numpy array of token codes,
        # produced from the reference audio at init time and reused for
        # every per-session session.set_voice_prompt_tokens(...).
        self._voice_prompt_tokens: Optional["np.ndarray"] = None

        # Per-IPC-session state. Single concurrent turn per session
        # (matches OpenMOSS's "batch_size = 1" support note in the
        # fast_api.py reference).
        self._sessions: Dict[str, _RealtimeSession] = {}
        self._sessions_lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        self.name = name or config.node_id
        self.is_streaming = True

        logger.info(
            "MossTTSRealtimeNode constructed: device=%s codec=%s sr=%d",
            self.device, self.codec_repo, self.sample_rate,
        )

    # ────── multiprocess lifecycle ─────────────────────────────────

    async def initialize(self) -> None:
        if not _ML_DEPS_AVAILABLE:
            cause = _ML_IMPORT_ERROR
            detail = (
                f"{type(cause).__name__}: {cause}" if cause is not None
                else "unknown import failure"
            )
            raise RuntimeError(
                f"MossTTSRealtimeNode ML stack failed to import — {detail}. "
                "Required packages: transformers, torch, torchaudio, accelerate. "
                "The model itself ships its `mossttsrealtime` Python package "
                "via HF `trust_remote_code` (loaded from the model snapshot)."
            ) from cause
        if self._initialized:
            return

        self.publish_progress(
            "loading_model",
            f"Loading MOSS-TTS-Realtime from {self.hf_repo} on {self.device}",
        )
        logger.info("[%s] loading %s on %s", self.node_id, self.hf_repo, self.device)

        def _load() -> Dict[str, Any]:
            # Vendored copy of the OpenMOSS `mossttsrealtime` package (no
            # PyPI release upstream — see module docstring for the
            # re-sync recipe).
            from remotemedia._vendor.mossttsrealtime.modeling_mossttsrealtime import (
                MossTTSRealtime,
            )
            from remotemedia._vendor.mossttsrealtime.processing_mossttsrealtime import (
                MossTTSRealtimeProcessor,
            )
            from remotemedia._vendor.mossttsrealtime.streaming_mossttsrealtime import (
                AudioStreamDecoder,
                MossTTSRealtimeInference,
                MossTTSRealtimeStreamingSession,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.hf_repo)
            processor = MossTTSRealtimeProcessor(tokenizer)

            dtype, attn = _select_dtype_and_attn(self.device)
            try:
                model = MossTTSRealtime.from_pretrained(
                    self.hf_repo,
                    attn_implementation=attn,
                    torch_dtype=dtype,
                ).to(self.device)
            except TypeError:
                # Some on-disk modeling files predate the attn_implementation kw.
                model = MossTTSRealtime.from_pretrained(
                    self.hf_repo, torch_dtype=dtype,
                ).to(self.device)
            model.eval()

            codec = AutoModel.from_pretrained(self.codec_repo, trust_remote_code=True).eval()
            codec = codec.to(self.device)

            return {
                "tokenizer": tokenizer,
                "processor": processor,
                "model": model,
                "codec": codec,
                "AudioStreamDecoder": AudioStreamDecoder,
                "MossTTSRealtimeInference": MossTTSRealtimeInference,
                "MossTTSRealtimeStreamingSession": MossTTSRealtimeStreamingSession,
            }

        loaded = await asyncio.to_thread(_load)
        self._tokenizer = loaded["tokenizer"]
        self._processor = loaded["processor"]
        self._model = loaded["model"]
        self._codec = loaded["codec"]
        self._init_classes = {
            "AudioStreamDecoder": loaded["AudioStreamDecoder"],
            "MossTTSRealtimeInference": loaded["MossTTSRealtimeInference"],
            "MossTTSRealtimeStreamingSession": loaded["MossTTSRealtimeStreamingSession"],
        }

        # Pre-encode the reference voice if one was configured. Failure
        # here is logged but non-fatal — subsequent push_text turns will
        # run without voice conditioning until `audio.in.reference`
        # updates it successfully.
        if self.voice_prompt_path or self.voice_prompt_url:
            try:
                await asyncio.to_thread(self._reload_voice_prompt_from_config)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s] failed to load voice prompt at init: %s",
                    self.node_id, exc,
                )

        self._initialized = True
        self.publish_progress("ready", "MOSS-TTS-Realtime ready")
        logger.info("[%s] MOSS-TTS-Realtime ready", self.node_id)

        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def cleanup(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Tear down every live session's codec.streaming(...) context.
        with self._sessions_lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            await asyncio.to_thread(self._destroy_session, sid)

        self._model = None
        self._processor = None
        self._codec = None
        self._tokenizer = None
        self._init_classes.clear()
        self._initialized = False
        logger.info("[%s] cleaned up", self.node_id)

    async def _cleanup_expired_sessions(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                now = datetime.now()
                with self._sessions_lock:
                    expired = [
                        sid for sid, s in self._sessions.items()
                        if (now - s.last_accessed).total_seconds() / 60
                        > self.session_timeout_minutes
                    ]
                for sid in expired:
                    logger.info("[%s] expiring session %s", self.node_id, sid)
                    await asyncio.to_thread(self._destroy_session, sid)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 — keep cleaner alive
                logger.error("[%s] cleanup error: %s", self.node_id, exc)

    # ────── voice-prompt management ────────────────────────────────

    def _reload_voice_prompt_from_config(self) -> None:
        path = self.voice_prompt_path
        if not path and self.voice_prompt_url:
            path = _download_url_to_cache(self.voice_prompt_url)
        if not path:
            return
        self._voice_prompt_tokens = self._encode_voice_prompt(path)
        logger.info(
            "[%s] voice prompt loaded from %s (%d tokens)",
            self.node_id, path,
            0 if self._voice_prompt_tokens is None else self._voice_prompt_tokens.shape[0],
        )

    def _encode_voice_prompt(self, audio_path: str) -> "np.ndarray":
        """Load + resample + codec-encode reference audio → token codes."""
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        with torch.inference_mode():
            result = self._codec.encode(
                wav.unsqueeze(0).to(self.device),
                chunk_duration=self.chunk_duration,
            )
        # ``encode`` returns {"audio_codes": [B, C, T]}. The reference
        # scripts squeeze (1) (the codec channel dim) then cpu+numpy.
        codes = result["audio_codes"]
        try:
            codes = codes.squeeze(1)
        except Exception:  # noqa: BLE001 — shape may already be [T, C]
            pass
        return codes.cpu().numpy().squeeze(0) if codes.ndim == 3 else codes.cpu().numpy()

    def _encode_voice_prompt_samples(self, samples: "np.ndarray", sample_rate: int) -> "np.ndarray":
        wav = torch.from_numpy(samples.astype(np.float32, copy=False)).reshape(1, -1)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        with torch.inference_mode():
            result = self._codec.encode(
                wav.unsqueeze(0).to(self.device),
                chunk_duration=self.chunk_duration,
            )
        codes = result["audio_codes"]
        try:
            codes = codes.squeeze(1)
        except Exception:  # noqa: BLE001
            pass
        return codes.cpu().numpy().squeeze(0) if codes.ndim == 3 else codes.cpu().numpy()

    # ────── per-IPC-session bookkeeping ────────────────────────────

    def _get_or_create_session(self, session_id: str) -> _RealtimeSession:
        with self._sessions_lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.touch()
                return sess

        sess = self._create_session_sync(session_id)
        with self._sessions_lock:
            self._sessions[session_id] = sess
        return sess

    def _create_session_sync(self, session_id: str) -> _RealtimeSession:
        cls = self._init_classes
        Inferencer = cls["MossTTSRealtimeInference"]
        StreamingSession = cls["MossTTSRealtimeStreamingSession"]
        AudioStreamDecoder = cls["AudioStreamDecoder"]

        inferencer = Inferencer(self._model, self._tokenizer, max_length=self.max_length)
        inferencer.reset_generation_state(keep_cache=False)

        session = StreamingSession(
            inferencer,
            self._processor,
            codec=self._codec,
            codec_sample_rate=self.sample_rate,
            codec_encode_kwargs={"chunk_duration": self.chunk_duration},
            prefill_text_len=self._processor.delay_tokens_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            repetition_window=self.repetition_window,
        )
        if self._voice_prompt_tokens is not None:
            try:
                session.set_voice_prompt_tokens(self._voice_prompt_tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] set_voice_prompt_tokens failed: %s", self.node_id, exc,
                )

        decoder = AudioStreamDecoder(
            self._codec,
            chunk_frames=self.decode_chunk_frames,
            overlap_frames=self.decode_overlap_frames,
            decode_kwargs={"chunk_duration": -1},
            device=torch.device(self.device),
        )

        # `codec.streaming(...)` is a context manager that has to wrap
        # every push_text / end_text / drain / decoder.flush call. The
        # OpenMOSS reference scripts use `with codec.streaming(...)`
        # around the whole inner loop. We need it to persist across
        # async calls though, so we manually __enter__ on first use and
        # __exit__ on session destroy.
        codec_streaming_ctx = self._codec.streaming(batch_size=1)

        codebook_size = int(getattr(getattr(self._codec, "config", self._codec),
                                    "codebook_size", 1024))
        audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))

        sess = _RealtimeSession(
            session_id=session_id,
            session=session,
            decoder=decoder,
            codec_streaming_ctx=codec_streaming_ctx,
            codebook_size=codebook_size,
            audio_eos_token=audio_eos_token,
        )
        logger.info("[%s] created realtime session %s", self.node_id, session_id)
        return sess

    def _destroy_session(self, session_id: str) -> None:
        with self._sessions_lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        # Flush + tear down decoder; exit streaming context if entered.
        try:
            if sess.streaming_ctx_entered:
                # Best-effort final flush before exiting the codec stream.
                try:
                    final = sess.decoder.flush()
                    if final is not None and getattr(final, "numel", lambda: 0)() > 0:
                        logger.debug(
                            "[%s] dropped %d trailing samples on session %s teardown",
                            self.node_id, int(final.numel()), session_id,
                        )
                except Exception:  # noqa: BLE001 — best-effort
                    pass
                try:
                    sess.codec_streaming_ctx.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        logger.info("[%s] destroyed realtime session %s", self.node_id, session_id)

    def _ensure_streaming_ctx(self, sess: _RealtimeSession) -> None:
        if not sess.streaming_ctx_entered:
            sess.codec_streaming_ctx.__enter__()
            sess.streaming_ctx_entered = True

    def _start_turn_if_needed(self, sess: _RealtimeSession,
                              user_text: Optional[str] = None) -> None:
        """Open a new TTS turn on the underlying streaming session.

        The vendored ``MossTTSRealtimeStreamingSession.reset_turn`` expects
        either ``(user_text, user_audio_tokens)`` — multi-turn dialogue
        where the user spoke and the assistant replies (see
        ``example_multiturn_stream_to_tts.py``) — or a pre-built
        ``input_ids`` covering ``system_prompt + assistant_prefix`` for
        single-turn read-aloud TTS (see ``example_llm_stream_to_tts.py``).

        Our pipelines hand us TTS text after the conversation has been
        synthesized upstream (Whisper transcript, LLM token stream,
        etc.) — there's no separate user audio to fold into the
        session's chat context. So we always take the single-turn path
        and let ``push_text(...)`` feed the assistant content. This also
        lets us cope with a missing voice prompt: ``make_ensemble(None)``
        emits a system turn without a reference clip, which is what
        :class:`MossTTSRealtimeProcessor` does by default.
        """
        if sess.in_turn:
            return

        first_turn = sess.turn_count == 0
        try:
            system_prompt = self._processor.make_ensemble(self._voice_prompt_tokens)
            assistant_prefix_ids = self._tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            assistant_prefix = np.full(
                (len(assistant_prefix_ids), system_prompt.shape[1]),
                fill_value=self._processor.audio_channel_pad,
                dtype=np.int64,
            )
            assistant_prefix[:, 0] = assistant_prefix_ids
            input_ids = np.concatenate([system_prompt, assistant_prefix], axis=0)

            sess.session.reset_turn(
                input_ids=input_ids,
                include_system_prompt=False,
                reset_cache=first_turn,
            )
        except Exception as exc:  # noqa: BLE001 — surface the actual cause
            logger.exception(
                "[%s] reset_turn failed (voice_prompt_loaded=%s, first_turn=%s)",
                self.node_id, self._voice_prompt_tokens is not None, first_turn,
            )
            raise

        sess.in_turn = True
        sess.turn_count += 1
        logger.info(
            "[%s] session=%s turn=%d started (first=%s, voice_prompt=%s)",
            self.node_id, sess.session_id, sess.turn_count,
            first_turn, self._voice_prompt_tokens is not None,
        )

    # ────── aux-port envelope ──────────────────────────────────────

    def _extract_envelope(self, data: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        blob = self._to_dict_or_none(data)
        if not isinstance(blob, dict):
            return None
        port = blob.get(AUX_PORT_KEY)
        if not isinstance(port, str) or not port:
            return None
        payload = blob.get("payload")
        if not isinstance(payload, dict):
            payload = {"text": str(payload)} if payload is not None else {}
        return port, payload

    def _to_dict_or_none(self, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            s = data.strip()
            if s.startswith("{"):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    return None
            return None
        if _HAS_RUNTIME_DATA and RuntimeData is not None and isinstance(data, RuntimeData):
            try:
                if data.is_text():
                    return self._to_dict_or_none(data.as_text())
                if data.is_json():
                    return data.as_json()
            except Exception:  # noqa: BLE001
                return None
        return None

    def _handle_aux_port(self, port: str, payload: Dict[str, Any]) -> str:
        """Return a string command for the main loop to act on:

            ``""``           — no follow-up action (state updated in place).
            ``"flush:<sid>"``— end_text + drain the named session.
            ``"reset:<sid>"``— tear down the named session.
            ``"new_turn:<sid>"`` — close current turn so the next push
                                   starts a new one.
        """
        short = port.split(".")[-1]   # accept "audio.in.X" or "X"
        if short in ("reference", "voice_prompt"):
            try:
                self._update_voice_prompt_from_payload(payload)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] reference update failed: %s", self.node_id, exc)
                return ""
            # Invalidate every session so the next push reinitialises
            # with the new voice prompt.
            with self._sessions_lock:
                sids = list(self._sessions.keys())
            for sid in sids:
                self._destroy_session(sid)
            return ""
        if short == "reset":
            sid = payload.get("session_id") or "*"
            return f"reset:{sid}"
        if short == "flush":
            sid = payload.get("session_id") or "*"
            return f"flush:{sid}"
        if short == "new_turn":
            sid = payload.get("session_id") or "*"
            return f"new_turn:{sid}"
        if short == "system_prompt":
            # Realtime model bakes system prompt into the chat template.
            # Acknowledge but don't act — log and keep going.
            logger.info("[%s] system_prompt aux port: informational only on realtime", self.node_id)
            return ""
        logger.warning("[%s] unknown aux port %r", self.node_id, port)
        return ""

    def _update_voice_prompt_from_payload(self, payload: Dict[str, Any]) -> None:
        if "tokens" in payload:
            tokens = payload["tokens"]
            self._voice_prompt_tokens = np.asarray(tokens, dtype=np.int64)
            return
        if "audio_path" in payload:
            self._voice_prompt_tokens = self._encode_voice_prompt(payload["audio_path"])
            return
        if "audio_url" in payload:
            path = _download_url_to_cache(payload["audio_url"])
            self._voice_prompt_tokens = self._encode_voice_prompt(path)
            return
        if "samples" in payload and "sample_rate" in payload:
            samples = np.asarray(payload["samples"], dtype=np.float32)
            self._voice_prompt_tokens = self._encode_voice_prompt_samples(
                samples, int(payload["sample_rate"]),
            )
            return
        raise ValueError(
            "voice prompt payload must include one of: tokens, audio_path, audio_url, "
            "or (samples, sample_rate)"
        )

    # ────── main processing entrypoint ─────────────────────────────

    async def process(self, data: Any) -> AsyncGenerator[Any, None]:
        if not _HAS_RUNTIME_DATA or RuntimeData is None:
            logger.error(
                "[%s] RuntimeData bindings unavailable — cannot emit audio", self.node_id,
            )
            return

        if not self._initialized:
            await self.initialize()

        session_id = (
            data.session_id
            if hasattr(data, "session_id") and data.session_id else "default"
        )

        envelope = self._extract_envelope(data)
        if envelope is not None:
            port, payload = envelope
            cmd = self._handle_aux_port(port, payload)
            if cmd.startswith("reset:"):
                sids = self._expand_session_target(cmd[len("reset:"):])
                for sid in sids:
                    await asyncio.to_thread(self._destroy_session, sid)
                return
            if cmd.startswith("flush:"):
                sids = self._expand_session_target(cmd[len("flush:"):])
                for sid in sids:
                    sess = self._sessions.get(sid)
                    if sess is None:
                        continue
                    async for chunk in self._flush_turn(sess):
                        yield chunk
                return
            if cmd.startswith("new_turn:"):
                sids = self._expand_session_target(cmd[len("new_turn:"):])
                for sid in sids:
                    sess = self._sessions.get(sid)
                    if sess is None:
                        continue
                    if sess.in_turn:
                        # close the current turn cleanly before next push
                        async for chunk in self._flush_turn(sess):
                            yield chunk
                return
            return

        # Normalise input → (text_delta, final_flag, user_text). Dict
        # callers can override `final` explicitly (e.g. set False on
        # streaming-LLM tokens); plain text callers inherit `auto_flush`.
        text_delta: Optional[str] = None
        final_flag = self.auto_flush
        user_text: Optional[str] = None

        blob = self._to_dict_or_none(data)
        if isinstance(blob, dict):
            text_delta = blob.get("text")
            if "final" in blob:
                final_flag = bool(blob["final"])
            user_text = blob.get("user_text")
        elif hasattr(data, "is_text") and data.is_text():
            try:
                text_delta = data.as_text()
            except Exception:  # noqa: BLE001
                text_delta = None
        elif isinstance(data, str):
            text_delta = data

        if text_delta is None and not final_flag:
            kind = getattr(data, "data_type", lambda: type(data).__name__)()
            logger.error("[%s] expected text/dict input, got %s", self.node_id, kind)
            yield RuntimeData.text(f"ERROR: expected text or dict input, got {kind}")
            return

        # Flush-marker shortcut.
        if text_delta is not None and text_delta in _FLUSH_MARKERS:
            sess = self._sessions.get(session_id)
            if sess is not None:
                async for chunk in self._flush_turn(sess):
                    yield chunk
            return

        sess = await asyncio.to_thread(self._get_or_create_session, session_id)
        self._ensure_streaming_ctx(sess)
        self._start_turn_if_needed(sess, user_text=user_text)

        if text_delta:
            try:
                async for chunk in self._push_text_and_yield(sess, text_delta):
                    yield chunk
            except RuntimeError as exc:
                if "CUDA" in str(exc):
                    logger.exception("[%s] CUDA error on push_text", self.node_id)
                    raise
                logger.exception("[%s] push_text failed", self.node_id)
                yield RuntimeData.text(f"ERROR: {exc}")
                # destroy this session — recoverable on next call.
                await asyncio.to_thread(self._destroy_session, session_id)
                return

        if final_flag:
            async for chunk in self._flush_turn(sess):
                yield chunk

    def _expand_session_target(self, target: str) -> List[str]:
        if target == "*" or not target:
            with self._sessions_lock:
                return list(self._sessions.keys())
        return [target]

    async def _push_text_and_yield(
        self, sess: _RealtimeSession, text_delta: str,
    ) -> AsyncGenerator[Any, None]:
        # Run push_text + per-frame decode in a worker thread so the
        # async pipeline keeps draining IPC. Capture frames + decoded
        # numpy chunks together so we don't keep crossing the GIL barrier.
        chunks: List["np.ndarray"] = await asyncio.to_thread(
            self._push_text_sync, sess, text_delta,
        )
        for arr in chunks:
            if arr.size == 0:
                continue
            if numpy_to_audio is not None:
                yield numpy_to_audio(arr, self.sample_rate, channels=1)
            else:
                yield RuntimeData.audio(arr, self.sample_rate, channels=1)
            await asyncio.sleep(0)

    def _push_text_sync(self, sess: _RealtimeSession, text_delta: str) -> List["np.ndarray"]:
        with torch.inference_mode():
            frames = sess.session.push_text(text_delta)
        return list(_decode_audio_frames(frames, sess.decoder,
                                         sess.codebook_size, sess.audio_eos_token))

    async def _flush_turn(self, sess: _RealtimeSession) -> AsyncGenerator[Any, None]:
        """Run end_text + drain + decoder.flush and yield all remaining audio."""
        if not sess.in_turn:
            return
        try:
            chunks: List["np.ndarray"] = await asyncio.to_thread(
                self._flush_turn_sync, sess,
            )
            for arr in chunks:
                if arr.size == 0:
                    continue
                if numpy_to_audio is not None:
                    yield numpy_to_audio(arr, self.sample_rate, channels=1)
                else:
                    yield RuntimeData.audio(arr, self.sample_rate, channels=1)
                await asyncio.sleep(0)
        finally:
            sess.in_turn = False
        # Terminal marker for downstream cut-over (matches lfm2_audio).
        yield RuntimeData.text("<|audio_end|>")

    def _flush_turn_sync(self, sess: _RealtimeSession) -> List["np.ndarray"]:
        out: List["np.ndarray"] = []
        with torch.inference_mode():
            frames = sess.session.end_text()
        out.extend(_decode_audio_frames(frames, sess.decoder,
                                        sess.codebook_size, sess.audio_eos_token))
        # Drain — at most a few thousand steps in normal operation, but
        # short-circuit on inferencer.is_finished to avoid runaway loops
        # if the model refuses to emit EOS.
        for _ in range(10_000):
            with torch.inference_mode():
                frames = sess.session.drain(max_steps=1)
            if not frames:
                break
            out.extend(_decode_audio_frames(frames, sess.decoder,
                                            sess.codebook_size, sess.audio_eos_token))
            if getattr(sess.session.inferencer, "is_finished", False):
                break
        # Decoder.flush — the tail of the overlap window.
        try:
            final = sess.decoder.flush()
            if final is not None and getattr(final, "numel", lambda: 0)() > 0:
                arr = final.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                out.append(arr)
        except Exception:  # noqa: BLE001 — flush is best-effort
            pass
        return out

    # ────── introspection ──────────────────────────────────────────

    def get_config(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": "MossTTSRealtimeNode",
            "hf_repo": self.hf_repo,
            "codec_repo": self.codec_repo,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "decode_chunk_frames": self.decode_chunk_frames,
            "decode_overlap_frames": self.decode_overlap_frames,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "repetition_window": self.repetition_window,
            "voice_prompt_loaded": self._voice_prompt_tokens is not None,
            "active_sessions": len(self._sessions),
        }


# ───────────────────── module-level helpers ───────────────────────────


def _resolve_device(req_device: Optional[str]) -> str:
    """Same rationale as in :mod:`moss_tts`: pin to a concrete index so
    accelerate doesn't shard sub-modules across GPUs."""
    if req_device is None:
        if _ML_DEPS_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if req_device == "cuda":
        return "cuda:0"
    return req_device


def _select_dtype_and_attn(device: str) -> Tuple[Any, str]:
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn = "sdpa"
        try:
            import importlib.util
            if importlib.util.find_spec("flash_attn") is not None:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8 and dtype in (torch.float16, torch.bfloat16):
                    attn = "flash_attention_2"
        except Exception:  # noqa: BLE001
            pass
        return dtype, attn
    return torch.float32, "eager"


def _sanitize_tokens(
    tokens: "torch.Tensor", codebook_size: int, audio_eos_token: int,
) -> Tuple["torch.Tensor", bool]:
    """Trim out-of-range codes / cut at first EOS row. Mirrors the
    reference example_llm_stream_to_tts.py helper exactly."""
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens, False
    eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
    stop_idx = None
    if eos_rows.numel() > 0:
        stop_idx = int(eos_rows[0].item())
    if invalid_rows.any():
        invalid_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)
    if stop_idx is not None:
        return tokens[:stop_idx], True
    return tokens, False


def _decode_audio_frames(
    audio_frames: List["torch.Tensor"],
    decoder: Any,
    codebook_size: int,
    audio_eos_token: int,
):
    """Generator: codec token frames → decoded float32 numpy chunks."""
    for frame in audio_frames:
        tokens = frame
        if tokens.dim() == 3:
            tokens = tokens[0]
        if tokens.dim() != 2:
            logger.warning("expected [T, C] audio tokens, got %s", tuple(tokens.shape))
            continue
        tokens, _ = _sanitize_tokens(tokens, codebook_size, audio_eos_token)
        if tokens.numel() == 0:
            continue
        decoder.push_tokens(tokens.detach())
        for wav in decoder.audio_chunks():
            if wav.numel() == 0:
                continue
            yield wav.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


# ───────────────────── full-path registry alias ──────────────────────
#
# See moss_tts.py for the full rationale. `loader.register_node_class()`
# tells the FFI subprocess runner to look up the class by full module
# path, but `@register_node` registers only the bare name. Mirror the
# bare-name registration under the full-path key here.
try:
    from remotemedia.core.multiprocessing import _NODE_REGISTRY as _MP_REGISTRY
    _MP_REGISTRY[f"{MossTTSRealtimeNode.__module__}.{MossTTSRealtimeNode.__name__}"] = MossTTSRealtimeNode
except ImportError:
    pass


def _download_url_to_cache(url: str) -> str:
    """Download a URL to a stable cache location and return the path.

    Used so a voice-prompt URL only needs to be fetched once even if the
    node is restarted or the session is recycled mid-stream.
    """
    import hashlib
    cache = Path.home() / ".cache" / "remotemedia" / "moss_tts_realtime"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".wav"
    target = cache / f"{digest}{suffix}"
    if target.exists():
        return str(target)
    logger.info("downloading voice prompt: %s → %s", url, target)
    urlretrieve(url, str(target))
    return str(target)
