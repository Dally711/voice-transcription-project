import json  # :)
import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import difflib
from librosa.sequence import dtw
from pydub import AudioSegment
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow_hub as hub
import re

SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 2.0
MIN_FRAMES = 10
DTW_THRESHOLD = 0.65
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
# Loosen YAMNet threshold but gate by agreement with other signals.
YAMNET_MAX_DISTANCE = 0.4   # reject noisy matches above this
YAMNET_MIN_MARGIN = 0.04    # require separation from runner-up
YAMNET_CACHE_FILE = "yamnet_cache.npz"


class CommandRecognizer:
    """Combines transcript keywords, DTW templates, and YAMNet embeddings."""

    _yamnet_model = None

    def __init__(
        self,
        data_dir: Path,
        command_json: str = "commands.json",
        sample_subdirs: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.command_json_path = self.data_dir / command_json
        self.sample_subdirs = sample_subdirs or ["samples"]
        self.commands = self._load_commands()
        self.command_names = [cmd.lower() for cmd in self.commands.keys()]
        self.sample_db = self._build_sample_embeddings()

    # ---------------------------
    # Loading helpers
    # ---------------------------
    @classmethod
    def _load_yamnet(cls):
        if cls._yamnet_model is None:
            print("Loading YAMNet model for command embeddings...")
            cls._yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
            print("YAMNet model ready.")
        return cls._yamnet_model

    def _load_commands(self) -> Dict[str, dict]:
        if not self.command_json_path.exists():
            print(f"Command JSON not found: {self.command_json_path}")
            return {}
        try:
            with self.command_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data = self._merge_numbered_labels(data)
            data = self._filter_bad_samples(data)
            print(f"Loaded {len(data)} commands from {self.command_json_path.name}.")
            return data
        except json.JSONDecodeError as exc:
            print(f"Failed to parse {self.command_json_path}: {exc}")
            return {}

    def _merge_numbered_labels(self, data: Dict[str, dict]) -> Dict[str, dict]:
        """Merge keys ending in a trailing number into their base name."""
        merged: Dict[str, dict] = {}
        for key, value in data.items():
            base = re.sub(r"\s+\d+$", "", key).strip()
            target = merged.setdefault(base, {"samples": []})
            # Merge samples
            samples = value.get("samples", [])
            if isinstance(samples, list):
                target.setdefault("samples", [])
                target["samples"].extend(samples)
            # Preserve other fields if not already present
            for k, v in value.items():
                if k == "samples":
                    continue
                target.setdefault(k, v)
        return merged

    def _filter_bad_samples(self, data: Dict[str, dict]) -> Dict[str, dict]:
        """Remove samples that would break DTW, and report them."""
        removed = []
        for cmd, entry in data.items():
            samples = entry.get("samples", []) or []
            cleaned = []
            for idx, sample in enumerate(samples):
                prepared = self._prepare_sample(sample)
                if (
                    prepared is None
                    or prepared.ndim != 2
                    or prepared.shape[1] == 0
                    or not np.isfinite(prepared).all()
                ):
                    removed.append((cmd, idx, "invalid shape"))
                    continue
                try:
                    # Self-check to ensure DTW will accept this sample.
                    dtw(prepared, prepared, metric="cosine")
                except Exception as exc:
                    removed.append((cmd, idx, f"dtw failed: {exc}"))
                    continue
                cleaned.append(sample)
            entry["samples"] = cleaned
        if removed:
            print(f"Removed {len(removed)} invalid DTW samples.")
            for cmd, idx, reason in removed[:10]:
                print(f"  - {cmd} sample {idx}: {reason}")
        return data

    def _load_cached_embeddings(self) -> Optional[Dict[str, List[np.ndarray]]]:
        cache_path = self.data_dir / YAMNET_CACHE_FILE
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            obj = data.get("db", None)
            if obj is None:
                return None
            db = obj.item()
            if isinstance(db, dict):
                return db
        except Exception as exc:
            print(f"Failed to load YAMNet cache: {exc}")
        return None

    def _save_cached_embeddings(self, db: Dict[str, List[np.ndarray]]) -> None:
        cache_path = self.data_dir / YAMNET_CACHE_FILE
        try:
            np.savez(cache_path, db=np.array(db, dtype=object))
            print(f"Saved YAMNet cache to {cache_path.name}.")
        except Exception as exc:
            print(f"Failed to save YAMNet cache: {exc}")

    def _build_sample_embeddings(self) -> Dict[str, List[np.ndarray]]:
        cached = self._load_cached_embeddings()
        if cached is not None:
            print(f"Loaded cached embeddings for {len(cached)} commands from {YAMNET_CACHE_FILE}.")
            return cached

        db: Dict[str, List[np.ndarray]] = {}
        any_audio = False
        bases = [self.data_dir / sub for sub in self.sample_subdirs]
        for base in bases:
            if not base.exists():
                continue
            for entry in base.iterdir():
                if entry.is_dir():
                    label = entry.name
                    files = list(entry.glob("*"))
                    embeddings = [self._embed_file(f) for f in files if f.is_file()]
                    embeddings = [emb for emb in embeddings if emb is not None]
                    if embeddings:
                        db.setdefault(label, []).extend(embeddings)
                        any_audio = True
                elif entry.is_file():
                    label = entry.stem.replace("_", " ").strip()
                    emb = self._embed_file(entry)
                    if emb is not None:
                        db.setdefault(label, []).append(emb)
                        any_audio = True
        if any_audio:
            print(f"Loaded embeddings for {len(db)} commands from audio folders.")
            self._save_cached_embeddings(db)
        return db

    def _embed_file(self, path: Path) -> Optional[np.ndarray]:
        try:
            audio = self._load_audio_any_format(path)
            return self._extract_embedding(audio)
        except Exception as exc:
            print(f"Failed to embed {path.name}: {exc}")
            return None

    # ---------------------------
    # Audio + feature helpers
    # ---------------------------
    def _load_audio_any_format(self, path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(audio.array_type).max
        return samples

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        model = self._load_yamnet()
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = model(audio_tensor)
        return tf.reduce_mean(embeddings, axis=0).numpy()

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        audio = np.array(audio, dtype=np.float32)
        target_len = int(MIN_AUDIO_SEC * SAMPLE_RATE)
        if audio.size < target_len:
            audio = np.pad(audio, (0, target_len - audio.size))
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        return audio

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            raise ValueError("Empty audio input")
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
        if mfcc.ndim != 2 or mfcc.shape[1] == 0:
            raise ValueError("Invalid MFCC shape")
        if mfcc.shape[1] < 2:
            mfcc = np.pad(mfcc, ((0, 0), (0, MIN_FRAMES - mfcc.shape[1])), mode="edge")
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        if combined.ndim != 2 or combined.shape[0] == 0 or combined.shape[1] == 0:
            raise ValueError("Invalid combined feature shape")
        return combined.T  # (frames, features)

    def _prepare_sample(self, sample: List[List[float]]) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(sample, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return None
        if not np.isfinite(arr).all():
            return None
        try:
            # Pad very short sequences so delta computation doesn't fail.
            if arr.shape[0] < 3:
                arr = np.pad(arr, ((0, 3 - arr.shape[0]), (0, 0)), mode="edge")
            if arr.shape[1] == 13:
                delta = librosa.feature.delta(arr.T).T
                delta2 = librosa.feature.delta(arr.T, order=2).T
                arr = np.hstack([arr, delta, delta2])
            if arr.shape[0] < MIN_FRAMES:
                arr = np.pad(arr, ((0, MIN_FRAMES - arr.shape[0]), (0, 0)), mode="edge")
        except Exception:
            return None
        return arr

    def _dtw_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        f1 = np.atleast_2d(f1)
        f2 = np.atleast_2d(f2)
        if not np.isfinite(f1).all() or not np.isfinite(f2).all():
            return -1.0
        if f1.ndim != 2 or f2.ndim != 2:
            return -1.0
        min_feat = min(f1.shape[1], f2.shape[1])
        if min_feat == 0:
            return -1.0
        f1 = f1[:, :min_feat]
        f2 = f2[:, :min_feat]
        try:
            D, _ = dtw(f1, f2, metric="cosine")
        except Exception:
            # Skip malformed pairs quietly to avoid noisy logs.
            return -1.0
        return 1 - D[-1, -1] / max(D.shape)

    # ---------------------------
    # Public API
    # ---------------------------
    def match(self, audio: np.ndarray, transcript: Optional[str] = None) -> Dict[str, Optional[dict]]:
        result: Dict[str, Optional[dict]] = {"transcript": None, "dtw": None, "yamnet": None, "text_fuzzy": None}
        processed_audio = self._preprocess_audio(audio)

        # 1) Transcript exact/substring match + fuzzy match
        if transcript:
            lower_text = transcript.lower().strip()
            for cmd in self.command_names:
                if cmd in lower_text:
                    result["transcript"] = {"command": cmd, "score": 1.0}
                    break
            # Fuzzy map to nearest command
            best_ratio = 0.0
            best_cmd = None
            for cmd in self.command_names:
                ratio = difflib.SequenceMatcher(None, lower_text, cmd).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cmd = cmd
            if best_cmd and best_ratio >= 0.50:
                result["text_fuzzy"] = {"command": best_cmd, "score": float(best_ratio)}

        # 2) DTW template matching
        if processed_audio.size and self.commands:
            features = self._extract_features(processed_audio)
            best_match = None
            best_score = -1.0
            second_best = -1.0
            for cmd, data in self.commands.items():
                for sample in data["samples"]:
                    sample_mfcc = self._prepare_sample(sample)
                    if sample_mfcc is None or sample_mfcc.shape[1] == 0:
                        continue
                    sim = self._dtw_similarity(features, sample_mfcc)
                    if sim < 0:
                        continue
                    if sim > best_score:
                        second_best = best_score
                        best_score = sim
                        best_match = cmd

            if best_match and best_score >= DTW_THRESHOLD and best_score - second_best > 0.06:
                result["dtw"] = {
                    "command": best_match,
                    "score": float(best_score),
                    "margin": float(best_score - second_best),
                }

        # 3) YAMNet embedding similarity
        if processed_audio.size and self.sample_db:
            query_emb = self._extract_embedding(processed_audio)
            best_cmd = None
            best_distance = None
            second_distance = None
            for cmd, embeddings in self.sample_db.items():
                if not embeddings:
                    continue
                avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
                dist = cosine(query_emb, avg_emb)
                if best_distance is None or dist < best_distance:
                    second_distance = best_distance
                    best_distance = dist
                    best_cmd = cmd
                elif second_distance is None or dist < second_distance:
                    second_distance = dist
            if best_cmd is not None and best_distance is not None:
                allow_set = set()
                if result.get("transcript"):
                    allow_set.add(result["transcript"]["command"])
                if result.get("dtw"):
                    allow_set.add(result["dtw"]["command"])
                passes_threshold = best_distance <= YAMNET_MAX_DISTANCE
                passes_margin = (
                    second_distance is None
                    or (second_distance - best_distance) >= YAMNET_MIN_MARGIN
                )
                agrees_with_other = not allow_set or best_cmd in allow_set
                if passes_threshold and passes_margin and agrees_with_other:
                    result["yamnet"] = {
                        "command": best_cmd,
                        "distance": float(best_distance),
                        "score": float(1 - best_distance),
                    }

        return result

