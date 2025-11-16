import json  # :)
import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
from librosa.sequence import dtw
from pydub import AudioSegment
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow_hub as hub

SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 2.0
MIN_FRAMES = 10
DTW_THRESHOLD = 0.7
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"


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
            print(f"Loaded {len(data)} commands from {self.command_json_path.name}.")
            return data
        except json.JSONDecodeError as exc:
            print(f"Failed to parse {self.command_json_path}: {exc}")
            return {}

    def _build_sample_embeddings(self) -> Dict[str, List[np.ndarray]]:
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
        if mfcc.shape[1] < 2:
            mfcc = np.pad(mfcc, ((0, 0), (0, MIN_FRAMES - mfcc.shape[1])), mode="edge")
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        return combined.T

    def _prepare_sample(self, sample: List[List[float]]) -> np.ndarray:
        arr = np.array(sample, dtype=np.float32)
        if arr.ndim != 2:
            arr = arr.reshape((-1, arr.shape[-1]))
        if arr.shape[1] == 13:
            delta = librosa.feature.delta(arr.T).T
            delta2 = librosa.feature.delta(arr.T, order=2).T
            arr = np.hstack([arr, delta, delta2])
        if arr.shape[0] < MIN_FRAMES:
            arr = np.pad(arr, ((0, MIN_FRAMES - arr.shape[0]), (0, 0)), mode="edge")
        return arr

    def _dtw_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        f1 = np.atleast_2d(f1)
        f2 = np.atleast_2d(f2)
        if f1.shape[1] == 0 or f2.shape[1] == 0:
            return -1.0
        min_feat = min(f1.shape[1], f2.shape[1])
        f1 = f1[:, :min_feat]
        f2 = f2[:, :min_feat]
        try:
            D, _ = dtw(f1, f2, metric="cosine")
        except Exception as exc:
            print(f"DTW similarity failed: {exc}")
            return -1.0
        return 1 - D[-1, -1] / max(D.shape)

    # ---------------------------
    # Public API
    # ---------------------------
    def match(self, audio: np.ndarray, transcript: Optional[str] = None) -> Dict[str, Optional[dict]]:
        result: Dict[str, Optional[dict]] = {"transcript": None, "dtw": None, "yamnet": None}
        processed_audio = self._preprocess_audio(audio)

        # 1) Transcript keyword match
        if transcript:
            lower_text = transcript.lower()
            for cmd in self.command_names:
                if cmd in lower_text:
                    result["transcript"] = {"command": cmd, "score": 1.0}
                    break

        # 2) DTW template matching
        if processed_audio.size and self.commands:
            features = self._extract_features(processed_audio)
            best_match = None
            best_score = -1.0
            second_best = -1.0
            for cmd, payload in self.commands.items():
                for sample in payload.get("samples", []):
                    sample_arr = self._prepare_sample(sample)
                    sim = self._dtw_similarity(features, sample_arr)
                    if sim > best_score:
                        second_best = best_score
                        best_score = sim
                        best_match = cmd
            if best_match and best_score >= DTW_THRESHOLD and best_score - second_best > 0.05:
                result["dtw"] = {"command": best_match, "score": float(best_score)}

        # 3) YAMNet embedding similarity
        if processed_audio.size and self.sample_db:
            query_emb = self._extract_embedding(processed_audio)
            best_cmd = None
            best_distance = None
            for cmd, embeddings in self.sample_db.items():
                if not embeddings:
                    continue
                avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
                dist = cosine(query_emb, avg_emb)
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_cmd = cmd
            if best_cmd is not None and best_distance is not None:
                result["yamnet"] = {
                    "command": best_cmd,
                    "distance": float(best_distance),
                    "score": float(1 - best_distance),
                }

        return result

