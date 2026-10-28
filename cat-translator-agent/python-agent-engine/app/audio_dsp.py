"""
Real-Time Low-Latency Audio & Video DSP Pipeline.

Optimized for microservice deployment via Docker/Kubernetes.
Executes sub-second signal processing using fast spectral noise gating,
Butterworth bandpass filtering, and VAD silence trimming.
"""

import logging
from pathlib import Path
import subprocess

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, sosfilt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class AudioProcessor:
    """Fast DSP Engine optimized for real-time mobile backend API routing."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 250.0,
        f_max: float = 8000.0,
        top_db: float = 25.0,
        target_duration_sec: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.top_db = top_db
        self.target_duration_sec = target_duration_sec

    def _extract_audio_from_video_stream(self, video_path: Path) -> np.ndarray:
        """Extracts mono PCM audio in-memory via C-level FFmpeg pipe."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "pipe:1",
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error("FFmpeg extraction failed: %s", stderr.decode())
                raise RuntimeError(f"FFmpeg process error: {stderr.decode()}")

            audio_buffer = np.frombuffer(stdout, dtype=np.int16)
            return audio_buffer.astype(np.float32) / 32768.0

        except Exception as err:
            logger.error("Error executing FFmpeg stream: %s", err)
            raise RuntimeError(
                f"Failed to strip audio from video stream: {err}"
            ) from err

    def load_and_preprocess(self, file_path: str | Path) -> np.ndarray:
        """Fast sub-second multi-stage audio clean pipeline."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")

        # 1. Direct Audio Decoding / Extraction
        if path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            y = self._extract_audio_from_video_stream(path)
        else:
            y, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)

        # 2. Fast Non-Stationary Spectral Noise Reduction (Parallelized OpenMP execution)
        y_denoised = nr.reduce_noise(
            y=y,
            sr=self.sample_rate,
            stationary=True,  # Fast stationary profile gating for ambient room noise
            prop_decrease=0.80,
            n_fft=1024,  # Smaller FFT window for sub-second execution speed
            n_jobs=-1,  # Utilize all available CPU cores inside Docker container
        ).astype(np.float32)

        # 3. Butterworth Bandpass Filter (Isolates 250Hz - 8000Hz)
        nyquist = 0.5 * self.sample_rate
        low = self.f_min / nyquist
        high = self.f_max / nyquist
        sos = butter(N=5, Wn=[low, high], btype="bandpass", output="sos")
        y_filtered = sosfilt(sos, y_denoised).astype(np.float32)

        # 4. Fast VAD Silence Trimming
        trimmed_y, _ = librosa.effects.trim(
            y_filtered,
            top_db=self.top_db,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )
        return trimmed_y if len(trimmed_y) > 0 else y_filtered

    def normalize_length(self, y: np.ndarray) -> np.ndarray:
        """Pads or center-crops audio signal duration to maintain uniform dimensions."""
        target_samples = int(self.target_duration_sec * self.sample_rate)
        current_samples = len(y)

        if current_samples == target_samples:
            return y

        if current_samples > target_samples:
            start = (current_samples - target_samples) // 2
            return y[start : start + target_samples]

        pad_left = (target_samples - current_samples) // 2
        pad_right = target_samples - current_samples - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant")

    def extract_log_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Computes Z-Score normalized 2D Log Mel-Spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec) + 1e-6
        return ((log_mel_spec - mean) / std).astype(np.float32)

    def process_file_to_tensor_input(
        self, file_path: str | Path
    ) -> dict[str, np.ndarray | float]:
        """Transforms raw audio or video files into standardized CNN input tensors in sub-second latency."""
        y_clean = self.load_and_preprocess(file_path)
        y_normalized = self.normalize_length(y_clean)

        spectrogram = self.extract_log_mel_spectrogram(y_normalized)
        tensor_ready_spec = np.expand_dims(spectrogram, axis=0)

        return {
            "spectrogram": tensor_ready_spec,
            "duration_processed": float(len(y_normalized) / self.sample_rate),
            "sample_rate": self.sample_rate,
        }

    def save_cleaned_audio(
        self, file_path: str | Path, output_wav_path: str | Path
    ) -> str:
        """Exports the isolated and cleaned audio file to disk."""
        y_clean = self.load_and_preprocess(file_path)
        out_path = Path(output_wav_path)
        sf.write(str(out_path), y_clean, self.sample_rate)
        logger.info("Saved cleaned audio output to: %s", out_path)
        return str(out_path)
