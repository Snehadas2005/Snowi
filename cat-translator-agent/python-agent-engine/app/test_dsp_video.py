"""
# changes directly will go to the next step
from app.audio_dsp import AudioProcessor

processor = AudioProcessor()

# Works seamlessly across formats
video_file = "../test_samples/blind_cat_video.mp4"
audio_file = "test_samples/kitten_meowing_audio.mp3"

for media_path in [video_file, audio_file]:
    try:
        res = processor.process_file_to_tensor_input(media_path)
        print(f"File processed successfully: {media_path}")
        print(f"Output shape: {res['spectrogram'].shape}\n")
    except Exception as e:
        print(f"Failed processing {media_path}: {e}")
"""
"""
# changes seen in form of graph
import matplotlib.pyplot as plt
from app.audio_dsp import AudioProcessor

processor = AudioProcessor()

# Media files
media_files = {
    "video": "../test_samples/blind_cat_video.mp4",
    "audio": "../test_samples/kitten_meowing_audio.mp3"
}

for media_type, file_path in media_files.items():
    try:
        res = processor.process_file_to_tensor_input(file_path)
        spec = res["spectrogram"][0]  # Shape: (128, 130)

        # Create plot
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Cleaned Mel-Spectrogram ({media_type.capitalize()} Source)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()

        # Save distinct PNG file for each format
        output_image = f"../test_samples/spectrogram_{media_type}.png"
        plt.savefig(output_image)
        plt.close()  # Clear memory
        
        print(f"Processed {media_type} successfully -> Saved plot: {output_image}")

    except Exception as e:
        print(f"Failed processing {media_type} ({file_path}): {e}")
"""

# changes seen in form of audio or video file
from app.audio_dsp import AudioProcessor

processor = AudioProcessor()

media_files = [
    ("../test_samples/blind_cat_video.mp4", "../test_samples/cleaned_from_video.wav"),
    ("../test_samples/kitten_meowing_audio.mp3", "../test_samples/cleaned_from_audio.wav"),
]

for src_file, out_wav in media_files:
    try:
        # 1. Inspect PyTorch-ready tensor data
        res = processor.process_file_to_tensor_input(src_file)
        print(f"File processed for model: {src_file}")
        print(f"Tensor Shape ready for PyTorch CNN: {res['spectrogram'].shape}")

        # 2. Save the cleaned audio file to disk
        saved_file = processor.save_cleaned_audio(src_file, out_wav)
        print(f"Listen to cleaned audio at: {saved_file}\n")

    except Exception as e:
        print(f"Failed processing {src_file}: {e}\n")