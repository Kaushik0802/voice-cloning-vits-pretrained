import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from TTS.api import TTS

class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = config["model"]["base_model"]
        self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)

        # Available speaker list (from VCTK)
        self.speakers = [
            "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234",
            "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246",
            "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256",
            "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266",
            "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276",
            "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286",
            "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300",
            "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311",
            "p312", "p313", "p314", "p316", "p317", "p318", "p323", "p326", "p329", "p330",
            "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347",
            "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"
        ]

    def synthesize_all(self, limit_speakers=100, limit_sentences=20):
        output_base = os.path.join(self.config["project"]["output_dir"], "generated_sentences")
        os.makedirs(output_base, exist_ok=True)

        # Sentences list
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "She sells seashells by the seashore.",
            "Voice cloning technology is amazing.",
            "Welcome to the world of machine learning.",
            "Python is a very powerful programming language.",
            "This project focuses on text to speech synthesis.",
            "Data science and AI are shaping our future.",
            "Natural language processing enables human-machine communication.",
            "Deep learning models require a lot of data.",
            "I love experimenting with new AI models.",
            "Music and voice synthesis have a lot in common.",
            "Speech generation is a fascinating field.",
            "TensorFlow and PyTorch are popular frameworks.",
            "The future is bright for AI researchers.",
            "Zero-shot learning enables new capabilities.",
            "Fast speech synthesis models are game-changers.",
            "Exploring TTS technology is really exciting.",
            "Audio quality matters in cloned voices.",
            "Different speakers have unique vocal features."
        ]

        for s_idx, sentence in enumerate(sentences[:limit_sentences], start=1):
            sentence_dir = os.path.join(output_base, f"sentence_{s_idx:02d}")
            os.makedirs(sentence_dir, exist_ok=True)

            print(f"\n🔵 Generating voices for Sentence {s_idx}: {sentence}")

            for sp_idx, speaker in enumerate(self.speakers[:limit_speakers]):
                print(f"    [{sp_idx+1}/{limit_speakers}] Speaker: {speaker}")

                wav = self.tts.tts(sentence, speaker=speaker)

                output_path = os.path.join(sentence_dir, f"{speaker}.wav")
                sf.write(output_path, wav, samplerate=self.config["dataset"]["sampling_rate"])

        print(f"\n Done generating {limit_sentences} folders × {limit_speakers} voices each!")

    def synthesize_single(self, text, speaker, out_path="temp.wav"):
        wav = self.tts.tts(text, speaker=speaker)
        sf.write(out_path, wav, samplerate=self.config["dataset"]["sampling_rate"])
        self._plot_waveform_and_spectrogram(wav, out_path)
        return out_path, wav

    def _plot_waveform_and_spectrogram(self, wav, save_path):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(wav)
        axs[0].set_title("Waveform")

        axs[1].specgram(wav, Fs=self.config["dataset"]["sampling_rate"])
        axs[1].set_title("Spectrogram")

        fig.tight_layout()
        plot_path = save_path.replace('.wav', '_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Saved waveform and spectrogram plot at: {plot_path}")
