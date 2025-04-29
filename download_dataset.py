from TTS.utils.downloaders import download_ljspeech

def download_ljspeech_dataset(output_dir="data/"):
    download_ljspeech(output_dir)
    print(f"LJSpeech downloaded to {output_dir}")
