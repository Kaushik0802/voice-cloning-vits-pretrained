# Voice Cloning with VITS – Multi-Speaker Batch Synthesis

This project demonstrates real-time and batch voice cloning using the Coqui TTS library's pretrained **VITS model** trained on the **VCTK dataset**. It allows generation of personalized synthetic voices across 109 unique speaker identities, with two usage modes: **interactive single-sentence cloning** and **bulk generation of 20 sentences × 100 speakers**.

---

## Project Overview
- **Goal**: Generate diverse synthetic voices for a given text using pretrained speaker embeddings.
- **Model**: `tts_models/en/vctk/vits` (multi-speaker VITS model from Coqui TTS)
- **Dataset Used**: **VCTK** (not LJ Speech)
  - Includes over 100 British English speakers with distinct identities (male/female)

Note: While LJ Speech is a single-speaker dataset, VCTK allows simulation of multiple personalized voices.

---

## Features
- Clone any sentence into 109 different voices
- Batch mode: generate 100 voices for 20 predefined sentences
- Gradio UI with two tabs:
  - Single Sentence Mode: Choose a speaker and synthesize live
  - Batch Mode: Bulk synthesize and save outputs to folders
- Outputs structured by sentence folders
- Spectrogram and waveform plots for each generated audio file

---

## Project Structure
```
├── data/               # Input/output audio
│   ├── raw/            # Optional raw inputs
│   └── generated/      # Cloned voices structured by sentence folders
│
├── src/                # Core source code
│   ├── inference_engine.py
│   ├── prepare_dataset.py
│   └── run_inference.py
│
├── config.yaml         # Model and inference parameters
├── requirements.txt    # Python dependencies
├── README.md           # Project summary and structure
└── voice_cloning_vits.ipynb  # Optional Colab interface
```

---

## Installation (Colab)
```bash
pip install -r requirements.txt
apt-get install -y espeak
```

---

## Run the App
```bash
python run_inference.py --config config.yaml
```
This launches a Gradio interface with:
- Text input and speaker dropdown
- Audio playback and download
- Option to trigger 20 × 100 voice synthesis batch

---

## Output Structure
```
outputs/
├── generated_sentences/
│   ├── sentence_01/
│   │   ├── p225.wav
│   │   ├── p226.wav
│   │   └── ...
│   ├── sentence_02/
│   └── ...
```

---

## Why VCTK (not LJ Speech)?
- LJ Speech is single speaker and better suited for fine-tuning
- VCTK is multi-speaker and ideal for fast cloning of different voices

This project generates personalized synthetic voices by leveraging these built-in speaker embeddings.

Note: To truly clone someone's voice (for example, from 10 real samples), you would need fine-tuning which is beyond the scope of this demo.

---

## References and Resources
- [Coqui TTS Documentation](https://tts.readthedocs.io/)
- [VCTK Corpus Info](https://datashare.ed.ac.uk/handle/10283/2651)
- [Colab Notebook](#) <!-- add link here -->
- [GitHub Repo](#) <!-- add link here -->

---

## Acknowledgements
- Coqui AI team for their open-source TTS models
- VCTK dataset contributors

---

## Example Use Cases
- Assistive voice technologies
- Multilingual virtual agents
- Accessibility and narration demos

---

## Future Extensions
- Upload your own samples and fine-tune (e.g., with YourTTS)
- Speaker adaptation from user data
- Real-time TTS APIs or chatbot integration

