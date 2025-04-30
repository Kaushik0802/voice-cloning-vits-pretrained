import argparse
import yaml
import gradio as gr
from inference_engine import InferenceEngine

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    engine = InferenceEngine(config)

    def synthesize_single_gradio(text, speaker):
        out_path, wav = engine.synthesize_single(text, speaker, out_path="gradio_output.wav")
        return out_path

    def run_batch():
        engine.synthesize_all(limit_speakers=100, limit_sentences=20)
        return "Batch synthesis complete. Check your output folder."

    speaker_dropdown = gr.Dropdown(choices=engine.speakers, label="Select Speaker")
    text_input = gr.Textbox(label="Enter text to synthesize", lines=2, value="The quick brown fox jumps over the lazy dog.")

    single_tab = gr.Interface(
        fn=synthesize_single_gradio,
        inputs=[text_input, speaker_dropdown],
        outputs=gr.Audio(type="filepath", label="Synthesized Audio"),
        allow_flagging="never",
        title="Single Sentence Voice Cloning",
        description="Enter a sentence and select a speaker to hear the cloned voice."
    )

    batch_tab = gr.Interface(
        fn=run_batch,
        inputs=[],
        outputs="text",
        allow_flagging="never",
        title="Batch Synthesis",
        description="Generate voices for 20 sentences across 100 speakers."
    )

    gr.TabbedInterface([single_tab, batch_tab], tab_names=["Single Speaker Mode", "Batch Mode"]).launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio app for multispeaker voice cloning.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
