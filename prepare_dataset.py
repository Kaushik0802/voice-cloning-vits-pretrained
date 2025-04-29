import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_ljspeech_metadata(dataset_path, output_path, val_split=0.1):
    """
    Prepare train/val CSVs for LJSpeech dataset.
    """
    metadata_file = os.path.join(dataset_path, "metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    metadata = pd.read_csv(metadata_file, sep='|', header=None, names=['id', 'text', 'normalized_text'])
    metadata['wav_path'] = metadata['id'].apply(lambda x: os.path.join(dataset_path, "wavs", f"{x}.wav"))
    metadata = metadata[['wav_path', 'text']]

    train_meta, val_meta = train_test_split(metadata, test_size=val_split, random_state=42)

    train_meta.to_csv(os.path.join(output_path, "metadata_train.csv"), index=False)
    val_meta.to_csv(os.path.join(output_path, "metadata_val.csv"), index=False)

    print(f"Prepared metadata: {len(train_meta)} train samples, {len(val_meta)} val samples.")
