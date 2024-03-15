import torch
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from functools import partial
from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

# Constants (you can modify these if needed)
BATCH_SIZE = 126
SHUFFLE = True
MODEL_NAME = 'skipgram'

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file='techcrunch_sp.model')

# Implement a tokenizer function using SentencePiece
def sp_tokenize(text):
    return sp.encode(text, out_type=int)

def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


# Create a dataset from your text file
class TextFileDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

# Initialize your dataset
dataset = TextFileDataset('/root/MLXWeek2_copy/merged_full.txt')

# We don't need to build vocab since SentencePiece handles it
# But we need a text processing pipeline which converts text to integer IDs
text_pipeline = lambda x: sp.encode(x, out_type=int)

# Adjust the collate functions if necessary
# Depending on your implementation details you might want to customize the collate functions

# Use your dataset and collate function to create a DataLoader

def sentencepiece_dataloader(file_path, batch_size, shuffle, sp_model_path, collate_fn_name):
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    # Initialize your dataset
    dataset = TextFileDataset(file_path)

    # We don't need to build vocab since SentencePiece handles it
    # But we need a text processing pipeline which converts text to integer IDs
    text_pipeline = lambda x: sp.encode(x, out_type=int)

    # Define collate functions
    if collate_fn_name == 'cbow':
        collate_fn = collate_cbow
    elif collate_fn_name == 'skipgram':
        collate_fn = collate_skipgram
    else:
        raise ValueError("Collate function name must be 'cbow' or 'skipgram'")

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline)
    )

    return dataloader