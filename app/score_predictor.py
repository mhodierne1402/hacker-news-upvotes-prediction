import torch

def score_predictor(title, score_predictor_model):
    # Tokenize the title using the sentencepiece tokenizer
    sp = spm.SentencePieceProcessor(model_file='/root/MLXWeek2_copy/weights/skipgram_TechCrunch/techcrunch_sp.model')
    title_tokens = sp.encode_as_pieces(title)
    title_ids = [sp.piece_to_id(token) for token in title_tokens]

    # Convert to tensor and add batch dimension
    title_tensor = torch.tensor(title_ids).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():  # Ensure no gradients are computed
        score = score_predictor_model(title_tensor)

    return score.item()