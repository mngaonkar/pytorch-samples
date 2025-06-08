import torch
import torch.nn as nn
from collections import Counter
import random
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

NUM_NEGATIVES = 5
NUM_EPOCHS = 100
EMBEDDING_DIM = 100 # Dimension of the embedding vectors
WINDOW_SIZE = 2

corpus = "the quick brown fox jumps over the lazy dog".split()
logger.info(f"Corpus: {corpus}")
word_counts = Counter(corpus)
logger.info(f"Word counts: {word_counts}")

vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
logger.info(f"Vocabulary: {vocab}")
logger.info(f"Vocabulary size: {len(vocab)}")

reverse_vocab = sorted(vocab.items(), key=lambda x: x[0], reverse=True)
logger.info(f"Reverse vocabulary: {reverse_vocab}")

def generate_skipgram_pairs(corpus:str, window_size:int=2 ) -> list[tuple]:
    pairs = []

    for i, word in enumerate(corpus):
        if i > len(corpus) - window_size - 1:
            continue
        left_range = i - window_size if i - window_size >=0 else 0
        right_range = i + window_size if i + window_size < len(corpus) else len(corpus) - 1
        for j in range(left_range, right_range + 1):
            i_index = vocab[word]
            j_index = vocab[corpus[j]]
            pairs.append((i_index, j_index)) if i != j else None

    return pairs

skipgram_pairs = generate_skipgram_pairs(corpus, window_size=2)
logger.debug(f"Skip-gram pairs: {skipgram_pairs}")

def get_negative_samples(target: int, context: int, num_negatives:int, vocab_size:int) -> torch.tensor:
    neg_samples = []
    while len(neg_samples) < num_negatives:
        index = random.randint(0, vocab_size - 1)
        if index != context and index != target:
            neg_samples.append(index)

    return torch.tensor(neg_samples, dtype=torch.long)

logger.debug(f"Negative samples for target 0, context 1: {get_negative_samples(0, 1, 5, len(vocab))}")

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target, context):
        target_embeds = self.target_embeddings(target)
        context_embeds = self.context_embeddings(context)
        scores = torch.sum(target_embeds * context_embeds, dim=1)
        return scores

def train_skipgram(corpus, vocab_size, embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES, epochs=NUM_EPOCHS):
    pairs = generate_skipgram_pairs(corpus, window_size)
    model = SkipGram(vocab_size, embedding_dim)
    print(f"Model info: {model}") 

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Loss function with sigmoid activation on logits followed by binary cross-entropy
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        logger.debug(f"pairs: {pairs}")
        for target, context in pairs:
            # Create tensors for target and context
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)

            # Reset gradients
            optimizer.zero_grad()

            # Model forward pass for positive sample
            positive_scores = model(target_tensor, context_tensor)
            postive_labels = torch.ones(positive_scores.size(0))
            logger.debug(f"Scores for positive samples of target {target} and context {context}: {positive_scores}")

            # Mode forward pass for negative samples
            negative_samples = get_negative_samples(target, context, num_negatives, vocab_size)
            negative_scores = model(target_tensor.repeat(num_negatives), negative_samples)
            negative_labels = torch.zeros(len(negative_samples))

            # Combine positive and negative scores/labels
            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([postive_labels, negative_labels])

            # Compute loss and backpropagate
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss every 10 epochs
        if epoch %10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(pairs):.4f}")

    return model

def consine_similarity(vector_a, vector_b):
    return torch.dot(vector_a, vector_b) / (torch.norm(vector_a) * torch.norm(vector_b))

if __name__ == "__main__":
    model = train_skipgram(corpus, len(vocab), embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES, epochs=NUM_EPOCHS)

    # Print learned embeddings
    embeddings = model.target_embeddings.weight.detach().numpy()
    # logger.warning(f"Learned embeddings: {embeddings}")
    print(f"Shape of embeddings: {embeddings.shape}")

    word = "brown"
    print(f"Embedding for the word {word}: {embeddings[vocab[word]]}")

    word_1 = "brown"
    word_2 = "fox"
    similarties = consine_similarity(torch.tensor(embeddings[vocab[word_1]]), torch.tensor(embeddings[vocab[word_2]]))
    print(f"Cosine similarity between {word_1} and {word_2}: {similarties:.4f}")
