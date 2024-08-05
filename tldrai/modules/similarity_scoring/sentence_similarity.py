import logging

from sentence_transformers import SentenceTransformer, util

from tldrai.modules.utils.gpu import check_gpu_memory
from tldrai.modules.utils.logging import configure_logging


class SentenceSimilarity:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        verbose=False,
    ):
        """
        Initializes the SentenceSimilarity class with a specified transformer model.

        Args:
            model_name (str): The name of the sentence transformer model to be used.
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        self.verbose = verbose
        configure_logging(logging.DEBUG if self.verbose else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_memory_used = 0.5  # GB
        free_memory = check_gpu_memory()

        # None lets it run another internal check
        device = None if free_memory >= self.model_memory_used else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode_sentences(self, sentences):
        """
        Encodes a list of sentences into embeddings using the loaded model.

        Args:
            sentences (List[str]): A list of sentences to encode.

        Returns:
            np.ndarray: The embeddings of the provided sentences.
        """
        return self.model.encode(sentences)

    def compare_base_to_others(self, base_sentence, other_sentences):
        """
        Compares a base sentence to a list of other sentences to find their similarity scores.

        Args:
            base_sentence (str): The base sentence to compare against.
            other_sentences (List[str]): A list of other sentences to compare to the base sentence.

        Returns:
            np.ndarray: A 2D numpy array of similarity scores between the base sentence and each of the other sentences.
        """
        # Encode the base sentence and other sentences
        base_embedding = self.encode_sentences([base_sentence])
        other_embeddings = self.encode_sentences(other_sentences)

        # Calculate and return the cosine similarity scores
        cosine_scores = util.cos_sim(base_embedding, other_embeddings)
        return cosine_scores
