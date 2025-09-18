from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingUse:
    def __init__(self):
        pass

    def create_embeddings(self, embedding_model, cuda_device):
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': cuda_device}
        )
        return embeddings
