from chonkie import SemanticChunker
def chunkie(text: str):
    chunker = SemanticChunker(
        max_chunk_size=200,
        overlap=50
    )
    return chunker.split(text)
