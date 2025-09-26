def chunk_text(texts, chunk_size=10):
    """
    Split one or more texts into chunks of ~chunk_size words.
    texts: str or list of str
    Returns a list of chunks.
    """
    if isinstance(texts, str):
        texts = [texts]

    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    return chunks
