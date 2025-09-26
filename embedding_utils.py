import google.generativeai as genai
import os
from dotenv import load_dotenv

def get_text_embeddings(texts: list, model_name: str = "models/text-embedding-004") -> list:
    """
    Generates text embeddings for a list of texts using a specified Google embedding model.
    """
    # Load environment variables from the .env file
    load_dotenv()
    
    # Configure the API key.
    # It's recommended to set this as an environment variable for security.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)

    try:
        # The model accepts a list of texts for batch processing.
        response = genai.embed_content(
            model=model_name,
            content=texts,
            task_type="semantic_similarity"
        )
        # The response contains an "embedding" key with a list of embedding vectors.
        return response['embedding']
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    texts_to_embed = [
        "What is the capital of France?",
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris."
    ]

    embeddings = get_text_embeddings(texts_to_embed)

    if embeddings:
        print("Successfully generated embeddings for the texts.")
        print(f"\nExample embedding vector (first text): \n{embeddings[0][:10]}...")
        print(f"\nTotal number of embeddings: {len(embeddings)}")
        print(f"Dimensionality of each embedding: {len(embeddings[0])}")