import google.generativeai as genai
import os
from dotenv import load_dotenv

def generate_answer_from_prompt(prompt: str, model_name: str = "models/gemini-2.5-flash") -> str:
    """
    Generates text from a given prompt using a specified Google generative model.

    Args:
        prompt (str): The text prompt to send to the model.
        model_name (str): The name of the generative model to use.
                          Defaults to "models/gemini-2.5-flash".

    Returns:
        str: The generated text response. Returns None if an error occurs.
    """
    # Load environment variables from the .env file
    load_dotenv()
    
    # Configure the API key from the environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)

    try:
       # Tweak the prompt with system instruction for a concise, direct answer
        response = genai.GenerativeModel(model_name).generate_content(
            f"You are a concise, helpful assistant. Using ONLY the provided context, answer the user's question. If the context does not contain the answer, state that you cannot answer based on the provided information.\n\nContext:\n{prompt}",
        )
        return response.text
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # Make sure your .env file has GOOGLE_API_KEY="YOUR_API_KEY"
    
    user_prompt = "What are the key benefits of using a text embedding model like text-embedding-004?"
    
    print("Generating response...")
    generated_answer = generate_answer_from_prompt(user_prompt)
    
    if generated_answer:
        print("\nGenerated Answer:")
        print(generated_answer)
    else:
        print("\nFailed to generate a response.")