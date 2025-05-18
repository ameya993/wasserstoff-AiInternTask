from openai import OpenAI
import os
import traceback

# Example context and question for testing
context = """
[Page 1, Paragraph 1] Hydrogen is a clean fuel that, when consumed in a fuel cell, produces only water.
[Page 2, Paragraph 2] Hydrogen can be produced from a variety of resources, such as natural gas, nuclear power, biomass, and renewable power like solar and wind.
"""
query = "What is hydrogen and how can it be produced?"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

try:
    print("[DEBUG] Calling Groq Llama 3 API...")
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always cite page and paragraph numbers from the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely and cite page and paragraph numbers where relevant."}
        ],
        max_tokens=300,
        temperature=0.2,
    )
    print("[DEBUG] Raw Groq API response:", completion)
    answer = completion.choices[0].message.content.strip()
    print("[DEBUG] Synthesized answer:", answer)
except Exception as e:
    print(f"[ERROR] Groq Llama 3 API call failed: {e}")
    traceback.print_exc()
    answer = "Sorry, the Llama 3 API call failed. Please check your Groq API key and try again."
