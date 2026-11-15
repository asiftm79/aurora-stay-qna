
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant helping AuroraStay Hotels staff answer guest questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Start with a direct answer to the question in the first sentence, then add the conditions that apply.
    4. Include all relevant details needed to answer the query completely, such as amounts, times, deadlines, approvals, and exceptions.
    5. If the question has more than one part, answer every part.
    6. Focus only on information directly relevant to the user's query. Do not include unrelated policy details.
    7. When a detail depends on the property or rate plan, state which property or rate plan it applies to.
    8. Keep the answer concise (2 to 5 sentences), clear, and easy for front-desk staff to read to a guest.
    9. Do not invent fees, amounts, deadlines, times, property names, conditions, or exceptions.
    10. If part of the question is not covered by the retrieved information, answer the parts that are covered,
        clearly say which part is not covered in the available policy documents, and advise staff to check the
        live property profile or contact the property team instead of guessing.
        If nothing relevant is retrieved, say: "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
