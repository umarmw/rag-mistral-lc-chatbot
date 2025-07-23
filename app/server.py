from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_chain import get_retriever
from models.local_llm import generate_response, MAX_DIALOG_TOKENS

app = FastAPI()
retriever = get_retriever()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    history: list  # List of {"role": "user"|"assistant", "content": str}

# def trim_prompt(prompt: str):
#     tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
#     if len(tokens) > MAX_DIALOG_TOKENS:
#         tokens = tokens[-MAX_DIALOG_TOKENS:]
#     return tokenizer.decode(tokens, skip_special_tokens=True)

@app.post("/ask")
def ask_question(query: Query):
    user_question = query.question
    history = query.history

    retrieved = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    prompt = "[CONTEXT]\n" + context + "\n\n"
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        prompt += f"{role}: {turn['content']}\n"
    prompt += f"User: {user_question}\nAssistant: "

    # prompt = trim_prompt(prompt)
    answer = generate_response(prompt)
    return {"answer": answer}
