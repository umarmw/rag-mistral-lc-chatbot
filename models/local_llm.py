from llama_cpp import Llama

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_ctx=4096,
    n_threads=8,  # Tune for your CPU
    n_gpu_layers=0,  # Use 0 for CPU only; increase for GPU acceleration
)

MAX_DIALOG_TOKENS = 2048

def generate_response(prompt: str) -> str:
    response = llm(
        prompt,
        max_tokens=512,
        stop=["</s>", "###", "User:", "Assistant:"],
        echo=False,
    )
    return response["choices"][0]["text"].strip()