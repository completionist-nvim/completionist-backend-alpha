from fastapi import FastAPI
from pydantic import BaseModel
from exllamav2_wrapper import ExllamaWrapper
from typing import List
import uvicorn
import os

class CompletionRequest(BaseModel):
    id: str
    context: str

class ChatRequest(BaseModel):
    id: str
    """
    The chat history.
    ```json
    {
        "history": [
            {
                "role": "user",
                "text": "How do I echo something to stdout in python?"
            },
            {
                "role": "assistant",
                "text": "You can use the `print` function."
            },
            {
                 "role": "user",
                 "text": "How do I echo something to stderr in python?"
             }
        ]
    }
    ```
    """
    history: List[dict]

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.post("/alpha/complete")
async def post_completion(completion_request: CompletionRequest):
    completion, generated_tokens, time = wrapper.complete(completion_request.context)
    return {
            "id": completion_request.id,
            "completion": completion,
            "generated_tokens": generated_tokens,
            "time": time
            }

@app.post("/alpha/chat")
async def post_chat(chat_request: ChatRequest):
    reply, generated_tokens, time = wrapper.chat(chat_request.history)
    return {
           "id": chat_request.id,
           "assistant_reply": reply,
           "generated_tokens": generated_tokens,
           "time": time
           }

if __name__ == "__main__":
    model = os.environ.get("MODEL")
    if model is None:
        raise Exception("Please set the MODEL environment variable to the path of the model")
    wrapper = ExllamaWrapper(model)
    uvicorn.run(app, host="0.0.0.0", port=8000)
