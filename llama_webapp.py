#SANTILLO NICOLA
#LUMSA University, 24.12.2023.


import gradio as gr
from llama_cpp import Llama

# GGUF model downloaded from HuggingFace 
model_path = "./llama-2-7b-chat.Q2_K.gguf"

# Llama model creation
model = Llama(model_path=model_path, chat_format= "llama-2")

SYSTEM_PROMPT = """<s>[INST] <<SYS>> This is an app to test LLama. Please don't share false information. <</SYS>>"""

# Model parameters
max_tokens = 100

#PARAMETERS = {
#   "temperature": 0.9,
#   "top_p": 0.95,
#   "repetition_penalty": 1.2,
#   "top_k": 50,
#   "top_k": 50,
#   "top_k": 50,
#   "top_k": 50,
#   "truncate": 1000,
#   "max_new_tokens": 1024,
#   "seed": 42,
#   "stop_sequences": ["</s>"],
#}

# Initializing history list 

history = []

# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    memory_limit= memory_limit -1
    # Keep len(history) <= memory_limit
    print("Lenght History : ",len(history))
    print("Memory Limit : ",memory_limit)
    
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message


# Extracting text after last occurrence of [/INST]
def extract_substring(input_string):
    # Find the index of "[/INST]"
    index = input_string.rfind("[/INST]")
    
    # Check if the substring is found
    if index != -1:
        # Extract the substring after "[/INST]"
        result = input_string[index + len("[/INST]"):].strip()
        return result
    else:
        # Return a message indicating that the substring was not found
        return "Substring not found in the input string"


# Returns response from Llama starting from current message and history
def get_llama_response(message: str, history: list) -> str:
   
    query = format_message(message, history)
   
    print("Formatted Input Message: ",query)

    output = model(query, max_tokens=max_tokens, echo=True)
    output_txt= output['choices'][0]['text']
    output_last= extract_substring (output_txt)

    print("Llama Output Dictionary:",output)
    print("Text Element in Dictionary:", output_txt)
    print("Last Answer from Llama:", output_last)
    
    return output_last

# Gradio chat interface
iface = gr.ChatInterface(get_llama_response).queue()

iface.launch(server_name= "0.0.0.0")