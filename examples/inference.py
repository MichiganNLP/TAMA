import torch
from transformers import pipeline

# Initialize the pipeline
pipe = pipeline(
    task="text-generation",
    model="MichiganNLP/tama-5e-7",
    torch_dtype=torch.float16,
    device=0
)

def generate_with_template(instruction, input_text, question):
    # Format the prompt using your template
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Question:
{question}

### Response:
"""

    # Generate text with the formatted prompt
    result = pipe(prompt, 
                  max_new_tokens=512,
                  do_sample=True,
                  temperature=0.7,
                  top_p=0.9,
                  return_full_text=False)  # Set to True if you want to include the prompt in the output
    
    # Extract generated text
    generated_text = result[0]["generated_text"]
    return generated_text

# Example usage
instruction = "You are an expert botanist. Explain biological processes in plants in detail."
input_text = "Plants have various mechanisms to create and store energy."
question = "How do plants create energy and what is this process called?"

response = generate_with_template(instruction, input_text, question)
print(response)