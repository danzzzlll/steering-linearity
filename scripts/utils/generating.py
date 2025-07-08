from transformers import AutoModelForCausalLM

def generate(
        multipliers: tuple, 
        llm: AutoModelForCausalLM, 
        tokenizer, 
        prompt,
        context_manager_steering,
        max_new_tokens,
        temperature,
        layers
    ):

        answers = {}
        for multiplier in multipliers:
            with context_manager_steering.apply(llm, multiplier=multiplier, layers=[14]):
                input_tensor = tokenizer.encode(prompt, return_tensors="pt")
                outputs = llm.generate(
                    input_tensor.to(llm.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )

                result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
                answers[multiplier] = result
                print("="* 50)
                print(f"Multiplier: {multiplier}")
                print(f"steered model: {result}")
        
        return answers