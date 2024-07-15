import argparse
import re
import logging
from typing import Tuple, Optional, List
import os
import json
from transformers import (
    pipeline,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.history: List[str] = []
        self.max_history = max_history

    def add_to_history(self, entry: str) -> None:
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_full_context(self) -> str:
        return "\n".join(self.history)

    def clear_history(self) -> None:
        self.history.clear()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text generation using a trained model.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory containing trained actor model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=10,
        help="Maximum number of conversation turns to keep in history",
    )
    return parser.parse_args()

def get_generator(path: str) -> pipeline:
    cache_dir = "/scratch/hemanth/LLMs/"
    token = "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"

    try:
        if os.path.exists(path):
            model_json_path = os.path.join(path, "config.json")
            if os.path.exists(model_json_path):
                with open(model_json_path, 'r') as f:
                    model_json = json.load(f)
                model_name = model_json.get("_name_or_path", path)
            else:
                model_name = path
        else:
            model_name = path

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            fast_tokenizer=True,
            cache_dir=cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(path, token=token)
        model_class = AutoModelForCausalLM.from_config(model_config)
        model = model_class.from_pretrained(
            path,
            from_tf=bool(".ckpt" in path),
            config=model_config,
        ).half()

        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        return pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")
    except Exception as e:
        logging.error(f"Error in get_generator: {str(e)}")
        raise


def get_user_input(conversation_manager: ConversationManager) -> Tuple[str, bool, bool]:
    try:
        tmp = input("Human: ")
        new_input = f"Human: {tmp}"
        conversation_manager.add_to_history(new_input)
        return tmp, tmp == "quit", tmp == "clear"
    except EOFError:
        return "", True, False

def get_model_response(generator: pipeline, context: str, max_new_tokens: int) -> Optional[list]:
    try:
        return generator(context, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7)
    except Exception as e:
        logging.error(f"Error in get_model_response: {str(e)}")
        return None

def process_response(response: list, conversation_manager: ConversationManager) -> str:
    if not response:
        return "Error: No response generated."
    
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "").strip()
    
    # Extract only the assistant's response
    assistant_response = output.split("Assistant:")[-1].strip()
    
    conversation_manager.add_to_history(f"Assistant: {assistant_response}")
    
    return assistant_response

def main(args: argparse.Namespace) -> None:
    try:
        generator = get_generator(args.path)
        set_seed(42)

        conversation_manager = ConversationManager(max_history=args.max_history)
        num_rounds = 0

        print("Start chatting with the AI. Type 'quit' to exit or 'clear' to reset the conversation.")
        
        while True:
            num_rounds += 1
            user_input, quit, clear = get_user_input(conversation_manager)

            if quit:
                break
            if clear:
                conversation_manager.clear_history()
                num_rounds = 0
                print("Conversation history cleared.")
                continue

            context = conversation_manager.get_full_context()
            response = get_model_response(generator, context, args.max_new_tokens)
            output = process_response(response, conversation_manager)

            print(f"Assistant: {output}")

    except KeyboardInterrupt:
        print("\nExiting the program.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    args = parse_args()
    main(args)