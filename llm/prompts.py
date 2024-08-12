from typing import Literal


class Prompt:
    prompt: str
    add_bos: bool
    add_eos: bool
    encode_special_tokens: bool
    stop_conditions: list


class Llama3(Prompt):
    """Llama 3 prompt format

    Supported roles: `system`, `user`, `assistant`
    """
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.add_bos = False
        self.add_eos = False
        self.encode_special_tokens = True
        self.stop_conditions = [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|eot_id|>"),
            tokenizer.single_id("<|start_header_id|>")
        ]

        if messages[0]["role"] not in {"system", "user"}:
            raise ValueError("The first message must have the role 'system' or 'user'.")

        self.prompt = "<|begin_of_text|>\n"

        for i, message in enumerate(messages):
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role '{message['role']}' for Llama 3 prompts.")

            if i:
                if message["role"] == messages[i - 1]["role"]:
                    raise ValueError("Consecutive messages cannot have the same roles.")

                elif message["role"] == "system":
                    raise ValueError("The system role message must be the first.")

                elif message["role"] == "assistant":
                    if i == 1 and messages[0]["role"] == "system":
                        raise ValueError("Messages with the 'system' role must be followed by a message with the 'user' role.")

            self.prompt += (
                    f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                    f"{message['content']}<|eot_id|>\n"
                )

            self.prompt += "<|start_header_id|>assistant<|end_header_id|>"


class MistralInstruct(Prompt):
    """Mistral-Instruct prompt format

    Supported roles: `system`, `user`, `assistant`
    """
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.add_bos = True
        self.add_eos = False
        self.encode_special_tokens = False
        self.stop_conditions = [
            tokenizer.eos_token_id,
            tokenizer.single_id("</s>"), # not sure about this
        ]

        if messages[0]["role"] not in {"system", "user"}:
            raise ValueError("The first message must have the role 'system' or 'user'.")

        self.prompt = "[INST] " if len(messages) == 1 else "<s>[INST] "

        for i, message in enumerate(messages):
            if i and message["role"] == messages[i - 1]["role"]:
                raise ValueError("Consecutive messages cannot have the same roles.")

            if message["role"] == "system":
                if i:
                    raise ValueError("The system role message must be the first message.")

                self.prompt += f"{message['content']}\n\n"

            elif message["role"] == "user":
                if i == 1 and messages[0]["role"] != "system":
                    raise ValueError("Messages with the 'system' role must be followed by a message with the 'user' role.")

                self.prompt += f"{message['content']} [/INST]"

            elif message["role"] == "assistant":
                self.prompt += f" {message['content']} [INST] "

            else:
                raise ValueError(f"Unsupported role '{message['role']}' for mistral instruct prompts.")


class Gemma(Prompt):
    """Gemma prompt format (should work for Gemma 2)

    Supported roles: `user`, `model`
    """
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.add_bos = False
        self.add_eos = False
        self.encode_special_tokens = True
        self.stop_conditions = [
            tokenizer.eos_token_id,
            "</s>",
            "<end_of_turn>",
        ]

        if messages[0]["role"] != "user":
            raise ValueError("The first message must have the role 'user'.")

        self.prompt = ""

        for i, message in enumerate(messages):
            if message["role"] not in {"user", "model"}:
                raise ValueError(f"Unsupported role '{message['role']}' for Gemma prompts.")

            if i and message["role"] == messages[i - 1]["role"]:
                raise ValueError("Message roles must alternate between 'user' and 'model'.")

            if message["role"] == "user":
                self.prompt += "<bos>"

            self.prompt += (
                f"<start_of_turn>{message['role']}\n"
                f"{message['content']}<end_of_turn>\n"
            )

        self.prompt += "<start_of_turn>model\n"


if __name__ == "__main__":
    print(Gemma([
        {"role": "user", "content": "Question 1"},
        {"role": "model", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]).prompt)
