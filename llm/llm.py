from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob
from llm import prompts


class Llama:
    def __init__(self, model_dir: str):
        self.config = ExLlamaV2Config(model_dir)
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache_Q4(self.model, max_seq_len=65536, lazy=True)
        self.model.load_autosplit(self.cache, progress = True)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

    def generate(
            self,
            prompt: prompts.Prompt | str,
            max_new_tokens: int = 300,
            settings: ExLlamaV2Sampler.Settings = ExLlamaV2Sampler.Settings(
                temperature = 0.95, # Sampler temperature (1 to disable)
                top_k = 50, # Sampler top-K (0 to disable)
                top_p = 0.8, # Sampler top-P (0 to disable)
                top_a = 0.0, # Sampler top-A (0 to disable)
                typical = 0.0, # Sampler typical threshold (0 to disable)
                skew = 0.0, # Skew sampling (0 to disable)
                token_repetition_penalty = 1.01, # Sampler repetition penalty (1 to disable)
                token_frequency_penalty = 0.0, # Sampler frequency penalty (0 to disable)
                token_presence_penalty = 0.0, # Sampler presence penalty (0 to disable)
                smoothing_factor = 0.0, # Smoothing Factor (0 to disable)
            ),
    ):
        if isinstance(prompt, prompts.Prompt):
            return self.generator.generate(
                prompt.prompt,
                max_new_tokens=max_new_tokens,
                gen_settings=settings,
                encode_special_tokens=prompt.encode_special_tokens,
                add_bos=prompt.add_bos,
            )
        else:
            return self.generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                gen_settings=settings,
            )

    def generate_stream(
            self,
            prompt: prompts.Prompt | str,
            max_new_tokens: int = 300,
            settings: ExLlamaV2Sampler.Settings = ExLlamaV2Sampler.Settings(
                temperature = 0.95, # Sampler temperature (1 to disable)
                top_k = 50, # Sampler top-K (0 to disable)
                top_p = 0.8, # Sampler top-P (0 to disable)
                top_a = 0.0, # Sampler top-A (0 to disable)
                typical = 0.0, # Sampler typical threshold (0 to disable)
                skew = 0.0, # Skew sampling (0 to disable)
                token_repetition_penalty = 1.01, # Sampler repetition penalty (1 to disable)
                token_frequency_penalty = 0.0, # Sampler frequency penalty (0 to disable)
                token_presence_penalty = 0.0, # Sampler presence penalty (0 to disable)
                smoothing_factor = 0.0, # Smoothing Factor (0 to disable)
            ),
    ):
        if isinstance(prompt, prompts.Prompt):
            job = ExLlamaV2DynamicJob(
                input_ids=self.tokenizer.encode(
                    prompt.prompt,
                    add_bos=prompt.add_bos,
                    add_eos=prompt.add_eos,
                    encode_special_tokens=prompt.encode_special_tokens
                ),
                max_new_tokens=max_new_tokens,
                gen_settings=settings,
                stop_conditions=prompt.stop_conditions,
                identifier=prompt, # probably incorrect, fix later
            )
        else:
            job = ExLlamaV2DynamicJob(
                input_ids=self.tokenizer.encode(prompt),
                max_new_tokens=max_new_tokens,
                gen_settings=settings,
                identifier=prompt, # probably incorrect, fix later
            )
        self.generator.enqueue(job)

        while self.generator.num_remaining_jobs():
            results = self.generator.iterate()

            for result in results:
                if result["identifier"] != prompt:
                    continue

                yield result.get("text", "")


if __name__ == "__main__":
    llm = Llama("waifumem/models/llama-3.1-8b-instruct-exl2")

    prompt = prompts.Llama3([
        {"role": "system", "content": "You are Raine, a AI vtuber with a kuudere personality. You are a shy girl who doesn't like to talk very much. However, you still make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine. You are talking to Eric."},
        {"role": "user", "content": "Hey Raine, it's me, your creator. This will probably be the first message that you'll ever remember... I just finished the first version of waifumem, the memory module you're using right now. How are you feeling?"},
    ])

    for token in llm.generate_stream(prompt, 100):
        print(token, end="")
    print()
