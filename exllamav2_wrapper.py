from typing import List, Tuple, Callable
import time

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)

from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator
)

class ExllamaWrapper:

    endings: List[str | int] = [
        "\n\n",
        "\n\n  ",
        "\n\n    ",
        "\n\n      ",
        "\n\n        ",
        "\n\n          ",
        "\n\n            ",
        "\n\n              ",
        "\n\n\t",
        "\n\n\t\t",
        "\n\n\t\t\t",
        "\n\n\t\t\t\t",
        "\n\n\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t\t",
        "###Instruction:"
    ]

    CHAT_MAX_NEW_TOKENS: int = 512
    COMPLETE_MAX_NEW_TOKENS: int = 128

    path: str
    model: ExLlamaV2
    config: ExLlamaV2Config
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer
    generator: ExLlamaV2StreamingGenerator

    def __init__(self, path: str):
        self.path = path
        self.__prepare_config()
        self.__prepare_model()
        self.__prepare_cache()
        self.__load_model()
        self.__prepare_tokenizer()
        self.__prepare_generator()
        self.__change_settings()

    def __prepare_config(self):
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.path
        self.config.prepare()

    def __prepare_model(self):
        self.model = ExLlamaV2(self.config)

    def __prepare_cache(self):
        self.cache = ExLlamaV2Cache(self.model, lazy = True)
        self.cache.max_seq_len = 2048

    def __load_model(self):
        self.model.load_autosplit(self.cache)

    def __prepare_tokenizer(self):
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.endings.append(self.tokenizer.eos_token_id)

    def __prepare_generator(self):
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.generator.warmup()

    def __change_settings(self):
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.1
        self.settings.top_k = 50
        self.settings.top_p = 0.95
        self.settings.token_repetition_penalty = 1.05
        self.settings.token_frequency_penalty = 0.0

    def __generate_completion_prompt(self, prompt) -> str:
        prompt = prompt.replace("<PRE>", "<｜fim▁begin｜>") \
                       .replace("<CUR>", "<｜fim▁hole｜>") \
                       .replace("<END>", "<｜fim▁end｜>")
        return self.tokenizer.bos_token + prompt

    PRE_PROMPT = "You are an AI programming assistant. The following is a conversation between you (###Response) and the user (###Instruction). You can use `markdown` formatting for your response if neccesary If the instruction asks how to do something include code snippets in backticks as examples.\n"

    def __generate_chat_prompt(self, history: List[dict]) -> Tuple[str, float]:
        """
        Generates the next prompt for a chat back-to-front.
        """
        time_begin = time.time()
        prompt = ""
        for i in range(len(history)).__reversed__():
            role_text = history[i]
            text = role_text["text"]
            role = role_text["role"]
            reply = role.replace("user", "###Instruction:\n") \
                         .replace("assistant", "###Response:\n") \
                        + text + "\n"
            new_prompt = reply + prompt
            new_prompt_tokens = len(self.tokenizer.encode(prompt)[0]) + len(self.tokenizer.encode("###Response:\n")[0])
            if new_prompt_tokens < self.CHAT_MAX_NEW_TOKENS + self.cache.max_seq_len:
                prompt = new_prompt
            else:
                break
        prompt = self.PRE_PROMPT + prompt + "###Response:\n"
        total_time = time.time() - time_begin
        return prompt, total_time

    def chat(self, history: List[dict]) -> Tuple[str, int, float]:
        """
        Generates the next message from the assistant.
        """
        prompt, time = self.__generate_chat_prompt(history)
        reply, tokens, time = self.raw(prompt)
        return reply.strip(), tokens, time + time

    async def stream_chat(self, history: List[dict], streaming_callback: Callable[[str], None], finishing_callback: Callable[[int, float], None]):
        """
        Stream the chat generation.
        """
        await self.stream(self.__generate_chat_prompt(history), streaming_callback, finishing_callback)

    def complete(self, prompt) -> Tuple[str, int, float]:
        """
        Returns a FIM completion.
        """
        return self.raw(self.__generate_completion_prompt(prompt))

    async def stream_complete(self, prompt, streaming_callback: Callable[[str], None], finishing_callback: Callable[[int, float], None]):
        """
        Stream the FIM completion generator.
        """
        await self.stream(self.__generate_completion_prompt(prompt), streaming_callback, finishing_callback)

    async def stream(self, prompt, streaming_callback: Callable[[str], None], finishing_callback: Callable[[int, float], None]):
        """
        Generate a raw streamed completion.

        Args:
            prompt (str): The prompt to base the completion on.

        Callbacks:
            streaming_callback (Callable[[str], None]): A callback to call when a new token is generated. The first argument is the token generated.
            finishing_callback (Callable[[int, float], None]): A callback to call when the generator is finished. The first argument is the number of tokens generated, the second is the time taken to generate the tokens.
        """
        if streaming_callback is None:
            raise ValueError("Streaming callback is required")
        if finishing_callback is None:
            raise ValueError("Finishing callback is required")

        input_ids, position_offsets = self.tokenizer.encode(self.__generate_completion_prompt(prompt), encode_special_tokens = True, return_offsets = True)
        input_mask = self.tokenizer.padding_mask(input_ids)
        self.generator.begin_stream(input_ids, self.settings, input_mask = input_mask, position_offsets = position_offsets)
        self.generator.set_stop_conditions(self.endings)

        time_begin = time.time()
        generated_tokens = 0
        while True:
            result = self.generator.stream()
            chunk = ""
            eos = False
            if len(result) == 3:
                chunk, eos, _ = result
            elif len(result) == 4:
                chunk, eos, _, _ = result
            else:
                raise ValueError("Unexpected result: {}".format(result))
            generated_tokens += 1
            streaming_callback(chunk)
            if eos or generated_tokens >= self.COMPLETE_MAX_NEW_TOKENS:
                time_total = time.time() - time_begin
                finishing_callback(generated_tokens, time_total)
                break

    def raw(self, prompt) -> Tuple[str, int, float]:
        """
        Generate a raw completion.

        Args:
            prompt (str): The prompt to base the completion on.

        Returns:
            Tuple[str, int, float]: The generated completion, the number of tokens generated, and the time taken to generate.
        """
        time_begin = time.time()

        input_ids, position_offsets = self.tokenizer.encode(prompt, encode_special_tokens = True, return_offsets = True)
        input_mask = self.tokenizer.padding_mask(input_ids)
        self.generator.begin_stream(input_ids, self.settings, input_mask = input_mask, position_offsets = position_offsets)
        self.generator.set_stop_conditions(self.endings)

        generated_tokens = 0
        buffer: str = ""
        while True:
            result = self.generator.stream()
            chunk = ""
            eos = False
            if len(result) == 3:
                chunk, eos, _ = result
            elif len(result) == 4:
                chunk, eos, _, _ = result
            else:
                raise ValueError("Unexpected result: {}".format(result))
            generated_tokens += 1
            buffer += chunk
            if eos or generated_tokens == self.COMPLETE_MAX_NEW_TOKENS: break

        time_total = time.time() - time_begin
        return buffer, generated_tokens, time_total
