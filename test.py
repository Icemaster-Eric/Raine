# really mid testing code
def test_llm():
    from llm import Llama, prompts

    llm_model = Llama("models/gemma-2-9b-it-8bpw")

    prompt = prompts.Llama3([
        {"role": "system", "content": "You are a helpful AI assistant."}
    ])

    llm_model.generate(prompt)

    return True


def test_tts():
    from tts import TTS

    return True


def test_waifumem():
    from waifumem import WaifuMem

    return True


if __name__ == "__main__":
    assert test_llm()
    assert test_tts()
    assert test_waifumem()
