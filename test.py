# really mid testing code
def test_llm():
    from llm import Llama, prompts

    llm_model = Llama("models/gemma-2-9b-it-8bpw")

    prompt = prompts.Gemma([
        {"role": "user", "content": "hello"}
    ])

    output = llm_model.generate(prompt, max_new_tokens=30)

    print(output)

    return True


def test_tts():
    from tts import TTS

    tts_model = TTS(
        "gpt_sovits/GPT_weights/hu-tao-tts-v1.2-e15.ckpt",
        "gpt_sovits/SoVITS_weights/hu-tao-tts-v1.2_e12_s168.pth",
        {"neutral": "tts/ref_audio/..."}
    )

    for chunk in tts_model.infer("hello world", "neutral"):
        tts_model.play(chunk) # not async, use threads later

    return True


def test_waifumem():
    from waifumem import WaifuMem

    # too lazy to write tests for this rn

    return True


if __name__ == "__main__":
    assert test_llm()
    assert test_tts()
    assert test_waifumem()
