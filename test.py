# really mid testing code
def test_llm():
    from llm import Llama, prompts

    llm_model = Llama("models/gemma-2-9b-it-sppo-8bpw")

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


def test_vts():
    import pyvts
    import asyncio

    async def vts_func():
        vts = pyvts.vts()

        await vts.connect()

        # authenticate
        await vts.request_authenticate_token()
        await vts.request_authenticate()

        response_data = await vts.request(vts.vts_request.requestHotKeyList())

        for hotkey in response_data['data']['availableHotkeys']:
            print(hotkey)
            request = vts.vts_request.requestTriggerHotKey(hotkey)
            await vts.request(request)
            break

        await vts.close()

        return True

    return asyncio.run(vts_func())


if __name__ == "__main__":
    #assert test_llm()
    #assert test_tts()
    #assert test_waifumem()
    assert test_vts()
