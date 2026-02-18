namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for XTTS v2 (GPT-2 based multilingual zero-shot TTS with VQ-VAE audio tokens).</summary>
public class XTTSv2Options : CodecTtsOptions { public XTTSv2Options() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 8192; CodecFrameRate = 22; LLMDim = 1024; NumLLMLayers = 30; LanguageModelName = "GPT-2"; } }
