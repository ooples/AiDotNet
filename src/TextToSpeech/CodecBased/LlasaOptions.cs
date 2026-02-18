namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Llasa.</summary>
public class LlasaOptions : CodecTtsOptions { public LlasaOptions() { SampleRate = 16000; NumCodebooks = 1; CodebookSize = 8192; CodecFrameRate = 50; LLMDim = 2048; NumLLMLayers = 24; LanguageModelName = "LLaMA"; } }
