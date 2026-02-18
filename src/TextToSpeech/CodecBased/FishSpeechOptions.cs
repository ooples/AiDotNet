namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for FishSpeech.</summary>
public class FishSpeechOptions : CodecTtsOptions { public FishSpeechOptions() { SampleRate = 44100; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 42; LLMDim = 1024; NumLLMLayers = 24; LanguageModelName = "LLaMA"; } }
