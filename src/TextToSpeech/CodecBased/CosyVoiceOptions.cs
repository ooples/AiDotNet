namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for CosyVoice.</summary>
public class CosyVoiceOptions : CodecTtsOptions { public CosyVoiceOptions() { SampleRate = 22050; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 14; } }
