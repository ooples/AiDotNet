namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for CosyVoice2.</summary>
public class CosyVoice2Options : CodecTtsOptions { public CosyVoice2Options() { SampleRate = 22050; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 14; } }
