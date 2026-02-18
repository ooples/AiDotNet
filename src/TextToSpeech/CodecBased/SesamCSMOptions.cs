namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for SesamCSM.</summary>
public class SesamCSMOptions : CodecTtsOptions { public SesamCSMOptions() { SampleRate = 24000; NumCodebooks = 32; CodebookSize = 2048; CodecFrameRate = 12; LLMDim = 2048; NumLLMLayers = 24; } }
