namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Zonos.</summary>
public class ZonosOptions : CodecTtsOptions { public ZonosOptions() { SampleRate = 44100; NumCodebooks = 9; CodebookSize = 1024; CodecFrameRate = 86; LLMDim = 1024; NumLLMLayers = 16; } }
