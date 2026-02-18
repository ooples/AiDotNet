namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Chatterbox.</summary>
public class ChatterboxOptions : CodecTtsOptions { public ChatterboxOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 16; } }
