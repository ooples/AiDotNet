namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for GPTSoVITS.</summary>
public class GPTSoVITSOptions : CodecTtsOptions { public GPTSoVITSOptions() { SampleRate = 32000; NumCodebooks = 1; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 512; NumLLMLayers = 12; } }
