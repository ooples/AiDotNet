namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for OuteTTS.</summary>
public class OuteTTSOptions : CodecTtsOptions { public OuteTTSOptions() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 75; LLMDim = 1024; NumLLMLayers = 12; } }
