namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for MaskGCT.</summary>
public class MaskGCTOptions : CodecTtsOptions { public MaskGCTOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 16; } }
