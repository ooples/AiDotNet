namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for VALLEX.</summary>
public class VALLEXOptions : CodecTtsOptions { public VALLEXOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 75; LLMDim = 1024; NumLLMLayers = 12; } }
