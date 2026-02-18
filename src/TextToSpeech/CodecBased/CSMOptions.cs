namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for CSM (Conversational Speech Model).</summary>
public class CSMOptions : CodecTtsOptions { public CSMOptions() { SampleRate = 24000; NumCodebooks = 32; CodebookSize = 2048; CodecFrameRate = 12; LLMDim = 2048; NumLLMLayers = 24; } }
