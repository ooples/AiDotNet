namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for FireRedTTS.</summary>
public class FireRedTTSOptions : CodecTtsOptions { public FireRedTTSOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 2048; NumLLMLayers = 24; } }
