namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for MARS5TTS.</summary>
public class MARS5TTSOptions : CodecTtsOptions { public MARS5TTSOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 75; LLMDim = 1536; NumLLMLayers = 24; } }
