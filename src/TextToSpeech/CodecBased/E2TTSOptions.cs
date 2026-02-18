namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for E2TTS.</summary>
public class E2TTSOptions : CodecTtsOptions { public E2TTSOptions() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 75; LLMDim = 1024; NumLLMLayers = 24; } }
