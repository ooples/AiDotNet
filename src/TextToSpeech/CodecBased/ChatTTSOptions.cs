namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for ChatTTS.</summary>
public class ChatTTSOptions : CodecTtsOptions { public ChatTTSOptions() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 626; CodecFrameRate = 21; LLMDim = 768; NumLLMLayers = 20; } }
