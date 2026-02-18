using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.DescriptionBased;
/// <summary>Options for ParlerTTS.</summary>
public class ParlerTTSOptions : CodecTtsOptions { public ParlerTTSOptions() { SampleRate = 44100; NumCodebooks = 9; CodebookSize = 1024; CodecFrameRate = 86; LLMDim = 1024; NumLLMLayers = 24; } }
