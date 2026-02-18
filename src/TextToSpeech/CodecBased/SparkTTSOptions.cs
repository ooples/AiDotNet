namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for SparkTTS.</summary>
public class SparkTTSOptions : CodecTtsOptions { public SparkTTSOptions() { SampleRate = 16000; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 25; LLMDim = 1536; NumLLMLayers = 28; LanguageModelName = "Qwen2.5"; } }
