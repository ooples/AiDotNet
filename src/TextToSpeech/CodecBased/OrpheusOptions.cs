namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Orpheus.</summary>
public class OrpheusOptions : CodecTtsOptions { public OrpheusOptions() { SampleRate = 24000; NumCodebooks = 3; CodebookSize = 4096; CodecFrameRate = 12; LLMDim = 3200; NumLLMLayers = 28; LanguageModelName = "LLaMA"; } }
