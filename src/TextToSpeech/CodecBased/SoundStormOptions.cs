namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for SoundStorm (parallel MaskGIT-style audio generation with SoundStream tokens).</summary>
public class SoundStormOptions : CodecTtsOptions { public SoundStormOptions() { SampleRate = 24000; NumCodebooks = 12; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 12; } public int NumMaskGITSteps { get; set; } = 8; }
