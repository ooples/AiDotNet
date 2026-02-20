namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for YourTTS (multilingual zero-shot multi-speaker VITS variant with speaker and language conditioning).</summary>
public class YourTTSOptions : EndToEndTtsOptions { public YourTTSOptions() { SampleRate = 16000; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } public int SpeakerEmbeddingDim { get; set; } = 256; public int NumLanguages { get; set; } = 16; }
