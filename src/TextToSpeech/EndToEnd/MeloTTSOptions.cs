namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for MeloTTS (multilingual VITS-based TTS with BERT-enhanced text processing and mixed-language support).</summary>
public class MeloTTSOptions : EndToEndTtsOptions { public MeloTTSOptions() { SampleRate = 44100; MelChannels = 80; HopSize = 512; HiddenDim = 192; NumFlowSteps = 4; } public double SpeedFactor { get; set; } = 1.0; }
