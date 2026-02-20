namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for Kokoro (lightweight StyleTTS2-inspired TTS with style tokens and ISTFTNet decoder).</summary>
public class KokoroOptions : EndToEndTtsOptions { public KokoroOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; HiddenDim = 512; NumFlowSteps = 0; } public int StyleDim { get; set; } = 128; }
