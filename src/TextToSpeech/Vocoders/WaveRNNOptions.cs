namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveRNN (efficient autoregressive vocoder with subscale generation).</summary>
public class WaveRNNOptions : VocoderOptions { public WaveRNNOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; } public int RnnDim { get; set; } = 512; public int FcDim { get; set; } = 512; public int Bits { get; set; } = 10; }
