namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveRNN (efficient autoregressive vocoder with subscale generation).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WaveRNN model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveRNNOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveRNNOptions(WaveRNNOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        RnnDim = other.RnnDim;
        FcDim = other.FcDim;
        Bits = other.Bits;
    }
 public WaveRNNOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; } public int RnnDim { get; set; } = 512; public int FcDim { get; set; } = 512; public int Bits { get; set; } = 10; }
