namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for WaveRNN (efficient autoregressive vocoder with subscale generation).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WaveRNN model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveRNNOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveRNNOptions(WaveRNNOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        RnnDim = other.RnnDim;
        FcDim = other.FcDim;
        Bits = other.Bits;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public WaveRNNOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 256;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the width of the recurrent layer.</summary>
    public int RnnDim { get; set; } = 512;
    /// <summary>Gets or sets the width of the paper's fully connected output layers. The native layers do not read it.</summary>
    public int FcDim { get; set; } = 512;
    /// <summary>Gets or sets the sample bit depth of the paper's mu-law output. The native model predicts samples directly and does not read it.</summary>
    public int Bits { get; set; } = 10;
    /// <summary>Gets or sets the gradient-norm clip of the default optimizer; 0 disables clipping.</summary>
    public double MaxGradientNorm { get; set; } = 4.0;
}
