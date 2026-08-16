namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Vocos (ConvNeXt-based Fourier vocoder predicting STFT magnitude and instantaneous frequency).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Vocos model. Default values follow the original paper settings.</para>
/// </remarks>
public class VocosOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VocosOptions(VocosOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ConvNeXtDim = other.ConvNeXtDim;
        NumBackboneBlocks = other.NumBackboneBlocks;
        IntermediateDim = other.IntermediateDim;
    }

    public VocosOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 2e-4;
    }

    public int ConvNeXtDim { get; set; } = 512;

    /// <summary>Gets or sets the number of isotropic ConvNeXt backbone blocks.</summary>
    public int NumBackboneBlocks { get; set; } = 8;

    /// <summary>Gets or sets the hidden width of each ConvNeXt pointwise MLP.</summary>
    public int IntermediateDim { get; set; } = 1536;
}
