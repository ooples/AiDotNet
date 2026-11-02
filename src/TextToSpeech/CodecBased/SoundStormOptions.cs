namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SoundStorm (parallel MaskGIT-style audio generation with SoundStream tokens).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SoundStorm model. Default values follow the original paper settings.</para>
/// </remarks>
public class SoundStormOptions : CodecTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SoundStormOptions(SoundStormOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumMaskGITSteps = other.NumMaskGITSteps;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public SoundStormOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 12;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1024;
        NumLLMLayers = 12;
    }

    /// <summary>Gets or sets the number of MaskGIT parallel decoding iterations of the paper's configuration. The native decoder does not read it.</summary>
    public int NumMaskGITSteps { get; set; } = 8;
}
