namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoiceV2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OpenVoiceV2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class OpenVoiceV2Options : VoiceCloningOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenVoiceV2Options(OpenVoiceV2Options other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        WeightDecay = other.WeightDecay;
    }

    public OpenVoiceV2Options()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
        LearningRate = 3e-4;
        WeightDecay = 0.01;
    }

    public int EncoderDim { get; set; } = 512;
    public int DecoderDim { get; set; } = 192;

    /// <summary>First AdamW moment coefficient used by the released recipe.</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Second AdamW moment coefficient used by the released recipe.</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW numerical-stability epsilon used by the released recipe.</summary>
    public double Epsilon { get; set; } = 1e-9;

}
