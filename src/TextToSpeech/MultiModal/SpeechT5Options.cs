using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for SpeechT5 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeechT5 model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeechT5Options : EndToEndTtsOptions
{
    public SpeechT5Options()
    {
        SampleRate = 16000;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.15;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Initializes an independent copy of another SpeechT5 configuration.</summary>
    public SpeechT5Options(SpeechT5Options other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        MaxGradientNorm = other.MaxGradientNorm;
        WarmupUpdates = other.WarmupUpdates;
    }

    /// <summary>Adam first-moment coefficient from the official TTS recipe.</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Adam second-moment coefficient from the official TTS recipe.</summary>
    public double OptimizerBeta2 { get; set; } = 0.98;

    /// <summary>Adam numerical-stability constant.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

    /// <summary>Global gradient-norm clipping threshold from the official TTS recipe.</summary>
    public double MaxGradientNorm { get; set; } = 25.0;

    /// <summary>Number of linear-warmup updates before inverse-square-root decay.</summary>
    public int WarmupUpdates { get; set; } = 10000;
}
