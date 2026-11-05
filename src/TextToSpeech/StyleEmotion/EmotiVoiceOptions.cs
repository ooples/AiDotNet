using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for EmotiVoice TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EmotiVoice model. Default values follow the original paper settings.</para>
/// </remarks>
public class EmotiVoiceOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmotiVoiceOptions(EmotiVoiceOptions other)
        : base(other)
    {
        // base(other) runs before this body and throws ArgumentNullException
        // if other is null, so a local null-check here is unreachable.
        EmotionDim = other.EmotionDim;
        NumEmotionLayers = other.NumEmotionLayers;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        LearningRateSchedulerGamma = other.LearningRateSchedulerGamma;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public EmotiVoiceOptions()
    {
        SampleRate = 16000;
        HiddenDim = 384;
        EncoderDim = 384;
        DecoderDim = 384;
        InterChannels = 384;
        FilterChannels = 1536;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 8;
        DropoutRate = 0.2;
        EmotionDim = 384;
        LearningRate = 1.25e-5;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the width of the emotion embedding.</summary>
    public int EmotionDim { get; set; } = 384;
    /// <summary>Gets or sets the number of layers in the emotion encoder.</summary>
    public int NumEmotionLayers { get; set; } = 3;
    /// <summary>Gets or sets the first-moment decay (beta1) of the default Adam optimizer.</summary>
    public double OptimizerBeta1 { get; set; } = 0.5;
    /// <summary>Gets or sets the second-moment decay (beta2) of the default Adam optimizer.</summary>
    public double OptimizerBeta2 { get; set; } = 0.9;
    /// <summary>Gets or sets the epsilon of the default Adam optimizer.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-9;
    /// <summary>Gets or sets the per-step decay factor of the default exponential learning-rate schedule; a value outside (0, 1) disables the schedule.</summary>
    public double LearningRateSchedulerGamma { get; set; } = 0.999875;
}
