namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for DiffWave (diffusion-based vocoder using denoising score matching).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DiffWave model. Default values follow the original paper settings.</para>
/// </remarks>
public class DiffWaveOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffWaveOptions(DiffWaveOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResLayers = other.NumResLayers;
        ResChannels = other.ResChannels;
        OptimizerBatchSize = other.OptimizerBatchSize;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    public DiffWaveOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumDiffusionSteps = 50;
        // Kong et al. use Adam with a batch size of 16 and a fixed 2e-4 learning rate.
        LearningRate = 2e-4;
        WeightDecay = 0.0;
    }

    public int NumResLayers { get; set; } = 30;
    public int ResChannels { get; set; } = 64;

    /// <summary>Gets or sets the Adam mini-batch size. The paper default is 16.</summary>
    public int OptimizerBatchSize { get; set; } = 16;

    /// <summary>Gets or sets Adam's first-moment decay.</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay.</summary>
    public double OptimizerBeta2 { get; set; } = 0.999;

    /// <summary>Gets or sets Adam's numerical-stability epsilon.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the gradient clipping norm. A value less than or equal to zero disables
    /// clipping, matching the paper's default; users may set a positive value explicitly.
    /// </summary>
    public double MaxGradientNorm { get; set; } = 0.0;
}
