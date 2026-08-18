using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for UNITER (Universal Image-TExt Representation).
/// </summary>
/// <remarks>
/// <para>UNITER (Chen et al., ECCV 2020) uses conditional masking pre-training where either
/// image regions or text tokens are masked, forcing the model to learn cross-modal alignment.
/// It uses a single-stream transformer for joint image-text encoding.</para>
/// </remarks>
public class UNITEROptions : FoundationalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UNITEROptions(UNITEROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        TextDim = other.TextDim;
        FusionDim = other.FusionDim;
        NumVisionLayers = other.NumVisionLayers;
        NumTextLayers = other.NumTextLayers;
        NumFusionLayers = other.NumFusionLayers;
        NumHeads = other.NumHeads;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabSize = other.VocabSize;
        DropoutRate = other.DropoutRate;
        FusionType = other.FusionType;
        VisualFeatureType = other.VisualFeatureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalTrainingSteps = other.TotalTrainingSteps;
        WarmupInitialLearningRate = other.WarmupInitialLearningRate;
        EndLearningRate = other.EndLearningRate;
        MaxGradientNorm = other.MaxGradientNorm;
        ImageMaskProbability = other.ImageMaskProbability;
        TextMaskProbability = other.TextMaskProbability;
        MaxImageRegions = other.MaxImageRegions;
    }

    /// <summary>Gets or sets the image region masking probability during training.</summary>
    public double ImageMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the text token masking probability during training.</summary>
    public double TextMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the maximum number of image regions.</summary>
    public int MaxImageRegions { get; set; } = 36;

    /// <summary>Gets or sets the number of linear learning-rate warmup steps.</summary>
    /// <remarks>The official all-data UNITER-base pre-training recipe uses 10,000 steps.</remarks>
    public int WarmupSteps { get; set; } = 10_000;

    /// <summary>Gets or sets the training horizon used by the post-warmup linear decay.</summary>
    /// <remarks>The official all-data UNITER-base pre-training recipe uses 200,000 steps.</remarks>
    public int TotalTrainingSteps { get; set; } = 200_000;

    /// <summary>
    /// Gets or sets the learning rate at the first warmup update, or <see langword="null"/> to
    /// derive the first positive increment from <see cref="FoundationalVLMOptions.LearningRate"/>
    /// and <see cref="WarmupSteps"/>.
    /// </summary>
    /// <remarks>
    /// The derived default avoids an inert first optimizer step while continuing to scale correctly
    /// when an advanced user customizes either the peak rate or the warmup duration.
    /// </remarks>
    public double? WarmupInitialLearningRate { get; set; }

    /// <summary>Gets or sets the final learning rate after linear decay.</summary>
    public double EndLearningRate { get; set; } = 0.0;

    /// <summary>Gets or sets the global gradient-norm clipping threshold.</summary>
    /// <remarks>The official all-data UNITER-base pre-training recipe uses 5.0.</remarks>
    public double MaxGradientNorm { get; set; } = 5.0;

    /// <summary>Validates the native optimizer and learning-rate schedule contract.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when a rate is non-finite or outside its valid range, the training horizon is
    /// invalid, warmup exceeds that horizon, or gradient clipping is not positive.
    /// </exception>
    public void Validate()
    {
        if (!double.IsFinite(LearningRate) || LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be finite and greater than zero.");
        if (!double.IsFinite(WeightDecay) || WeightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay), WeightDecay,
                "WeightDecay must be finite and non-negative.");
        if (WarmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(WarmupSteps), WarmupSteps,
                "WarmupSteps must be non-negative.");
        if (TotalTrainingSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(TotalTrainingSteps), TotalTrainingSteps,
                "TotalTrainingSteps must be greater than zero.");
        if (WarmupSteps > TotalTrainingSteps)
            throw new ArgumentOutOfRangeException(nameof(WarmupSteps), WarmupSteps,
                "WarmupSteps cannot exceed TotalTrainingSteps.");
        if (WarmupInitialLearningRate is double warmupRate
            && (!double.IsFinite(warmupRate) || warmupRate < 0.0))
        {
            throw new ArgumentOutOfRangeException(nameof(WarmupInitialLearningRate), warmupRate,
                "WarmupInitialLearningRate must be finite and non-negative when specified.");
        }
        if (!double.IsFinite(EndLearningRate) || EndLearningRate < 0.0)
            throw new ArgumentOutOfRangeException(nameof(EndLearningRate), EndLearningRate,
                "EndLearningRate must be finite and non-negative.");
        if (!double.IsFinite(MaxGradientNorm) || MaxGradientNorm <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(MaxGradientNorm), MaxGradientNorm,
                "MaxGradientNorm must be finite and greater than zero.");
    }

    public UNITEROptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
        // Official UNITER-base all-data pre-training configuration.
        LearningRate = 5e-5;
        WeightDecay = 0.01;
    }
}
