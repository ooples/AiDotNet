using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for SegNeXt (Guo et al. 2022).
/// </summary>
/// <remarks>
/// <para>
/// The training defaults are the paper's ADE20K recipe (§4.1, and the authors' and mmsegmentation's configs): AdamW at
/// 6e-5 with weight decay 0.01, a linear warm-up over 1500 iterations from 1e-6 of the rate, then a poly (power 1)
/// decay to zero over 160k iterations. The configs also train the decode head at ten times the rate and exempt
/// parameters named "norm" from weight decay. The architecture follows <see cref="ModelSize"/> (MSCAN-T/S/B/L with a
/// LightHamHead of 256/256/512/1024 channels).
/// </para>
/// <para>
/// <b>For Beginners:</b> SegNeXt labels every pixel of an image with a class. These options pick its size and how it
/// trains.
/// </para>
/// </remarks>
public class SegNeXtOptions : SegmentationModelOptions
{
    /// <summary>Creates the paper's defaults (MSCAN-T, 150 ADE20K classes).</summary>
    public SegNeXtOptions()
    {
        NumClasses = 150;
        // The decode head's Dropout2d before the classifier (dropout_ratio 0.1 in every official config).
        DropRate = 0.1;
        ModelSize = SegNeXtModelSize.Tiny;
    }

    /// <summary>Copies every option.</summary>
    public SegNeXtOptions(SegNeXtOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        DropPathRate = other.DropPathRate;
        MatrixRank = other.MatrixRank;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        HeadLearningRateMultiplier = other.HeadLearningRateMultiplier;
        WarmupIterations = other.WarmupIterations;
        WarmupStartFactor = other.WarmupStartFactor;
        TrainingIterations = other.TrainingIterations;
        RandomBasesPerForward = other.RandomBasesPerForward;
        EmbedDims = other.EmbedDims is null ? null : (int[])other.EmbedDims.Clone();
        Depths = other.Depths is null ? null : (int[])other.Depths.Clone();
        HeadChannels = other.HeadChannels;
    }

    /// <summary>MSCAN-T, -S, -B or -L.</summary>
    public SegNeXtModelSize ModelSize { get; set; }

    /// <summary>The encoder's stochastic-depth rate; null takes the size's (0.1, and 0.3 for Large).</summary>
    public double? DropPathRate { get; set; }

    /// <summary>
    /// Whether the Hamburger NMF draws new random bases on every forward pass (the official code's rand_init=True,
    /// which makes predictions vary slightly between calls), instead of bases drawn once when the model is built and
    /// saved with it (rand_init=False, the default here, so predictions are repeatable).
    /// </summary>
    public bool RandomBasesPerForward { get; set; }

    /// <summary>The Hamburger NMF's rank (16).</summary>
    public int MatrixRank { get; set; } = 16;

    /// <summary>AdamW's base learning rate (6e-5).</summary>
    public double LearningRate { get; set; } = 6e-5;

    /// <summary>AdamW's weight decay (0.01).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>The decode head's learning-rate multiplier (10).</summary>
    public double HeadLearningRateMultiplier { get; set; } = 10.0;

    /// <summary>Linear warm-up iterations (1500).</summary>
    public int WarmupIterations { get; set; } = 1500;

    /// <summary>The fraction of the rate the warm-up starts from (1e-6).</summary>
    public double WarmupStartFactor { get; set; } = 1e-6;

    /// <summary>The iteration the poly decay reaches zero at (160 000).</summary>
    public int TrainingIterations { get; set; } = 160_000;

    /// <summary>Stage widths overriding the size's; null keeps them.</summary>
    public int[]? EmbedDims { get; set; }

    /// <summary>Blocks per stage overriding the size's; null keeps them.</summary>
    public int[]? Depths { get; set; }

    /// <summary>The head's channels (squeeze, Hamburger and align) overriding the size's; null keeps them.</summary>
    public int? HeadChannels { get; set; }

    /// <summary>Checks the shared segmentation options.</summary>
    public void Validate() => ValidateSegmentationCore();
}
