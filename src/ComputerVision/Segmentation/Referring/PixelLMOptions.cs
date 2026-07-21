using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for PixelLM pixel-level reasoning segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PixelLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class PixelLMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public PixelLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PixelLMOptions(PixelLMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        OptimizerBatchSize = other.OptimizerBatchSize;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
    }

    /// <summary>Gets or sets the AdamW learning rate. The paper default is 3e-4.</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets decoupled weight decay. The paper default is zero.</summary>
    public double WeightDecay { get; set; } = 0.0;

    /// <summary>Gets or sets the optimizer batch size. The paper default is 16.</summary>
    public int OptimizerBatchSize { get; set; } = 16;

    /// <summary>Gets or sets AdamW's first-moment decay.</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment decay. The paper default is 0.95.</summary>
    public double OptimizerBeta2 { get; set; } = 0.95;

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

}
