using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for CAT-Seg cost aggregation open-vocabulary segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CATSeg model. Default values follow the original paper settings.</para>
/// </remarks>
public class CATSegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CATSegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CATSegOptions(CATSegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>
    /// Gets or sets the base AdamW learning rate.
    /// </summary>
    /// <value>Defaults to 0.0002, matching the official CAT-Seg training configuration.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the decoupled AdamW weight decay.
    /// </summary>
    /// <value>Defaults to 0.0001, matching the official CAT-Seg training configuration.</value>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the maximum global gradient norm used by AdamW.
    /// </summary>
    /// <value>Defaults to 0.01, matching CAT-Seg's official full-model gradient clipping.</value>
    public double MaxGradientNorm { get; set; } = 0.01;
}
