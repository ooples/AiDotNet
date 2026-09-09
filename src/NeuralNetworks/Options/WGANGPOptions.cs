using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the WGANGP neural network.
/// </summary>
public class WGANGPOptions : GanOptions
{
    /// <summary>Initializes the paper defaults.</summary>
    public WGANGPOptions()
    {
        GradientPenaltyCoefficient = 10.0;
        CriticIterations = 5;
    }

    /// <summary>Initializes an independent copy of another configuration.</summary>
    public WGANGPOptions(WGANGPOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        CriticIterations = other.CriticIterations;
        GradientPenaltyCoefficient = other.GradientPenaltyCoefficient;
    }

    /// <summary>Gets or sets Adam's learning rate (Algorithm 1 default: 1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets Adam's first-moment decay (Algorithm 1 default: 0).</summary>
    public double Beta1 { get; set; } = 0.0;

    /// <summary>Gets or sets Adam's second-moment decay (Algorithm 1 default: 0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the gradient penalty coefficient.
    /// </summary>
    public double GradientPenaltyCoefficient { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
