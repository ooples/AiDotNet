using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the HawkLanguageModel.
/// </summary>
public class HawkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the RG-LRU width. The default 2560 is the published 1.3B
    /// Griffin/Hawk configuration for a model width of 2048.
    /// </summary>
    public int RecurrenceDimension { get; set; } = 2560;

    /// <summary>
    /// Gets or sets the AdamW learning rate. The paper tunes this value by
    /// model scale rather than publishing a single universal value.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the decoupled AdamW weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>Gets or sets AdamW's first-moment decay.</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment decay.</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Gets or sets whether global-norm gradient clipping is enabled.</summary>
    public bool EnableGradientClipping { get; set; } = true;

    /// <summary>Gets or sets the maximum gradient norm.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Initializes an options instance with default values.</summary>
    public HawkOptions() { }

    /// <summary>Initializes an options instance by copying inherited configuration.</summary>
    /// <param name="other">The source options.</param>
    public HawkOptions(HawkOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        RecurrenceDimension = other.RecurrenceDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        EnableGradientClipping = other.EnableGradientClipping;
        MaxGradientNorm = other.MaxGradientNorm;
    }
}
