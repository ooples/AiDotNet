using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the DeepQNetwork.
/// </summary>
public class DeepQNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with the published DQN training defaults.
    /// </summary>
    public DeepQNetworkOptions()
    {
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeepQNetworkOptions(DeepQNetworkOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        // Inherited from ModelOptions: a clone that drops it initializes and trains differently
        // from the configuration it was copied from.
        Seed = other.Seed;
        LearningRate = other.LearningRate;
        GradientMomentum = other.GradientMomentum;
        SquaredGradientMomentum = other.SquaredGradientMomentum;
        MinSquaredGradient = other.MinSquaredGradient;
    }

    /// <summary>Gets or sets the RMSProp learning rate.</summary>
    /// <remarks>
    /// <para>
    /// Mnih et al. 2015, Extended Data Table 1: RMSProp at 0.00025.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate controls how large a step the network takes
    /// each time it learns from experience. Too large and the estimates swing wildly; too small and
    /// learning crawls. This default is the value the published DQN used.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.00025;

    /// <summary>Gets or sets the RMSProp gradient momentum.</summary>
    /// <remarks>
    /// <para>
    /// Mnih et al. 2015, Extended Data Table 1: gradient momentum 0.95.
    /// </para>
    /// <para><b>For Beginners:</b> Momentum lets an update keep some of the direction of the
    /// updates before it, which smooths out noisy steps rather than reacting to each one alone.</para>
    /// </remarks>
    public double GradientMomentum { get; set; } = 0.95;

    /// <summary>Gets or sets the RMSProp squared-gradient decay.</summary>
    /// <remarks>
    /// <para>
    /// Mnih et al. 2015, Extended Data Table 1: squared gradient momentum 0.95.
    /// </para>
    /// <para><b>For Beginners:</b> RMSProp keeps a running average of how large recent gradients
    /// were and divides by it, so parameters with consistently large gradients take smaller steps.
    /// This controls how quickly that running average forgets older values.</para>
    /// </remarks>
    public double SquaredGradientMomentum { get; set; } = 0.95;

    /// <summary>Gets or sets the RMSProp stability constant.</summary>
    /// <remarks>
    /// <para>
    /// Mnih et al. 2015, Extended Data Table 1: min squared gradient 0.01. This is far larger than
    /// a generic optimizer's epsilon, and deliberately so - it bounds the effective step when the
    /// running average of squared gradients is small.
    /// </para>
    /// <para><b>For Beginners:</b> A small floor added before dividing, so the step size cannot
    /// blow up when recent gradients have been tiny.</para>
    /// </remarks>
    public double MinSquaredGradient { get; set; } = 0.01;

    /// <summary>
    /// Validates the training settings, throwing on values the optimizer cannot use.
    /// </summary>
    /// <remarks>
    /// These four values are handed straight to the RMSProp optimizer, so they are external input
    /// to it and are checked before it is built. Non-finite values matter as much as out-of-range
    /// ones: NaN fails every ordered comparison, so a bare range test would let it through and the
    /// first update would poison every parameter.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a value cannot describe a working
    /// optimizer.</exception>
    public void Validate()
    {
        if (double.IsNaN(LearningRate) || double.IsInfinity(LearningRate) || LearningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be a positive, finite number.");
        }

        if (double.IsNaN(GradientMomentum) || double.IsInfinity(GradientMomentum)
            || GradientMomentum < 0.0 || GradientMomentum >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(GradientMomentum), GradientMomentum,
                "GradientMomentum must be finite and within [0, 1).");
        }

        if (double.IsNaN(SquaredGradientMomentum) || double.IsInfinity(SquaredGradientMomentum)
            || SquaredGradientMomentum < 0.0 || SquaredGradientMomentum >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(SquaredGradientMomentum), SquaredGradientMomentum,
                "SquaredGradientMomentum must be finite and within [0, 1).");
        }

        if (double.IsNaN(MinSquaredGradient) || double.IsInfinity(MinSquaredGradient)
            || MinSquaredGradient <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(MinSquaredGradient), MinSquaredGradient,
                "MinSquaredGradient must be a positive, finite number.");
        }
    }
}
