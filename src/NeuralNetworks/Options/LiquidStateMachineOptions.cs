using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the LiquidStateMachine.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Liquid State Machine is a reservoir model whose pool behaves like
/// ripples in water — an input disturbs the surface, and the pattern of ripples carries a memory
/// of it. Only the readout is trained. The values here are the ones it ships with.
/// </para>
/// </remarks>
public class LiquidStateMachineOptions : ReservoirOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public LiquidStateMachineOptions()
    {
        ConnectionProbability = 0.1;
        SpectralRadius = 0.9;
        InputScaling = 1.0;
        LeakingRate = 0.3;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public LiquidStateMachineOptions(LiquidStateMachineOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        ConnectionProbability = other.ConnectionProbability;
        SpectralRadius = other.SpectralRadius;
        InputScaling = other.InputScaling;
        LeakingRate = other.LeakingRate;
        ReadoutLearningRate = other.ReadoutLearningRate;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the probability that any two reservoir neurons are connected. Default: 0.1.
    /// </summary>
    public double ConnectionProbability { get; set; }

    /// <summary>
    /// Gets or sets how strongly the input is scaled before entering the reservoir. Default: 1.0.
    /// </summary>
    public double InputScaling { get; set; }

    /// <summary>
    /// Learning rate for the readout layer. Per Maass et al. 2002, the reservoir is fixed
    /// and only the readout is trained. A low LR prevents overfitting/divergence since
    /// the readout is a simple linear mapping.
    /// </summary>
    public double ReadoutLearningRate { get; set; } = 0.0001;

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required value is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// The leaking rate default of 0.3 is deliberate: Jaeger and Haas (2004) require it below 1.0
    /// for the temporal dynamics this model depends on.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateCore();
        Require(ConnectionProbability, nameof(ConnectionProbability));
        Require(InputScaling, nameof(InputScaling));
    }
}
