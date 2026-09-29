using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the EchoStateNetwork.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An Echo State Network keeps a large pool of randomly connected neurons
/// fixed and trains only a linear readout on top, which makes it very fast to train. The values
/// here are the ones it ships with.
/// </para>
/// </remarks>
public class EchoStateNetworkOptions : ReservoirOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public EchoStateNetworkOptions()
    {
        SpectralRadius = 0.9;
        Sparsity = 0.1;
        LeakingRate = 1.0;
        Regularization = 1e-4;
        WarmupPeriod = 10;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public EchoStateNetworkOptions(EchoStateNetworkOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        SpectralRadius = other.SpectralRadius;
        Sparsity = other.Sparsity;
        LeakingRate = other.LeakingRate;
        Regularization = other.Regularization;
        WarmupPeriod = other.WarmupPeriod;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the fraction of reservoir connections that are non-zero. Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only about a tenth of the possible connections exist. A sparse
    /// pool is both cheaper and, in practice, better behaved than a fully connected one.</para>
    /// </remarks>
    public double Sparsity { get; set; }

    /// <summary>
    /// Gets or sets the ridge regularization strength for the readout. Default: 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Zero is a legitimate setting — plain least squares with no ridge term — so this is
    /// deliberately not required positive.
    /// </para>
    /// </remarks>
    public double Regularization { get; set; }

    /// <summary>
    /// Gets or sets how many initial steps are discarded before readout training. Default: 10.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The pool starts from an arbitrary state, so its first few
    /// outputs say more about that starting point than about the input. Those are thrown away.
    /// Zero is valid and is what the parameterless constructor path uses, so this is not required
    /// positive.</para>
    /// </remarks>
    public int WarmupPeriod { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required value is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// Only Sparsity is added to the family's requirements. Regularization and WarmupPeriod may
    /// both legitimately be zero.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateCore();
        Require(Sparsity, nameof(Sparsity));
    }
}
