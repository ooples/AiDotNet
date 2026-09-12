namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the reservoir computing models — the Echo State Network and the
/// Liquid State Machine.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Reservoir models keep a large pool of randomly connected neurons fixed
/// and train only a simple readout on top. The settings here shape the pool's dynamics, which is
/// where all the model's memory of past inputs lives.
/// </para>
/// <para>
/// The reservoir SIZE is not here. It is a required constructor argument with no default on both
/// models — there is no published value to fall back on, so it stays a parameter the caller must
/// supply rather than an option with an invented default.
/// </para>
/// </remarks>
public abstract class ReservoirOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the spectral radius of the reservoir's weight matrix. Default: 0.9.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how long an input echoes around the pool. Below 1.0
    /// the echoes fade, which is what keeps the model stable; at or above 1.0 they can grow
    /// without bound. 0.9 sits just under the edge, giving a long memory that still settles.</para>
    /// </remarks>
    public double SpectralRadius { get; set; }

    /// <summary>
    /// Gets or sets the leaking rate. Default: per model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much of each neuron's previous state carries over to the
    /// next step. A rate of 1.0 replaces the state entirely each step; lower values make the pool
    /// react more slowly and remember longer.</para>
    /// </remarks>
    public double LeakingRate { get; set; }

    /// <summary>
    /// Throws if a dimension every reservoir model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required value is zero or negative, which means the derived options class
    /// did not assign its published defaults.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Both values must be positive to describe a working reservoir: a spectral radius of zero is
    /// a dead pool, and a leaking rate of zero is a pool that never updates.
    /// </para>
    /// </remarks>
    protected void ValidateCore()
    {
        Require(SpectralRadius, nameof(SpectralRadius));
        Require(LeakingRate, nameof(LeakingRate));
    }
}
