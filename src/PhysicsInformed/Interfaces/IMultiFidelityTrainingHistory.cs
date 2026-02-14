using System.Collections.Generic;

namespace AiDotNet.PhysicsInformed.Interfaces;

/// <summary>
/// Extended training history interface for multi-fidelity PINN training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// For Beginners:
/// Multi-fidelity training uses data from multiple sources with different accuracy levels:
/// - Low-fidelity: Cheap but less accurate (e.g., coarse simulations, simplified models)
/// - High-fidelity: Expensive but accurate (e.g., fine simulations, experiments)
///
/// This interface tracks metrics for each fidelity level during training.
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MultiFidelityTrainingHistory")]
public interface IMultiFidelityTrainingHistory<T>
{
    /// <summary>
    /// Gets the total losses per epoch (combined from all fidelity levels).
    /// </summary>
    List<T> Losses { get; }

    /// <summary>
    /// Gets the low-fidelity data losses per epoch.
    /// </summary>
    List<T> LowFidelityLosses { get; }

    /// <summary>
    /// Gets the high-fidelity data losses per epoch.
    /// </summary>
    List<T> HighFidelityLosses { get; }

    /// <summary>
    /// Gets the correlation losses per epoch (measures agreement between fidelity levels).
    /// </summary>
    List<T> CorrelationLosses { get; }

    /// <summary>
    /// Gets the PDE residual losses per epoch.
    /// </summary>
    List<T> PhysicsLosses { get; }

    /// <summary>
    /// Records metrics for a training epoch.
    /// </summary>
    /// <param name="totalLoss">Combined loss from all components.</param>
    /// <param name="lowFidelityLoss">Loss from low-fidelity data fitting.</param>
    /// <param name="highFidelityLoss">Loss from high-fidelity data fitting.</param>
    /// <param name="correlationLoss">Loss measuring fidelity correlation.</param>
    /// <param name="physicsLoss">PDE residual loss.</param>
    void AddEpoch(T totalLoss, T lowFidelityLoss, T highFidelityLoss, T correlationLoss, T physicsLoss);
}
