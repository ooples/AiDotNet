using System.Collections.Generic;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Training history for multi-fidelity PINN training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// For Beginners:
/// This class tracks the training progress of a multi-fidelity PINN.
/// It extends the base TrainingHistory with additional metrics specific to
/// multi-fidelity learning:
///
/// - LowFidelityLosses: Error on cheap/approximate data
/// - HighFidelityLosses: Error on expensive/accurate data
/// - CorrelationLosses: How well the model captures the relationship between fidelity levels
/// - PhysicsLosses: PDE residual errors
///
/// Typical Training Dynamics:
/// 1. Early training: Low-fidelity loss dominates (most data)
/// 2. Mid training: High-fidelity becomes important (precision matters)
/// 3. Late training: Physics loss should be low (PDE satisfied)
/// </remarks>
public class MultiFidelityTrainingHistory<T> : TrainingHistory<T>, IMultiFidelityTrainingHistory<T>
{
    /// <summary>
    /// Gets the low-fidelity data losses per epoch.
    /// </summary>
    public List<T> LowFidelityLosses { get; } = new List<T>();

    /// <summary>
    /// Gets the high-fidelity data losses per epoch.
    /// </summary>
    public List<T> HighFidelityLosses { get; } = new List<T>();

    /// <summary>
    /// Gets the correlation losses per epoch.
    /// </summary>
    public List<T> CorrelationLosses { get; } = new List<T>();

    /// <summary>
    /// Gets the PDE residual losses per epoch.
    /// </summary>
    public List<T> PhysicsLosses { get; } = new List<T>();

    /// <summary>
    /// Records metrics for a training epoch.
    /// </summary>
    /// <param name="totalLoss">Combined loss from all components.</param>
    /// <param name="lowFidelityLoss">Loss from low-fidelity data fitting.</param>
    /// <param name="highFidelityLoss">Loss from high-fidelity data fitting.</param>
    /// <param name="correlationLoss">Loss measuring fidelity correlation.</param>
    /// <param name="physicsLoss">PDE residual loss.</param>
    public void AddEpoch(T totalLoss, T lowFidelityLoss, T highFidelityLoss, T correlationLoss, T physicsLoss)
    {
        AddEpoch(totalLoss); // Base class tracks total loss
        LowFidelityLosses.Add(lowFidelityLoss);
        HighFidelityLosses.Add(highFidelityLoss);
        CorrelationLosses.Add(correlationLoss);
        PhysicsLosses.Add(physicsLoss);
    }
}
