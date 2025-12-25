using System.Collections.Generic;

namespace AiDotNet.PhysicsInformed.Interfaces;

/// <summary>
/// Extended training history interface for domain decomposition PINN training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// For Beginners:
/// Domain decomposition divides a large problem domain into smaller subdomains.
/// Each subdomain has its own neural network, and interface conditions ensure
/// continuity between neighboring subdomains.
///
/// This interface tracks:
/// - Per-subdomain losses
/// - Interface continuity losses
/// - Overall convergence metrics
/// </remarks>
public interface IDomainDecompositionTrainingHistory<T>
{
    /// <summary>
    /// Gets the total losses per epoch (combined from all subdomains).
    /// </summary>
    List<T> Losses { get; }

    /// <summary>
    /// Gets the losses per subdomain per epoch.
    /// Outer list: epochs, Inner list: subdomain losses.
    /// </summary>
    List<List<T>> SubdomainLosses { get; }

    /// <summary>
    /// Gets the interface continuity losses per epoch.
    /// </summary>
    List<T> InterfaceLosses { get; }

    /// <summary>
    /// Gets the PDE residual losses per epoch (sum across all subdomains).
    /// </summary>
    List<T> PhysicsLosses { get; }

    /// <summary>
    /// Gets the number of subdomains.
    /// </summary>
    int SubdomainCount { get; }

    /// <summary>
    /// Records metrics for a training epoch.
    /// </summary>
    /// <param name="totalLoss">Combined loss from all subdomains and interfaces.</param>
    /// <param name="subdomainLosses">Individual losses per subdomain.</param>
    /// <param name="interfaceLoss">Interface continuity loss.</param>
    /// <param name="physicsLoss">Total PDE residual loss.</param>
    void AddEpoch(T totalLoss, List<T> subdomainLosses, T interfaceLoss, T physicsLoss);
}
