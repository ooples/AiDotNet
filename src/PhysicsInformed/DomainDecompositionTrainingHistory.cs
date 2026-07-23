using System;
using System.Collections.Generic;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Training history for domain decomposition PINN training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// For Beginners:
/// This class tracks the training progress of a domain decomposition PINN.
/// It extends the base TrainingHistory with additional metrics specific to
/// domain decomposition:
///
/// - SubdomainLosses: Per-subdomain PDE and data losses
/// - InterfaceLosses: Continuity errors at subdomain boundaries
/// - PhysicsLosses: Total PDE residual across all subdomains
///
/// Key Observations During Training:
/// 1. Subdomain losses should decrease independently
/// 2. Interface losses ensure solution continuity
/// 3. If one subdomain loss is much higher, it may need more capacity
/// 4. Interface losses are critical for global solution quality
/// </remarks>
public class DomainDecompositionTrainingHistory<T> : TrainingHistory<T>, IDomainDecompositionTrainingHistory<T>
{
    /// <summary>
    /// Gets the losses per subdomain per epoch.
    /// </summary>
    public List<List<T>> SubdomainLosses { get; } = new List<List<T>>();

    /// <summary>
    /// Gets the interface continuity losses per epoch.
    /// </summary>
    public List<T> InterfaceLosses { get; } = new List<T>();

    /// <summary>
    /// Gets the PDE residual losses per epoch.
    /// </summary>
    public List<T> PhysicsLosses { get; } = new List<T>();

    /// <summary>
    /// Gets the number of subdomains.
    /// </summary>
    public int SubdomainCount { get; }

    /// <summary>
    /// Initializes a new instance with the specified number of subdomains.
    /// </summary>
    /// <param name="subdomainCount">Number of subdomains in the decomposition (must be at least 1).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when subdomainCount is less than 1.</exception>
    public DomainDecompositionTrainingHistory(int subdomainCount)
    {
        if (subdomainCount < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(subdomainCount),
                subdomainCount,
                "Domain decomposition requires at least 1 subdomain.");
        }

        SubdomainCount = subdomainCount;
    }

    /// <summary>
    /// Records metrics for a training epoch.
    /// </summary>
    /// <param name="totalLoss">Combined loss from all subdomains and interfaces.</param>
    /// <param name="subdomainLosses">Individual losses per subdomain (must have exactly SubdomainCount elements).</param>
    /// <param name="interfaceLoss">Interface continuity loss.</param>
    /// <param name="physicsLoss">Total PDE residual loss.</param>
    /// <exception cref="ArgumentNullException">Thrown when subdomainLosses is null.</exception>
    /// <exception cref="ArgumentException">Thrown when subdomainLosses count doesn't match SubdomainCount.</exception>
    public void AddEpoch(T totalLoss, List<T> subdomainLosses, T interfaceLoss, T physicsLoss)
    {
        if (subdomainLosses is null)
        {
            throw new ArgumentNullException(nameof(subdomainLosses));
        }

        if (subdomainLosses.Count != SubdomainCount)
        {
            throw new ArgumentException(
                $"Expected {SubdomainCount} subdomain losses, but received {subdomainLosses.Count}.",
                nameof(subdomainLosses));
        }

        AddEpoch(totalLoss); // Base class tracks total loss
        SubdomainLosses.Add(new List<T>(subdomainLosses));
        InterfaceLosses.Add(interfaceLoss);
        PhysicsLosses.Add(physicsLoss);
    }
}
