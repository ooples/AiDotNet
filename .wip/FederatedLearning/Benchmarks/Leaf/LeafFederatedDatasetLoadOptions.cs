namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Options controlling how LEAF federated benchmark JSON files are loaded.
/// </summary>
/// <remarks>
/// <para>
/// LEAF datasets can be large. These options allow callers to load a smaller subset for
/// quick experiments and CI-friendly tests.
/// </para>
/// <para><b>For Beginners:</b> Think of these as "load settings" for a dataset file:
/// you can choose to load all clients/users or just the first N to keep the run fast.
/// </para>
/// </remarks>
public sealed class LeafFederatedDatasetLoadOptions
{
    /// <summary>
    /// Gets or sets the maximum number of users/clients to load (null loads all users).
    /// </summary>
    public int? MaxUsers { get; set; }

    /// <summary>
    /// Gets or sets whether to validate that each user's declared <c>num_samples</c> matches the actual sample count.
    /// </summary>
    public bool ValidateDeclaredSampleCounts { get; set; } = true;
}

