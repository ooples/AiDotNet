namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Represents a LEAF dataset with optional train/test splits.
/// </summary>
/// <typeparam name="TInput">The client feature type.</typeparam>
/// <typeparam name="TOutput">The client label type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Many datasets ship with a training split (used to learn)
/// and a testing split (used to evaluate). This type groups those together.
/// </para>
/// </remarks>
public sealed class LeafFederatedDataset<TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LeafFederatedDataset{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="train">The train split.</param>
    /// <param name="test">The optional test split.</param>
    public LeafFederatedDataset(
        LeafFederatedSplit<TInput, TOutput> train,
        LeafFederatedSplit<TInput, TOutput>? test = null)
    {
        Train = train ?? throw new ArgumentNullException(nameof(train));
        Test = test;
    }

    /// <summary>
    /// Gets the training split.
    /// </summary>
    public LeafFederatedSplit<TInput, TOutput> Train { get; }

    /// <summary>
    /// Gets the optional test split.
    /// </summary>
    public LeafFederatedSplit<TInput, TOutput>? Test { get; }
}

