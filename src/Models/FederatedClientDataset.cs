namespace AiDotNet.Models;

/// <summary>
/// Represents a single client's local dataset for federated learning.
/// </summary>
/// <remarks>
/// This type is intentionally simple: it holds the client's local features and labels
/// plus the sample count used for weighting during aggregation.
/// </remarks>
/// <typeparam name="TInput">The type of the input data (e.g., Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The type of the output data (e.g., Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
public sealed class FederatedClientDataset<TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FederatedClientDataset{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="features">The client's local feature data.</param>
    /// <param name="labels">The client's local label/target data.</param>
    /// <param name="sampleCount">The number of samples contained in this dataset.</param>
    public FederatedClientDataset(TInput features, TOutput labels, int sampleCount)
    {
        if (sampleCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be non-negative.");
        }

        Features = features;
        Labels = labels;
        SampleCount = sampleCount;
    }

    /// <summary>
    /// Gets the client's local feature data.
    /// </summary>
    public TInput Features { get; }

    /// <summary>
    /// Gets the client's local label/target data.
    /// </summary>
    public TOutput Labels { get; }

    /// <summary>
    /// Gets the number of samples contained in this dataset.
    /// </summary>
    public int SampleCount { get; }
}

