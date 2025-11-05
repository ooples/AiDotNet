using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Tasks;

/// <summary>
/// Default implementation of an episode for N-way K-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
public class Episode<T> : IEpisode<T> where T : struct
{
    /// <inheritdoc/>
    public Tensor<T> SupportData { get; }

    /// <inheritdoc/>
    public Tensor<T> SupportLabels { get; }

    /// <inheritdoc/>
    public Tensor<T> QueryData { get; }

    /// <inheritdoc/>
    public Tensor<T> QueryLabels { get; }

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueries { get; }

    /// <inheritdoc/>
    public string TaskId { get; }

    /// <summary>
    /// Creates a new episode with support and query sets.
    /// </summary>
    /// <param name="supportData">Support set data tensor</param>
    /// <param name="supportLabels">Support set labels tensor</param>
    /// <param name="queryData">Query set data tensor</param>
    /// <param name="queryLabels">Query set labels tensor</param>
    /// <param name="numWays">Number of classes (N)</param>
    /// <param name="numShots">Number of support examples per class (K)</param>
    /// <param name="numQueries">Number of query examples per class</param>
    /// <param name="taskId">Task identifier</param>
    public Episode(
        Tensor<T> supportData,
        Tensor<T> supportLabels,
        Tensor<T> queryData,
        Tensor<T> queryLabels,
        int numWays,
        int numShots,
        int numQueries,
        string taskId = "")
    {
        if (numWays <= 0)
            throw new ArgumentException("Number of ways must be positive", nameof(numWays));
        if (numShots <= 0)
            throw new ArgumentException("Number of shots must be positive", nameof(numShots));
        if (numQueries <= 0)
            throw new ArgumentException("Number of queries must be positive", nameof(numQueries));

        SupportData = supportData ?? throw new ArgumentNullException(nameof(supportData));
        SupportLabels = supportLabels ?? throw new ArgumentNullException(nameof(supportLabels));
        QueryData = queryData ?? throw new ArgumentNullException(nameof(queryData));
        QueryLabels = queryLabels ?? throw new ArgumentNullException(nameof(queryLabels));
        NumWays = numWays;
        NumShots = numShots;
        NumQueries = numQueries;
        TaskId = taskId ?? Guid.NewGuid().ToString();

        ValidateEpisode();
    }

    private void ValidateEpisode()
    {
        var expectedSupportSize = NumWays * NumShots;
        var expectedQuerySize = NumWays * NumQueries;

        if (SupportData.Shape[0] != expectedSupportSize)
            throw new ArgumentException(
                $"Support data size {SupportData.Shape[0]} doesn't match N*K={expectedSupportSize}");

        if (SupportLabels.Shape[0] != expectedSupportSize)
            throw new ArgumentException(
                $"Support labels size {SupportLabels.Shape[0]} doesn't match N*K={expectedSupportSize}");

        if (QueryData.Shape[0] != expectedQuerySize)
            throw new ArgumentException(
                $"Query data size {QueryData.Shape[0]} doesn't match N*Q={expectedQuerySize}");

        if (QueryLabels.Shape[0] != expectedQuerySize)
            throw new ArgumentException(
                $"Query labels size {QueryLabels.Shape[0]} doesn't match N*Q={expectedQuerySize}");
    }
}
