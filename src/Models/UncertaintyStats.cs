namespace AiDotNet.Models;

/// <summary>
/// Represents uncertainty-quantification diagnostics aggregated over a dataset.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This container is designed to integrate with the existing AiDotNet evaluation pipeline by living alongside
/// <see cref="ErrorStats{T}"/> and <see cref="PredictionStats{T}"/> within <see cref="DataSetStats{T, TInput, TOutput}"/>.
/// </para>
/// <para><b>For Beginners:</b> This stores summary uncertainty metrics (like average entropy) for an entire dataset,
/// similar to how accuracy or error metrics summarize model quality.</para>
/// </remarks>
public sealed class UncertaintyStats<T>
{
    /// <summary>
    /// Gets a dictionary of aggregate uncertainty metrics for the dataset.
    /// </summary>
    /// <remarks>
    /// Keys are stable so consumers do not have to branch on missing keys. When a metric is not computable,
    /// it should be populated with a sensible default (typically 0).
    /// </remarks>
    public Dictionary<string, T> Metrics { get; } = new();

    /// <summary>
    /// Creates an empty <see cref="UncertaintyStats{T}"/> instance.
    /// </summary>
    public static UncertaintyStats<T> Empty()
    {
        return new UncertaintyStats<T>();
    }
}

