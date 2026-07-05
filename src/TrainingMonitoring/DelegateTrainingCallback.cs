using AiDotNet.Interfaces;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// Adapts a simple per-epoch delegate into a full <see cref="ITrainingCallback{T}"/>.
/// </summary>
/// <remarks>
/// This is the backing type for the one-line
/// <c>AiModelBuilder.ConfigureTrainingCallback(Func&lt;TrainingProgress&lt;T&gt;, bool&gt;)</c>
/// overload: it forwards <see cref="OnEpochEnd"/> to the supplied delegate and makes
/// <see cref="OnTrainBegin"/> / <see cref="OnTrainEnd"/> no-ops.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
internal sealed class DelegateTrainingCallback<T> : ITrainingCallback<T>
{
    private readonly Func<TrainingProgress<T>, bool> _onEpochEnd;

    /// <summary>
    /// Initializes a new adapter around the given per-epoch delegate.
    /// </summary>
    /// <param name="onEpochEnd">
    /// Invoked once per epoch; return <c>false</c> to request an early stop.
    /// </param>
    public DelegateTrainingCallback(Func<TrainingProgress<T>, bool> onEpochEnd)
    {
        _onEpochEnd = onEpochEnd ?? throw new ArgumentNullException(nameof(onEpochEnd));
    }

    /// <inheritdoc/>
    public void OnTrainBegin(TrainingProgress<T> progress)
    {
    }

    /// <inheritdoc/>
    public bool OnEpochEnd(TrainingProgress<T> progress) => _onEpochEnd(progress);

    /// <inheritdoc/>
    public void OnTrainEnd(TrainingProgress<T> progress)
    {
    }
}
