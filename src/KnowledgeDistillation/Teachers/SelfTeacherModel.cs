using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Self teacher model that uses the student's own predictions from earlier training.
/// </summary>
public class SelfTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private Vector<T>[]? _cachedPredictions;
    private readonly int _outputDim;

    public override int OutputDimension => _outputDim;

    public SelfTeacherModel(int outputDimension)
    {
        _outputDim = outputDimension;
    }

    public void CachePredictions(Vector<T>[] predictions)
    {
        if (predictions == null)
            throw new ArgumentNullException(nameof(predictions));

        if (predictions.Length == 0)
            throw new ArgumentException("Predictions array cannot be empty", nameof(predictions));

        // Validate all predictions have correct dimension
        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == null)
                throw new ArgumentException($"Prediction at index {i} is null", nameof(predictions));

            if (predictions[i].Length != _outputDim)
                throw new ArgumentException(
                    $"Prediction at index {i} has dimension {predictions[i].Length}, expected {_outputDim}",
                    nameof(predictions));
        }

        _cachedPredictions = predictions;
    }

    public override Vector<T> GetLogits(Vector<T> input)
    {
        throw new InvalidOperationException("Self teacher uses cached predictions, not direct input");
    }

    /// <summary>
    /// Gets a cached prediction by index.
    /// </summary>
    /// <param name="index">Index of the cached prediction.</param>
    /// <returns>The cached logits for the specified index.</returns>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw cached logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public Vector<T> GetCachedPrediction(int index)
    {
        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative");
        if (_cachedPredictions == null || index >= _cachedPredictions.Length)
            throw new InvalidOperationException("Predictions not cached or index out of range");
        return _cachedPredictions[index];
    }
}
