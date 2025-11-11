using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Distributed teacher model that aggregates predictions from multiple distributed workers.
/// </summary>
public class DistributedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _workers;
    private readonly AggregationMode _aggregation;

    public override int OutputDimension => _workers[0].OutputDimension;

    public DistributedTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] workers,
        AggregationMode aggregation = AggregationMode.Average)
    {
        _workers = workers ?? throw new ArgumentNullException(nameof(workers));

        // Validate workers array is non-empty
        if (_workers.Length == 0)
            throw new ArgumentException("Workers array cannot be empty", nameof(workers));

        // Validate no worker is null
        for (int i = 0; i < _workers.Length; i++)
        {
            if (_workers[i] == null)
                throw new ArgumentException($"Worker at index {i} is null", nameof(workers));
        }

        // Validate all workers have the same output dimension
        int expectedOutputDim = _workers[0].OutputDimension;
        for (int i = 1; i < _workers.Length; i++)
        {
            if (_workers[i].OutputDimension != expectedOutputDim)
                throw new ArgumentException(
                    $"Worker at index {i} has OutputDimension {_workers[i].OutputDimension}, " +
                    $"but expected {expectedOutputDim} (from worker 0). " +
                    $"All workers must have the same OutputDimension.",
                    nameof(workers));
        }

        _aggregation = aggregation;
    }

    /// <summary>
    /// Gets aggregated logits from all distributed workers.
    /// </summary>
    /// <param name="input">Input data.</param>
    /// <returns>Aggregated logits from all workers.</returns>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw aggregated logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        int n = _workers[0].OutputDimension;
        var aggregated = new Vector<T>(n);

        switch (_aggregation)
        {
            case AggregationMode.Average:
                for (int j = 0; j < n; j++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < _workers.Length; i++)
                    {
                        var logits = _workers[i].GetLogits(input);
                        sum = NumOps.Add(sum, logits[j]);
                    }
                    aggregated[j] = NumOps.Divide(sum, NumOps.FromDouble(_workers.Length));
                }
                break;

            case AggregationMode.Voting:
                // For simplicity, use average as voting
                goto case AggregationMode.Average;
        }

        return aggregated;
    }
}

public enum AggregationMode
{
    Average,
    Voting
}
