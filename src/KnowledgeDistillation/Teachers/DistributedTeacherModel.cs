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
        _aggregation = aggregation;
    }

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

    protected override Vector<T> ApplyTemperatureSoftmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }
}

public enum AggregationMode
{
    Average,
    Voting
}
