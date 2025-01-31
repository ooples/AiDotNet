namespace AiDotNet.OutlierRemoval;

public class MADOutlierRemoval<T> : IOutlierRemoval<T>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    public MADOutlierRemoval(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? GetDefaultThreshold();
    }

    public (Matrix<T> CleanedInputs, Vector<T> CleanedOutputs) RemoveOutliers(Matrix<T> inputs, Vector<T> outputs)
    {
        var cleanedInputs = new List<Vector<T>>();
        var cleanedOutputs = new List<T>();

        for (int i = 0; i < outputs.Length; i++)
        {
            bool isOutlier = false;
            for (int j = 0; j < inputs.Columns; j++)
            {
                var column = inputs.GetColumn(j);
                var median = StatisticsHelper<T>.CalculateMedian(column);
                var deviations = column.Select(x => _numOps.Abs(_numOps.Subtract(x, median)));
                var mad = StatisticsHelper<T>.CalculateMedian(deviations);
                var modifiedZScore = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(0.6745), _numOps.Subtract(column[i], median)), mad);

                if (_numOps.GreaterThan(_numOps.Abs(modifiedZScore), _threshold))
                {
                    isOutlier = true;
                    break;
                }
            }

            if (!isOutlier)
            {
                cleanedInputs.Add(inputs.GetRow(i));
                cleanedOutputs.Add(outputs[i]);
            }
        }

        return (new Matrix<T>(cleanedInputs), new Vector<T>(cleanedOutputs));
    }

    private T GetDefaultThreshold()
    {
        return _numOps.FromDouble(3.5);
    }
}