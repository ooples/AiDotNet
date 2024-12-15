namespace AiDotNet.OutlierRemoval;

public class ZScoreOutlierRemoval<T> : IOutlierRemoval<T>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    public ZScoreOutlierRemoval(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? FindDefaultThreshold();
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
                (var mean, var std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
                var zScore = _numOps.Divide(_numOps.Subtract(column[i], mean), std);

                if (_numOps.GreaterThan(_numOps.Abs(zScore), _threshold))
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

        return (new Matrix<T>(cleanedInputs, _numOps), new Vector<T>(cleanedOutputs, _numOps));
    }

    private T FindDefaultThreshold()
    {
        return _numOps.FromDouble(3.0);
    }
}