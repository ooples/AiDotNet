namespace AiDotNet.OutlierRemoval;

public class IQROutlierRemoval<T> : IOutlierRemoval<T>
{
    private readonly T _iqrMultiplier;
    private readonly INumericOperations<T> _numOps;

    public IQROutlierRemoval(T iqrMultiplier)
    {
        _iqrMultiplier = iqrMultiplier;
        _numOps = MathHelper.GetNumericOperations<T>();
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
                var quartiles = new Quartile<T>(column, _numOps);
                var q1 = quartiles.Q1;
                var q3 = quartiles.Q3;
                var iqr = _numOps.Subtract(q3, q1);
                var lowerBound = _numOps.Subtract(q1, _numOps.Multiply(_iqrMultiplier, iqr));
                var upperBound = _numOps.Add(q3, _numOps.Multiply(_iqrMultiplier, iqr));

                if (_numOps.LessThan(column[i], lowerBound) || _numOps.GreaterThan(column[i], upperBound))
                {
                    isOutlier = true;
                    break;
                }
            }

            if (!isOutlier)
            {
                cleanedInputs.Add(new Vector<T>(inputs.GetRow(i), _numOps));
                cleanedOutputs.Add(outputs[i]);
            }
        }

        return (new Matrix<T>(cleanedInputs, _numOps), new Vector<T>(cleanedOutputs, _numOps));
    }
}