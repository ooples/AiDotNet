﻿namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Removes outliers from the data using the threshold method. This method is not recommended for data sets with less than 15 data points.
/// </summary>
public class ThresholdOutlierRemoval<T> : IOutlierRemoval<T>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    public ThresholdOutlierRemoval(T threshold)
    {
        _threshold = threshold;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Matrix<T> CleanedInputs, Vector<T> CleanedOutputs) RemoveOutliers(Matrix<T> inputs, Vector<T> outputs)
    {
        var cleanedInputs = new List<Vector<T>>();
        var cleanedOutputs = new List<T>();

        for (int j = 0; j < inputs.Columns; j++)
        {
            var column = inputs.GetColumn(j);
            var median = StatisticsHelper<T>.CalculateMedian(column);
            var deviations = column.Select(x => _numOps.Abs(_numOps.Subtract(x, median))).OrderBy(x => x).ToList();
            var medianDeviation = deviations[deviations.Count / 2];
            var threshold = _numOps.Multiply(_threshold, medianDeviation);

            for (int i = 0; i < inputs.Rows; i++)
            {
                if (_numOps.LessThanOrEquals(_numOps.Abs(_numOps.Subtract(inputs[i, j], median)), threshold))
                {
                    if (j == 0) // Only add to cleaned data once per row
                    {
                        cleanedInputs.Add(inputs.GetRow(i));
                        cleanedOutputs.Add(outputs[i]);
                    }
                }
            }
        }

        return (new Matrix<T>(cleanedInputs, _numOps), new Vector<T>(cleanedOutputs, _numOps));
    }
}