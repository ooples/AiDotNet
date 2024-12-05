namespace AiDotNet.Normalizers;

public class RobustScalingNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double median = CalculateMedian(vector);
        double q1 = CalculateQuantile(vector, 0.25);
        double q3 = CalculateQuantile(vector, 0.75);
        double iqr = q3 - q1;
        if (iqr == 0) iqr = 1;

        var normalizedVector = vector.Subtract(median).Divide(iqr);
        var parameters = new NormalizationParameters { Median = median, IQR = iqr, Method = NormalizationMethod.RobustScaling };

        return (normalizedVector, parameters);
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var normalizedMatrix = Matrix<double>.CreateZeros(matrix.Rows, matrix.Columns);
        var parametersList = new List<NormalizationParameters>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedMatrix.SetColumn(i, normalizedColumn);
            parametersList.Add(parameters);
        }

        return (normalizedMatrix, parametersList);
    }

    public Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters)
    {
        return vector.Multiply(parameters.IQR).Add(parameters.Median);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select(p => yParams.IQR / p.IQR).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double denormalizedIntercept = yParams.Median;
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept -= coefficients[i] * xParams[i].Median * (yParams.IQR / xParams[i].IQR);
        }

        return denormalizedIntercept;
    }

    private static double CalculateMedian(Vector<double> vector)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);
        int n = sortedVector.Length;
        if (n % 2 == 0)
        {
            return (sortedVector[n / 2 - 1] + sortedVector[n / 2]) / 2;
        }

        return sortedVector[n / 2];
    }

    private static double CalculateQuantile(Vector<double> vector, double quantile)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);
        int n = sortedVector.Length;
        double index = quantile * (n - 1);
        int lowerIndex = (int)Math.Floor(index);
        int upperIndex = (int)Math.Ceiling(index);

        if (lowerIndex == upperIndex)
        {
            return sortedVector[lowerIndex];
        }

        double lowerValue = sortedVector[lowerIndex];
        double upperValue = sortedVector[upperIndex];

        return lowerValue + (upperValue - lowerValue) * (index - lowerIndex);
    }
}