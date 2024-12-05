namespace AiDotNet.Normalizers;

public class LogMeanVarianceNormalizer : INormalizer
{
    private const double Epsilon = 1e-10;

    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double minValue = vector.Min();
        double shift = minValue > 0 ? 0 : -minValue + 1 + Epsilon;

        var logVector = vector.Transform(x => Math.Log(x + shift));
        double mean = logVector.Average();
        
        double variance = logVector.Select(x => Math.Pow(x - mean, 2)).Average();
        double stdDev = Math.Sqrt(Math.Max(variance, Epsilon));

        var normalizedVector = logVector.Transform(x => (x - mean) / stdDev);
        normalizedVector = normalizedVector.Transform(x => double.IsNaN(x) ? 0 : x);

        var parameters = new NormalizationParameters { Shift = shift, Mean = mean, StdDev = stdDev };
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
        return vector
            .Multiply(parameters.StdDev)
            .Add(parameters.Mean)
            .Transform(x => Math.Exp(x) - parameters.Shift);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(
            Vector<double>.FromArray(xParams.Select(p => yParams.StdDev / Math.Max(p.StdDev, Epsilon)).ToArray())
        );
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double denormalizedLogIntercept = yParams.Mean;
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedLogIntercept -= coefficients[i] * (xParams[i].Mean * yParams.StdDev / Math.Max(xParams[i].StdDev, Epsilon));
        }

        return Math.Exp(denormalizedLogIntercept) - yParams.Shift;
    }
}