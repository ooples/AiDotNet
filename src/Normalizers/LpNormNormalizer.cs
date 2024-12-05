namespace AiDotNet.Normalizers;

public class LpNormNormalizer(double p = 2) : INormalizer
{
    private readonly double _p = p;

    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double norm = Math.Pow(vector.Select(x => Math.Pow(Math.Abs(x), _p)).Sum(), 1 / _p);

        var normalizedVector = vector.Divide(norm);
        var parameters = new NormalizationParameters { Scale = norm, P = _p, Method = NormalizationMethod.LpNorm };
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
        return vector.Multiply(parameters.Scale);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select(p => yParams.Scale / p.Scale).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return 0;
    }
}