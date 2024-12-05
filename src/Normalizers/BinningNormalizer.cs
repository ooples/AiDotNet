namespace AiDotNet.Normalizers;

public class BinningNormalizer : INormalizer
{
    private const int DefaultBinCount = 10;

    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);

        var bins = new List<double>();
        for (int i = 0; i <= DefaultBinCount; i++)
        {
            bins.Add(sortedVector[(int)((i / (double)DefaultBinCount) * (sortedVector.Length - 1))]);
        }

        var normalizedVector = vector.Transform(x => 
        {
            int binIndex = bins.FindIndex(b => x <= b);
            return (double)(binIndex == -1 ? DefaultBinCount - 1 : binIndex) / (DefaultBinCount - 1);
        });

        var parameters = new NormalizationParameters { Method = NormalizationMethod.Binning, Bins = bins };
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
        if (parameters.Bins == null || parameters.Bins.Count == 0)
        {
            throw new ArgumentException("Invalid bin parameters. Bins list is null or empty.");
        }

        return vector.Transform(x => 
        {
            // Ensure x is within [0, 1] range
            x = Math.Max(0, Math.Min(1, x));
        
            // Calculate the bin index
            int binIndex = (int)(x * (parameters.Bins.Count - 1));
        
            // Ensure binIndex is within valid range
            binIndex = Math.Max(0, Math.Min(parameters.Bins.Count - 2, binIndex));

            // Return the average of the current bin and the next bin
            return (parameters.Bins[binIndex] + parameters.Bins[binIndex + 1]) / 2;
        });
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        // Denormalizing coefficients for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select((p, i) => 
            (yParams.Bins.Last() - yParams.Bins.First()) / (p.Bins.Last() - p.Bins.First())).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        // Denormalizing y-intercept for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        double denormalizedIntercept = (yParams.Bins.First() + yParams.Bins.Last()) / 2;
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept -= coefficients[i] * ((xParams[i].Bins.First() + xParams[i].Bins.Last()) / 2);
        }

        return denormalizedIntercept;
    }
}