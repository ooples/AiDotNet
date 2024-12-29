namespace AiDotNet.Helpers;

public static class TimeSeriesHelper<T>
{
    public static Vector<T> DifferenceSeries(Vector<T> y, int d, INumericOperations<T> numOps)
    {
        Vector<T> result = y;
        for (int i = 0; i < d; i++)
        {
            Vector<T> temp = new Vector<T>(result.Length - 1, numOps);
            for (int j = 1; j < result.Length; j++)
            {
                temp[j - 1] = numOps.Subtract(result[j], result[j - 1]);
            }
            result = temp;
        }

        return result;
    }

    public static Vector<T> EstimateARCoefficients(Vector<T> y, int p, MatrixDecompositionType decompositionType, INumericOperations<T> numOps)
    {
        Matrix<T> X = new Matrix<T>(y.Length - p, p, numOps);
        Vector<T> Y = new Vector<T>(y.Length - p, numOps);

        for (int i = p; i < y.Length; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i - p, j] = y[i - j - 1];
            }
            Y[i - p] = y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(X, Y, decompositionType);
    }

    public static Vector<T> CalculateARResiduals(Vector<T> y, Vector<T> arCoefficients, INumericOperations<T> numOps)
    {
        int n = y.Length;
        int p = arCoefficients.Length;
        Vector<T> residuals = new Vector<T>(n - p, numOps);

        for (int i = p; i < n; i++)
        {
            T predicted = numOps.Zero;
            for (int j = 0; j < p; j++)
            {
                predicted = numOps.Add(predicted, numOps.Multiply(arCoefficients[j], y[i - j - 1]));
            }
            residuals[i - p] = numOps.Subtract(y[i], predicted);
        }

        return residuals;
    }

    public static Vector<T> EstimateMACoefficients(Vector<T> residuals, int q, INumericOperations<T> numOps)
    {
        Vector<T> maCoefficients = new Vector<T>(q, numOps);
        for (int i = 0; i < q; i++)
        {
            maCoefficients[i] = CalculateAutoCorrelation(residuals, i + 1, numOps);
        }

        return maCoefficients;
    }

    public static T CalculateAutoCorrelation(Vector<T> y, int lag, INumericOperations<T> numOps)
    {
        T sum = numOps.Zero;
        T sumSquared = numOps.Zero;
        int n = y.Length;

        for (int i = 0; i < n - lag; i++)
        {
            sum = numOps.Add(sum, numOps.Multiply(y[i], y[i + lag]));
            sumSquared = numOps.Add(sumSquared, numOps.Multiply(y[i], y[i]));
        }

        return numOps.Divide(sum, sumSquared);
    }
}