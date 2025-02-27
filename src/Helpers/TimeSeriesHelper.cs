namespace AiDotNet.Helpers;

public static class TimeSeriesHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static Vector<T> DifferenceSeries(Vector<T> y, int d)
    {
        Vector<T> result = y;
        for (int i = 0; i < d; i++)
        {
            Vector<T> temp = new Vector<T>(result.Length - 1);
            for (int j = 1; j < result.Length; j++)
            {
                temp[j - 1] = NumOps.Subtract(result[j], result[j - 1]);
            }
            result = temp;
        }

        return result;
    }

    public static Vector<T> EstimateARCoefficients(Vector<T> y, int p, MatrixDecompositionType decompositionType)
    {
        Matrix<T> X = new Matrix<T>(y.Length - p, p);
        Vector<T> Y = new Vector<T>(y.Length - p);

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

    public static Vector<T> CalculateARResiduals(Vector<T> y, Vector<T> arCoefficients)
    {
        int n = y.Length;
        int p = arCoefficients.Length;
        Vector<T> residuals = new Vector<T>(n - p);

        for (int i = p; i < n; i++)
        {
            T predicted = NumOps.Zero;
            for (int j = 0; j < p; j++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(arCoefficients[j], y[i - j - 1]));
            }
            residuals[i - p] = NumOps.Subtract(y[i], predicted);
        }

        return residuals;
    }

    public static Vector<T> EstimateMACoefficients(Vector<T> residuals, int q)
    {
        Vector<T> maCoefficients = new Vector<T>(q);
        for (int i = 0; i < q; i++)
        {
            maCoefficients[i] = CalculateAutoCorrelation(residuals, i + 1);
        }

        return maCoefficients;
    }

    public static T CalculateAutoCorrelation(Vector<T> y, int lag)
    {
        T sum = NumOps.Zero;
        T sumSquared = NumOps.Zero;
        int n = y.Length;

        for (int i = 0; i < n - lag; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(y[i], y[i + lag]));
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(y[i], y[i]));
        }

        return NumOps.Divide(sum, sumSquared);
    }
}