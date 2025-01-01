namespace AiDotNet.Helpers;

public static class RegressionHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static (Matrix<T> xScaled, Vector<T> yScaled, Vector<T> xMean, Vector<T> xStd, T yStd) CenterAndScale(Matrix<T> x, Vector<T> y)
    {
        Vector<T> xMean = new(x.Columns, NumOps);
        for (int j = 0; j < x.Columns; j++)
        {
            xMean[j] = x.GetColumn(j).Mean();
        }

        T yMean = y.Mean();
        Vector<T> yMeanVector = new([yMean], NumOps);

        Vector<T> xStd = new(x.Columns, NumOps);
        for (int j = 0; j < x.Columns; j++)
        {
            xStd[j] = StatisticsHelper<T>.CalculateStandardDeviation(x.GetColumn(j));
        }

        T yStd = StatisticsHelper<T>.CalculateStandardDeviation(y);

        Matrix<T> xScaled = new(x.Rows, x.Columns, NumOps);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                xScaled[i, j] = NumOps.Divide(NumOps.Subtract(x[i, j], xMean[j]), xStd[j]);
            }
        }

        Vector<T> yScaled = y.Transform(yi => NumOps.Divide(NumOps.Subtract(yi, yMean), yStd));

        return (xScaled, yScaled, xMean, xStd, yStd);
    }
}