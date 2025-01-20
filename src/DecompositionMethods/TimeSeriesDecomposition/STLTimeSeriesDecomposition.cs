namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class STLTimeSeriesDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly STLDecompositionOptions<T> _options;

    public STLTimeSeriesDecomposition(Vector<T> timeSeries, STLDecompositionOptions<T> options)
        : base(timeSeries)
    {
        _options = options;
        Decompose();
    }

    public void Decompose()
    {
        var stlDecomposition = new STLDecomposition<T>(_options);
        stlDecomposition.Train(new Matrix<T>(TimeSeries.Length, 1, NumOps), TimeSeries);

        Vector<T> trend = stlDecomposition.GetTrend();
        Vector<T> seasonal = stlDecomposition.GetSeasonal();
        Vector<T> residual = stlDecomposition.GetResidual();

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }
}