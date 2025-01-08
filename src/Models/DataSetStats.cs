namespace AiDotNet.Models;

public class DataSetStats<T>
{
    public ErrorStats<T> ErrorStats { get; set; } = ErrorStats<T>.Empty();
    public BasicStats<T> ActualBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> PredictedBasicStats { get; set; } = BasicStats<T>.Empty();
    public PredictionStats<T> PredictionStats { get; set; } = PredictionStats<T>.Empty();
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();
    public Matrix<T> Features { get; set; } = Matrix<T>.Empty();
}