namespace AiDotNet.Models.Options;

public class VARModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int Lag { get; set; } = 1;
    public int OutputDimension { get; set; } = 1;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
}