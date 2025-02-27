namespace AiDotNet.Models.Options;

public class ARIMAXModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int AROrder { get; set; } = 1;
    public int MAOrder { get; set; } = 1;
    public int DifferenceOrder { get; set; } = 0;
    public int ExogenousVariables { get; set; } = 1;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}