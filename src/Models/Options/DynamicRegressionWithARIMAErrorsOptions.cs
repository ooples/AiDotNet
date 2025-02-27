namespace AiDotNet.Models.Options;

public class DynamicRegressionWithARIMAErrorsOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int ExternalRegressors { get; set; } = 1;
    public int AROrder { get; set; } = 1;
    public int MAOrder { get; set; } = 1;
    public int DifferenceOrder { get; set; } = 0;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
    public IRegularization<T>? Regularization { get; set; }
}