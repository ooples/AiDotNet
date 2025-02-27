namespace AiDotNet.Models.Options;

public class TransferFunctionOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int AROrder { get; set; } = 1;
    public int MAOrder { get; set; } = 1;
    public int InputLagOrder { get; set; } = 1;
    public int OutputLagOrder { get; set; } = 1;
    public IOptimizer<T>? Optimizer { get; set; }
}