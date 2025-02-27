namespace AiDotNet.Models.Options;

public class InterventionAnalysisOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int AROrder { get; set; } = 1;
    public int MAOrder { get; set; } = 1;
    public List<InterventionInfo> Interventions { get; set; } = [];
    public IOptimizer<T>? Optimizer { get; set; }
}