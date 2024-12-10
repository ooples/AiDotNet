namespace AiDotNet.Models;

public class BasicStats
{
    public double Mean { get; set; }
    public double Variance { get; set; }
    public double StandardDeviation { get; set; }
    public double Skewness { get; set; }
    public double Kurtosis { get; set; }
    public double Min { get; set; }
    public double Max { get; set; }
    public int N { get; set; }
    public double Median { get; set; }
    public double UpperConfidenceLevel { get; set; }
    public double LowerConfidenceLevel { get; set; }
    public double UpperCredibleLevel { get; set; }
    public double LowerCredibleLevel { get; set; }
    public DistributionFitResult BestDistributionFit { get; set; } = new();
}