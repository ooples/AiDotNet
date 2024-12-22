namespace AiDotNet.Models;

public class ResidualAnalysisFitDetectorOptions
{
    public double MeanThreshold { get; set; } = 0.1;
    public double StdThreshold { get; set; } = 0.2;
    public double MapeThreshold { get; set; } = 0.1;
    public double R2Threshold { get; set; } = 0.1;
}