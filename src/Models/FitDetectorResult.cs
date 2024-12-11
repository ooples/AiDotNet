namespace AiDotNet.Models;

public class FitDetectorResult
{
    public FitType FitType { get; set; }
    public double ConfidenceLevel { get; set; }
    public List<string> Recommendations { get; set; } = [];
}