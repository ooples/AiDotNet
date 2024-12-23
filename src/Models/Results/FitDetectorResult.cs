namespace AiDotNet.Models.Results;

public class FitDetectorResult<T>
{
    public FitType FitType { get; set; }
    public T? ConfidenceLevel { get; set; }
    public List<string> Recommendations { get; set; }
    public Dictionary<string, object> AdditionalInfo { get; set; } = [];

    public FitDetectorResult()
    {
        Recommendations = [];
    }

    public FitDetectorResult(FitType fitType, T confidenceLevel)
    {
        FitType = fitType;
        ConfidenceLevel = confidenceLevel;
        Recommendations = [];
    }
}