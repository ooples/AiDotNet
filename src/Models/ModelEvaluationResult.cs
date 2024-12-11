namespace AiDotNet.Models;

public class ModelEvaluationResult
{
    public Dictionary<string, double> TrainingMetrics { get; set; } = [];
    public Dictionary<string, double> ValidationMetrics { get; set; } = [];
    public Dictionary<string, double> TestMetrics { get; set; } = [];
}