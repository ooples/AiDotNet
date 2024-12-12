namespace AiDotNet.Models;

public class ModelEvaluationResult<T>
{
    public Dictionary<string, T> TrainingMetrics { get; set; } = [];
    public Dictionary<string, T> ValidationMetrics { get; set; } = [];
    public Dictionary<string, T> TestMetrics { get; set; } = [];
}