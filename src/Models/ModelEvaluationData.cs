namespace AiDotNet.Models;

public class ModelEvaluationData<T>
{
    public ErrorStats<T> TrainingErrorStats { get; set; } = ErrorStats<T>.Empty();
    public ErrorStats<T> ValidationErrorStats { get; set; } = ErrorStats<T>.Empty();
    public ErrorStats<T> TestErrorStats { get; set; } = ErrorStats<T>.Empty();
    public BasicStats<T> TrainingPredictedBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> ValidationPredictedBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> TestPredictedBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> TrainingActualBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> ValidationActualBasicStats { get; set; } = BasicStats<T>.Empty();
    public BasicStats<T> TestActualBasicStats { get; set; } = BasicStats<T>.Empty();
    public PredictionStats<T> TrainingPredictionStats { get; set; } = PredictionStats<T>.Empty();
    public PredictionStats<T> ValidationPredictionStats { get; set; } = PredictionStats<T>.Empty();
    public PredictionStats<T> TestPredictionStats { get; set; } = PredictionStats<T>.Empty();
    public ModelStats<T> ModelStats { get; set; } = ModelStats<T>.Empty();
}