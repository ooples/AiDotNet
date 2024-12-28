namespace AiDotNet.Models.Results;

public struct ModelResult<T>
{
    public ISymbolicModel<T> Solution { get; set; }
    public T Fitness { get; set; }
    public FitDetectorResult<T> FitDetectionResult { get; set; }
    public Vector<T> TrainingPredictions { get; set; }
    public Vector<T> ValidationPredictions { get; set; }
    public Vector<T> TestPredictions { get; set; }
    public ModelEvaluationData<T> EvaluationData { get; set; }
    public List<Vector<T>> SelectedFeatures { get; set; }
    public Matrix<T> TrainingFeatures { get; set; }
    public Matrix<T> ValidationFeatures { get; set; }
    public Matrix<T> TestFeatures { get; set; }
}