namespace AiDotNet.Models.Results;

public struct ModelResult<T>
{
    public ISymbolicModel<T> Solution { get; set; }
    public T Fitness { get; set; }
    public FitDetectorResult<T> FitDetectionResult { get; set; }
    public ModelEvaluationData<T> EvaluationData { get; set; }
    public List<Vector<T>> SelectedFeatures { get; set; }
}