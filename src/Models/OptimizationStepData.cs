namespace AiDotNet.Models;

public class OptimizationStepData<T>
{
    public ISymbolicModel<T> Solution { get; set; }
    public List<Vector<T>> SelectedFeatures { get; set; } = [];
    public Matrix<T> XTrainSubset { get; set; } = Matrix<T>.Empty();
    public Matrix<T> XValSubset { get; set; } = Matrix<T>.Empty();
    public Matrix<T> XTestSubset { get; set; } = Matrix<T>.Empty();
    public T FitnessScore { get; set; }
    public FitDetectorResult<T> FitDetectionResult { get; set; } = new();
    public ModelEvaluationData<T> EvaluationData { get; set; } = new();

    public OptimizationStepData()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        FitnessScore = numOps.Zero;
        Solution = new VectorModel<T>(Vector<T>.Empty());
    }
}