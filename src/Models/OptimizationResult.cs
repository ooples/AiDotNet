namespace AiDotNet.Models;

public class OptimizationResult
{
    public Vector<double> BestCoefficients { get; set; } = Vector<double>.Empty();
    public double BestIntercept { get; set; }
    public double FitnessScore { get; set; }
    public int Iterations { get; set; }
    public Vector<double> FitnessHistory { get; set; } = Vector<double>.Empty();
    public List<Vector<double>> SelectedFeatures { get; set; } = [];
    
    public DatasetResult TrainingResult { get; set; } = new();
    public DatasetResult ValidationResult { get; set; } = new();
    public DatasetResult TestResult { get; set; } = new();
    
    public FitDetectorResult FitDetectionResult { get; set; } = new();
    
    public Vector<double> CoefficientLowerBounds { get; set; } = Vector<double>.Empty();
    public Vector<double> CoefficientUpperBounds { get; set; } = Vector<double>.Empty();

    public class DatasetResult
    {
        public Matrix<double> X { get; set; } = Matrix<double>.Empty();
        public Vector<double> Y { get; set; } = Vector<double>.Empty();
        public Vector<double> Predictions { get; set; } = Vector<double>.Empty();
        public ErrorStats ErrorStats { get; set; } = ErrorStats.Empty();
        public BasicStats BasicStats { get; set; } = BasicStats.Empty();
    }
}