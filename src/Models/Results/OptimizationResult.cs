namespace AiDotNet.Models.Results;

public class OptimizationResult<T>
{
    public ISymbolicModel<T> BestSolution { get; set; }
    public T BestIntercept { get; set; }
    public T BestFitnessScore { get; set; }
    public int Iterations { get; set; }
    public Vector<T> FitnessHistory { get; set; }
    public List<Vector<T>> SelectedFeatures { get; set; }

    public DatasetResult TrainingResult { get; set; }
    public DatasetResult ValidationResult { get; set; }
    public DatasetResult TestResult { get; set; }

    public FitDetectorResult<T> FitDetectionResult { get; set; }

    public Vector<T> CoefficientLowerBounds { get; set; }
    public Vector<T> CoefficientUpperBounds { get; set; }

    private readonly INumericOperations<T> _numOps;

    public OptimizationResult()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        BestSolution = SymbolicModelFactory<T>.CreateRandomModel(false, 1); // Default to vector model
        FitnessHistory = Vector<T>.Empty();
        SelectedFeatures = [];
        TrainingResult = new DatasetResult();
        ValidationResult = new DatasetResult();
        TestResult = new DatasetResult();
        FitDetectionResult = new FitDetectorResult<T>();
        CoefficientLowerBounds = Vector<T>.Empty();
        CoefficientUpperBounds = Vector<T>.Empty();
        BestIntercept = _numOps.Zero;
        BestFitnessScore = _numOps.Zero;
    }

    public class DatasetResult
    {
        public Matrix<T> X { get; set; }
        public Vector<T> Y { get; set; }
        public Vector<T> Predictions { get; set; }
        public ErrorStats<T> ErrorStats { get; set; }
        public PredictionStats<T> PredictionStats { get; set; }
        public BasicStats<T> ActualBasicStats { get; set; }
        public BasicStats<T> PredictedBasicStats { get; set; }

        public DatasetResult()
        {
            X = Matrix<T>.Empty();
            Y = Vector<T>.Empty();
            Predictions = Vector<T>.Empty();
            ErrorStats = ErrorStats<T>.Empty();
            PredictionStats = PredictionStats<T>.Empty();
            ActualBasicStats = BasicStats<T>.Empty();
            PredictedBasicStats = BasicStats<T>.Empty();
        }
    }
}