using AiDotNet.Models.Options;

namespace AiDotNet.Statistics;

public class ModelStats<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ModelStatsOptions _options;

    public int FeatureCount { get; private set; }
    public Matrix<T> CorrelationMatrix { get; private set; }
    public List<T> VIFList { get; private set; }
    public T ConditionNumber { get; private set; }
    public T LogPointwisePredictiveDensity { get; set; }
    public List<T> LeaveOneOutPredictiveDensities { get; set; }
    public T ObservedTestStatistic { get; set; }
    public List<T> PosteriorPredictiveSamples { get; set; }
    public T MarginalLikelihood { get; set; }
    public T ReferenceModelMarginalLikelihood { get; set; }
    public T LogLikelihood { get; set; }
    public T EffectiveNumberOfParameters { get; set; }
    public Vector<T> Actual { get; }
    public Vector<T> Predicted { get; }
    public Matrix<T> FeatureMatrix { get; }
    public IPredictiveModel<T>? Model { get; }
    public List<string> FeatureNames { get; set; }
    public Dictionary<string, Vector<T>> FeatureValues { get; set; }

    public ModelStats(ModelStatsInputs<T> inputs, ModelStatsOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new ModelStatsOptions(); // Use default options if not provided
        FeatureCount = inputs.FeatureCount;
        ConditionNumber = _numOps.Zero;
        VIFList = [];
        CorrelationMatrix = Matrix<T>.Empty();
        LogPointwisePredictiveDensity = _numOps.Zero;
        LeaveOneOutPredictiveDensities = [];
        ObservedTestStatistic = _numOps.Zero;
        PosteriorPredictiveSamples = [];
        MarginalLikelihood = _numOps.Zero;
        ReferenceModelMarginalLikelihood = _numOps.Zero;
        LogLikelihood = _numOps.Zero;
        EffectiveNumberOfParameters = _numOps.Zero;
        Actual = inputs.Actual;
        Predicted = inputs.Predicted;
        FeatureMatrix = inputs.XMatrix;
        Model = inputs.Model;
        FeatureNames = inputs.FeatureNames ?? [];
        FeatureValues = inputs.FeatureValues ?? [];

        CalculateModelStats(inputs);
    }

    public static ModelStats<T> Empty()
    {
        return new ModelStats<T>(new());
    }

    private void CalculateModelStats(ModelStatsInputs<T> inputs)
    {
        var matrix = inputs.XMatrix;

        CorrelationMatrix = StatisticsHelper<T>.CalculateCorrelationMatrix(matrix, _options);
        VIFList = StatisticsHelper<T>.CalculateVIF(matrix, _options);
        ConditionNumber = StatisticsHelper<T>.CalculateConditionNumber(matrix, _options);
        LogPointwisePredictiveDensity = StatisticsHelper<T>.CalculateLogPointwisePredictiveDensity(inputs.Actual, inputs.Predicted);

        if (inputs.FitFunction != null)
        {
            LeaveOneOutPredictiveDensities = StatisticsHelper<T>.CalculateLeaveOneOutPredictiveDensities(matrix, inputs.Actual, inputs.FitFunction);
        }
        
        ObservedTestStatistic = StatisticsHelper<T>.CalculateObservedTestStatistic(inputs.Actual, inputs.Predicted);
        PosteriorPredictiveSamples = StatisticsHelper<T>.CalculatePosteriorPredictiveSamples(inputs.Actual, inputs.Predicted, inputs.FeatureCount);
        MarginalLikelihood = StatisticsHelper<T>.CalculateMarginalLikelihood(inputs.Actual, inputs.Predicted, inputs.FeatureCount);
        ReferenceModelMarginalLikelihood = StatisticsHelper<T>.CalculateReferenceModelMarginalLikelihood(inputs.Actual);
        LogLikelihood = StatisticsHelper<T>.CalculateLogLikelihood(inputs.Actual, inputs.Predicted);
        EffectiveNumberOfParameters = StatisticsHelper<T>.CalculateEffectiveNumberOfParameters(matrix, inputs.Coefficients);
    }
}