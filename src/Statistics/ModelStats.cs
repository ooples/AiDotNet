using System;

namespace AiDotNet.Statistics;

public class ModelStats<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ModelStatsOptions _options;

    public int FeatureCount { get; private set; }
    public Matrix<T> CorrelationMatrix { get; private set; }
    public Matrix<T> CovarianceMatrix { get; private set; }
    public List<T> VIFList { get; private set; }
    public T ConditionNumber { get; private set; }
    public T LogPointwisePredictiveDensity { get; private set; }
    public List<T> LeaveOneOutPredictiveDensities { get; private set; }
    public T ObservedTestStatistic { get; private set; }
    public List<T> PosteriorPredictiveSamples { get; private set; }
    public T MarginalLikelihood { get; private set; }
    public T ReferenceModelMarginalLikelihood { get; private set; }
    public T LogLikelihood { get; private set; }
    public T EffectiveNumberOfParameters { get; private set; }
    public Vector<T> Actual { get; }
    public Vector<T> Predicted { get; }
    public Matrix<T> FeatureMatrix { get; }
    public IPredictiveModel<T>? Model { get; }
    public List<string> FeatureNames { get; private set; }
    public Dictionary<string, Vector<T>> FeatureValues { get; private set; }
    public T EuclideanDistance { get; private set; }
    public T ManhattanDistance { get; private set; }
    public T CosineSimilarity { get; private set; }
    public T JaccardSimilarity { get; private set; }
    public T HammingDistance { get; private set; }
    public T MahalanobisDistance { get; private set; }
    public T MutualInformation { get; private set; }
    public T NormalizedMutualInformation { get; private set; }
    public T VariationOfInformation { get; private set; }
    public T SilhouetteScore { get; private set; }
    public T CalinskiHarabaszIndex { get; private set; }
    public T DaviesBouldinIndex { get; private set; }
    public T MeanAveragePrecision { get; private set; }
    public T NormalizedDiscountedCumulativeGain { get; private set; }
    public T MeanReciprocalRank { get; private set; }
    public Vector<T> AutoCorrelationFunction { get; private set; }
    public Vector<T> PartialAutoCorrelationFunction { get; private set; }

    public ModelStats(ModelStatsInputs<T> inputs, ModelStatsOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new ModelStatsOptions(); // Use default options if not provided
        FeatureCount = inputs.FeatureCount;
        ConditionNumber = _numOps.Zero;
        VIFList = [];
        CorrelationMatrix = Matrix<T>.Empty();
        CovarianceMatrix = Matrix<T>.Empty();
        AutoCorrelationFunction = Vector<T>.Empty();
        PartialAutoCorrelationFunction = Vector<T>.Empty();
        MeanAveragePrecision = _numOps.Zero;
        NormalizedDiscountedCumulativeGain = _numOps.Zero;
        MeanReciprocalRank = _numOps.Zero;
        SilhouetteScore = _numOps.Zero;
        CalinskiHarabaszIndex = _numOps.Zero;
        DaviesBouldinIndex = _numOps.Zero;
        MutualInformation = _numOps.Zero;
        NormalizedMutualInformation = _numOps.Zero;
        VariationOfInformation = _numOps.Zero;
        EuclideanDistance = _numOps.Zero;
        ManhattanDistance = _numOps.Zero;
        CosineSimilarity = _numOps.Zero;
        JaccardSimilarity = _numOps.Zero;
        HammingDistance = _numOps.Zero;
        MahalanobisDistance = _numOps.Zero;
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
        var actual = inputs.Actual;
        var predicted = inputs.Predicted;
        var featureCount = inputs.FeatureCount;

        CorrelationMatrix = StatisticsHelper<T>.CalculateCorrelationMatrix(matrix, _options);
        CovarianceMatrix = StatisticsHelper<T>.CalculateCovarianceMatrix(matrix);
        VIFList = StatisticsHelper<T>.CalculateVIF(matrix, _options);
        ConditionNumber = StatisticsHelper<T>.CalculateConditionNumber(matrix, _options);
        LogPointwisePredictiveDensity = StatisticsHelper<T>.CalculateLogPointwisePredictiveDensity(actual, predicted);

        if (inputs.FitFunction != null)
        {
            LeaveOneOutPredictiveDensities = StatisticsHelper<T>.CalculateLeaveOneOutPredictiveDensities(matrix, actual, inputs.FitFunction);
        }
        
        ObservedTestStatistic = StatisticsHelper<T>.CalculateObservedTestStatistic(actual, predicted);
        PosteriorPredictiveSamples = StatisticsHelper<T>.CalculatePosteriorPredictiveSamples(actual, predicted, featureCount);
        MarginalLikelihood = StatisticsHelper<T>.CalculateMarginalLikelihood(actual, predicted, featureCount);
        ReferenceModelMarginalLikelihood = StatisticsHelper<T>.CalculateReferenceModelMarginalLikelihood(actual);
        LogLikelihood = StatisticsHelper<T>.CalculateLogLikelihood(actual, predicted);
        EffectiveNumberOfParameters = StatisticsHelper<T>.CalculateEffectiveNumberOfParameters(matrix, inputs.Coefficients);
        MutualInformation = StatisticsHelper<T>.CalculateMutualInformation(actual, predicted);
        NormalizedMutualInformation = StatisticsHelper<T>.CalculateNormalizedMutualInformation(actual, predicted);
        VariationOfInformation = StatisticsHelper<T>.CalculateVariationOfInformation(actual, predicted);
        SilhouetteScore = StatisticsHelper<T>.CalculateSilhouetteScore(matrix, predicted);
        CalinskiHarabaszIndex = StatisticsHelper<T>.CalculateCalinskiHarabaszIndex(matrix, predicted);
        DaviesBouldinIndex = StatisticsHelper<T>.CalculateDaviesBouldinIndex(matrix, predicted);
        MeanAveragePrecision = StatisticsHelper<T>.CalculateMeanAveragePrecision(actual, predicted, _options.MapTopK);
        NormalizedDiscountedCumulativeGain = StatisticsHelper<T>.CalculateNDCG(actual, predicted, _options.NdcgTopK);
        MeanReciprocalRank = StatisticsHelper<T>.CalculateMeanReciprocalRank(actual, predicted);

        var residuals = StatisticsHelper<T>.CalculateResiduals(inputs.Actual, inputs.Predicted);
        AutoCorrelationFunction = StatisticsHelper<T>.CalculateAutoCorrelationFunction(residuals, _options.AcfMaxLag);
        PartialAutoCorrelationFunction = StatisticsHelper<T>.CalculatePartialAutoCorrelationFunction(residuals, _options.PacfMaxLag);

        EuclideanDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Euclidean);
        ManhattanDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Manhattan);
        CosineSimilarity = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Cosine);
        JaccardSimilarity = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Jaccard);
        HammingDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Hamming);
        MahalanobisDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Mahalanobis, CovarianceMatrix);
    }
}