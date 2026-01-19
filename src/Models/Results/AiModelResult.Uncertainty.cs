using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.UncertaintyQuantification.Interfaces;
using AiDotNet.UncertaintyQuantification.Layers;
using Newtonsoft.Json;

namespace AiDotNet.Models.Results;

public partial class AiModelResult<T, TInput, TOutput>
{
    private const string PredictiveEntropyMetricKey = "predictive_entropy";
    private const string MutualInformationMetricKey = "mutual_information";

    private readonly INumericOperations<T> _uqNumOps = MathHelper.GetNumericOperations<T>();
    private readonly object _monteCarloDropoutLock = new();
    private readonly object _parameterSamplingLock = new();

    [JsonProperty]
    private List<IFullModel<T, TInput, TOutput>>? _deepEnsembleModels;

    [JsonProperty]
    internal UncertaintyQuantificationOptions? UncertaintyQuantificationOptions { get; private set; }

    [JsonProperty]
    internal bool HasConformalRegression { get; private set; }

    [JsonProperty]
    internal T ConformalRegressionQuantile { get; private set; } = default!;

    [JsonProperty]
    internal bool HasConformalClassification { get; private set; }

    [JsonProperty]
    internal T ConformalClassificationThreshold { get; private set; } = default!;

    [JsonProperty]
    internal int ConformalClassificationNumClasses { get; private set; }

    [JsonProperty]
    internal bool HasAdaptiveConformalClassification { get; private set; }

    [JsonProperty]
    internal double[]? ConformalClassificationAdaptiveBinEdges { get; private set; }

    [JsonProperty]
    internal Vector<T>? ConformalClassificationAdaptiveThresholds { get; private set; }

    [JsonProperty]
    internal bool HasTemperatureScaling { get; private set; }

    [JsonProperty]
    internal T TemperatureScalingTemperature { get; private set; } = default!;

    [JsonProperty]
    internal bool HasPlattScaling { get; private set; }

    [JsonProperty]
    internal Vector<T>? PlattScalingA { get; private set; }

    [JsonProperty]
    internal Vector<T>? PlattScalingB { get; private set; }

    [JsonProperty]
    internal bool HasIsotonicRegressionCalibration { get; private set; }

    [JsonProperty]
    internal Vector<T>? IsotonicCalibrationX { get; private set; }

    [JsonProperty]
    internal Vector<T>? IsotonicCalibrationY { get; private set; }

    [JsonProperty]
    internal bool HasLaplacePosterior { get; private set; }

    [JsonProperty]
    internal Vector<T>? LaplacePosteriorMean { get; private set; }

    [JsonProperty]
    internal Vector<T>? LaplacePosteriorVarianceDiag { get; private set; }

    [JsonProperty]
    internal bool HasSwagPosterior { get; private set; }

    [JsonProperty]
    internal Vector<T>? SwagPosteriorMean { get; private set; }

    [JsonProperty]
    internal Vector<T>? SwagPosteriorVarianceDiag { get; private set; }

    [JsonProperty]
    internal bool HasExpectedCalibrationError { get; private set; }

    [JsonProperty]
    internal T ExpectedCalibrationError { get; private set; } = default!;

    internal void SetUncertaintyQuantificationOptions(UncertaintyQuantificationOptions? options)
    {
        options?.Normalize();
        UncertaintyQuantificationOptions = options;
    }

    internal void SetUncertaintyCalibrationArtifacts(UncertaintyCalibrationArtifacts<T> artifacts)
    {
        HasConformalRegression = artifacts.HasConformalRegression;
        ConformalRegressionQuantile = artifacts.ConformalRegressionQuantile;
        HasConformalClassification = artifacts.HasConformalClassification;
        ConformalClassificationThreshold = artifacts.ConformalClassificationThreshold;
        ConformalClassificationNumClasses = artifacts.ConformalClassificationNumClasses;
        HasAdaptiveConformalClassification = artifacts.HasAdaptiveConformalClassification;
        ConformalClassificationAdaptiveBinEdges = artifacts.ConformalClassificationAdaptiveBinEdges;
        ConformalClassificationAdaptiveThresholds = artifacts.ConformalClassificationAdaptiveThresholds;
        HasTemperatureScaling = artifacts.HasTemperatureScaling;
        TemperatureScalingTemperature = artifacts.TemperatureScalingTemperature;
        HasPlattScaling = artifacts.HasPlattScaling;
        PlattScalingA = artifacts.PlattScalingA;
        PlattScalingB = artifacts.PlattScalingB;
        HasIsotonicRegressionCalibration = artifacts.HasIsotonicRegressionCalibration;
        IsotonicCalibrationX = artifacts.IsotonicCalibrationX;
        IsotonicCalibrationY = artifacts.IsotonicCalibrationY;
        HasLaplacePosterior = artifacts.HasLaplacePosterior;
        LaplacePosteriorMean = artifacts.LaplacePosteriorMean;
        LaplacePosteriorVarianceDiag = artifacts.LaplacePosteriorVarianceDiag;
        HasSwagPosterior = artifacts.HasSwagPosterior;
        SwagPosteriorMean = artifacts.SwagPosteriorMean;
        SwagPosteriorVarianceDiag = artifacts.SwagPosteriorVarianceDiag;
        HasExpectedCalibrationError = artifacts.HasExpectedCalibrationError;
        ExpectedCalibrationError = artifacts.ExpectedCalibrationError;
    }

    internal void SetDeepEnsembleModels(List<IFullModel<T, TInput, TOutput>> models)
        => _deepEnsembleModels = models ?? throw new ArgumentNullException(nameof(models));

    public UncertaintyPredictionResult<T, TOutput> PredictWithUncertainty(TInput newData, int? numSamples = null)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (NormalizationInfo.Normalizer == null)
        {
            throw new InvalidOperationException("Normalizer is not initialized.");
        }

        var uq = UncertaintyQuantificationOptions;
        if (uq is not { Enabled: true })
        {
            var deterministic = Predict(newData);
            var metrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);
            return new UncertaintyPredictionResult<T, TOutput>(
                methodUsed: UncertaintyQuantificationMethod.Auto,
                prediction: deterministic,
                variance: CreateZeroLikeOutput(deterministic),
                metrics: metrics);
        }

        var method = uq.Method;
        if (method == UncertaintyQuantificationMethod.Auto)
        {
            if (HasConformalRegression || HasConformalClassification)
            {
                method = UncertaintyQuantificationMethod.ConformalPrediction;
            }
            else if (_deepEnsembleModels is { Count: > 0 })
            {
                method = UncertaintyQuantificationMethod.DeepEnsemble;
            }
            else if (HasSwagPosterior && SwagPosteriorMean is { Length: > 0 } && SwagPosteriorVarianceDiag is { Length: > 0 })
            {
                method = UncertaintyQuantificationMethod.Swag;
            }
            else if (HasLaplacePosterior && LaplacePosteriorMean is { Length: > 0 } && LaplacePosteriorVarianceDiag is { Length: > 0 })
            {
                method = UncertaintyQuantificationMethod.LaplaceApproximation;
            }
            else if (Model is IUncertaintyEstimator<T>)
            {
                method = UncertaintyQuantificationMethod.BayesianNeuralNetwork;
            }
            else
            {
                method = UncertaintyQuantificationMethod.MonteCarloDropout;
            }
        }

        if (method == UncertaintyQuantificationMethod.ConformalPrediction)
        {
            return PredictWithConformal(newData, uq, method);
        }

        if (method == UncertaintyQuantificationMethod.DeepEnsemble)
        {
            return PredictWithDeepEnsemble(newData, uq, method);
        }

        if (method == UncertaintyQuantificationMethod.LaplaceApproximation)
        {
            return PredictWithDiagonalGaussianPosterior(newData, uq, method, LaplacePosteriorMean, LaplacePosteriorVarianceDiag);
        }

        if (method == UncertaintyQuantificationMethod.Swag)
        {
            return PredictWithDiagonalGaussianPosterior(newData, uq, method, SwagPosteriorMean, SwagPosteriorVarianceDiag);
        }

        if (method == UncertaintyQuantificationMethod.BayesianNeuralNetwork)
        {
            return PredictWithBayesianNeuralNetwork(newData, uq, method);
        }

        var effectiveSamples = numSamples ?? uq.NumSamples;
        if (effectiveSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be at least 1.");
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeInput(newData);

        var mcDropoutLayers = GetMonteCarloDropoutLayers(Model);
        if (mcDropoutLayers.Count == 0)
        {
            var deterministic = Predict(newData);
            var metrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);
            return new UncertaintyPredictionResult<T, TOutput>(
                methodUsed: method,
                prediction: deterministic,
                variance: CreateZeroLikeOutput(deterministic),
                metrics: metrics);
        }

        lock (_monteCarloDropoutLock)
        {
            SetMonteCarloMode(mcDropoutLayers, enabled: true);
            try
            {
                var (meanTensor, varianceTensor, predictiveEntropy, mutualInformation) = ComputeMonteCarloMomentsAndMetrics(
                    normalizedNewData,
                    effectiveSamples,
                    mcDropoutLayers,
                    uq.RandomSeed);

                var meanOutput = ConvertFromTensor(meanTensor);
                var denormalizedMean = NormalizationInfo.Normalizer.Denormalize(meanOutput, NormalizationInfo.YParams);
                denormalizedMean = ApplyProbabilityCalibrationIfEnabled(denormalizedMean, uq);

                if (uq.DenormalizeUncertainty)
                {
                    varianceTensor = DenormalizeVarianceIfSupported(varianceTensor, NormalizationInfo.YParams);
                }

                var varianceOutput = ConvertFromTensor(varianceTensor);
                var metrics = CreateDefaultUncertaintyMetrics(denormalizedMean, mutualInformation);
                metrics[PredictiveEntropyMetricKey] = predictiveEntropy;
                metrics[MutualInformationMetricKey] = mutualInformation;

                return new UncertaintyPredictionResult<T, TOutput>(
                    methodUsed: method,
                    prediction: denormalizedMean,
                    variance: varianceOutput,
                    metrics: metrics);
            }
            finally
            {
                SetMonteCarloMode(mcDropoutLayers, enabled: false);
            }
        }
    }

    private UncertaintyPredictionResult<T, TOutput> PredictWithConformal(
        TInput newData,
        UncertaintyQuantificationOptions uq,
        UncertaintyQuantificationMethod method)
    {
        var deterministic = Predict(newData);
        deterministic = ApplyProbabilityCalibrationIfEnabled(deterministic, uq);

        var metrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);

        RegressionConformalInterval<TOutput>? regressionInterval = null;
        ClassificationConformalPredictionSet? classificationSet = null;

        if (HasConformalRegression)
        {
            var lower = AddScalarToOutput(deterministic, _uqNumOps.Negate(ConformalRegressionQuantile));
            var upper = AddScalarToOutput(deterministic, ConformalRegressionQuantile);
            regressionInterval = new RegressionConformalInterval<TOutput>(lower, upper);
        }

        if (HasConformalClassification)
        {
            var probsTensor = ConversionsHelper.ConvertToTensor<T>(deterministic!).Clone();
            var (batch, classes) = InferBatchAndClasses(probsTensor, ConformalClassificationNumClasses);
            if (batch > 0 && classes > 1)
            {
                var probsFlat = probsTensor.ToVector();

                if (HasAdaptiveConformalClassification &&
                    ConformalClassificationAdaptiveBinEdges is { Length: > 1 } edges &&
                    ConformalClassificationAdaptiveThresholds is { Length: > 0 } thresholds)
                {
                    var perSampleThresholds = ComputeAdaptiveThresholdsPerSample(probsFlat, batch, classes, edges, thresholds);
                    classificationSet = new ClassificationConformalPredictionSet(
                        BuildPredictionSets(probsFlat, batch, classes, perSampleThresholds));
                }
                else
                {
                    classificationSet = new ClassificationConformalPredictionSet(
                        BuildPredictionSets(probsFlat, batch, classes, ConformalClassificationThreshold));
                }

                metrics[PredictiveEntropyMetricKey] = new Tensor<T>([batch], ComputePerSampleEntropy(probsFlat, batch, classes));
                metrics[MutualInformationMetricKey] = CreateZeroVectorTensor(batch);
            }
        }

        return new UncertaintyPredictionResult<T, TOutput>(
            methodUsed: method,
            prediction: deterministic,
            variance: CreateZeroLikeOutput(deterministic),
            metrics: metrics,
            regressionInterval: regressionInterval,
            classificationSet: classificationSet);
    }

    private UncertaintyPredictionResult<T, TOutput> PredictWithDeepEnsemble(
        TInput newData,
        UncertaintyQuantificationOptions uq,
        UncertaintyQuantificationMethod method)
    {
        if (_deepEnsembleModels is not { Count: > 0 })
        {
            var deterministic = Predict(newData);
            var fallbackMetrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);
            return new UncertaintyPredictionResult<T, TOutput>(
                methodUsed: method,
                prediction: deterministic,
                variance: CreateZeroLikeOutput(deterministic),
                metrics: fallbackMetrics);
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer!.NormalizeInput(newData);

        var numOps = MathHelper.GetNumericOperations<T>();
        var samples = new List<Tensor<T>>(_deepEnsembleModels.Count);

        for (int i = 0; i < _deepEnsembleModels.Count; i++)
        {
            var normalizedPrediction = _deepEnsembleModels[i].Predict(normalizedNewData);
            samples.Add(ConversionsHelper.ConvertToTensor<T>(normalizedPrediction!).Clone());
        }

        var first = samples[0];
        var firstVector = first.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(first, firstVector);

        if (treatAsProbabilities && classes > 1)
        {
            for (int i = 0; i < samples.Count; i++)
            {
                var sampleOutput = ConvertFromTensor(samples[i]);
                var calibratedOutput = ApplyProbabilityCalibrationIfEnabled(sampleOutput, uq);
                samples[i] = ConversionsHelper.ConvertToTensor<T>(calibratedOutput!).Clone();
            }
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanOutput = ConvertFromTensor(meanTensor);
        var denormalizedMean = NormalizationInfo.Normalizer!.Denormalize(meanOutput, NormalizationInfo.YParams);

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor, NormalizationInfo.YParams);
        }

        var varianceOutput = ConvertFromTensor(varianceTensor);

        var predictiveEntropy = CreateZeroVectorTensor(batch);
        var mutualInformation = CreateZeroVectorTensor(batch);

        if (treatAsProbabilities && classes > 1)
        {
            var expectedEntropySum = new Vector<T>(batch);
            foreach (var sample in samples)
            {
                var sampleEntropy = ComputePerSampleEntropy(sample.ToVector(), batch, classes);
                for (int b = 0; b < batch; b++)
                {
                    expectedEntropySum[b] = numOps.Add(expectedEntropySum[b], sampleEntropy[b]);
                }
            }

            var meanVector = meanTensor.ToVector();
            var predictiveEntropyVec = ComputePerSampleEntropy(meanVector, batch, classes);
            var expectedEntropyVec = new Vector<T>(batch);
            for (int b = 0; b < batch; b++)
            {
                expectedEntropyVec[b] = numOps.Divide(expectedEntropySum[b], numOps.FromDouble(samples.Count));
            }

            var miVec = new Vector<T>(batch);
            for (int b = 0; b < batch; b++)
            {
                var mi = numOps.Subtract(predictiveEntropyVec[b], expectedEntropyVec[b]);
                if (numOps.LessThan(mi, numOps.Zero))
                {
                    mi = numOps.Zero;
                }
                miVec[b] = mi;
            }

            predictiveEntropy = new Tensor<T>([batch], predictiveEntropyVec);
            mutualInformation = new Tensor<T>([batch], miVec);
        }

        var metrics = CreateDefaultUncertaintyMetrics(denormalizedMean, mutualInformation);
        metrics[PredictiveEntropyMetricKey] = predictiveEntropy;
        metrics[MutualInformationMetricKey] = mutualInformation;

        return new UncertaintyPredictionResult<T, TOutput>(
            methodUsed: method,
            prediction: denormalizedMean,
            variance: varianceOutput,
            metrics: metrics);
    }

    private UncertaintyPredictionResult<T, TOutput> PredictWithDiagonalGaussianPosterior(
        TInput newData,
        UncertaintyQuantificationOptions uq,
        UncertaintyQuantificationMethod method,
        Vector<T>? posteriorMean,
        Vector<T>? posteriorVarianceDiag)
    {
        if (posteriorMean == null || posteriorVarianceDiag == null || posteriorMean.Length == 0 || posteriorMean.Length != posteriorVarianceDiag.Length)
        {
            var deterministic = Predict(newData);
            var fallbackMetrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);
            return new UncertaintyPredictionResult<T, TOutput>(
                methodUsed: method,
                prediction: deterministic,
                variance: CreateZeroLikeOutput(deterministic),
                metrics: fallbackMetrics);
        }

        var effectiveSamples = uq.NumSamples;
        if (effectiveSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(uq.NumSamples), "Number of samples must be at least 1.");
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer!.NormalizeInput(newData);
        var numOps = MathHelper.GetNumericOperations<T>();
        var rng = uq.RandomSeed.HasValue ? RandomHelper.CreateSeededRandom(uq.RandomSeed.Value) : RandomHelper.CreateSecureRandom();

        var samples = new List<Tensor<T>>(effectiveSamples);

        lock (_parameterSamplingLock)
        {
            var originalParameters = Model!.GetParameters().Clone();
            try
            {
                for (int s = 0; s < effectiveSamples; s++)
                {
                    var sampledParams = new Vector<T>(posteriorMean.Length);
                    for (int i = 0; i < posteriorMean.Length; i++)
                    {
                        var std = numOps.Sqrt(posteriorVarianceDiag[i]);
                        var z = rng.NextGaussian();
                        var noise = numOps.Multiply(std, numOps.FromDouble(z));
                        sampledParams[i] = numOps.Add(posteriorMean[i], noise);
                    }

                    Model.SetParameters(sampledParams);
                    var normalizedPrediction = Model.Predict(normalizedNewData);
                    samples.Add(ConversionsHelper.ConvertToTensor<T>(normalizedPrediction!).Clone());
                }
            }
            finally
            {
                Model.SetParameters(originalParameters);
            }
        }

        var first = samples[0];
        var firstVector = first.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(first, firstVector);

        if (treatAsProbabilities && classes > 1)
        {
            for (int i = 0; i < samples.Count; i++)
            {
                var sampleOutput = ConvertFromTensor(samples[i]);
                var calibratedOutput = ApplyProbabilityCalibrationIfEnabled(sampleOutput, uq);
                samples[i] = ConversionsHelper.ConvertToTensor<T>(calibratedOutput!).Clone();
            }
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanOutput = ConvertFromTensor(meanTensor);
        var denormalizedMean = NormalizationInfo.Normalizer!.Denormalize(meanOutput, NormalizationInfo.YParams);

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor, NormalizationInfo.YParams);
        }

        var varianceOutput = ConvertFromTensor(varianceTensor);

        var predictiveEntropy = CreateZeroVectorTensor(batch);
        var mutualInformation = CreateZeroVectorTensor(batch);

        var meanVector = meanTensor.ToVector();
        var (treatMeanAsProbabilities, meanBatch, meanClasses) = InferProbabilityDistributionLayout(meanTensor, meanVector);
        if (treatMeanAsProbabilities && meanClasses > 1)
        {
            var expectedEntropySum = new Vector<T>(meanBatch);
            foreach (var sample in samples)
            {
                var sampleEntropy = ComputePerSampleEntropy(sample.ToVector(), meanBatch, meanClasses);
                for (int b = 0; b < meanBatch; b++)
                {
                    expectedEntropySum[b] = numOps.Add(expectedEntropySum[b], sampleEntropy[b]);
                }
            }

            var predictiveEntropyVec = ComputePerSampleEntropy(meanVector, meanBatch, meanClasses);
            var expectedEntropyVec = new Vector<T>(meanBatch);
            for (int b = 0; b < meanBatch; b++)
            {
                expectedEntropyVec[b] = numOps.Divide(expectedEntropySum[b], numOps.FromDouble(samples.Count));
            }

            var miVec = new Vector<T>(meanBatch);
            for (int b = 0; b < meanBatch; b++)
            {
                var mi = numOps.Subtract(predictiveEntropyVec[b], expectedEntropyVec[b]);
                if (numOps.LessThan(mi, numOps.Zero))
                {
                    mi = numOps.Zero;
                }
                miVec[b] = mi;
            }

            predictiveEntropy = new Tensor<T>([meanBatch], predictiveEntropyVec);
            mutualInformation = new Tensor<T>([meanBatch], miVec);
        }

        var metrics = CreateDefaultUncertaintyMetrics(denormalizedMean, mutualInformation);
        metrics[PredictiveEntropyMetricKey] = predictiveEntropy;
        metrics[MutualInformationMetricKey] = mutualInformation;

        return new UncertaintyPredictionResult<T, TOutput>(
            methodUsed: method,
            prediction: denormalizedMean,
            variance: varianceOutput,
            metrics: metrics);
    }

    private UncertaintyPredictionResult<T, TOutput> PredictWithBayesianNeuralNetwork(
        TInput newData,
        UncertaintyQuantificationOptions uq,
        UncertaintyQuantificationMethod method)
    {
        var estimator = Model as IUncertaintyEstimator<T>;
        if (estimator == null)
        {
            var deterministic = Predict(newData);
            var fallbackMetrics = CreateDefaultUncertaintyMetrics(deterministic, mutualInformation: null);
            return new UncertaintyPredictionResult<T, TOutput>(
                methodUsed: method,
                prediction: deterministic,
                variance: CreateZeroLikeOutput(deterministic),
                metrics: fallbackMetrics);
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer!.NormalizeInput(newData);
        var inputTensor = ConversionsHelper.ConvertToTensor<T>(normalizedNewData!).Clone();

        var uqResult = estimator.PredictWithUncertainty(inputTensor);

        var meanOutput = ConvertFromTensor(uqResult.Prediction);
        var denormalizedMean = NormalizationInfo.Normalizer!.Denormalize(meanOutput, NormalizationInfo.YParams);
        denormalizedMean = ApplyProbabilityCalibrationIfEnabled(denormalizedMean, uq);

        var varianceTensor = uqResult.Variance != null
            ? uqResult.Variance.Clone()
            : new Tensor<T>(
                uqResult.Prediction.Shape,
                Vector<T>.CreateDefault(uqResult.Prediction.Length, MathHelper.GetNumericOperations<T>().Zero));

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor, NormalizationInfo.YParams);
        }

        var varianceOutput = ConvertFromTensor(varianceTensor);

        var metrics = CreateDefaultUncertaintyMetrics(denormalizedMean, mutualInformation: null);
        if (uqResult.Metrics != null)
        {
            foreach (var kvp in uqResult.Metrics)
            {
                metrics[kvp.Key] = kvp.Value;
            }
        }

        return new UncertaintyPredictionResult<T, TOutput>(
            methodUsed: method,
            prediction: denormalizedMean,
            variance: varianceOutput,
            metrics: metrics);
    }

    private static (int batch, int classes) InferBatchAndClasses(Tensor<T> probabilities, int configuredClasses)
    {
        var classes = configuredClasses > 0
            ? configuredClasses
            : probabilities.Shape[probabilities.Shape.Length - 1];

        if (classes <= 0)
        {
            return (1, 0);
        }

        var batch = probabilities.Rank == 1 ? 1 : probabilities.Length / classes;
        return batch <= 0 ? (1, classes) : (batch, classes);
    }

    private static int[][] BuildPredictionSets(Vector<T> probsFlat, int batch, int classes, T threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var sets = new int[batch][];

        for (int b = 0; b < batch; b++)
        {
            var indices = new List<int>();
            var baseIndex = b * classes;
            for (int c = 0; c < classes; c++)
            {
                if (numOps.GreaterThanOrEquals(probsFlat[baseIndex + c], threshold))
                {
                    indices.Add(c);
                }
            }

            if (indices.Count == 0)
            {
                var best = 0;
                var bestProb = probsFlat[baseIndex];
                for (int c = 1; c < classes; c++)
                {
                    var p = probsFlat[baseIndex + c];
                    if (numOps.GreaterThan(p, bestProb))
                    {
                        bestProb = p;
                        best = c;
                    }
                }
                indices.Add(best);
            }

            sets[b] = indices.ToArray();
        }

        return sets;
    }

    private static int[][] BuildPredictionSets(Vector<T> probsFlat, int batch, int classes, Vector<T> thresholdsPerSample)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var sets = new int[batch][];

        for (int b = 0; b < batch; b++)
        {
            var threshold = b < thresholdsPerSample.Length
                ? thresholdsPerSample[b]
                : thresholdsPerSample[thresholdsPerSample.Length - 1];

            var indices = new List<int>();
            var baseIndex = b * classes;
            for (int c = 0; c < classes; c++)
            {
                if (numOps.GreaterThanOrEquals(probsFlat[baseIndex + c], threshold))
                {
                    indices.Add(c);
                }
            }

            if (indices.Count == 0)
            {
                var best = 0;
                var bestProb = probsFlat[baseIndex];
                for (int c = 1; c < classes; c++)
                {
                    var p = probsFlat[baseIndex + c];
                    if (numOps.GreaterThan(p, bestProb))
                    {
                        bestProb = p;
                        best = c;
                    }
                }
                indices.Add(best);
            }

            sets[b] = indices.ToArray();
        }

        return sets;
    }

    private static Vector<T> ComputeAdaptiveThresholdsPerSample(
        Vector<T> probsFlat,
        int batch,
        int classes,
        double[] edges,
        Vector<T> thresholds)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var bins = Math.Max(1, Math.Min(thresholds.Length, Math.Max(1, edges.Length - 1)));
        var perSample = new Vector<T>(batch);

        for (int b = 0; b < batch; b++)
        {
            var baseIndex = b * classes;
            var best = probsFlat[baseIndex];
            for (int c = 1; c < classes; c++)
            {
                var p = probsFlat[baseIndex + c];
                if (numOps.GreaterThan(p, best))
                {
                    best = p;
                }
            }

            var conf = numOps.ToDouble(best);
            if (conf < 0.0) conf = 0.0;
            if (conf > 1.0) conf = 1.0;

            var bin = (int)Math.Floor(conf * bins);
            if (bin == bins) bin = bins - 1;

            perSample[b] = thresholds[bin];
        }

        return perSample;
    }

    private TOutput AddScalarToOutput(TOutput output, T scalar)
    {
        var tensor = ConversionsHelper.ConvertToTensor<T>(output!).Clone();
        var vec = tensor.ToVector();
        for (int i = 0; i < vec.Length; i++)
        {
            vec[i] = _uqNumOps.Add(vec[i], scalar);
        }

        var updated = new Tensor<T>(tensor.Shape, vec);
        return ConvertFromTensor(updated);
    }

    private TOutput ApplyTemperatureScalingToOutputProbabilities(TOutput output, T temperature)
    {
        var probsTensor = ConversionsHelper.ConvertToTensor<T>(output!).Clone();
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (!treatAsProbabilities || classes <= 1)
        {
            return output;
        }

        var scaledTensor = ApplyTemperatureScalingToProbabilityTensor(probsTensor, temperature, batch, classes);
        return ConvertFromTensor(scaledTensor);
    }

    private TOutput ApplyProbabilityCalibrationIfEnabled(TOutput output, UncertaintyQuantificationOptions uq)
    {
        var method = uq.CalibrationMethod;
        if (method == ProbabilityCalibrationMethod.None)
        {
            return output;
        }

        var probsTensor = ConversionsHelper.ConvertToTensor<T>(output!).Clone();
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (classes <= 1)
        {
            return output;
        }

        if (!treatAsProbabilities)
        {
            probsTensor = SoftmaxTensor(probsTensor, batch, classes);
            output = ConvertFromTensor(probsTensor);
            probsVector = probsTensor.ToVector();
            treatAsProbabilities = true;
        }

        if (method == ProbabilityCalibrationMethod.Auto)
        {
            if (classes == 2 && HasIsotonicRegressionCalibration && uq.EnableIsotonicRegressionCalibration)
            {
                method = ProbabilityCalibrationMethod.IsotonicRegression;
            }
            else if (classes == 2 && HasPlattScaling && uq.EnablePlattScaling)
            {
                method = ProbabilityCalibrationMethod.PlattScaling;
            }
            else if (HasTemperatureScaling && uq.EnableTemperatureScaling)
            {
                method = ProbabilityCalibrationMethod.TemperatureScaling;
            }
            else
            {
                return output;
            }
        }

        return method switch
        {
            ProbabilityCalibrationMethod.TemperatureScaling when HasTemperatureScaling && uq.EnableTemperatureScaling
                => ApplyTemperatureScalingToOutputProbabilities(output, TemperatureScalingTemperature),
            ProbabilityCalibrationMethod.PlattScaling when HasPlattScaling && uq.EnablePlattScaling
                => ApplyPlattScalingToOutputProbabilities(output),
            ProbabilityCalibrationMethod.IsotonicRegression when HasIsotonicRegressionCalibration && uq.EnableIsotonicRegressionCalibration
                => ApplyIsotonicCalibrationToOutputProbabilities(output),
            _ => output
        };
    }

    private TOutput ApplyPlattScalingToOutputProbabilities(TOutput output)
    {
        if (PlattScalingA is not { Length: > 0 } aVec || PlattScalingB is not { Length: > 0 } bVec)
        {
            return output;
        }

        var probsTensor = ConversionsHelper.ConvertToTensor<T>(output!).Clone();
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (!treatAsProbabilities || classes != 2 || batch <= 0)
        {
            return output;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;
        var outVec = new Vector<T>(batch * 2);

        var a = aVec[0];
        var b = bVec[0];

        for (int i = 0; i < batch; i++)
        {
            var p1 = numOps.ToDouble(probsVector[i * 2 + 1]);
            if (p1 < eps) p1 = eps;
            if (p1 > 1.0 - eps) p1 = 1.0 - eps;

            var logit = Math.Log(p1 / (1.0 - p1));
            var z = numOps.Add(numOps.Multiply(a, numOps.FromDouble(logit)), b);
            var p1Cal = 1.0 / (1.0 + Math.Exp(-numOps.ToDouble(z)));

            var p1T = numOps.FromDouble(p1Cal);
            outVec[i * 2 + 1] = p1T;
            outVec[i * 2] = numOps.Subtract(numOps.One, p1T);
        }

        var calibrated = new Tensor<T>([batch, 2], outVec).Reshape(probsTensor.Shape);
        return ConvertFromTensor(calibrated);
    }

    private TOutput ApplyIsotonicCalibrationToOutputProbabilities(TOutput output)
    {
        if (IsotonicCalibrationX is not { Length: > 0 } x || IsotonicCalibrationY is not { Length: > 0 } y)
        {
            return output;
        }

        var probsTensor = ConversionsHelper.ConvertToTensor<T>(output!).Clone();
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (!treatAsProbabilities || classes != 2 || batch <= 0)
        {
            return output;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = 1e-12;
        var outVec = new Vector<T>(batch * 2);

        for (int i = 0; i < batch; i++)
        {
            var p1 = numOps.ToDouble(probsVector[i * 2 + 1]);
            if (p1 < eps) p1 = eps;
            if (p1 > 1.0 - eps) p1 = 1.0 - eps;

            var p1T = numOps.FromDouble(p1);
            var p1Cal = EvaluateIsotonic(p1T, x, y, numOps);
            outVec[i * 2 + 1] = p1Cal;
            outVec[i * 2] = numOps.Subtract(numOps.One, p1Cal);
        }

        var calibrated = new Tensor<T>([batch, 2], outVec).Reshape(probsTensor.Shape);
        return ConvertFromTensor(calibrated);
    }

    private static T EvaluateIsotonic(T p, Vector<T> x, Vector<T> y, INumericOperations<T> numOps)
    {
        if (x.Length == 0)
        {
            return p;
        }

        var spanX = x.AsSpan();
        var spanY = y.AsSpan();
        for (int i = 0; i < spanX.Length; i++)
        {
            if (numOps.LessThanOrEquals(p, spanX[i]))
            {
                return spanY[i];
            }
        }

        return spanY[spanY.Length - 1];
    }

    private static Tensor<T> SoftmaxTensor(Tensor<T> tensor, int batch, int classes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (classes <= 1)
        {
            return tensor;
        }

        var flat = tensor.ToVector();
        var output = new Vector<T>(batch * classes);

        for (int b = 0; b < batch; b++)
        {
            var baseIndex = b * classes;
            var max = flat[baseIndex];
            for (int c = 1; c < classes; c++)
            {
                var v = flat[baseIndex + c];
                if (numOps.GreaterThan(v, max))
                {
                    max = v;
                }
            }

            var sumExp = numOps.Zero;
            for (int c = 0; c < classes; c++)
            {
                var ex = numOps.Exp(numOps.Subtract(flat[baseIndex + c], max));
                output[baseIndex + c] = ex;
                sumExp = numOps.Add(sumExp, ex);
            }

            for (int c = 0; c < classes; c++)
            {
                output[baseIndex + c] = numOps.Divide(output[baseIndex + c], sumExp);
            }
        }

        return new Tensor<T>([batch, classes], output).Reshape(tensor.Shape);
    }

    private Tensor<T> DenormalizeVarianceIfSupported(Tensor<T> normalizedVariance, NormalizationParameters<T> yParams)
    {
        if (!TryGetOutputDenormalizationScale(yParams, out var scale))
        {
            return normalizedVariance;
        }

        var scaleSquared = _uqNumOps.Multiply(scale, scale);

        var normalizedVector = normalizedVariance.ToVector();
        var varianceVector = new Vector<T>(normalizedVector.Length);
        for (int i = 0; i < normalizedVector.Length; i++)
        {
            varianceVector[i] = _uqNumOps.Multiply(normalizedVector[i], scaleSquared);
        }

        var varianceTensor = Tensor<T>.FromVector(varianceVector);
        if (normalizedVariance.Shape.Length > 1)
        {
            varianceTensor = varianceTensor.Reshape(normalizedVariance.Shape);
        }

        return varianceTensor;
    }

    private bool TryGetOutputDenormalizationScale(NormalizationParameters<T> yParams, out T scale)
    {
        switch (yParams.Method)
        {
            case NormalizationMethod.None:
                scale = _uqNumOps.One;
                return true;
            case NormalizationMethod.MinMax:
                scale = _uqNumOps.Subtract(yParams.Max, yParams.Min);
                return true;
            case NormalizationMethod.ZScore:
                scale = yParams.StdDev;
                return true;
            default:
                scale = default!;
                return false;
        }
    }

    private static TOutput ConvertFromTensor(Tensor<T> tensor)
    {
        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)tensor;
        }

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)tensor.ToVector();
        }

        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            return (TOutput)(object)tensor.ToMatrix();
        }

        return (TOutput)(object)tensor;
    }

    private static TOutput? CreateZeroLikeOutput(TOutput output)
    {
        if (output is Tensor<T> tensor)
        {
            return (TOutput)(object)new Tensor<T>(tensor.Shape);
        }

        if (output is Vector<T> vector)
        {
            return (TOutput)(object)Vector<T>.CreateDefault(vector.Length, MathHelper.GetNumericOperations<T>().Zero);
        }

        if (output is Matrix<T> matrix)
        {
            return (TOutput)(object)new Matrix<T>(matrix.Rows, matrix.Columns);
        }

        if (output is T)
        {
            var zero = MathHelper.GetNumericOperations<T>().Zero;
            return (TOutput)(object)zero!;
        }

        return default;
    }

    private static Tensor<T> CreateZeroVectorTensor(int length)
        => new([length], Vector<T>.CreateDefault(length, MathHelper.GetNumericOperations<T>().Zero));

    private static Dictionary<string, Tensor<T>> CreateDefaultUncertaintyMetrics(TOutput prediction, Tensor<T>? mutualInformation)
    {
        return new Dictionary<string, Tensor<T>>
        {
            [PredictiveEntropyMetricKey] = CreateZeroVectorTensor(1),
            [MutualInformationMetricKey] = mutualInformation ?? CreateZeroVectorTensor(1)
        };
    }

    private static List<MCDropoutLayer<T>> GetMonteCarloDropoutLayers(IFullModel<T, TInput, TOutput> model)
    {
        if (model is Models.NeuralNetworkModel<T> nn)
        {
            return nn.Network.LayersReadOnly.OfType<MCDropoutLayer<T>>().ToList();
        }

        if (model is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> baseNetwork)
        {
            return baseNetwork.LayersReadOnly.OfType<MCDropoutLayer<T>>().ToList();
        }

        return [];
    }

    private static void SetMonteCarloMode(List<MCDropoutLayer<T>> layers, bool enabled)
    {
        foreach (var layer in layers)
        {
            layer.MonteCarloMode = enabled;
        }
    }

    private static (Tensor<T> mean, Tensor<T> variance) ComputeMeanAndVariance(List<Tensor<T>> samples)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var length = samples[0].Length;
        var meanVec = new Vector<T>(length);

        foreach (var sample in samples)
        {
            var vec = sample.ToVector();
            for (int i = 0; i < length; i++)
            {
                meanVec[i] = numOps.Add(meanVec[i], vec[i]);
            }
        }

        var inv = numOps.Divide(numOps.One, numOps.FromDouble(samples.Count));
        for (int i = 0; i < length; i++)
        {
            meanVec[i] = numOps.Multiply(meanVec[i], inv);
        }

        var varianceVec = new Vector<T>(length);
        foreach (var sample in samples)
        {
            var vec = sample.ToVector();
            for (int i = 0; i < length; i++)
            {
                var diff = numOps.Subtract(vec[i], meanVec[i]);
                varianceVec[i] = numOps.Add(varianceVec[i], numOps.Multiply(diff, diff));
            }
        }

        var varianceInv = numOps.Divide(numOps.One, numOps.FromDouble(samples.Count));
        for (int i = 0; i < length; i++)
        {
            varianceVec[i] = numOps.Multiply(varianceVec[i], varianceInv);
        }

        var shape = samples[0].Shape;
        return (new Tensor<T>(shape, meanVec), new Tensor<T>(shape, varianceVec));
    }

    private static Tensor<T> ApplyTemperatureScalingToProbabilityTensor(Tensor<T> probabilities, T temperature, int batch, int classes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = numOps.FromDouble(1e-12);
        var scaled = new Vector<T>(batch * classes);
        var flat = probabilities.ToVector();

        for (int i = 0; i < batch; i++)
        {
            var maxLogit = default(T)!;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[i * classes + c];
                if (numOps.LessThan(p, eps))
                {
                    p = eps;
                }
                var logit = numOps.Divide(numOps.Log(p), temperature);
                if (c == 0 || numOps.GreaterThan(logit, maxLogit))
                {
                    maxLogit = logit;
                }
            }

            var sumExp = numOps.Zero;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[i * classes + c];
                if (numOps.LessThan(p, eps))
                {
                    p = eps;
                }
                var logit = numOps.Divide(numOps.Log(p), temperature);
                var exp = numOps.Exp(numOps.Subtract(logit, maxLogit));
                scaled[i * classes + c] = exp;
                sumExp = numOps.Add(sumExp, exp);
            }

            for (int c = 0; c < classes; c++)
            {
                scaled[i * classes + c] = numOps.Divide(scaled[i * classes + c], sumExp);
            }
        }

        return new Tensor<T>([batch, classes], scaled).Reshape(probabilities.Shape);
    }

    private static (bool treatAsProbabilities, int batch, int classes) InferProbabilityDistributionLayout(Tensor<T> tensor, Vector<T> flat)
    {
        if (tensor.Rank == 0 || tensor.Length == 0)
        {
            return (false, 0, 0);
        }

        var classes = tensor.Shape[tensor.Shape.Length - 1];
        if (classes <= 1)
        {
            return (false, 1, classes);
        }

        var batch = tensor.Rank == 1 ? 1 : tensor.Length / classes;
        if (batch <= 0)
        {
            return (false, 0, classes);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var tolerance = numOps.FromDouble(1e-2);
        for (int b = 0; b < batch; b++)
        {
            var sum = numOps.Zero;
            var baseIndex = b * classes;
            for (int c = 0; c < classes; c++)
            {
                var p = flat[baseIndex + c];
                if (numOps.LessThan(p, numOps.Zero) || numOps.GreaterThan(p, numOps.One))
                {
                    return (false, batch, classes);
                }
                sum = numOps.Add(sum, p);
            }
            if (numOps.GreaterThan(numOps.Abs(numOps.Subtract(sum, numOps.One)), tolerance))
            {
                return (false, batch, classes);
            }
        }

        return (true, batch, classes);
    }

    private static Vector<T> ComputePerSampleEntropy(Vector<T> probsFlat, int batch, int classes)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = numOps.FromDouble(1e-12);
        var entropy = new Vector<T>(batch);

        for (int b = 0; b < batch; b++)
        {
            var sum = numOps.Zero;
            var baseIndex = b * classes;
            for (int c = 0; c < classes; c++)
            {
                var p = probsFlat[baseIndex + c];
                if (numOps.LessThan(p, eps))
                {
                    continue;
                }
                sum = numOps.Subtract(sum, numOps.Multiply(p, numOps.Log(p)));
            }
            entropy[b] = sum;
        }

        return entropy;
    }

    private (Tensor<T> mean, Tensor<T> variance, Tensor<T> predictiveEntropy, Tensor<T> mutualInformation) ComputeMonteCarloMomentsAndMetrics(
        TInput normalizedInput,
        int numSamples,
        List<MCDropoutLayer<T>> dropoutLayers,
        int? baseSeed)
    {
        var samples = new List<Tensor<T>>(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            if (baseSeed.HasValue)
            {
                // Ensure deterministic dropout masks per call when a seed is supplied.
                // We reset each layer's RNG for every sample so repeated PredictWithUncertainty()
                // calls produce identical Monte Carlo sequences (important for testing and reproducibility).
                for (int layerIndex = 0; layerIndex < dropoutLayers.Count; layerIndex++)
                {
                    dropoutLayers[layerIndex].ResetRng(unchecked(baseSeed.Value + (i + 1) * 10007 + (layerIndex + 1) * 1009));
                }
            }

            var normalizedPrediction = Model!.Predict(normalizedInput);
            samples.Add(ConversionsHelper.ConvertToTensor<T>(normalizedPrediction!).Clone());
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanVector = meanTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(meanTensor, meanVector);
        if (!treatAsProbabilities || classes <= 1)
        {
            return (meanTensor, varianceTensor, CreateZeroVectorTensor(batch), CreateZeroVectorTensor(batch));
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var predictiveEntropyVec = ComputePerSampleEntropy(meanVector, batch, classes);
        var expectedEntropySum = new Vector<T>(batch);
        foreach (var sample in samples)
        {
            var sampleEntropy = ComputePerSampleEntropy(sample.ToVector(), batch, classes);
            for (int b = 0; b < batch; b++)
            {
                expectedEntropySum[b] = numOps.Add(expectedEntropySum[b], sampleEntropy[b]);
            }
        }

        var expectedEntropyVec = new Vector<T>(batch);
        for (int b = 0; b < batch; b++)
        {
            expectedEntropyVec[b] = numOps.Divide(expectedEntropySum[b], numOps.FromDouble(samples.Count));
        }

        var miVec = new Vector<T>(batch);
        for (int b = 0; b < batch; b++)
        {
            var mi = numOps.Subtract(predictiveEntropyVec[b], expectedEntropyVec[b]);
            if (numOps.LessThan(mi, numOps.Zero))
            {
                mi = numOps.Zero;
            }
            miVec[b] = mi;
        }

        return (
            meanTensor,
            varianceTensor,
            new Tensor<T>([batch], predictiveEntropyVec),
            new Tensor<T>([batch], miVec));
    }
}
