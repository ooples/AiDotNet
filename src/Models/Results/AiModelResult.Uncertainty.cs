using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Preprocessing;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.UncertaintyQuantification.Interfaces;
using AiDotNet.UncertaintyQuantification.Layers;
using Newtonsoft.Json;
using AiDotNet.Validation;

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

#pragma warning disable CS8601, CS8618 // T defaults to default(T) - used with value types
    [JsonProperty]
    internal T ConformalRegressionQuantile { get; private set; } = default;
#pragma warning restore CS8601, CS8618

    [JsonProperty]
    internal bool HasConformalClassification { get; private set; }

#pragma warning disable CS8601, CS8618 // T defaults to default(T) - used with value types
    [JsonProperty]
    internal T ConformalClassificationThreshold { get; private set; } = default;
#pragma warning restore CS8601, CS8618

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

#pragma warning disable CS8601, CS8618 // T defaults to default(T) - used with value types
    [JsonProperty]
    internal T TemperatureScalingTemperature { get; private set; } = default;
#pragma warning restore CS8601, CS8618

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

#pragma warning disable CS8601, CS8618 // T defaults to default(T) - used with value types
    [JsonProperty]
    internal T ExpectedCalibrationError { get; private set; } = default;
#pragma warning restore CS8601, CS8618

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
        { Guard.NotNull(models); _deepEnsembleModels = models; }

    /// <summary>
    /// Makes a prediction together with an estimate of how uncertain the model is about it.
    /// </summary>
    /// <param name="newData">The input features to predict on.</param>
    /// <param name="numSamples">Number of stochastic forward passes for Monte Carlo dropout, or <c>null</c> to use
    /// the configured <c>NumSamples</c>. Ignored by the other uncertainty methods.</param>
    /// <returns>The prediction, its variance, the method actually used, and uncertainty metrics.</returns>
    /// <exception cref="InvalidOperationException">The model is not initialized.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Monte Carlo dropout is used and the effective sample count is
    /// less than 1.</exception>
    /// <remarks>
    /// <para>
    /// The method comes from the uncertainty quantification options configured on the builder. With
    /// <c>Auto</c>, the first available of these is used: conformal prediction (when calibrated), a deep ensemble,
    /// a SWAG posterior, a Laplace posterior, the model's own Bayesian estimator, and finally Monte Carlo dropout.
    /// When uncertainty quantification is not enabled, or Monte Carlo dropout finds no dropout layers, this returns
    /// the ordinary <see cref="Predict(TInput)"/> result with zero variance.
    /// </para>
    /// <para><b>For Beginners:</b> A normal prediction gives you one answer. This also tells you how much to trust
    /// it: a large variance means the model is unsure, which is a signal to review the case or gather more data.
    /// A zero variance does not by itself mean the model is confident. Conformal prediction reports its uncertainty as
    /// an interval or, for classification, a set of possible classes, and can return zero variance alongside a wide
    /// interval or a large set; zero variance is also what you get when no uncertainty method was available. Check
    /// the method actually used and its interval or set, not only the variance.
    /// </para>
    /// </remarks>
    public UncertaintyPredictionResult<T, TOutput> PredictWithUncertainty(TInput newData, int? numSamples = null)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
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

        var normalizedNewData = PreprocessingInfo?.IsFitted == true
            ? PreprocessingInfo.TransformFeatures(newData)
            : newData;

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
                var denormalizedMean = PreprocessingInfo is not null
                    ? PreprocessingInfo.InverseTransformPredictions(meanOutput)
                    : meanOutput;
                denormalizedMean = ApplyProbabilityCalibrationIfEnabled(denormalizedMean, uq);

                if (uq.DenormalizeUncertainty)
                {
                    varianceTensor = DenormalizeVarianceIfSupported(varianceTensor);
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
            var probsTensor = ConvertToTensorSafe(deterministic, "deterministic prediction");
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

        var normalizedNewData = PreprocessingInfo?.IsFitted == true
            ? PreprocessingInfo.TransformFeatures(newData)
            : newData;

        var samples = new List<Tensor<T>>(_deepEnsembleModels.Count);

        for (int i = 0; i < _deepEnsembleModels.Count; i++)
        {
            var normalizedPrediction = _deepEnsembleModels[i].Predict(normalizedNewData);
            samples.Add(ConvertToTensorSafe(normalizedPrediction, "normalized prediction"));
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
                samples[i] = ConvertToTensorSafe(calibratedOutput, "calibrated output");
            }
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanOutput = ConvertFromTensor(meanTensor);
        var denormalizedMean = PreprocessingInfo is not null
            ? PreprocessingInfo.InverseTransformPredictions(meanOutput)
            : meanOutput;

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor);
        }

        var varianceOutput = ConvertFromTensor(varianceTensor);

        var predictiveEntropy = CreateZeroVectorTensor(batch);
        var mutualInformation = CreateZeroVectorTensor(batch);

        if (treatAsProbabilities && classes > 1)
        {
            (predictiveEntropy, mutualInformation) = ComputeEntropyAndMutualInformation(samples, meanTensor.ToVector(), batch, classes);
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

        var normalizedNewData = PreprocessingInfo?.IsFitted == true
            ? PreprocessingInfo.TransformFeatures(newData)
            : newData;
        var rng = uq.RandomSeed.HasValue ? RandomHelper.CreateSeededRandom(uq.RandomSeed.Value) : RandomHelper.CreateSecureRandom();

        var samples = new List<Tensor<T>>(effectiveSamples);
        var meanTensorParameters = Tensor<T>.FromVector(posteriorMean);
        var posteriorStd = ResultEngine.TensorSqrt(Tensor<T>.FromVector(posteriorVarianceDiag));
        int parameterCount = posteriorMean.Length;

        lock (_parameterSamplingLock)
        {
            var originalParameters = new AiDotNet.Training.ModelParameterCheckpoint<T, TInput, TOutput>(InterfaceGuard.Parameterizable(EnsureModel));
            originalParameters.Capture();
            try
            {
                for (int s = 0; s < effectiveSamples; s++)
                {
                    // theta = mu + sqrt(diag(Sigma)) * z with z ~ N(0, I); the standard normal draws keep their
                    // original order, one per parameter.
                    var draws = new double[parameterCount];
                    for (int i = 0; i < parameterCount; i++)
                    {
                        draws[i] = rng.NextGaussian();
                    }

                    var z = new Tensor<double>(draws, new[] { parameterCount }).Cast<T>();
                    var sampledParams = ResultEngine.TensorAdd(meanTensorParameters, ResultEngine.TensorMultiply(posteriorStd, z)).ToVector();

                    InterfaceGuard.Parameterizable(EnsureModel).SetParameters(sampledParams);
                    var normalizedPrediction = EnsureModel.Predict(normalizedNewData);
                    samples.Add(ConvertToTensorSafe(normalizedPrediction, "normalized prediction"));
                }
            }
            finally
            {
                originalParameters.Restore();
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
                samples[i] = ConvertToTensorSafe(calibratedOutput, "calibrated output");
            }
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanOutput = ConvertFromTensor(meanTensor);
        var denormalizedMean = PreprocessingInfo is not null
            ? PreprocessingInfo.InverseTransformPredictions(meanOutput)
            : meanOutput;

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor);
        }

        var varianceOutput = ConvertFromTensor(varianceTensor);

        var predictiveEntropy = CreateZeroVectorTensor(batch);
        var mutualInformation = CreateZeroVectorTensor(batch);

        var meanVector = meanTensor.ToVector();
        var (treatMeanAsProbabilities, meanBatch, meanClasses) = InferProbabilityDistributionLayout(meanTensor, meanVector);
        if (treatMeanAsProbabilities && meanClasses > 1)
        {
            (predictiveEntropy, mutualInformation) = ComputeEntropyAndMutualInformation(samples, meanVector, meanBatch, meanClasses);
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

        var normalizedNewData = PreprocessingInfo?.IsFitted == true
            ? PreprocessingInfo.TransformFeatures(newData)
            : newData;
        var inputTensor = ConvertToTensorSafe(normalizedNewData, "normalized input data");

        var uqResult = estimator.PredictWithUncertainty(inputTensor);

        var meanOutput = ConvertFromTensor(uqResult.Prediction);
        var denormalizedMean = PreprocessingInfo is not null
            ? PreprocessingInfo.InverseTransformPredictions(meanOutput)
            : meanOutput;
        denormalizedMean = ApplyProbabilityCalibrationIfEnabled(denormalizedMean, uq);

        var varianceTensor = uqResult.Variance != null
            ? uqResult.Variance.Clone()
            : new Tensor<T>(
                uqResult.Prediction._shape,
                Vector<T>.CreateDefault(uqResult.Prediction.Length, MathHelper.GetNumericOperations<T>().Zero));

        if (uq.DenormalizeUncertainty)
        {
            varianceTensor = DenormalizeVarianceIfSupported(varianceTensor);
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
        var probabilities = ProbabilityRows(probsFlat, batch, classes);
        var thresholds = ResultEngine.TensorFullLike(probabilities, threshold);
        return BuildPredictionSetsFromMask(probabilities, ResultEngine.TensorGreaterEqual(probabilities, thresholds), batch, classes);
    }

    private static int[][] BuildPredictionSets(Vector<T> probsFlat, int batch, int classes, Vector<T> thresholdsPerSample)
    {
        // Sample b uses threshold b, or the last threshold when there are fewer thresholds than samples.
        var thresholdIndex = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            thresholdIndex[b] = b < thresholdsPerSample.Length ? b : thresholdsPerSample.Length - 1;
        }

        var probabilities = ProbabilityRows(probsFlat, batch, classes);
        var perSample = ResultEngine.TensorTake(Tensor<T>.FromVector(thresholdsPerSample), new Tensor<int>(thresholdIndex, new[] { batch }));
        var thresholds = ResultEngine.TensorBroadcastTo(ResultEngine.Reshape(perSample, new[] { batch, 1 }), new[] { batch, classes });
        return BuildPredictionSetsFromMask(probabilities, ResultEngine.TensorGreaterEqual(probabilities, thresholds), batch, classes);
    }

    /// <summary>The first <paramref name="batch"/> x <paramref name="classes"/> probabilities as a [batch, classes] tensor.</summary>
    private static Tensor<T> ProbabilityRows(Vector<T> probsFlat, int batch, int classes)
    {
        var flat = Tensor<T>.FromVector(probsFlat);
        if (probsFlat.Length != batch * classes)
        {
            flat = ResultEngine.TensorSlice(flat, new[] { 0 }, new[] { batch * classes });
        }

        return ResultEngine.Reshape(flat, new[] { batch, classes });
    }

    /// <summary>
    /// Prediction sets from a [batch, classes] 0/1 inclusion mask: the included classes in ascending order, or the
    /// single most probable class (first strict maximum) when a sample includes none.
    /// </summary>
    private static int[][] BuildPredictionSetsFromMask(Tensor<T> probabilities, Tensor<T> included, int batch, int classes)
    {
        int[] mask = included.Cast<int>().ToArray();
        int[] best = ResultEngine.TensorArgMax(probabilities, 1).ToArray();
        var sets = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            var indices = new List<int>();
            for (int c = 0; c < classes; c++)
            {
                if (mask[b * classes + c] != 0)
                {
                    indices.Add(c);
                }
            }

            if (indices.Count == 0)
            {
                indices.Add(best[b]);
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

        // Confidence is the top-class probability (first strict maximum), clamped to [0, 1]; its bin is
        // floor(confidence * bins), with confidence 1 folded into the last bin. The bin index is compared in int.
        var probabilities = ProbabilityRows(probsFlat, batch, classes);
        int[] best = ResultEngine.TensorArgMax(probabilities, 1).ToArray();
        var confidence = ResultEngine.TensorClamp(
            ResultEngine.TensorTakeAlongDim(probabilities, ResultEngine.Reshape(new Tensor<int>(best, new[] { batch }), new[] { batch, 1 }), 1),
            numOps.Zero,
            numOps.One);
        int[] bin = ResultEngine.TensorFloor(ResultEngine.TensorMultiplyScalar(confidence, numOps.FromDouble(bins))).Cast<int>().ToArray();
        for (int b = 0; b < batch; b++)
        {
            if (bin[b] >= bins)
            {
                bin[b] = bins - 1;
            }
        }

        return ResultEngine.TensorTake(Tensor<T>.FromVector(thresholds), new Tensor<int>(bin, new[] { batch })).ToVector();
    }

    private TOutput AddScalarToOutput(TOutput output, T scalar)
    {
        var tensor = ConvertToTensorSafe(output, "model output");
        return ConvertFromTensor(ResultEngine.TensorAddScalar(tensor, scalar));
    }

    private TOutput ApplyTemperatureScalingToOutputProbabilities(TOutput output, T temperature)
    {
        var probsTensor = ConvertToTensorSafe(output, "model output");
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

        var probsTensor = ConvertToTensorSafe(output, "model output");
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

        var probsTensor = ConvertToTensorSafe(output, "model output");
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (!treatAsProbabilities || classes != 2 || batch <= 0)
        {
            return output;
        }

        var a = aVec[0];
        var b = bVec[0];

        // Platt (1999): p1' = sigmoid(a * logit(p1) + b), with p1 kept inside (0, 1) so the logit is finite.
        var p1 = ClampProbability(PositiveClassProbabilities(probsTensor, batch));
        var logit = ResultEngine.TensorLog(ResultEngine.TensorDivide(p1, ResultEngine.TensorSubtract(ResultEngine.TensorOnesLike(p1), p1)));
        var calibratedPositive = ResultEngine.TensorSigmoid(ResultEngine.TensorAddScalar(ResultEngine.TensorMultiplyScalar(logit, a), b));
        return ConvertFromTensor(TwoClassProbabilities(calibratedPositive, batch, probsTensor._shape));
    }

    private TOutput ApplyIsotonicCalibrationToOutputProbabilities(TOutput output)
    {
        if (IsotonicCalibrationX is not { Length: > 0 } x || IsotonicCalibrationY is not { Length: > 0 } y)
        {
            return output;
        }

        var probsTensor = ConvertToTensorSafe(output, "model output");
        var probsVector = probsTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(probsTensor, probsVector);
        if (!treatAsProbabilities || classes != 2 || batch <= 0)
        {
            return output;
        }

        // The calibrated value is y[i] for the first breakpoint i with p1 <= x[i], or the last y when p1 exceeds them all.
        var p1 = ClampProbability(PositiveClassProbabilities(probsTensor, batch));
        int points = x.Length;
        var shape = new[] { batch, points };
        var below = ResultEngine.TensorLessEqual(
            ResultEngine.TensorBroadcastTo(ResultEngine.Reshape(p1, new[] { batch, 1 }), shape),
            ResultEngine.TensorBroadcastTo(ResultEngine.Reshape(Tensor<T>.FromVector(x), new[] { 1, points }), shape));
        int[] firstMatch = ResultEngine.TensorArgMax(below, 1).ToArray();
        int[] matches = ResultEngine.ReduceSum(below.Cast<int>(), new[] { 1 }, false).ToArray();
        var index = new int[batch];
        for (int i = 0; i < batch; i++)
        {
            index[i] = matches[i] > 0 ? firstMatch[i] : y.Length - 1;
        }

        var calibratedPositive = ResultEngine.TensorTake(Tensor<T>.FromVector(y), new Tensor<int>(index, new[] { batch }));
        return ConvertFromTensor(TwoClassProbabilities(calibratedPositive, batch, probsTensor._shape));
    }

    /// <summary>The positive-class (column 1) probabilities of a [batch, 2] probability tensor.</summary>
    private static Tensor<T> PositiveClassProbabilities(Tensor<T> probabilities, int batch)
    {
        var rows = ResultEngine.Reshape(probabilities, new[] { batch, 2 });
        return ResultEngine.Reshape(ResultEngine.TensorSlice(rows, new[] { 0, 1 }, new[] { batch, 1 }), new[] { batch });
    }

    /// <summary>[1 - p, p] per sample, reshaped to <paramref name="shape"/>.</summary>
    private static Tensor<T> TwoClassProbabilities(Tensor<T> positive, int batch, int[] shape)
    {
        var negative = ResultEngine.TensorSubtract(ResultEngine.TensorOnesLike(positive), positive);
        return ResultEngine.Reshape(ResultEngine.TensorStack(new[] { negative, positive }, 1), shape);
    }

    /// <summary>
    /// Clamps probabilities to [eps, 1 - eps] with eps = 1e-12 for double, and the spacing just below 1 for float
    /// (2^-24) and Half (2^-11), whose 1 - 1e-12 rounds back to 1 and would give an infinite logit.
    /// </summary>
    private static Tensor<T> ClampProbability(Tensor<T> probabilities)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double epsilon = typeof(T) == typeof(Half) ? 4.8828125e-4 : typeof(T) == typeof(float) ? 5.9604644775390625e-8 : 1e-12;
        T eps = numOps.FromDouble(epsilon);
        return ResultEngine.TensorClamp(probabilities, eps, numOps.Subtract(numOps.One, eps));
    }

    private static Tensor<T> SoftmaxTensor(Tensor<T> tensor, int batch, int classes)
    {
        if (classes <= 1)
        {
            return tensor;
        }

        var rows = ResultEngine.Reshape(tensor, new[] { batch, classes });
        return ResultEngine.Reshape(ResultEngine.TensorSoftmax(rows, 1), tensor._shape);
    }

    /// <summary>
    /// Attempts to denormalize variance based on the target preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When predictions are normalized, the variance also needs to be
    /// scaled by the square of the normalization scale factor. This method attempts to
    /// extract scaling information from the preprocessing pipeline to properly denormalize
    /// the variance estimates.
    /// </para>
    /// <para>
    /// If the target pipeline is not configured or does not expose scaling information,
    /// the original normalized variance is returned unchanged.
    /// </para>
    /// </remarks>
    private Tensor<T> DenormalizeVarianceIfSupported(Tensor<T> normalizedVariance)
    {
        // With the new PreprocessingPipeline pattern, scaling parameters are not directly
        // exposed. Future enhancement: add GetScaleFactor() to PreprocessingPipeline
        // to support variance denormalization.
        // For now, return the normalized variance unchanged.
        return normalizedVariance;
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
            return (TOutput)(object)new Tensor<T>(tensor._shape);
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
            object boxedZero = zero ?? throw new InvalidOperationException("Numeric zero value was unexpectedly null.");
            return (TOutput)boxedZero;
        }

        return default;
    }

    /// <summary>
    /// Safely converts a potentially nullable output value to a Tensor, throwing if the value is null.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <param name="context">Description of the value for error reporting.</param>
    /// <returns>The converted Tensor.</returns>
    private static Tensor<T> ConvertToTensorSafe<TValue>(TValue value, string context = "value")
    {
        if (value is null)
        {
            throw new InvalidOperationException(
                $"Cannot convert null {context} to Tensor. The model prediction returned null.");
        }

        return ConversionsHelper.ConvertToTensor<T>(value).Clone();
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

        var shape = samples[0]._shape;
        int length = samples[0].Length;
        var numOps = MathHelper.GetNumericOperations<T>();
        T count = numOps.FromDouble(samples.Count);

        // Elementwise mean and population variance over the samples.
        var stacked = ResultEngine.TensorStack(
            samples.Select(sample => ResultEngine.Reshape(sample, new[] { length })).ToArray(), 0);
        var mean = ResultEngine.TensorDivideScalar(ResultEngine.ReduceSum(stacked, new[] { 0 }, false), count);
        var deviations = ResultEngine.TensorSubtract(stacked, ResultEngine.TensorBroadcastTo(ResultEngine.Reshape(mean, new[] { 1, length }), new[] { samples.Count, length }));

        // Divide the deviations by their largest finite magnitude before squaring so the squares cannot overflow
        // (65504 in Half), then restore the scale; NaN deviations still propagate to their own element.
        var magnitudes = ResultEngine.TensorAbs(deviations);
        var finite = ResultEngine.TensorWhere(ResultEngine.TensorEquals(magnitudes, magnitudes), magnitudes, ResultEngine.TensorZerosLike(magnitudes));
        T scale = ResultEngine.TensorMaxValue(finite);
        Tensor<T> variance;
        if (numOps.Equals(scale, numOps.Zero) || numOps.IsInfinity(scale))
        {
            variance = ResultEngine.TensorDivideScalar(ResultEngine.ReduceSum(ResultEngine.TensorSquare(deviations), new[] { 0 }, false), count);
        }
        else
        {
            var scaled = ResultEngine.TensorSquare(ResultEngine.TensorDivideScalar(deviations, scale));
            var scaledVariance = ResultEngine.TensorDivideScalar(ResultEngine.ReduceSum(scaled, new[] { 0 }, false), count);
            variance = ResultEngine.TensorMultiplyScalar(ResultEngine.TensorMultiplyScalar(scaledVariance, scale), scale);
        }

        return (ResultEngine.Reshape(mean, shape), ResultEngine.Reshape(variance, shape));
    }

    private static Tensor<T> ApplyTemperatureScalingToProbabilityTensor(Tensor<T> probabilities, T temperature, int batch, int classes)
    {
        // Guo et al. (2017): softmax(log(p) / T) per sample, with p floored at eps so the log stays finite.
        var rows = ResultEngine.Reshape(probabilities, new[] { batch, classes });
        var eps = ResultEngine.TensorFullLike(rows, MatrixExtensions.SafeTolerance<T>(1e-12));
        var floored = ResultEngine.TensorWhere(ResultEngine.TensorLessThan(rows, eps), eps, rows);
        var logits = ResultEngine.TensorDivideScalar(ResultEngine.TensorLog(floored), temperature);
        return ResultEngine.Reshape(ResultEngine.TensorSoftmax(logits, 1), probabilities._shape);
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

        // Every value must lie in [0, 1] and every row must sum to 1 within 1e-2.
        var numOps = MathHelper.GetNumericOperations<T>();
        var rows = ProbabilityRows(flat, batch, classes);
        if (ResultEngine.TensorCountNonzero(ResultEngine.TensorLessThan(rows, numOps.Zero)) > 0 ||
            ResultEngine.TensorCountNonzero(ResultEngine.TensorGreaterThan(rows, numOps.One)) > 0)
        {
            return (false, batch, classes);
        }

        var rowSums = ResultEngine.ReduceSum(rows, new[] { 1 }, false);
        var offBy = ResultEngine.TensorAbs(ResultEngine.TensorSubtractScalar(rowSums, numOps.One));
        if (ResultEngine.TensorCountNonzero(ResultEngine.TensorGreaterThan(offBy, numOps.FromDouble(1e-2))) > 0)
        {
            return (false, batch, classes);
        }

        return (true, batch, classes);
    }

    private static Vector<T> ComputePerSampleEntropy(Vector<T> probsFlat, int batch, int classes)
    {
        // H = -sum_c p_c log p_c per sample, skipping probabilities below eps (where p log p -> 0).
        var probabilities = ProbabilityRows(probsFlat, batch, classes);
        var eps = ResultEngine.TensorFullLike(probabilities, MatrixExtensions.SafeTolerance<T>(1e-12));
        var skipped = ResultEngine.TensorLessThan(probabilities, eps);
        var safe = ResultEngine.TensorWhere(skipped, ResultEngine.TensorOnesLike(probabilities), probabilities);
        var terms = ResultEngine.TensorMultiply(probabilities, ResultEngine.TensorLog(safe));
        var kept = ResultEngine.TensorWhere(skipped, ResultEngine.TensorZerosLike(probabilities), terms);
        return ResultEngine.TensorNegate(ResultEngine.ReduceSum(kept, new[] { 1 }, false)).ToVector();
    }

    /// <summary>
    /// Predictive entropy H(mean) and mutual information max(0, H(mean) - mean_s H(sample_s)) per sample
    /// (Gal, 2016, section 3.3; the BALD decomposition).
    /// </summary>
    private static (Tensor<T> PredictiveEntropy, Tensor<T> MutualInformation) ComputeEntropyAndMutualInformation(
        List<Tensor<T>> samples, Vector<T> meanVector, int batch, int classes)
    {
        var predictive = Tensor<T>.FromVector(ComputePerSampleEntropy(meanVector, batch, classes));
        var perSample = ResultEngine.TensorStack(
            samples.Select(sample => Tensor<T>.FromVector(ComputePerSampleEntropy(sample.ToVector(), batch, classes))).ToArray(), 0);
        var expected = ResultEngine.TensorDivideScalar(
            ResultEngine.ReduceSum(perSample, new[] { 0 }, false), MathHelper.GetNumericOperations<T>().FromDouble(samples.Count));
        var difference = ResultEngine.TensorSubtract(predictive, expected);
        var zeros = ResultEngine.TensorZerosLike(difference);
        var mutualInformation = ResultEngine.TensorWhere(ResultEngine.TensorLessThan(difference, zeros), zeros, difference);
        return (ResultEngine.Reshape(predictive, new[] { batch }), ResultEngine.Reshape(mutualInformation, new[] { batch }));
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

            var normalizedPrediction = EnsureModel.Predict(normalizedInput);
            samples.Add(ConvertToTensorSafe(normalizedPrediction, "normalized prediction"));
        }

        var (meanTensor, varianceTensor) = ComputeMeanAndVariance(samples);

        var meanVector = meanTensor.ToVector();
        var (treatAsProbabilities, batch, classes) = InferProbabilityDistributionLayout(meanTensor, meanVector);
        if (!treatAsProbabilities || classes <= 1)
        {
            return (meanTensor, varianceTensor, CreateZeroVectorTensor(batch), CreateZeroVectorTensor(batch));
        }

        var (predictiveEntropy, mutualInformation) = ComputeEntropyAndMutualInformation(samples, meanVector, batch, classes);
        return (meanTensor, varianceTensor, predictiveEntropy, mutualInformation);
    }

    private static IEngine ResultEngine => AiDotNetEngine.Current;
}
