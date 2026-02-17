using System.Text;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.AdversarialRobustness.Defenses;

/// <summary>
/// Implements Adaptive Randomized Smoothing with f-Differential Privacy (f-DP) certified defense.
/// </summary>
/// <remarks>
/// <para>
/// Extends standard Randomized Smoothing by adapting the noise distribution per-input based
/// on a local sensitivity estimate. Instead of using a fixed Gaussian sigma, the noise level
/// is scaled according to the model's local Lipschitz constant around each input. This provides
/// tighter certified radii for "easy" inputs while maintaining coverage for difficult ones.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard Randomized Smoothing adds the same amount of noise to every
/// input. But some inputs are "easy" (far from the decision boundary) and don't need much noise,
/// while "hard" inputs (near the boundary) need more. This adaptive version adjusts the noise
/// level for each input — like adjusting your seatbelt based on road conditions rather than
/// always wearing the tightest setting.
/// </para>
/// <para>
/// <b>References:</b>
/// - Adaptive Randomized Smoothing: f-DP certified defense (NeurIPS 2024)
/// - Certified Adversarial Robustness via Randomized Smoothing (Cohen et al., 2019)
/// - Tight second-order certificates for randomized smoothing (Mohapatra et al., 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class AdaptiveRandomizedSmoothing<T, TInput, TOutput> : ICertifiedDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private CertifiedDefenseOptions<T> _options;
    private readonly Random _random;
    private readonly double _minSigma;
    private readonly double _maxSigma;
    private readonly int _sensitivitySamples;

    /// <summary>
    /// Initializes a new instance of Adaptive Randomized Smoothing.
    /// </summary>
    /// <param name="options">Base certified defense configuration.</param>
    /// <param name="minSigma">Minimum noise sigma (for easy inputs). Default: 0.1.</param>
    /// <param name="maxSigma">Maximum noise sigma (for hard inputs). Default: 1.0.</param>
    /// <param name="sensitivitySamples">
    /// Number of samples for estimating local sensitivity. Default: 50.
    /// Higher = more accurate sensitivity estimate but slower.
    /// </param>
    public AdaptiveRandomizedSmoothing(
        CertifiedDefenseOptions<T> options,
        double minSigma = 0.1,
        double maxSigma = 1.0,
        int sensitivitySamples = 50)
    {
        Guard.NotNull(options);
        _options = options;
        _minSigma = minSigma;
        _maxSigma = maxSigma;
        _sensitivitySamples = sensitivitySamples;

        _random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSeededRandom(Environment.TickCount);
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T> CertifyPrediction(TInput input, IFullModel<T, TInput, TOutput> model)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (model == null) throw new ArgumentNullException(nameof(model));

        // Step 1: Estimate local sensitivity to adapt sigma
        double adaptiveSigma = EstimateAdaptiveSigma(input, model);
        var sigma = NumOps.FromDouble(adaptiveSigma);

        // Step 2: Convert input to vector for noise operations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Step 3: Sample predictions with adapted Gaussian noise
        var classCounts = new Dictionary<int, int>();

        for (int i = 0; i < _options.NumSamples; i++)
        {
            var noisyVector = AddGaussianNoise(vectorInput, sigma);
            var noisyInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(noisyVector, input);
            var output = model.Predict(noisyInput);
            var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);
            var sampledClass = ArgMax(outputVector);

            if (!classCounts.ContainsKey(sampledClass))
                classCounts[sampledClass] = 0;
            classCounts[sampledClass]++;
        }

        // Step 4: Find the most frequently predicted class
        int topClassLabel = -1;
        int topCount = -1;
        foreach (var kv in classCounts)
        {
            if (kv.Value > topCount)
            {
                topClassLabel = kv.Key;
                topCount = kv.Value;
            }
        }

        // Step 5: Compute certified radius using adapted sigma
        double pA = (double)topCount / _options.NumSamples;
        double pALower = ComputeLowerBound(pA, _options.NumSamples, _options.ConfidenceLevel);
        double pAUpper = ComputeUpperBound(pA, _options.NumSamples, _options.ConfidenceLevel);
        double certifiedRadius = ComputeRadius(pALower, adaptiveSigma);

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = topClassLabel,
            CertifiedRadius = NumOps.FromDouble(certifiedRadius),
            IsCertified = certifiedRadius > 0,
            Confidence = pA,
            LowerBound = pALower,
            UpperBound = pAUpper
        };

        result.CertificationDetails["SampleCount"] = _options.NumSamples;
        result.CertificationDetails["AdaptiveSigma"] = adaptiveSigma;
        result.CertificationDetails["TopClassCount"] = topCount;
        result.CertificationDetails["Method"] = "AdaptiveRandomizedSmoothing";

        return result;
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T>[] CertifyBatch(TInput[] inputs, IFullModel<T, TInput, TOutput> model)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));

        var results = new CertifiedPrediction<T>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = CertifyPrediction(inputs[i], model);
        }
        return results;
    }

    /// <inheritdoc/>
    public T ComputeCertifiedRadius(TInput input, IFullModel<T, TInput, TOutput> model)
    {
        var prediction = CertifyPrediction(input, model);
        return prediction.CertifiedRadius;
    }

    /// <inheritdoc/>
    public CertifiedAccuracyMetrics<T> EvaluateCertifiedAccuracy(
        TInput[] testData, TOutput[] labels, IFullModel<T, TInput, TOutput> model, T radius)
    {
        if (testData == null) throw new ArgumentNullException(nameof(testData));
        if (labels == null) throw new ArgumentNullException(nameof(labels));
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (testData.Length != labels.Length)
            throw new ArgumentException("Number of labels must match number of test samples.", nameof(labels));

        var metrics = new CertifiedAccuracyMetrics<T>();
        int cleanCorrect = 0, certified = 0;
        var certifiedRadii = new List<double>();

        for (int i = 0; i < testData.Length; i++)
        {
            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(labels[i]);
            int trueClass = ArgMax(labelVector);

            // Clean accuracy
            var cleanOutput = model.Predict(testData[i]);
            var cleanVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
            if (ArgMax(cleanVector) == trueClass) cleanCorrect++;

            // Certified prediction
            var certResult = CertifyPrediction(testData[i], model);
            if (certResult.PredictedClass == trueClass && NumOps.GreaterThanOrEquals(certResult.CertifiedRadius, radius))
                certified++;

            if (certResult.IsCertified)
                certifiedRadii.Add(NumOps.ToDouble(certResult.CertifiedRadius));
        }

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Length;
        metrics.CertifiedAccuracy = (double)certified / testData.Length;
        metrics.CertificationRadius = radius;
        metrics.CertificationRate = (double)certifiedRadii.Count / testData.Length;

        if (certifiedRadii.Count > 0)
        {
            metrics.AverageCertifiedRadius = NumOps.FromDouble(certifiedRadii.Average());
            certifiedRadii.Sort();
            int count = certifiedRadii.Count;
            int mid = count / 2;
            metrics.MedianCertifiedRadius = count % 2 == 0
                ? NumOps.FromDouble((certifiedRadii[mid - 1] + certifiedRadii[mid]) / 2.0)
                : NumOps.FromDouble(certifiedRadii[mid]);
        }

        return metrics;
    }

    /// <inheritdoc/>
    public CertifiedDefenseOptions<T> GetOptions() => _options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(_options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        var json = Encoding.UTF8.GetString(data);
        _options = JsonConvert.DeserializeObject<CertifiedDefenseOptions<T>>(json) ?? new CertifiedDefenseOptions<T>();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());

    /// <inheritdoc/>
    public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

    private double EstimateAdaptiveSigma(TInput input, IFullModel<T, TInput, TOutput> model)
    {
        // Estimate local sensitivity by probing model response to small perturbations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Get base prediction
        var baseOutput = model.Predict(input);
        var baseVector = ConversionsHelper.ConvertToVector<T, TOutput>(baseOutput);
        int baseClass = ArgMax(baseVector);

        // Count how many perturbed inputs change the prediction
        int changedPredictions = 0;
        double probeSigma = _minSigma;

        for (int i = 0; i < _sensitivitySamples; i++)
        {
            var noisy = AddGaussianNoise(vectorInput, NumOps.FromDouble(probeSigma));
            var noisyInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(noisy, input);
            var noisyOutput = model.Predict(noisyInput);
            var noisyVector = ConversionsHelper.ConvertToVector<T, TOutput>(noisyOutput);

            if (ArgMax(noisyVector) != baseClass)
            {
                changedPredictions++;
            }
        }

        // Sensitivity: fraction of predictions that changed
        double sensitivity = (double)changedPredictions / _sensitivitySamples;

        // Adapt sigma: high sensitivity (near boundary) → use more noise
        // Low sensitivity (far from boundary) → use less noise
        double adaptedSigma = _minSigma + sensitivity * (_maxSigma - _minSigma);

        return adaptedSigma;
    }

    private Vector<T> AddGaussianNoise(Vector<T> input, T sigma)
    {
        var noise = Engine.GenerateGaussianNoise<T>(input.Length, NumOps.Zero, sigma, _random.Next());
        var noisy = Engine.Add<T>(input, noise);
        return Engine.Clamp<T>(noisy, NumOps.Zero, NumOps.One);
    }

    private double ComputeRadius(double pA, double sigma)
    {
        if (pA <= 0.5) return 0.0;
        double inverseNormal = NumOps.ToDouble(StatisticsHelper<T>.CalculateInverseNormalCDF(NumOps.FromDouble(pA)));
        return Math.Max(sigma * inverseNormal, 0.0);
    }

    private double ComputeLowerBound(double pA, int n, double confidence)
    {
        int successes = (int)Math.Round(pA * n);
        var interval = StatisticsHelper<T>.CalculateClopperPearsonInterval(successes, n, NumOps.FromDouble(confidence));
        return NumOps.ToDouble(interval.Lower);
    }

    private double ComputeUpperBound(double pA, int n, double confidence)
    {
        int successes = (int)Math.Round(pA * n);
        var interval = StatisticsHelper<T>.CalculateClopperPearsonInterval(successes, n, NumOps.FromDouble(confidence));
        return NumOps.ToDouble(interval.Upper);
    }

    private static int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        T maxValue = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
