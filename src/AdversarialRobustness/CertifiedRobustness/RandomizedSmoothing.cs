using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using System.Text;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// Implements Randomized Smoothing for certified robustness.
/// </summary>
/// <remarks>
/// <para>
/// Randomized Smoothing creates a smoothed classifier by averaging predictions over
/// Gaussian noise, enabling provable robustness guarantees.
/// </para>
/// <para><b>For Beginners:</b> Randomized Smoothing is like asking multiple slightly
/// different versions of a question and taking the majority vote. By adding random noise
/// to the input many times and seeing what the model predicts each time, we can
/// mathematically prove that the prediction is robust to small changes.</para>
/// <para>
/// Original paper: "Certified Adversarial Robustness via Randomized Smoothing"
/// by Cohen et al. (2019)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class RandomizedSmoothing<T, TInput, TOutput> : ICertifiedDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private CertifiedDefenseOptions<T> options;
    private readonly Random random;

    /// <summary>
    /// Initializes a new instance of Randomized Smoothing.
    /// </summary>
    /// <param name="options">The certified defense configuration options.</param>
    /// <remarks>
    /// <para>
    /// If <see cref="CertifiedDefenseOptions{T}.RandomSeed"/> is set, the random number generator
    /// is initialized with that seed for reproducible results. Otherwise, a non-deterministic
    /// random generator is used for proper statistical validity of the certification.
    /// </para>
    /// </remarks>
    public RandomizedSmoothing(CertifiedDefenseOptions<T> options)
    {
        this.options = options ?? throw new ArgumentNullException(nameof(options));

        // Use the configured seed if provided, otherwise use non-deterministic random
        // for proper statistical validity of the certification
        this.random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSeededRandom(Environment.TickCount);
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T> CertifyPrediction(TInput input, IFullModel<T, TInput, TOutput> model)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var sigma = NumOps.FromDouble(options.NoiseSigma);

        // Convert input to vector for noise operations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Sample predictions with Gaussian noise
        var classCounts = new Dictionary<int, int>();

        for (int i = 0; i < options.NumSamples; i++)
        {
            var noisyVector = AddGaussianNoise(vectorInput, sigma);
            var noisyInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(noisyVector, input);
            var output = model.Predict(noisyInput);
            var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);
            var sampledClass = ArgMax(outputVector);

            if (!classCounts.ContainsKey(sampledClass))
            {
                classCounts[sampledClass] = 0;
            }
            classCounts[sampledClass]++;
        }

        // Find the most frequently predicted class
        var topClassLabel = -1;
        var topCount = -1;
        foreach (var kv in classCounts)
        {
            if (kv.Value > topCount)
            {
                topClassLabel = kv.Key;
                topCount = kv.Value;
            }
        }

        // Compute certified radius using the lower confidence bound on pA
        // Per Cohen et al. (2019), we must use the lower bound to provide guaranteed certification:
        // R = σ * Φ⁻¹(pA_lower) where pA_lower is the Clopper-Pearson lower bound
        // Using the point estimate pA would overestimate the certified radius
        var pA = (double)topCount / options.NumSamples;
        var pA_lower = ComputeLowerBound(pA, options.NumSamples, options.ConfidenceLevel);
        var pA_upper = ComputeUpperBound(pA, options.NumSamples, options.ConfidenceLevel);
        var certifiedRadius = ComputeCertifiedRadius(pA_lower, sigma);

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = topClassLabel,
            CertifiedRadius = NumOps.FromDouble(certifiedRadius),
            IsCertified = certifiedRadius > 0,
            Confidence = pA,
            LowerBound = pA_lower,
            UpperBound = pA_upper
        };

        result.CertificationDetails["SampleCount"] = options.NumSamples;
        result.CertificationDetails["Sigma"] = options.NoiseSigma;
        result.CertificationDetails["TopClassCount"] = topCount;

        return result;
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T>[] CertifyBatch(TInput[] inputs, IFullModel<T, TInput, TOutput> model)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

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
        TInput[] testData,
        TOutput[] labels,
        IFullModel<T, TInput, TOutput> model,
        T radius)
    {
        if (testData == null)
        {
            throw new ArgumentNullException(nameof(testData));
        }

        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (testData.Length != labels.Length)
        {
            throw new ArgumentException("Number of labels must match number of test samples.", nameof(labels));
        }

        var metrics = new CertifiedAccuracyMetrics<T>();
        int cleanCorrect = 0;
        int certified = 0;
        var certifiedRadii = new List<double>();

        for (int i = 0; i < testData.Length; i++)
        {
            var input = testData[i];
            var label = labels[i];
            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(label);
            var trueClass = ArgMax(labelVector);

            // Clean accuracy
            var cleanOutput = model.Predict(input);
            var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
            var cleanPrediction = ArgMax(cleanOutputVector);
            if (cleanPrediction == trueClass)
            {
                cleanCorrect++;
            }

            // Certified prediction
            var certResult = CertifyPrediction(input, model);

            if (certResult.PredictedClass == trueClass && NumOps.GreaterThanOrEquals(certResult.CertifiedRadius, radius))
            {
                certified++;
            }

            if (certResult.IsCertified)
            {
                certifiedRadii.Add(NumOps.ToDouble(certResult.CertifiedRadius));
            }
        }

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Length;
        metrics.CertifiedAccuracy = (double)certified / testData.Length;
        metrics.CertificationRadius = radius;
        metrics.CertificationRate = (double)certifiedRadii.Count / testData.Length;

        if (certifiedRadii.Count > 0)
        {
            metrics.AverageCertifiedRadius = NumOps.FromDouble(certifiedRadii.Average());
            certifiedRadii.Sort();

            // Proper median calculation: for even-sized lists, average the two middle elements
            var count = certifiedRadii.Count;
            var mid = count / 2;
            metrics.MedianCertifiedRadius = count % 2 == 0
                ? NumOps.FromDouble((certifiedRadii[mid - 1] + certifiedRadii[mid]) / 2.0) // Even: average two middle
                : NumOps.FromDouble(certifiedRadii[mid]); // Odd: middle element
        }

        return metrics;
    }

    /// <inheritdoc/>
    public CertifiedDefenseOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        options = JsonConvert.DeserializeObject<CertifiedDefenseOptions<T>>(json) ?? new CertifiedDefenseOptions<T>();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    private Vector<T> AddGaussianNoise(Vector<T> input, T sigma)
    {
        // Generate Gaussian noise using Engine
        var noise = Engine.GenerateGaussianNoise<T>(
            input.Length,
            NumOps.Zero,
            sigma,
            random.Next());

        // Add noise to input: noisy = input + noise
        var noisy = Engine.Add<T>(input, noise);

        // Clip to valid range [0, 1]
        return Engine.Clamp<T>(noisy, NumOps.Zero, NumOps.One);
    }

    private double ComputeCertifiedRadius(double pA, T sigma)
    {
        // Radius formula: σ * Φ^(-1)(pA)
        // where Φ^(-1) is the inverse normal CDF

        if (pA <= 0.5)
        {
            return 0.0; // No certification possible
        }

        var sigmaValue = NumOps.ToDouble(sigma);
        var radius = sigmaValue * InverseNormalCDF(pA);
        return Math.Max(radius, 0.0);
    }

    private double InverseNormalCDF(double p)
    {
        // Delegate to the centralized StatisticsHelper implementation
        // for consistency across the codebase
        return NumOps.ToDouble(StatisticsHelper<T>.CalculateInverseNormalCDF(NumOps.FromDouble(p)));
    }

    private double ComputeLowerBound(double pA, int n, double confidence)
    {
        // Use exact Clopper-Pearson confidence interval via Beta distribution
        // This provides guaranteed coverage, unlike the normal approximation
        int successes = (int)Math.Round(pA * n);
        var interval = StatisticsHelper<T>.CalculateClopperPearsonInterval(successes, n, NumOps.FromDouble(confidence));
        return NumOps.ToDouble(interval.Lower);
    }

    private double ComputeUpperBound(double pA, int n, double confidence)
    {
        // Use exact Clopper-Pearson confidence interval via Beta distribution
        // This provides guaranteed coverage, unlike the normal approximation
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
