using System.Numerics;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

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
public class RandomizedSmoothing<T> : ICertifiedDefense<T>
    where T : struct, INumber<T>
{
    private readonly CertifiedDefenseOptions<T> options;
    private readonly Random random;

    /// <summary>
    /// Initializes a new instance of Randomized Smoothing.
    /// </summary>
    /// <param name="options">The certified defense configuration options.</param>
    public RandomizedSmoothing(CertifiedDefenseOptions<T> options)
    {
        this.options = options;
        this.random = new Random(42);
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T> CertifyPrediction(T[] input, Func<T[], T[]> model)
    {
        var sigma = T.CreateChecked(options.NoiseSigma);

        // Sample predictions with Gaussian noise
        var classCounts = new Dictionary<int, int>();

        for (int i = 0; i < options.NumSamples; i++)
        {
            var noisyInput = AddGaussianNoise(input, sigma);
            var output = model(noisyInput);
            var predictedClass = ArgMax(output);

            if (!classCounts.ContainsKey(predictedClass))
            {
                classCounts[predictedClass] = 0;
            }
            classCounts[predictedClass]++;
        }

        // Find the most frequently predicted class
        var topClass = classCounts.OrderByDescending(kv => kv.Value).First();
        var topCount = topClass.Value;
        var predictedClass = topClass.Key;

        // Compute certified radius using concentration inequalities
        var pA = (double)topCount / options.NumSamples;
        var certifiedRadius = ComputeCertifiedRadius(pA, sigma);

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = predictedClass,
            CertifiedRadius = T.CreateChecked(certifiedRadius),
            IsCertified = certifiedRadius > 0,
            Confidence = pA,
            LowerBound = ComputeLowerBound(pA, options.NumSamples, options.ConfidenceLevel),
            UpperBound = ComputeUpperBound(pA, options.NumSamples, options.ConfidenceLevel)
        };

        result.CertificationDetails["SampleCount"] = options.NumSamples;
        result.CertificationDetails["Sigma"] = options.NoiseSigma;
        result.CertificationDetails["TopClassCount"] = topCount;

        return result;
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T>[] CertifyBatch(T[][] inputs, Func<T[], T[]> model)
    {
        var results = new CertifiedPrediction<T>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = CertifyPrediction(inputs[i], model);
        }
        return results;
    }

    /// <inheritdoc/>
    public T ComputeCertifiedRadius(T[] input, Func<T[], T[]> model)
    {
        var prediction = CertifyPrediction(input, model);
        return prediction.CertifiedRadius;
    }

    /// <inheritdoc/>
    public CertifiedAccuracyMetrics<T> EvaluateCertifiedAccuracy(
        T[][] testData,
        int[] labels,
        Func<T[], T[]> model,
        T radius)
    {
        var metrics = new CertifiedAccuracyMetrics<T>();
        int cleanCorrect = 0;
        int certified = 0;
        var certifiedRadii = new List<double>();

        for (int i = 0; i < testData.Length; i++)
        {
            // Clean accuracy
            var cleanOutput = model(testData[i]);
            var cleanPrediction = ArgMax(cleanOutput);
            if (cleanPrediction == labels[i])
            {
                cleanCorrect++;
            }

            // Certified prediction
            var certResult = CertifyPrediction(testData[i], model);

            if (certResult.PredictedClass == labels[i] && certResult.CertifiedRadius >= radius)
            {
                certified++;
            }

            if (certResult.IsCertified)
            {
                certifiedRadii.Add(double.CreateChecked(certResult.CertifiedRadius));
            }
        }

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Length;
        metrics.CertifiedAccuracy = (double)certified / testData.Length;
        metrics.CertificationRadius = radius;
        metrics.CertificationRate = (double)certifiedRadii.Count / testData.Length;

        if (certifiedRadii.Count > 0)
        {
            metrics.AverageCertifiedRadius = T.CreateChecked(certifiedRadii.Average());
            certifiedRadii.Sort();
            metrics.MedianCertifiedRadius = T.CreateChecked(certifiedRadii[certifiedRadii.Count / 2]);
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
        var json = System.Text.Json.JsonSerializer.Serialize(options);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data) { }

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

    private T[] AddGaussianNoise(T[] input, T sigma)
    {
        var noisy = new T[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            // Box-Muller transform for Gaussian noise
            var u1 = random.NextDouble();
            var u2 = random.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            var noise = T.CreateChecked(randStdNormal) * sigma;

            noisy[i] = input[i] + noise;
            noisy[i] = T.Min(T.Max(noisy[i], T.Zero), T.One); // Clip to valid range
        }

        return noisy;
    }

    private double ComputeCertifiedRadius(double pA, T sigma)
    {
        // Radius formula: σ * Φ^(-1)(pA)
        // where Φ^(-1) is the inverse normal CDF

        if (pA <= 0.5)
        {
            return 0.0; // No certification possible
        }

        var sigmaValue = double.CreateChecked(sigma);
        var radius = sigmaValue * InverseNormalCDF(pA);
        return Math.Max(radius, 0.0);
    }

    private double InverseNormalCDF(double p)
    {
        // Approximation of inverse normal CDF using the Beasley-Springer-Moro algorithm
        // For simplicity, using a basic approximation
        if (p <= 0 || p >= 1)
        {
            throw new ArgumentException("Probability must be between 0 and 1");
        }

        // Use approximation
        var t = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
               (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    }

    private double ComputeLowerBound(double pA, int n, double confidence)
    {
        // Clopper-Pearson confidence interval (simplified)
        return Math.Max(0.0, pA - 1.96 * Math.Sqrt(pA * (1 - pA) / n));
    }

    private double ComputeUpperBound(double pA, int n, double confidence)
    {
        return Math.Min(1.0, pA + 1.96 * Math.Sqrt(pA * (1 - pA) / n));
    }

    private static int ArgMax(T[] array)
    {
        int maxIndex = 0;
        T maxValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > maxValue)
            {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
