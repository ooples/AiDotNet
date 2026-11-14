using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Validation;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Certified Adversarial Robustness via Randomized Smoothing", "https://arxiv.org/abs/1902.02918", Year = 2019, Authors = "Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter")]
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
    /// If <see cref="CertifiedDefenseOptions{T}.Seed"/> is set, the random number generator
    /// is initialized with that seed for reproducible results. Otherwise, a non-deterministic
    /// random generator is used for proper statistical validity of the certification.
    /// </para>
    /// </remarks>
    public RandomizedSmoothing(CertifiedDefenseOptions<T> options)
    {
        Guard.NotNull(options);
        this.options = options;

        // Use the configured seed if provided, otherwise use non-deterministic random
        // for proper statistical validity of the certification
        this.random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// CERTIFY of Cohen et al. (2019): n0 = <see cref="CertifiedDefenseOptions{T}.SelectionSamples"/> noisy
    /// samples choose the top class c_A; n = <see cref="CertifiedDefenseOptions{T}.NumSamples"/> fresh samples
    /// count how often c_A wins, and the one-sided Clopper-Pearson bound at level alpha = 1 - confidence gives
    /// p_A_lower. The prediction is certified with radius sigma * Phi^-1(p_A_lower) when p_A_lower is at least
    /// 1/2; otherwise CERTIFY abstains, reported as <see cref="CertifiedPrediction{T}.IsCertified"/> = false.
    /// </para>
    /// </remarks>
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
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Selection and estimation use separate samples, so choosing c_A does not bias the estimate of p_A.
        var selectionCounts = SampleClassCounts(vectorInput, input, model, sigma, options.SelectionSamples);
        int topClass = RankByCount(selectionCounts).FirstOrDefault();
        var estimationCounts = SampleClassCounts(vectorInput, input, model, sigma, options.NumSamples);
        int topCount = estimationCounts.TryGetValue(topClass, out int observed) ? observed : 0;

        // The bounds, the radius and the frequency are computed in double: counts above 2048 are not exact in
        // Half, and a lower bound near 1 rounds to 1 there, where Phi^-1 is infinite.
        double lowerBound = SmoothingStatistics.ClopperPearsonLower(topCount, options.NumSamples, options.ConfidenceLevel);
        double upperBound = SmoothingStatistics.ClopperPearsonUpper(topCount, options.NumSamples, options.ConfidenceLevel);
        bool certified = lowerBound >= 0.5;
        double certifiedRadius = certified ? SmoothingStatistics.Radius(lowerBound, options.NoiseSigma) : 0.0;

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = topClass,
            CertifiedRadius = NumOps.FromDouble(certifiedRadius),
            IsCertified = certified,
            Confidence = NumOps.FromDouble((double)topCount / options.NumSamples),
            LowerBound = NumOps.FromDouble(lowerBound),
            UpperBound = NumOps.FromDouble(upperBound)
        };

        result.CertificationDetails["SampleCount"] = options.NumSamples;
        result.CertificationDetails["SelectionSampleCount"] = options.SelectionSamples;
        result.CertificationDetails["Sigma"] = options.NoiseSigma;
        result.CertificationDetails["TopClassCount"] = topCount;
        result.CertificationDetails["Abstained"] = !certified;

        return result;
    }

    /// <summary>
    /// PREDICT of Cohen et al. (2019): the smoothed classifier's class, or null when it abstains.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Draws <see cref="CertifiedDefenseOptions{T}.NumSamples"/> noisy samples, takes the two most frequent
    /// classes with counts n_A and n_B, and returns c_A only if the two-sided binomial test of n_A out of
    /// n_A + n_B against 1/2 has a p-value of at most alpha = 1 - confidence. The returned class then differs
    /// from the smoothed classifier's class with probability at most alpha.
    /// </para>
    /// <para><b>For Beginners:</b> If the vote between the top two classes is too close to call, the method
    /// says "I don't know" (null) instead of guessing.</para>
    /// </remarks>
    /// <param name="input">The input to classify.</param>
    /// <param name="model">The base classifier.</param>
    /// <returns>The predicted class, or null to abstain.</returns>
    public int? PredictWithAbstention(TInput input, IFullModel<T, TInput, TOutput> model)
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
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        var counts = SampleClassCounts(vectorInput, input, model, sigma, options.NumSamples);
        var ranked = RankByCount(counts);
        if (ranked.Count == 0)
        {
            return null;
        }

        int topCount = counts[ranked[0]];
        int runnerUpCount = ranked.Count > 1 ? counts[ranked[1]] : 0;
        double pValue = SmoothingStatistics.TwoSidedBinomialPValue(topCount, topCount + runnerUpCount);
        return pValue <= 1.0 - options.ConfidenceLevel ? ranked[0] : null;
    }

    /// <summary>
    /// The two-sided binomial test p-value of k successes in n trials against p = 1/2 (scipy binom_test),
    /// for k at least n / 2: min(1, 2 P(X >= k)), with P(X >= k) = I_{1/2}(k, n - k + 1).
    /// </summary>
    internal static T TwoSidedBinomialPValue(int successes, int trials)
        => NumOps.FromDouble(SmoothingStatistics.TwoSidedBinomialPValue(successes, trials));

    /// <summary>
    /// Classifies <paramref name="count"/> Gaussian-noise samples x + N(0, sigma^2 I) and counts each class.
    /// </summary>
    /// <remarks>
    /// The noise is drawn <see cref="CertifiedDefenseOptions{T}.BatchSize"/> samples at a time, one engine draw
    /// per batch, as Cohen et al.'s <c>_sample_noise</c> does.
    /// </remarks>
    private Dictionary<int, int> SampleClassCounts(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T sigma,
        int count)
    {
        int batchSize = ValidatedBatchSize(options.BatchSize);
        int dimension = vectorInput.Length;
        var counts = new Dictionary<int, int>();
        for (int first = 0; first < count; first += batchSize)
        {
            int batch = Math.Min(batchSize, count - first);
            var noise = Engine.GenerateGaussianNoise<T>(batch * dimension, NumOps.Zero, sigma, random.Next()).AsTensor();
            for (int b = 0; b < batch; b++)
            {
                // x + N(0, sigma^2 I), unclipped: clipping would smooth a different base classifier than the one
                // the radius is reported for.
                var row = Engine.TensorSlice(noise, new[] { b * dimension }, new[] { dimension }).ToVector();
                var noisyInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(Engine.Add<T>(vectorInput, row), referenceInput);
                int sampledClass = ArgMax(ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(noisyInput)));
                counts[sampledClass] = counts.TryGetValue(sampledClass, out int seen) ? seen + 1 : 1;
            }
        }

        return counts;
    }

    /// <summary>The configured noise batch size; it must be positive.</summary>
    internal static int ValidatedBatchSize(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CertifiedDefenseOptions<T>.BatchSize), batchSize, "BatchSize must be at least 1.");
        }

        return batchSize;
    }

    /// <summary>
    /// Classes by decreasing count; ties go to the smaller class index (numpy argmax).
    /// </summary>
    private static List<int> RankByCount(Dictionary<int, int> counts)
    {
        return counts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key).ToList();
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

        int cleanCorrect = 0;
        int certified = 0;
        var radii = new List<T>();
        int certifiedCount = 0;

        for (int i = 0; i < testData.Length; i++)
        {
            var (clean, certifiedAtRadius, result) = EvaluateOne(testData[i], labels[i], model, radius);
            cleanCorrect += clean ? 1 : 0;
            certified += certifiedAtRadius ? 1 : 0;
            certifiedCount += result.IsCertified ? 1 : 0;
            radii.Add(result.IsCertified ? result.CertifiedRadius : NumOps.Zero);
        }

        return CertificationMath<T>.Summarize(cleanCorrect, certified, certifiedCount, radius, radii);
    }

    /// <summary>
    /// Clean correctness and certified correctness at <paramref name="radius"/> for one input. An abstained
    /// prediction is never certified correct (Cohen et al. 2019, Sec. 4: certified accuracy counts inputs
    /// that CERTIFY classifies correctly with a radius of at least r).
    /// </summary>
    private (bool Clean, bool Certified, CertifiedPrediction<T> Result) EvaluateOne(
        TInput input,
        TOutput label,
        IFullModel<T, TInput, TOutput> model,
        T radius)
    {
        int trueClass = ArgMax(ConversionsHelper.ConvertToVector<T, TOutput>(label));
        bool clean = ArgMax(ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(input))) == trueClass;
        var result = CertifyPrediction(input, model);
        bool certifiedCorrect = result.IsCertified
            && result.PredictedClass == trueClass
            && NumOps.GreaterThanOrEquals(result.CertifiedRadius, radius);
        return (clean, certifiedCorrect, result);
    }

    /// <inheritdoc/>
    public CertifiedDefenseOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
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
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            File.WriteAllBytes(filePath, Serialize());
        }
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(File.ReadAllBytes(filePath));
        }
    }

    private static int ArgMax(Vector<T> vector) => CertificationMath<T>.ArgMax(vector);
}
