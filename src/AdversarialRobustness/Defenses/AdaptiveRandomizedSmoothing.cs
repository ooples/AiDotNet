using System.Text;
using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Certified Adversarial Robustness via Randomized Smoothing", "https://arxiv.org/abs/1902.02918", Year = 2019, Authors = "Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter")]
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
        if (minSigma <= 0) throw new ArgumentOutOfRangeException(nameof(minSigma), "minSigma must be positive.");
        if (maxSigma < minSigma) throw new ArgumentOutOfRangeException(nameof(maxSigma), "maxSigma must be >= minSigma.");
        if (sensitivitySamples <= 0) throw new ArgumentOutOfRangeException(nameof(sensitivitySamples), "sensitivitySamples must be positive.");

        _options = options;
        _minSigma = minSigma;
        _maxSigma = maxSigma;
        _sensitivitySamples = sensitivitySamples;

        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The noise level depends on the input, so the fixed-sigma certificate of Cohen et al. does not apply. The
    /// smoothed classifier here is the whole two-step mechanism: k probe queries x + N(0, sigma_min^2 I) choose
    /// sigma from the fraction of probes that change the clean prediction, then one query x + N(0, sigma^2 I) is
    /// classified. Every Monte Carlo sample runs both steps with fresh noise, so the counts estimate the
    /// probabilities of that mechanism and not those of one realized sigma.
    /// </para>
    /// <para>
    /// The steps are Gaussian mechanisms whose noise is chosen adaptively, so their composition (Gaussian
    /// differential privacy, Dong et al. 2019) holds only with a bound valid for every probe outcome (fully
    /// adaptive composition, Smith and Thakurta 2022): sigma is at least sigma_min, so the mechanism is
    /// r sqrt(k + 1) / sigma_min-GDP for inputs at L2 distance r, which gives the certified radius
    /// sigma_eff * Phi^-1(p_A_lower) with sigma_eff = sigma_min / sqrt(k + 1). The realized sigma(x) is not a
    /// sound substitute, because a nearby input can realize a smaller sigma. c_A is chosen from separate
    /// selection samples and p_A_lower is the one-sided Clopper-Pearson bound, as in CERTIFY.
    /// </para>
    /// </remarks>
    public CertifiedPrediction<T> CertifyPrediction(TInput input, IFullModel<T, TInput, TOutput> model)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (model == null) throw new ArgumentNullException(nameof(model));

        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        int baseClass = ArgMax(ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(input)));

        // CERTIFY over the two-step mechanism: selection samples pick c_A, fresh samples estimate p_A.
        var selection = SampleMechanism(vectorInput, input, model, baseClass, _options.SelectionSamples);
        int topClassLabel = selection.Counts
            .OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key).FirstOrDefault();
        var estimation = SampleMechanism(vectorInput, input, model, baseClass, _options.NumSamples);
        int topCount = estimation.Counts.TryGetValue(topClassLabel, out int observed) ? observed : 0;

        // Bounds and radius in double: counts above 2048 are not exact in Half, and a bound near 1 rounds to 1.
        double pALower = SmoothingStatistics.ClopperPearsonLower(topCount, _options.NumSamples, _options.ConfidenceLevel);
        double pAUpper = SmoothingStatistics.ClopperPearsonUpper(topCount, _options.NumSamples, _options.ConfidenceLevel);
        double effectiveSigma = EffectiveSigma();
        bool certified = pALower >= 0.5;
        double certifiedRadius = certified ? SmoothingStatistics.Radius(pALower, effectiveSigma) : 0.0;

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = topClassLabel,
            CertifiedRadius = NumOps.FromDouble(certifiedRadius),
            IsCertified = certified,
            Confidence = NumOps.FromDouble((double)topCount / _options.NumSamples),
            LowerBound = NumOps.FromDouble(pALower),
            UpperBound = NumOps.FromDouble(pAUpper)
        };

        result.CertificationDetails["SampleCount"] = _options.NumSamples;
        result.CertificationDetails["SelectionSampleCount"] = _options.SelectionSamples;
        result.CertificationDetails["AdaptiveSigma"] = estimation.MeanSigma;
        result.CertificationDetails["EffectiveSigma"] = effectiveSigma;
        result.CertificationDetails["TopClassCount"] = topCount;
        result.CertificationDetails["Abstained"] = !certified;
        result.CertificationDetails["Method"] = CertifiedDefenseMethod.AdaptiveRandomizedSmoothing;

        return result;
    }

    /// <summary>
    /// The single-Gaussian noise level that bounds the probe step (k queries at sigma_min) composed with the
    /// classification step (one query at a sigma of at least sigma_min) for every probe outcome:
    /// sigma_min / sqrt(k + 1).
    /// </summary>
    internal double EffectiveSigma() => _minSigma / Math.Sqrt(_sensitivitySamples + 1.0);

    /// <summary>
    /// Runs the two-step mechanism <paramref name="count"/> times and counts the classes; also returns the mean
    /// noise level the probes chose.
    /// </summary>
    private (Dictionary<int, int> Counts, double MeanSigma) SampleMechanism(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        int baseClass,
        int count)
    {
        var counts = new Dictionary<int, int>();
        double sigmaSum = 0.0;
        for (int i = 0; i < count; i++)
        {
            double sigma = ProbeSigma(vectorInput, referenceInput, model, baseClass);
            sigmaSum += sigma;
            int sampledClass = ClassifyNoisy(vectorInput, referenceInput, model, NumOps.FromDouble(sigma));
            counts[sampledClass] = counts.TryGetValue(sampledClass, out int seen) ? seen + 1 : 1;
        }

        return (counts, count > 0 ? sigmaSum / count : _minSigma);
    }

    private int ClassifyNoisy(Vector<T> vectorInput, TInput referenceInput, IFullModel<T, TInput, TOutput> model, T sigma)
    {
        var noisyInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(AddGaussianNoise(vectorInput, sigma), referenceInput);
        return ArgMax(ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(noisyInput)));
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
        if (testData.Length == 0)
            throw new ArgumentException("Test data must not be empty.", nameof(testData));
        if (testData.Length != labels.Length)
            throw new ArgumentException("Number of labels must match number of test samples.", nameof(labels));

        int cleanCorrect = 0, certified = 0;
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
    /// Clean and certified correctness for one input; an abstained prediction is never certified correct.
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
    public CertifiedDefenseOptions<T> GetOptions() => _options;

    /// <inheritdoc/>
    /// <remarks>
    /// No-op: this defense is stateless — the adaptive sigma is recomputed per-input
    /// during CertifyPrediction, so there is no trained state to reset.
    /// </remarks>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var state = new Newtonsoft.Json.Linq.JObject
        {
            ["options"] = Newtonsoft.Json.Linq.JToken.FromObject(_options),
            ["minSigma"] = _minSigma,
            ["maxSigma"] = _maxSigma,
            ["sensitivitySamples"] = _sensitivitySamples
        };
        var json = state.ToString(Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        if (data == null) throw new ArgumentNullException(nameof(data));
        var json = Encoding.UTF8.GetString(data);

        // Parse the JSON once
        var state = Newtonsoft.Json.Linq.JObject.Parse(json);

        // Check if state["options"] exists (new format)
        if (state["options"] is { } optionsToken)
        {
            _options = optionsToken.ToObject<CertifiedDefenseOptions<T>>() ?? new CertifiedDefenseOptions<T>();
            // Note: _minSigma, _maxSigma, _sensitivitySamples are readonly, so we can't restore them
            // after construction. The serialized values serve as documentation of the saved state.
            // For full round-trip fidelity, construct a new instance with the serialized parameters.
        }
        else
        {
            // Try to deserialize the entire state as legacy options-only format
            try
            {
                var legacyOptions = state.ToObject<CertifiedDefenseOptions<T>>();
                if (legacyOptions != null)
                {
                    _options = legacyOptions;
                }
                else
                {
                    throw new InvalidOperationException("Failed to deserialize state: neither new format (with 'options' key) nor legacy format (plain CertifiedDefenseOptions<T>) could be parsed.");
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to deserialize state: neither new format (with 'options' key) nor legacy format (plain CertifiedDefenseOptions<T>) could be parsed.", ex);
            }
        }
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            File.WriteAllBytes(filePath, Serialize());
        }
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(File.ReadAllBytes(filePath));
        }
    }

    /// <summary>
    /// The probe step: k queries x + N(0, sigma_min^2 I); sigma grows linearly from sigma_min to sigma_max with the
    /// fraction of probes whose class differs from the clean prediction (near a boundary, more noise).
    /// </summary>
    private double ProbeSigma(Vector<T> vectorInput, TInput input, IFullModel<T, TInput, TOutput> model, int baseClass)
    {
        int changedPredictions = 0;
        var probeSigma = NumOps.FromDouble(_minSigma);
        for (int i = 0; i < _sensitivitySamples; i++)
        {
            if (ClassifyNoisy(vectorInput, input, model, probeSigma) != baseClass)
            {
                changedPredictions++;
            }
        }

        double sensitivity = (double)changedPredictions / _sensitivitySamples;
        return _minSigma + sensitivity * (_maxSigma - _minSigma);
    }

    private Vector<T> AddGaussianNoise(Vector<T> input, T sigma)
    {
        // x + N(0, sigma^2 I), unclipped: the certificate is for the Gaussian-smoothed base classifier.
        var noise = Engine.GenerateGaussianNoise<T>(input.Length, NumOps.Zero, sigma, _random.Next());
        return Engine.Add<T>(input, noise);
    }

    private static int ArgMax(Vector<T> vector) => CertificationMath<T>.ArgMax(vector);
}
