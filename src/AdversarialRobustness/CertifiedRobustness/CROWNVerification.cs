using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Serialization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// Implements CROWN (Convex Relaxation based perturbation analysis Of Neural networks)
/// for computing certified robustness bounds.
/// </summary>
/// <remarks>
/// <para>
/// CROWN is a state-of-the-art neural network verification technique that computes
/// tighter certified bounds than IBP by using linear relaxation of non-linear activations.
/// It works by propagating linear upper and lower bounds backward through the network.
/// </para>
/// <para><b>For Beginners:</b> CROWN finds the smallest "safety box" around a prediction
/// where we can guarantee the model's answer won't change. It's more precise than IBP
/// because it uses smarter mathematical approximations.</para>
/// <para>
/// <b>Mathematical Foundation:</b>
/// For a ReLU activation σ(x) = max(0, x) with bounds [l, u]:
///
/// Case 1: l ≥ 0 (always active): σ(x) = x
/// Case 2: u ≤ 0 (always inactive): σ(x) = 0
/// Case 3: l &lt; 0 &lt; u (crossing): Use linear relaxation
///   - Upper bound: σ(x) ≤ u(x - l)/(u - l)
///   - Lower bound: σ(x) ≥ 0 or σ(x) ≥ x (choose tighter)
///
/// CROWN propagates these linear bounds backward through the network to get
/// tighter output bounds than IBP's forward propagation alone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Efficient Neural Network Robustness Certification with General Activation Functions", "https://arxiv.org/abs/1811.00866", Year = 2018, Authors = "Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel")]
public class CROWNVerification<T, TInput, TOutput> : ICertifiedDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private CertifiedDefenseOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the CROWNVerification class with default options.
    /// </summary>
    public CROWNVerification()
    {
        _options = new CertifiedDefenseOptions<T>
        {
            CertificationMethod = CertifiedDefenseMethod.Crown,
            UseTightBounds = true
        };
    }

    /// <summary>
    /// Initializes a new instance of the CROWNVerification class with specified options.
    /// </summary>
    /// <param name="options">The configuration options for CROWN.</param>
    public CROWNVerification(CertifiedDefenseOptions<T> options)
    {
        Guard.NotNull(options);
        _options = options;
        _options.CertificationMethod = CertifiedDefenseMethod.Crown;
        _options.UseTightBounds = true;
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T> CertifyPrediction(
        TInput input,
        IFullModel<T, TInput, TOutput> model)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        // Convert input to vector for bound computations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Get the perturbation radius from options
        T epsilon = NumOps.FromDouble(_options.NoiseSigma);

        // Get the clean prediction
        var cleanOutput = model.Predict(input);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int predictedClass = GetPredictedClass(cleanOutputVector);

        // Bound the network over the ball and check the margins of the predicted class
        var bounds = ComputeCROWNBounds(vectorInput, input, model, epsilon, predictedClass);

        // Compute the certified radius
        T certifiedRadius = ComputeCertifiedRadiusInternal(vectorInput, input, model, predictedClass);

        var result = new CertifiedPrediction<T>
        {
            PredictedClass = predictedClass,
            IsCertified = bounds.Certified,
            CertifiedRadius = certifiedRadius,
            Confidence = ComputeConfidence(bounds.MinimumMargin),
            LowerBound = bounds.Lower.AsSpan()[predictedClass],
            UpperBound = bounds.Upper.AsSpan()[predictedClass]
        };

        result.CertificationDetails["Method"] = CertifiedDefenseMethod.Crown;
        result.CertificationDetails["NormType"] = _options.NormType;
        if (bounds.MinimumMargin is { } marginLowerBound)
        {
            result.CertificationDetails["MarginLowerBound"] = marginLowerBound;
        }

        result.CertificationDetails["Sound"] = bounds.Sound;
        return result;
    }

    /// <inheritdoc/>
    public CertifiedPrediction<T>[] CertifyBatch(
        TInput[] inputs,
        IFullModel<T, TInput, TOutput> model)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var results = new CertifiedPrediction<T>[inputs.Length];

        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = CertifyPrediction(inputs[i], model);
        }

        return results;
    }

    /// <inheritdoc/>
    public T ComputeCertifiedRadius(
        TInput input,
        IFullModel<T, TInput, TOutput> model)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        // Convert input to vector
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        // Get the clean prediction
        var cleanOutput = model.Predict(input);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int predictedClass = GetPredictedClass(cleanOutputVector);

        return ComputeCertifiedRadiusInternal(vectorInput, input, model, predictedClass);
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
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        int cleanCorrect = 0, certifiedCorrect = 0, certifiedCount = 0;
        var radii = new List<T>();
        for (int i = 0; i < testData.Length; i++)
        {
            var trueClass = GetPredictedClass(ConversionsHelper.ConvertToVector<T, TOutput>(labels[i]));
            var certification = CertifyPrediction(testData[i], model);
            bool correct = certification.PredictedClass == trueClass;
            cleanCorrect += correct ? 1 : 0;
            certifiedCount += CertifiedSomewhere(certification) ? 1 : 0;
            certifiedCorrect += correct && CertifiedAt(certification, radius) ? 1 : 0;
            radii.Add(certification.CertifiedRadius);
        }

        return CertificationMath<T>.Summarize(cleanCorrect, certifiedCorrect, certifiedCount, radius, radii);
    }

    /// <summary>
    /// Certified at the evaluation radius: the certified radius reaches it. <see cref="CertifiedPrediction{T}.IsCertified"/>
    /// refers to the radius of the options (NoiseSigma), which need not be the evaluation radius.
    /// </summary>
    private static bool CertifiedAt(CertifiedPrediction<T> certification, T radius) =>
        CertifiedSomewhere(certification) && NumOps.GreaterThanOrEquals(certification.CertifiedRadius, radius);

    /// <summary>Certified at some radius: the certified radius is positive.</summary>
    private static bool CertifiedSomewhere(CertifiedPrediction<T> certification) =>
        NumOps.GreaterThan(certification.CertifiedRadius, NumOps.Zero);

    /// <inheritdoc/>
    public CertifiedDefenseOptions<T> GetOptions()
    {
        return _options;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _options = new CertifiedDefenseOptions<T>
        {
            CertificationMethod = CertifiedDefenseMethod.Crown,
            UseTightBounds = true
        };
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var data = new SerializationData
        {
            NoiseSigma = _options.NoiseSigma,
            ConfidenceLevel = _options.ConfidenceLevel,
            NumSamples = _options.NumSamples,
            NormType = _options.NormType,
            UseTightBounds = _options.UseTightBounds,
            BatchSize = _options.BatchSize,
            RandomSeed = _options.Seed
        };

        var json = JsonConvert.SerializeObject(data, Formatting.None);
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

        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.None,
            SerializationBinder = new SafeSerializationBinder()
        };

        var deserialized = JsonConvert.DeserializeObject<SerializationData>(json, settings);
        if (deserialized != null)
        {
            _options = new CertifiedDefenseOptions<T>
            {
                NoiseSigma = deserialized.NoiseSigma,
                ConfidenceLevel = deserialized.ConfidenceLevel,
                NumSamples = deserialized.NumSamples,
                NormType = deserialized.NormType,
                UseTightBounds = deserialized.UseTightBounds,
                BatchSize = deserialized.BatchSize,
                Seed = deserialized.RandomSeed,
                CertificationMethod = CertifiedDefenseMethod.Crown
            };
        }
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        Deserialize(File.ReadAllBytes(fullPath));
    }

    /// <summary>
    /// The outcome of bounding the network at one radius.
    /// </summary>
    private readonly struct BoundResult
    {
        public BoundResult(Vector<T> lower, Vector<T> upper, T minimumMargin, bool certified, bool sound)
        {
            Lower = lower;
            Upper = upper;
            MinimumMargin = minimumMargin;
            Certified = certified;
            Sound = sound;
        }

        /// <summary>Lower bounds of every network output.</summary>
        public Vector<T> Lower { get; }

        /// <summary>Upper bounds of every network output.</summary>
        public Vector<T> Upper { get; }

        /// <summary>Lower bound of min over j != y of z_y - z_j.</summary>
        public T MinimumMargin { get; }

        public bool Certified { get; }

        /// <summary>True for layer-wise CROWN (or IBP) bounds; false for the sampled (empirical, not sound) fallback.</summary>
        public bool Sound { get; }
    }

    /// <summary>
    /// Bounds the network over the perturbation ball and decides whether class <paramref name="predictedClass"/>
    /// is certified.
    /// </summary>
    /// <remarks>
    /// With layer access this is CROWN (Zhang et al. 2018): the margin specification z_y - z_j is propagated
    /// backwards through every affine layer and through linear lower/upper relaxations of every activation
    /// (adaptive ReLU lower slope, chord and tangent lines for sigmoid and tanh), with the pre-activation bounds
    /// of each hidden layer computed by CROWN itself, and the resulting linear function is bounded over the
    /// ball with the dual norm. A softmax inside the network has no element-wise relaxation, so such networks
    /// are bounded with IBP. Without layer access the bounds come from sampling, which is empirical and
    /// reported as not sound.
    /// </remarks>
    private BoundResult ComputeCROWNBounds(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon,
        int predictedClass)
    {
        var network = BoundNetwork<T>.FromModel(model, vectorInput.Length);
        int outputs = network?.OutputLength(vectorInput.Length) ?? 0;
        if (network is not null && predictedClass < outputs)
        {
            var x = vectorInput.AsTensor();
            var norm = _options.NormType;
            Tensor<T> topLogitLower;
            var margins = network.SupportsCrown
                ? network.CrownMargins(x, epsilon, norm, predictedClass, out topLogitLower)
                : network.IntervalMargins(x, epsilon, norm, predictedClass, out topLogitLower);
            var (lower, upper) = network.SupportsCrown
                ? network.CrownLogitBounds(x, epsilon, norm)
                : network.IntervalBounds(x, epsilon, norm, network.Operations.Count);
            T minimumMargin = Engine.TensorMinValue(margins);
            bool certified = NumOps.GreaterThan(minimumMargin, NumOps.Zero)
                && NumOps.GreaterThan(topLogitLower.GetFlat(0), NumOps.Zero);
            return new BoundResult(lower.ToVector(), upper.ToVector(), minimumMargin, certified, true);
        }

        var (sampledLower, sampledUpper) = ApproximateBoundsWithSampling(vectorInput, referenceInput, model, epsilon);
        T sampledMargin = ComputeMinimumMargin(sampledLower, sampledUpper, predictedClass);
        return new BoundResult(
            sampledLower,
            sampledUpper,
            sampledMargin,
            CheckCertification(sampledLower, sampledUpper, predictedClass),
            false);
    }

    /// <summary>
    /// Approximates output bounds using sampling when layer access is not available.
    /// </summary>
    /// <remarks>
    /// Perturbations are Gaussian draws clipped to the L-infinity ball or scaled into the L2 ball. The
    /// resulting interval is an inner estimate of the true output range, so it can not certify soundly.
    /// </remarks>
    private (Vector<T> lower, Vector<T> upper) ApproximateBoundsWithSampling(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        // Get a clean prediction to determine output dimension
        var cleanOutput = model.Predict(referenceInput);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);

        // Initialize with clean output values
        var lowerBounds = new Vector<T>(cleanOutputVector.AsSpan().ToArray());
        var upperBounds = new Vector<T>(cleanOutputVector.AsSpan().ToArray());

        // Sample points to approximate bounds
        int numSamples = _options.NumSamples;
        var random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int s = 0; s < numSamples; s++)
        {
            // Generate a random perturbation within the epsilon ball using Engine
            var noise = Engine.GenerateGaussianNoise<T>(
                vectorInput.Length,
                NumOps.Zero,
                epsilon,
                random.Next());

            noise = ProjectToBall(noise, epsilon);

            // Add perturbation to input
            var perturbedVector = Engine.Add<T>(vectorInput, noise);

            // Convert back to TInput and get prediction
            var perturbedInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbedVector, referenceInput);
            var perturbedOutput = model.Predict(perturbedInput);
            var perturbedOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(perturbedOutput);

            // Update bounds
            lowerBounds = Engine.Min<T>(lowerBounds, perturbedOutputVector);
            upperBounds = Engine.Max<T>(upperBounds, perturbedOutputVector);
        }

        return (lowerBounds, upperBounds);
    }

    /// <summary>
    /// Clips a perturbation to the L-infinity ball, or scales it into the L2 ball.
    /// </summary>
    private Vector<T> ProjectToBall(Vector<T> noise, T epsilon)
    {
        if (_options.NormType != AdversarialNormType.L2)
        {
            return Engine.Clamp<T>(noise, NumOps.Negate(epsilon), epsilon);
        }

        T length = Engine.Norm<T>(noise);
        return NumOps.GreaterThan(length, epsilon)
            ? Engine.Multiply<T>(noise, NumOps.Divide(epsilon, length))
            : noise;
    }

    /// <summary>
    /// Gets the predicted class from an output vector.
    /// </summary>
    private int GetPredictedClass(Vector<T> output)
    {
        if (output == null)
        {
            throw new ArgumentNullException(nameof(output));
        }

        if (output.Length == 0)
        {
            throw new ArgumentException("Output vector cannot be empty.", nameof(output));
        }

        return CertificationMath<T>.ArgMax(output);
    }

    /// <summary>
    /// Checks if the prediction is certifiably robust.
    /// </summary>
    /// <remarks>
    /// A prediction is certified if the lower bound of the predicted class
    /// is greater than the upper bound of all other classes.
    /// </remarks>
    private bool CheckCertification(Vector<T> lowerBounds, Vector<T> upperBounds, int predictedClass)
    {
        return NumOps.GreaterThan(ComputeMinimumMargin(lowerBounds, upperBounds, predictedClass), NumOps.Zero);
    }

    /// <summary>
    /// min over j != y of lower(z_y) - upper(z_j), from independent output intervals; the largest value of T
    /// when there is no other class.
    /// </summary>
    private T ComputeMinimumMargin(Vector<T> lowerBounds, Vector<T> upperBounds, int predictedClass)
    {
        var predictedLower = lowerBounds.AsSpan()[predictedClass];
        var margins = Engine.ScalarMinusTensor(predictedLower, upperBounds.AsTensor());
        var isPredicted = Vector<T>.CreateStandardBasis(upperBounds.Length, predictedClass).AsTensor();
        var largest = Engine.TensorAddScalar(Engine.TensorZerosLike(margins), NumOps.MaxValue);
        return Engine.TensorMinValue(Engine.TensorWhere(isPredicted, largest, margins));
    }

    /// <summary>
    /// Computes the certified radius: the largest radius at which the bounds certify the prediction.
    /// </summary>
    /// <remarks>
    /// The search doubles the radius from 1 while it stays certified (inputs need not lie in [0, 1], and a wide
    /// margin certifies beyond 1), up to 2^20, then bisects the last bracket 20 times. Only certified radii are
    /// kept, so the result is itself certified.
    /// </remarks>
    private T ComputeCertifiedRadiusInternal(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        int predictedClass)
    {
        const int maxDoublings = 20;
        // The brackets 2^0 .. 2^20, converted to T once.
        var powers = new double[maxDoublings + 1];
        for (int d = 0; d <= maxDoublings; d++)
        {
            powers[d] = Math.Pow(2.0, d);
        }

        var brackets = new Tensor<double>(powers, new[] { powers.Length }).Cast<T>().ToArray();
        T minRadius = NumOps.Zero;
        T maxRadius = brackets[0];
        for (int doubling = 0; doubling < maxDoublings; doubling++)
        {
            if (!ComputeCROWNBounds(vectorInput, referenceInput, model, maxRadius, predictedClass).Certified)
            {
                break;
            }

            minRadius = maxRadius;
            maxRadius = brackets[doubling + 1];
        }

        const int maxIterations = 20; // bisection steps over the last bracket
        for (int iter = 0; iter < maxIterations; iter++)
        {
            T midRadius = Midpoint(minRadius, maxRadius);
            if (ComputeCROWNBounds(vectorInput, referenceInput, model, midRadius, predictedClass).Certified)
            {
                // Still certified at this radius, try larger
                minRadius = midRadius;
            }
            else
            {
                // Not certified, try smaller
                maxRadius = midRadius;
            }
        }

        return minRadius;
    }

    private static T Midpoint(T a, T b) => NumOps.Multiply(NumOps.Add(a, b), NumOps.FromDouble(0.5));

    /// <summary>
    /// Confidence as sigmoid of the certified margin lower bound: above 0.5 exactly when certified.
    /// </summary>
    private T ComputeConfidence(T minimumMargin)
    {
        var margin = new Vector<T>(new[] { minimumMargin });
        return Engine.Sigmoid<T>(margin).AsSpan()[0];
    }

    /// <summary>
    /// Serialization data structure.
    /// </summary>
    private class SerializationData
    {
        public double NoiseSigma { get; set; }
        public double ConfidenceLevel { get; set; }
        public int NumSamples { get; set; }
        public AdversarialNormType NormType { get; set; } = AdversarialNormType.L2;
        public bool UseTightBounds { get; set; }
        public int BatchSize { get; set; }
        public int? RandomSeed { get; set; }
    }
}
