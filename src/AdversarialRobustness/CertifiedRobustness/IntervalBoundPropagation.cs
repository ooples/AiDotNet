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
/// Implements Interval Bound Propagation (IBP) for certifying neural network robustness.
/// </summary>
/// <remarks>
/// <para>
/// IBP is a formal verification method that propagates interval bounds through neural network
/// layers to compute guaranteed output bounds for all inputs within a specified perturbation region.
/// It provides provable robustness guarantees without requiring adversarial examples.
/// </para>
/// <para><b>For Beginners:</b> IBP is like asking "if my input can vary within a certain range,
/// what is the guaranteed range of possible outputs?" This helps certify that a model's
/// predictions are stable within a given perturbation radius.</para>
/// <para>
/// <b>Mathematical Foundation:</b>
/// For a neural network f(x), IBP computes bounds [y_L, y_U] such that:
/// ∀x ∈ B_p(x₀, ε): y_L ≤ f(x) ≤ y_U
///
/// For a linear layer with weights W and biases b:
/// - Input interval: [x_L, x_U]
/// - W⁺ = max(W, 0), W⁻ = min(W, 0)
/// - Lower bound: W⁺ · x_L + W⁻ · x_U + b
/// - Upper bound: W⁺ · x_U + W⁻ · x_L + b
///
/// For ReLU activation:
/// - Lower: max(0, x_L)
/// - Upper: max(0, x_U)
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
[ResearchPaper("On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models", "https://arxiv.org/abs/1810.12715", Year = 2018, Authors = "Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, Pushmeet Kohli")]
public class IntervalBoundPropagation<T, TInput, TOutput> : ICertifiedDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private CertifiedDefenseOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the IntervalBoundPropagation class with default options.
    /// </summary>
    public IntervalBoundPropagation()
    {
        _options = new CertifiedDefenseOptions<T>
        {
            CertificationMethod = CertifiedDefenseMethod.IntervalBoundPropagation
        };
    }

    /// <summary>
    /// Initializes a new instance of the IntervalBoundPropagation class with specified options.
    /// </summary>
    /// <param name="options">The configuration options for IBP.</param>
    public IntervalBoundPropagation(CertifiedDefenseOptions<T> options)
    {
        Guard.NotNull(options);
        _options = options;
        _options.CertificationMethod = CertifiedDefenseMethod.IntervalBoundPropagation;
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
        T epsilon = NumOps.FromDouble(_options.NoiseSigma); // Reusing NoiseSigma as perturbation radius

        // Get the clean prediction
        var cleanOutput = model.Predict(input);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int predictedClass = GetPredictedClass(cleanOutputVector);

        // Bound the network over the ball and check the margins of the predicted class
        var bounds = ComputeOutputBounds(vectorInput, input, model, epsilon, predictedClass);

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

        result.CertificationDetails["Method"] = CertifiedDefenseMethod.IntervalBoundPropagation;
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
            CertificationMethod = CertifiedDefenseMethod.IntervalBoundPropagation
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
            TypeNameHandling = TypeNameHandling.None
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
                CertificationMethod = CertifiedDefenseMethod.IntervalBoundPropagation
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

        /// <summary>True for layer-wise IBP bounds; false for the sampled (empirical, not sound) fallback.</summary>
        public bool Sound { get; }
    }

    /// <summary>
    /// Bounds the network over the perturbation ball and decides whether class <paramref name="predictedClass"/>
    /// is certified.
    /// </summary>
    /// <remarks>
    /// With layer access this is IBP as in Gowal et al. (2018): the first affine layer is bounded exactly over
    /// the ball (epsilon times the dual norm of each weight row), later layers by center/radius interval
    /// arithmetic, and the margin specification z_y - z_j is merged into the last affine layer (their Eq. 6) so
    /// the margins are bounded directly instead of from separate logit intervals. With
    /// <see cref="CertifiedDefenseOptions{T}.UseTightBounds"/> the margin bounds are also computed by CROWN and the
    /// larger (tighter) of the two is kept. Without layer access the bounds come from sampling, which is empirical
    /// and reported as not sound.
    /// </remarks>
    private BoundResult ComputeOutputBounds(
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
            var margins = network.IntervalMargins(x, epsilon, norm, predictedClass, out var topLogitLower);
            if (_options.UseTightBounds && network.SupportsCrown)
            {
                // Tight bounds: both lower bounds are sound, so their element-wise maximum is too, and it is at
                // least as tight as IBP alone (CROWN back-substitution, Zhang et al. 2018).
                var crownMargins = network.CrownMargins(x, epsilon, norm, predictedClass, out var crownTopLogitLower);
                margins = Engine.TensorMax(margins, crownMargins);
                topLogitLower = Engine.TensorMax(topLogitLower, crownTopLogitLower);
            }

            var (lower, upper) = network.IntervalBounds(x, epsilon, norm, network.Operations.Count);
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
            if (!ComputeOutputBounds(vectorInput, referenceInput, model, maxRadius, predictedClass).Certified)
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
            if (ComputeOutputBounds(vectorInput, referenceInput, model, midRadius, predictedClass).Certified)
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
