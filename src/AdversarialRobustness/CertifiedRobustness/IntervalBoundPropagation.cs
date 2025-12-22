using System.Text;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Serialization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

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
            CertificationMethod = "IBP"
        };
    }

    /// <summary>
    /// Initializes a new instance of the IntervalBoundPropagation class with specified options.
    /// </summary>
    /// <param name="options">The configuration options for IBP.</param>
    public IntervalBoundPropagation(CertifiedDefenseOptions<T> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.CertificationMethod = "IBP";
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

        // Compute interval bounds through the network
        var (lowerBounds, upperBounds) = ComputeOutputBounds(vectorInput, input, model, epsilon);

        // Get the clean prediction
        var cleanOutput = model.Predict(input);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int predictedClass = GetPredictedClass(cleanOutputVector);

        // Check if the prediction is certifiably robust
        bool isCertified = CheckCertification(lowerBounds, upperBounds, predictedClass);

        // Compute the certified radius
        T certifiedRadius = ComputeCertifiedRadiusInternal(vectorInput, input, model, predictedClass);

        // Compute confidence as the margin between the predicted class and runner-up
        double confidence = ComputeConfidence(lowerBounds, upperBounds, predictedClass);

        // Get scalar bounds for the predicted class
        double lowerBound = NumOps.ToDouble(lowerBounds[predictedClass]);
        double upperBound = NumOps.ToDouble(upperBounds[predictedClass]);

        return new CertifiedPrediction<T>
        {
            PredictedClass = predictedClass,
            IsCertified = isCertified,
            CertifiedRadius = certifiedRadius,
            Confidence = confidence,
            LowerBound = lowerBound,
            UpperBound = upperBound
        };
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

        int totalSamples = testData.Length;
        int correctPredictions = 0;
        int certifiedCorrect = 0;
        T totalCertifiedRadius = NumOps.Zero;

        for (int i = 0; i < totalSamples; i++)
        {
            var input = testData[i];
            var label = labels[i];
            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(label);
            var trueClass = GetPredictedClass(labelVector);

            var certification = CertifyPrediction(input, model);

            // Check if prediction is correct
            if (certification.PredictedClass == trueClass)
            {
                correctPredictions++;

                // Check if certified at the given radius
                if (certification.IsCertified &&
                    NumOps.GreaterThanOrEquals(certification.CertifiedRadius, radius))
                {
                    certifiedCorrect++;
                }
            }

            totalCertifiedRadius = NumOps.Add(totalCertifiedRadius, certification.CertifiedRadius);
        }

        double cleanAccuracy = (double)correctPredictions / totalSamples;
        double certifiedAccuracy = (double)certifiedCorrect / totalSamples;
        T averageRadius = NumOps.Divide(totalCertifiedRadius, NumOps.FromDouble(totalSamples));
        double certificationRate = (double)certifiedCorrect / Math.Max(1, correctPredictions);

        return new CertifiedAccuracyMetrics<T>
        {
            CleanAccuracy = cleanAccuracy,
            CertifiedAccuracy = certifiedAccuracy,
            CertificationRadius = radius,
            AverageCertifiedRadius = averageRadius,
            CertificationRate = certificationRate
        };
    }

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
            CertificationMethod = "IBP"
        };
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var data = new SerializationData
        {
            NoiseSigma = _options.NoiseSigma,
            ConfidenceLevel = _options.ConfidenceLevel,
            NumSamples = _options.NumSamples,
            NormType = _options.NormType,
            UseTightBounds = _options.UseTightBounds,
            BatchSize = _options.BatchSize,
            RandomSeed = _options.RandomSeed
        };

        var json = JsonConvert.SerializeObject(data, Formatting.None);
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
                RandomSeed = deserialized.RandomSeed,
                CertificationMethod = "IBP"
            };
        }
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
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
    /// Computes interval bounds for the neural network output.
    /// </summary>
    /// <param name="vectorInput">The input as a vector.</param>
    /// <param name="referenceInput">The original input for type conversion.</param>
    /// <param name="model">The model to analyze.</param>
    /// <param name="epsilon">The perturbation radius.</param>
    /// <returns>Tuple of lower and upper bound vectors.</returns>
    private (Vector<T> lower, Vector<T> upper) ComputeOutputBounds(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        // Initialize input bounds: [x - ε, x + ε] using vectorized operations
        var inputLower = Engine.Subtract<T>(vectorInput, Engine.Fill<T>(vectorInput.Length, epsilon));
        var inputUpper = Engine.Add<T>(vectorInput, Engine.Fill<T>(vectorInput.Length, epsilon));

        // Try to get layer-wise access if available
        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            if (architecture.Layers != null && architecture.Layers.Count > 0)
            {
                return PropagateIntervalBounds(inputLower, inputUpper, architecture.Layers);
            }
        }

        // Fallback: Use sampling-based approximation
        return ApproximateBoundsWithSampling(vectorInput, referenceInput, model, epsilon);
    }

    /// <summary>
    /// Propagates interval bounds through the network layers.
    /// </summary>
    /// <param name="lower">Lower bound of input interval.</param>
    /// <param name="upper">Upper bound of input interval.</param>
    /// <param name="layers">The network layers.</param>
    /// <returns>Tuple of output lower and upper bounds.</returns>
    private (Vector<T> lower, Vector<T> upper) PropagateIntervalBounds(
        Vector<T> lower,
        Vector<T> upper,
        List<ILayer<T>> layers)
    {
        var currentLower = lower;
        var currentUpper = upper;

        foreach (var layer in layers)
        {
            // Get layer parameters
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var activationTypes = layer.GetActivationTypes().ToList();

            // Propagate through linear transformation
            if (weights != null)
            {
                (currentLower, currentUpper) = PropagateLinearLayer(
                    currentLower, currentUpper, weights, biases);
            }

            // Propagate through activation function
            foreach (var activation in activationTypes)
            {
                (currentLower, currentUpper) = PropagateActivation(
                    currentLower, currentUpper, activation);
            }
        }

        return (currentLower, currentUpper);
    }

    /// <summary>
    /// Propagates interval bounds through a linear layer.
    /// </summary>
    /// <remarks>
    /// For a linear layer y = Wx + b with input interval [x_L, x_U]:
    /// - W⁺ = max(W, 0), W⁻ = min(W, 0)
    /// - y_L = W⁺ · x_L + W⁻ · x_U + b
    /// - y_U = W⁺ · x_U + W⁻ · x_L + b
    /// </remarks>
    private (Vector<T> lower, Vector<T> upper) PropagateLinearLayer(
        Vector<T> lower,
        Vector<T> upper,
        Tensor<T> weights,
        Tensor<T>? biases)
    {
        // Get weight matrix dimensions
        var shape = weights.Shape;
        if (shape.Length < 2)
        {
            return (lower, upper);
        }

        int outputDim = shape[0];
        int inputDim = shape[1];

        var outputLower = new Vector<T>(outputDim);
        var outputUpper = new Vector<T>(outputDim);

        // Initialize with biases if available
        if (biases != null)
        {
            for (int i = 0; i < outputDim && i < biases.Length; i++)
            {
                outputLower[i] = biases[i];
                outputUpper[i] = biases[i];
            }
        }

        // Propagate through weight matrix
        for (int i = 0; i < outputDim; i++)
        {
            T lowerSum = outputLower[i];
            T upperSum = outputUpper[i];

            for (int j = 0; j < Math.Min(inputDim, lower.Length); j++)
            {
                T w = weights[i * inputDim + j];

                // W⁺ = max(W, 0), W⁻ = min(W, 0)
                if (NumOps.GreaterThan(w, NumOps.Zero))
                {
                    // Positive weight: contributes W * x_L to lower, W * x_U to upper
                    lowerSum = NumOps.Add(lowerSum, NumOps.Multiply(w, lower[j]));
                    upperSum = NumOps.Add(upperSum, NumOps.Multiply(w, upper[j]));
                }
                else if (NumOps.LessThan(w, NumOps.Zero))
                {
                    // Negative weight: contributes W * x_U to lower, W * x_L to upper
                    lowerSum = NumOps.Add(lowerSum, NumOps.Multiply(w, upper[j]));
                    upperSum = NumOps.Add(upperSum, NumOps.Multiply(w, lower[j]));
                }
            }

            outputLower[i] = lowerSum;
            outputUpper[i] = upperSum;
        }

        return (outputLower, outputUpper);
    }

    /// <summary>
    /// Propagates interval bounds through an activation function.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) PropagateActivation(
        Vector<T> lower,
        Vector<T> upper,
        ActivationFunction activation)
    {
        var outputLower = new Vector<T>(lower.Length);
        var outputUpper = new Vector<T>(upper.Length);

        for (int i = 0; i < lower.Length; i++)
        {
            switch (activation)
            {
                case ActivationFunction.ReLU:
                    // ReLU: max(0, x)
                    // Lower: max(0, lower)
                    // Upper: max(0, upper)
                    outputLower[i] = NumOps.GreaterThan(lower[i], NumOps.Zero)
                        ? lower[i] : NumOps.Zero;
                    outputUpper[i] = NumOps.GreaterThan(upper[i], NumOps.Zero)
                        ? upper[i] : NumOps.Zero;
                    break;

                case ActivationFunction.Sigmoid:
                    // Sigmoid: 1 / (1 + e^(-x))
                    // Monotonically increasing, so bounds preserve order
                    outputLower[i] = ApplySigmoid(lower[i]);
                    outputUpper[i] = ApplySigmoid(upper[i]);
                    break;

                case ActivationFunction.Tanh:
                    // Tanh is monotonically increasing
                    outputLower[i] = ApplyTanh(lower[i]);
                    outputUpper[i] = ApplyTanh(upper[i]);
                    break;

                case ActivationFunction.LeakyReLU:
                    // LeakyReLU: x if x > 0, else α*x (typically α = 0.01)
                    T alpha = NumOps.FromDouble(0.01);
                    outputLower[i] = PropagateLeakyReLU(lower[i], alpha);
                    outputUpper[i] = PropagateLeakyReLU(upper[i], alpha);
                    // Handle crossing zero - ensure proper ordering when bounds straddle zero
                    if (NumOps.LessThan(lower[i], NumOps.Zero) &&
                        NumOps.GreaterThan(upper[i], NumOps.Zero) &&
                        NumOps.GreaterThan(outputLower[i], outputUpper[i]))
                    {
                        (outputLower[i], outputUpper[i]) = (outputUpper[i], outputLower[i]);
                    }
                    break;

                case ActivationFunction.Softmax:
                    // Softmax outputs are always in [0, 1] range
                    // Use conservative bounds since proper softmax IBP requires joint computation
                    outputLower[i] = NumOps.Zero;
                    outputUpper[i] = NumOps.One;
                    break;

                case ActivationFunction.Linear:
                case ActivationFunction.Identity:
                default:
                    // Identity function - bounds pass through unchanged
                    outputLower[i] = lower[i];
                    outputUpper[i] = upper[i];
                    break;
            }
        }

        return (outputLower, outputUpper);
    }

    /// <summary>
    /// Applies sigmoid activation to a single value.
    /// </summary>
    private T ApplySigmoid(T x)
    {
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        T onePlusExp = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, onePlusExp);
    }

    /// <summary>
    /// Applies tanh activation to a single value.
    /// </summary>
    private T ApplyTanh(T x)
    {
        T expX = NumOps.Exp(x);
        T expNegX = NumOps.Exp(NumOps.Negate(x));
        T numerator = NumOps.Subtract(expX, expNegX);
        T denominator = NumOps.Add(expX, expNegX);
        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Applies LeakyReLU activation.
    /// </summary>
    private T PropagateLeakyReLU(T x, T alpha)
    {
        if (NumOps.GreaterThan(x, NumOps.Zero))
        {
            return x;
        }
        return NumOps.Multiply(alpha, x);
    }

    /// <summary>
    /// Approximates output bounds using sampling when layer access is not available.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) ApproximateBoundsWithSampling(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        // Get a clean prediction to determine output dimension
        var cleanOutput = model.Predict(referenceInput);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int outputDim = cleanOutputVector.Length;

        var lowerBounds = new Vector<T>(outputDim);
        var upperBounds = new Vector<T>(outputDim);

        // Initialize with clean output values
        for (int i = 0; i < outputDim; i++)
        {
            lowerBounds[i] = cleanOutputVector[i];
            upperBounds[i] = cleanOutputVector[i];
        }

        // Sample points to approximate bounds
        int numSamples = _options.NumSamples;
        var random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int s = 0; s < numSamples; s++)
        {
            // Generate a random perturbation within the epsilon ball using Engine
            var noise = Engine.GenerateGaussianNoise<T>(
                vectorInput.Length,
                NumOps.Zero,
                epsilon,
                random.Next());

            // Clip perturbation to L-infinity ball
            noise = Engine.Clamp<T>(noise, NumOps.Negate(epsilon), epsilon);

            // Add perturbation to input
            var perturbedVector = Engine.Add<T>(vectorInput, noise);

            // Convert back to TInput and get prediction
            var perturbedInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbedVector, referenceInput);
            var perturbedOutput = model.Predict(perturbedInput);
            var perturbedOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(perturbedOutput);

            // Update bounds
            for (int i = 0; i < outputDim; i++)
            {
                if (NumOps.LessThan(perturbedOutputVector[i], lowerBounds[i]))
                {
                    lowerBounds[i] = perturbedOutputVector[i];
                }
                if (NumOps.GreaterThan(perturbedOutputVector[i], upperBounds[i]))
                {
                    upperBounds[i] = perturbedOutputVector[i];
                }
            }
        }

        return (lowerBounds, upperBounds);
    }

    /// <summary>
    /// Gets the predicted class from an output vector.
    /// </summary>
    private int GetPredictedClass(Vector<T> output)
    {
        int predictedClass = 0;
        T maxValue = output[0];

        for (int i = 1; i < output.Length; i++)
        {
            if (NumOps.GreaterThan(output[i], maxValue))
            {
                maxValue = output[i];
                predictedClass = i;
            }
        }

        return predictedClass;
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
        T predictedLower = lowerBounds[predictedClass];

        for (int i = 0; i < upperBounds.Length; i++)
        {
            // If any other class's upper bound >= predicted class's lower bound,
            // we cannot certify robustness
            if (i != predictedClass &&
                NumOps.GreaterThanOrEquals(upperBounds[i], predictedLower))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the certified radius using binary search.
    /// </summary>
    private T ComputeCertifiedRadiusInternal(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        int predictedClass)
    {
        T minRadius = NumOps.Zero;
        T maxRadius = NumOps.FromDouble(1.0);

        // Initial check: if not certified at max radius, use binary search
        // If certified at max radius, we could continue searching higher

        int maxIterations = 20; // For binary search precision

        for (int iter = 0; iter < maxIterations; iter++)
        {
            T midRadius = NumOps.Divide(
                NumOps.Add(minRadius, maxRadius),
                NumOps.FromDouble(2.0));

            var (lowerBounds, upperBounds) = ComputeOutputBounds(vectorInput, referenceInput, model, midRadius);

            if (CheckCertification(lowerBounds, upperBounds, predictedClass))
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

    /// <summary>
    /// Computes the confidence margin between predicted class and runner-up.
    /// </summary>
    private double ComputeConfidence(Vector<T> lowerBounds, Vector<T> upperBounds, int predictedClass)
    {
        T predictedLower = lowerBounds[predictedClass];
        T maxOtherUpper = NumOps.FromDouble(double.NegativeInfinity);

        for (int i = 0; i < upperBounds.Length; i++)
        {
            if (i != predictedClass &&
                NumOps.GreaterThan(upperBounds[i], maxOtherUpper))
            {
                maxOtherUpper = upperBounds[i];
            }
        }

        // Confidence is the margin between predicted class lower and max other upper
        T margin = NumOps.Subtract(predictedLower, maxOtherUpper);

        // Convert to a probability-like confidence in [0, 1]
        // Using sigmoid-like transformation
        double marginDouble = NumOps.ToDouble(margin);
        double confidence = 1.0 / (1.0 + Math.Exp(-marginDouble));

        return confidence;
    }

    /// <summary>
    /// Serialization data structure.
    /// </summary>
    private class SerializationData
    {
        public double NoiseSigma { get; set; }
        public double ConfidenceLevel { get; set; }
        public int NumSamples { get; set; }
        public string NormType { get; set; } = "L2";
        public bool UseTightBounds { get; set; }
        public int BatchSize { get; set; }
        public int? RandomSeed { get; set; }
    }
}
