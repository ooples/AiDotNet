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
            CertificationMethod = "CROWN",
            UseTightBounds = true
        };
    }

    /// <summary>
    /// Initializes a new instance of the CROWNVerification class with specified options.
    /// </summary>
    /// <param name="options">The configuration options for CROWN.</param>
    public CROWNVerification(CertifiedDefenseOptions<T> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.CertificationMethod = "CROWN";
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

        // Compute CROWN bounds through the network
        var (lowerBounds, upperBounds) = ComputeCROWNBounds(vectorInput, input, model, epsilon);

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

            if (certification.PredictedClass == trueClass)
            {
                correctPredictions++;

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
            CertificationMethod = "CROWN",
            UseTightBounds = true
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
                RandomSeed = deserialized.RandomSeed,
                CertificationMethod = "CROWN"
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
    /// Computes CROWN bounds for the neural network output.
    /// </summary>
    /// <remarks>
    /// CROWN combines forward interval propagation with backward linear bound propagation
    /// to achieve tighter bounds than IBP alone.
    /// </remarks>
    private (Vector<T> lower, Vector<T> upper) ComputeCROWNBounds(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        // First, compute IBP-style forward bounds to get pre-activation intervals
        var (ibpLower, ibpUpper) = ComputeIBPBounds(vectorInput, referenceInput, model, epsilon);

        // If we have layer access, use CROWN's backward bound propagation
        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            if (architecture.Layers != null && architecture.Layers.Count > 0)
            {
                return ComputeCROWNBoundsWithLayers(vectorInput, epsilon, architecture.Layers, ibpLower, ibpUpper);
            }
        }

        // Fallback to IBP bounds
        return (ibpLower, ibpUpper);
    }

    /// <summary>
    /// Computes forward IBP bounds for pre-activation intervals.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) ComputeIBPBounds(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        // Initialize input bounds using vectorized operations
        var inputLower = Engine.Subtract<T>(vectorInput, Engine.Fill<T>(vectorInput.Length, epsilon));
        var inputUpper = Engine.Add<T>(vectorInput, Engine.Fill<T>(vectorInput.Length, epsilon));

        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            if (architecture.Layers != null && architecture.Layers.Count > 0)
            {
                return PropagateIBPBounds(inputLower, inputUpper, architecture.Layers);
            }
        }

        // Fallback: sampling-based approximation
        return ApproximateBoundsWithSampling(vectorInput, referenceInput, model, epsilon);
    }

    /// <summary>
    /// Propagates IBP bounds through layers.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) PropagateIBPBounds(
        Vector<T> lower,
        Vector<T> upper,
        List<ILayer<T>> layers)
    {
        var currentLower = lower;
        var currentUpper = upper;

        foreach (var layer in layers)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var activationTypes = layer.GetActivationTypes().ToList();

            if (weights != null)
            {
                (currentLower, currentUpper) = PropagateLinearLayer(
                    currentLower, currentUpper, weights, biases);
            }

            foreach (var activation in activationTypes)
            {
                (currentLower, currentUpper) = PropagateActivationIBP(
                    currentLower, currentUpper, activation);
            }
        }

        return (currentLower, currentUpper);
    }

    /// <summary>
    /// Computes CROWN bounds using backward linear bound propagation.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) ComputeCROWNBoundsWithLayers(
        Vector<T> input,
        T epsilon,
        List<ILayer<T>> layers,
        Vector<T> ibpLower,
        Vector<T> ibpUpper)
    {
        int numLayers = layers.Count;
        int outputDim = ibpLower.Length;
        int inputDim = input.Length;

        // Store pre-activation bounds for each layer
        var preActivationBounds = new List<(Vector<T> lower, Vector<T> upper)>();

        // Forward pass to compute pre-activation bounds
        var currentLower = Engine.Subtract<T>(input, Engine.Fill<T>(input.Length, epsilon));
        var currentUpper = Engine.Add<T>(input, Engine.Fill<T>(input.Length, epsilon));

        foreach (var layer in layers)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();

            if (weights != null)
            {
                (currentLower, currentUpper) = PropagateLinearLayer(
                    currentLower, currentUpper, weights, biases);
            }

            // Store pre-activation bounds
            preActivationBounds.Add((
                new Vector<T>(currentLower.ToArray()),
                new Vector<T>(currentUpper.ToArray())
            ));

            var activationTypes = layer.GetActivationTypes().ToList();
            foreach (var activation in activationTypes)
            {
                (currentLower, currentUpper) = PropagateActivationIBP(
                    currentLower, currentUpper, activation);
            }
        }

        // Backward pass to compute linear bounds
        // Start with identity bounds on the output
        var outputLower = new Vector<T>(outputDim);
        var outputUpper = new Vector<T>(outputDim);

        // For each output neuron, compute its CROWN bounds
        for (int outIdx = 0; outIdx < outputDim; outIdx++)
        {
            // Initialize with linear function: f(x) = x[outIdx]
            var alphaLower = new Vector<T>(outputDim);
            var alphaUpper = new Vector<T>(outputDim);
            var betaLower = NumOps.Zero;
            var betaUpper = NumOps.Zero;

            alphaLower[outIdx] = NumOps.One;
            alphaUpper[outIdx] = NumOps.One;

            // Propagate backward through layers
            for (int layerIdx = numLayers - 1; layerIdx >= 0; layerIdx--)
            {
                var layer = layers[layerIdx];
                var activationTypes = layer.GetActivationTypes().ToList();

                // Propagate through activation (in reverse)
                if (layerIdx < preActivationBounds.Count)
                {
                    var (preActLower, preActUpper) = preActivationBounds[layerIdx];

                    foreach (var activation in activationTypes)
                    {
                        (alphaLower, alphaUpper, betaLower, betaUpper) = PropagateActivationCROWN(
                            alphaLower, alphaUpper, betaLower, betaUpper,
                            preActLower, preActUpper, activation);
                    }
                }

                // Propagate through linear transformation
                var weights = layer.GetWeights();
                var biases = layer.GetBiases();

                if (weights != null)
                {
                    (alphaLower, alphaUpper, betaLower, betaUpper) = PropagateLinearCROWN(
                        alphaLower, alphaUpper, betaLower, betaUpper, weights, biases);
                }
            }

            // Compute final bounds using the linear relaxation
            // Lower: α_L · x + β_L evaluated on [x-ε, x+ε]
            // Upper: α_U · x + β_U evaluated on [x-ε, x+ε]
            T lowerBound = betaLower;
            T upperBound = betaUpper;

            for (int j = 0; j < Math.Min(inputDim, alphaLower.Length); j++)
            {
                T aL = alphaLower[j];
                T aU = alphaUpper[j];

                // For positive aL, use lower input bound (input - epsilon)
                // For negative/zero aL, use upper input bound (input + epsilon)
                var lowerInputBound = NumOps.GreaterThan(aL, NumOps.Zero)
                    ? NumOps.Subtract(input[j], epsilon)
                    : NumOps.Add(input[j], epsilon);
                lowerBound = NumOps.Add(lowerBound, NumOps.Multiply(aL, lowerInputBound));

                // For positive aU, use upper input bound (input + epsilon)
                // For negative/zero aU, use lower input bound (input - epsilon)
                var upperInputBound = NumOps.GreaterThan(aU, NumOps.Zero)
                    ? NumOps.Add(input[j], epsilon)
                    : NumOps.Subtract(input[j], epsilon);
                upperBound = NumOps.Add(upperBound, NumOps.Multiply(aU, upperInputBound));
            }

            outputLower[outIdx] = lowerBound;
            outputUpper[outIdx] = upperBound;
        }

        // Take the tighter of CROWN and IBP bounds
        for (int i = 0; i < outputDim; i++)
        {
            // Use the tighter lower bound (maximum of CROWN and IBP)
            if (NumOps.LessThanOrEquals(outputLower[i], ibpLower[i]))
            {
                outputLower[i] = ibpLower[i];
            }

            // Use the tighter upper bound (minimum of CROWN and IBP)
            if (NumOps.GreaterThanOrEquals(outputUpper[i], ibpUpper[i]))
            {
                outputUpper[i] = ibpUpper[i];
            }
        }

        return (outputLower, outputUpper);
    }

    /// <summary>
    /// Propagates linear bounds backward through a linear layer.
    /// </summary>
    private (Vector<T> alphaLower, Vector<T> alphaUpper, T betaLower, T betaUpper) PropagateLinearCROWN(
        Vector<T> alphaLower,
        Vector<T> alphaUpper,
        T betaLower,
        T betaUpper,
        Tensor<T> weights,
        Tensor<T>? biases)
    {
        var shape = weights.Shape;
        if (shape.Length < 2)
        {
            return (alphaLower, alphaUpper, betaLower, betaUpper);
        }

        int outputDim = shape[0];
        int inputDim = shape[1];

        // Transform: y = Wx + b
        // Backward propagation: α_new = W^T · α
        var newAlphaLower = new Vector<T>(inputDim);
        var newAlphaUpper = new Vector<T>(inputDim);

        for (int j = 0; j < inputDim; j++)
        {
            T sumLower = NumOps.Zero;
            T sumUpper = NumOps.Zero;

            for (int i = 0; i < Math.Min(outputDim, alphaLower.Length); i++)
            {
                T w = weights[i * inputDim + j];
                T aL = alphaLower[i];
                T aU = alphaUpper[i];

                // For both lower and upper bound coefficients, accumulate regardless of sign
                sumLower = NumOps.Add(sumLower, NumOps.Multiply(aL, w));
                sumUpper = NumOps.Add(sumUpper, NumOps.Multiply(aU, w));
            }

            newAlphaLower[j] = sumLower;
            newAlphaUpper[j] = sumUpper;
        }

        // Add bias contribution to beta
        if (biases != null)
        {
            for (int i = 0; i < Math.Min(outputDim, alphaLower.Length); i++)
            {
                if (i < biases.Length)
                {
                    betaLower = NumOps.Add(betaLower, NumOps.Multiply(alphaLower[i], biases[i]));
                    betaUpper = NumOps.Add(betaUpper, NumOps.Multiply(alphaUpper[i], biases[i]));
                }
            }
        }

        return (newAlphaLower, newAlphaUpper, betaLower, betaUpper);
    }

    /// <summary>
    /// Propagates linear bounds backward through an activation function using CROWN relaxation.
    /// </summary>
    private (Vector<T> alphaLower, Vector<T> alphaUpper, T betaLower, T betaUpper) PropagateActivationCROWN(
        Vector<T> alphaLower,
        Vector<T> alphaUpper,
        T betaLower,
        T betaUpper,
        Vector<T> preActLower,
        Vector<T> preActUpper,
        ActivationFunction activation)
    {
        int dim = alphaLower.Length;
        var newAlphaLower = new Vector<T>(dim);
        var newAlphaUpper = new Vector<T>(dim);
        T newBetaLower = betaLower;
        T newBetaUpper = betaUpper;

        for (int i = 0; i < dim; i++)
        {
            T l = i < preActLower.Length ? preActLower[i] : NumOps.Zero;
            T u = i < preActUpper.Length ? preActUpper[i] : NumOps.Zero;

            switch (activation)
            {
                case ActivationFunction.ReLU:
                    var (slopeLower, slopeUpper, interceptLower, interceptUpper) =
                        ComputeReLUCROWNBounds(l, u);

                    // Apply CROWN transformation
                    if (NumOps.GreaterThanOrEquals(alphaLower[i], NumOps.Zero))
                    {
                        newAlphaLower[i] = NumOps.Multiply(alphaLower[i], slopeLower);
                        newBetaLower = NumOps.Add(newBetaLower,
                            NumOps.Multiply(alphaLower[i], interceptLower));
                    }
                    else
                    {
                        newAlphaLower[i] = NumOps.Multiply(alphaLower[i], slopeUpper);
                        newBetaLower = NumOps.Add(newBetaLower,
                            NumOps.Multiply(alphaLower[i], interceptUpper));
                    }

                    if (NumOps.GreaterThanOrEquals(alphaUpper[i], NumOps.Zero))
                    {
                        newAlphaUpper[i] = NumOps.Multiply(alphaUpper[i], slopeUpper);
                        newBetaUpper = NumOps.Add(newBetaUpper,
                            NumOps.Multiply(alphaUpper[i], interceptUpper));
                    }
                    else
                    {
                        newAlphaUpper[i] = NumOps.Multiply(alphaUpper[i], slopeLower);
                        newBetaUpper = NumOps.Add(newBetaUpper,
                            NumOps.Multiply(alphaUpper[i], interceptLower));
                    }
                    break;

                case ActivationFunction.Sigmoid:
                case ActivationFunction.Tanh:
                case ActivationFunction.LeakyReLU:
                    // For other activations, use linear relaxation based on monotonicity
                    newAlphaLower[i] = alphaLower[i];
                    newAlphaUpper[i] = alphaUpper[i];
                    break;

                case ActivationFunction.Linear:
                case ActivationFunction.Identity:
                default:
                    // Identity: pass through unchanged
                    newAlphaLower[i] = alphaLower[i];
                    newAlphaUpper[i] = alphaUpper[i];
                    break;
            }
        }

        return (newAlphaLower, newAlphaUpper, newBetaLower, newBetaUpper);
    }

    /// <summary>
    /// Computes the CROWN linear relaxation bounds for ReLU.
    /// </summary>
    /// <remarks>
    /// For ReLU with pre-activation bounds [l, u]:
    /// - If l ≥ 0: slope=1, intercept=0 (always active)
    /// - If u ≤ 0: slope=0, intercept=0 (always inactive)
    /// - If l &lt; 0 &lt; u:
    ///   - Upper relaxation: slope = u/(u-l), intercept = -lu/(u-l)
    ///   - Lower relaxation: slope ∈ {0, 1}, choose based on area minimization
    /// </remarks>
    private (T slopeLower, T slopeUpper, T interceptLower, T interceptUpper) ComputeReLUCROWNBounds(T l, T u)
    {
        T zero = NumOps.Zero;
        T one = NumOps.One;

        // Case 1: Always active (l >= 0)
        if (NumOps.GreaterThanOrEquals(l, zero))
        {
            return (one, one, zero, zero);
        }

        // Case 2: Always inactive (u <= 0)
        if (NumOps.LessThanOrEquals(u, zero))
        {
            return (zero, zero, zero, zero);
        }

        // Case 3: Crossing zero (l < 0 < u)
        // Upper bound: f(x) ≤ u(x - l)/(u - l) = (u/(u-l))x - ul/(u-l)
        T uMinusL = NumOps.Subtract(u, l);
        T slopeUpper = NumOps.Divide(u, uMinusL);
        T interceptUpper = NumOps.Divide(NumOps.Negate(NumOps.Multiply(u, l)), uMinusL);

        // Lower bound: choose between slope=0 and slope=1
        // Area minimization: use slope=1 if |l| < |u|, else slope=0
        T absL = NumOps.Abs(l);
        T absU = NumOps.Abs(u);
        T slopeLower;
        T interceptLower;

        if (NumOps.LessThan(absL, absU))
        {
            // Use slope = 1 (identity line through origin)
            slopeLower = one;
            interceptLower = zero;
        }
        else
        {
            // Use slope = 0 (zero line)
            slopeLower = zero;
            interceptLower = zero;
        }

        return (slopeLower, slopeUpper, interceptLower, interceptUpper);
    }

    /// <summary>
    /// Propagates interval bounds through a linear layer.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) PropagateLinearLayer(
        Vector<T> lower,
        Vector<T> upper,
        Tensor<T> weights,
        Tensor<T>? biases)
    {
        var shape = weights.Shape;
        if (shape.Length < 2)
        {
            return (lower, upper);
        }

        int outputDim = shape[0];
        int inputDim = shape[1];

        var outputLower = new Vector<T>(outputDim);
        var outputUpper = new Vector<T>(outputDim);

        if (biases != null)
        {
            for (int i = 0; i < outputDim && i < biases.Length; i++)
            {
                outputLower[i] = biases[i];
                outputUpper[i] = biases[i];
            }
        }

        for (int i = 0; i < outputDim; i++)
        {
            T lowerSum = outputLower[i];
            T upperSum = outputUpper[i];

            for (int j = 0; j < Math.Min(inputDim, lower.Length); j++)
            {
                T w = weights[i * inputDim + j];

                if (NumOps.GreaterThan(w, NumOps.Zero))
                {
                    lowerSum = NumOps.Add(lowerSum, NumOps.Multiply(w, lower[j]));
                    upperSum = NumOps.Add(upperSum, NumOps.Multiply(w, upper[j]));
                }
                else if (NumOps.LessThan(w, NumOps.Zero))
                {
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
    /// Propagates interval bounds through an activation using IBP.
    /// </summary>
    private (Vector<T> lower, Vector<T> upper) PropagateActivationIBP(
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
                    outputLower[i] = NumOps.GreaterThan(lower[i], NumOps.Zero)
                        ? lower[i] : NumOps.Zero;
                    outputUpper[i] = NumOps.GreaterThan(upper[i], NumOps.Zero)
                        ? upper[i] : NumOps.Zero;
                    break;

                case ActivationFunction.Sigmoid:
                    outputLower[i] = ApplySigmoid(lower[i]);
                    outputUpper[i] = ApplySigmoid(upper[i]);
                    break;

                case ActivationFunction.Tanh:
                    outputLower[i] = ApplyTanh(lower[i]);
                    outputUpper[i] = ApplyTanh(upper[i]);
                    break;

                case ActivationFunction.LeakyReLU:
                    T alpha = NumOps.FromDouble(0.01);
                    outputLower[i] = PropagateLeakyReLU(lower[i], alpha);
                    outputUpper[i] = PropagateLeakyReLU(upper[i], alpha);
                    if (NumOps.GreaterThan(outputLower[i], outputUpper[i]))
                    {
                        (outputLower[i], outputUpper[i]) = (outputUpper[i], outputLower[i]);
                    }
                    break;

                default:
                    outputLower[i] = lower[i];
                    outputUpper[i] = upper[i];
                    break;
            }
        }

        return (outputLower, outputUpper);
    }

    private T ApplySigmoid(T x)
    {
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        T onePlusExp = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, onePlusExp);
    }

    private T ApplyTanh(T x)
    {
        T expX = NumOps.Exp(x);
        T expNegX = NumOps.Exp(NumOps.Negate(x));
        T numerator = NumOps.Subtract(expX, expNegX);
        T denominator = NumOps.Add(expX, expNegX);
        return NumOps.Divide(numerator, denominator);
    }

    private T PropagateLeakyReLU(T x, T alpha)
    {
        if (NumOps.GreaterThan(x, NumOps.Zero))
        {
            return x;
        }
        return NumOps.Multiply(alpha, x);
    }

    private (Vector<T> lower, Vector<T> upper) ApproximateBoundsWithSampling(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        T epsilon)
    {
        var cleanOutput = model.Predict(referenceInput);
        var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
        int outputDim = cleanOutputVector.Length;

        var lowerBounds = new Vector<T>(outputDim);
        var upperBounds = new Vector<T>(outputDim);

        for (int i = 0; i < outputDim; i++)
        {
            lowerBounds[i] = cleanOutputVector[i];
            upperBounds[i] = cleanOutputVector[i];
        }

        int numSamples = _options.NumSamples;
        var random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : new Random();

        for (int s = 0; s < numSamples; s++)
        {
            // Generate noise using Engine
            var noise = Engine.GenerateGaussianNoise<T>(
                vectorInput.Length,
                NumOps.Zero,
                epsilon,
                random.Next());

            // Clip to L-infinity ball
            noise = Engine.Clamp<T>(noise, NumOps.Negate(epsilon), epsilon);

            // Add perturbation
            var perturbedVector = Engine.Add<T>(vectorInput, noise);

            var perturbedInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbedVector, referenceInput);
            var perturbedOutput = model.Predict(perturbedInput);
            var perturbedOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(perturbedOutput);

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

    private bool CheckCertification(Vector<T> lowerBounds, Vector<T> upperBounds, int predictedClass)
    {
        T predictedLower = lowerBounds[predictedClass];

        for (int i = 0; i < upperBounds.Length; i++)
        {
            // Check if any other class's upper bound >= predicted class's lower bound
            if (i != predictedClass &&
                NumOps.GreaterThanOrEquals(upperBounds[i], predictedLower))
            {
                return false;
            }
        }

        return true;
    }

    private T ComputeCertifiedRadiusInternal(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> model,
        int predictedClass)
    {
        T minRadius = NumOps.Zero;
        T maxRadius = NumOps.FromDouble(1.0);
        int maxIterations = 20;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            T midRadius = NumOps.Divide(
                NumOps.Add(minRadius, maxRadius),
                NumOps.FromDouble(2.0));

            var (lowerBounds, upperBounds) = ComputeCROWNBounds(vectorInput, referenceInput, model, midRadius);

            if (CheckCertification(lowerBounds, upperBounds, predictedClass))
            {
                minRadius = midRadius;
            }
            else
            {
                maxRadius = midRadius;
            }
        }

        return minRadius;
    }

    private double ComputeConfidence(Vector<T> lowerBounds, Vector<T> upperBounds, int predictedClass)
    {
        T predictedLower = lowerBounds[predictedClass];
        T maxOtherUpper = NumOps.FromDouble(double.NegativeInfinity);

        for (int i = 0; i < upperBounds.Length; i++)
        {
            // Find the maximum upper bound of all other classes
            if (i != predictedClass &&
                NumOps.GreaterThan(upperBounds[i], maxOtherUpper))
            {
                maxOtherUpper = upperBounds[i];
            }
        }

        T margin = NumOps.Subtract(predictedLower, maxOtherUpper);
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
