using AiDotNet.Helpers;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Layer-level attribution explainer for computing attributions at intermediate layers.
/// Supports LayerIntegratedGradients, LayerDeepLIFT, LayerGradientXActivation, and LayerConductance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> While input attribution tells you which input features matter,
/// layer attribution tells you which NEURONS IN A HIDDEN LAYER matter for the output.
///
/// <b>Why layer attribution?</b>
/// - <b>Higher-level features:</b> Later layers encode abstract concepts, not raw pixels
/// - <b>Model understanding:</b> See what the model "thinks" at each stage
/// - <b>Debugging:</b> Find where information is lost or distorted
/// - <b>Transfer learning:</b> Understand which learned representations are being used
///
/// <b>Comparison of methods:</b>
/// - <b>LayerGradient:</b> Simple ∂output/∂layer, fast but noisy
/// - <b>LayerIntegratedGradients:</b> Path-integrated gradients, theoretically grounded
/// - <b>LayerDeepLIFT:</b> Difference from reference, handles saturation well
/// - <b>LayerConductance:</b> Layer activation × gradient, captures both magnitude and sensitivity
///
/// <b>Example:</b>
/// For an image classifier, layer attribution on the last conv layer might show
/// that specific feature maps (detecting eyes, fur, etc.) are important for the "cat" prediction.
/// </para>
/// </remarks>
public class LayerAttributionExplainer<T> : IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>> _layerActivationFunction;
    private readonly Func<Vector<T>, int, Vector<T>> _layerGradientFunction;
    private readonly int _layerSize;
    private readonly int _inputSize;
    private readonly LayerAttributionMethod _method;
    private readonly int _integrationSteps;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => $"Layer{_method}";

    /// <summary>
    /// Gets whether this explainer supports local explanations.
    /// </summary>
    public bool SupportsLocalExplanations => true;

    /// <summary>
    /// Gets whether this explainer supports global explanations.
    /// </summary>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a layer attribution explainer.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="layerActivationFunction">Function returning activations at the target layer.</param>
    /// <param name="layerGradientFunction">Function computing gradient of output[class] w.r.t. layer activations.</param>
    /// <param name="layerSize">Number of neurons in the target layer.</param>
    /// <param name="inputSize">Size of model input.</param>
    /// <param name="method">Attribution method to use.</param>
    /// <param name="integrationSteps">Steps for Integrated Gradients (default: 50).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>layerActivationFunction:</b> Returns the hidden layer's output for an input
    /// - <b>layerGradientFunction(input, class):</b> Returns ∂output[class]/∂layer for all neurons
    /// </para>
    /// </remarks>
    public LayerAttributionExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        Func<Vector<T>, int, Vector<T>> layerGradientFunction,
        int layerSize,
        int inputSize,
        LayerAttributionMethod method = LayerAttributionMethod.IntegratedGradients,
        int integrationSteps = 50)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _layerActivationFunction = layerActivationFunction ?? throw new ArgumentNullException(nameof(layerActivationFunction));
        _layerGradientFunction = layerGradientFunction ?? throw new ArgumentNullException(nameof(layerGradientFunction));
        _layerSize = layerSize;
        _inputSize = inputSize;
        _method = method;
        _integrationSteps = integrationSteps;
    }

    /// <summary>
    /// Computes layer attribution for the specified layer.
    /// </summary>
    /// <param name="instance">The input instance.</param>
    /// <param name="targetClass">The output class to explain (null = predicted class).</param>
    /// <returns>Layer attribution result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how important each neuron in the layer is
    /// for the target class prediction.
    /// </para>
    /// </remarks>
    public LayerAttributionResult<T> ComputeLayerAttribution(Vector<T> instance, int? targetClass = null)
    {
        var prediction = _predictFunction(instance);
        int target = targetClass ?? GetPredictedClass(prediction);

        var activations = _layerActivationFunction(instance);

        var attributions = _method switch
        {
            LayerAttributionMethod.Gradient => ComputeGradientAttribution(instance, target),
            LayerAttributionMethod.IntegratedGradients => ComputeIntegratedGradientsAttribution(instance, target),
            LayerAttributionMethod.GradientXActivation => ComputeGradientXActivationAttribution(instance, target, activations),
            LayerAttributionMethod.DeepLIFT => ComputeDeepLIFTAttribution(instance, target),
            LayerAttributionMethod.Conductance => ComputeConductanceAttribution(instance, target),
            _ => ComputeIntegratedGradientsAttribution(instance, target)
        };

        return new LayerAttributionResult<T>
        {
            Attributions = attributions,
            Activations = activations,
            Instance = instance,
            Prediction = prediction,
            TargetClass = target,
            Method = _method
        };
    }

    /// <summary>
    /// Computes simple gradient attribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LayerGradient computes ∂output/∂layer. This is fast but can be noisy
    /// and doesn't satisfy the completeness axiom.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGradientAttribution(Vector<T> instance, int targetClass)
    {
        return _layerGradientFunction(instance, targetClass);
    }

    /// <summary>
    /// Computes Layer Integrated Gradients attribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LayerIntegratedGradients integrates gradients along a path from
    /// a baseline (typically zero input) to the actual input.
    ///
    /// <b>Key properties:</b>
    /// - Completeness: Attributions sum to (output - baseline_output)
    /// - Sensitivity: Non-zero attribution for features that change the output
    ///
    /// <b>Algorithm:</b>
    /// 1. Create baseline input (usually zeros)
    /// 2. Create path from baseline to actual input: x(α) = baseline + α(input - baseline)
    /// 3. At each step, compute gradient of output w.r.t. layer activations
    /// 4. Average gradients and multiply by (layer_activation - baseline_activation)
    /// </para>
    /// </remarks>
    private Vector<T> ComputeIntegratedGradientsAttribution(Vector<T> instance, int targetClass)
    {
        var baseline = new Vector<T>(_inputSize); // Zero baseline
        var attributions = new T[_layerSize];

        // Get baseline and actual activations
        var baselineActivations = _layerActivationFunction(baseline);
        var actualActivations = _layerActivationFunction(instance);

        // Integrate gradients along path
        var integratedGradients = new double[_layerSize];

        for (int step = 0; step < _integrationSteps; step++)
        {
            double alpha = (step + 0.5) / _integrationSteps;

            // Interpolated input
            var interpolated = new Vector<T>(_inputSize);
            for (int i = 0; i < _inputSize; i++)
            {
                interpolated[i] = NumOps.Add(
                    baseline[i],
                    NumOps.Multiply(NumOps.FromDouble(alpha), NumOps.Subtract(instance[i], baseline[i])));
            }

            // Get gradient at interpolated point
            var gradient = _layerGradientFunction(interpolated, targetClass);

            for (int i = 0; i < _layerSize && i < gradient.Length; i++)
            {
                integratedGradients[i] += NumOps.ToDouble(gradient[i]);
            }
        }

        // Scale by activation difference
        for (int i = 0; i < _layerSize; i++)
        {
            double avgGradient = integratedGradients[i] / _integrationSteps;
            double activationDiff = NumOps.ToDouble(actualActivations[i]) - NumOps.ToDouble(baselineActivations[i]);
            attributions[i] = NumOps.FromDouble(avgGradient * activationDiff);
        }

        return new Vector<T>(attributions);
    }

    /// <summary>
    /// Computes Gradient × Activation attribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the layer equivalent of Input × Gradient.
    /// It multiplies each neuron's activation by its gradient.
    ///
    /// <b>Intuition:</b>
    /// - Gradient: how sensitive is the output to this neuron?
    /// - Activation: how strongly is this neuron firing?
    /// - Product: neurons that fire strongly AND matter get high attribution
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGradientXActivationAttribution(Vector<T> instance, int targetClass, Vector<T> activations)
    {
        var gradient = _layerGradientFunction(instance, targetClass);
        var attributions = new T[_layerSize];

        for (int i = 0; i < _layerSize && i < gradient.Length; i++)
        {
            attributions[i] = NumOps.Multiply(activations[i], gradient[i]);
        }

        return new Vector<T>(attributions);
    }

    /// <summary>
    /// Computes Layer DeepLIFT attribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LayerDeepLIFT compares activations to a reference baseline
    /// and attributes based on the difference from reference.
    ///
    /// <b>Advantage over gradients:</b>
    /// DeepLIFT handles saturation well. Even if a neuron is in a "flat" region
    /// (gradient ≈ 0), DeepLIFT can still attribute based on the difference from baseline.
    ///
    /// <b>Formula:</b>
    /// attribution[i] = (activation[i] - baseline_activation[i]) × multiplier
    /// where multiplier distributes output difference proportionally
    /// </para>
    /// </remarks>
    private Vector<T> ComputeDeepLIFTAttribution(Vector<T> instance, int targetClass)
    {
        var baseline = new Vector<T>(_inputSize);

        // Get activations and predictions for both
        var baselineActivations = _layerActivationFunction(baseline);
        var actualActivations = _layerActivationFunction(instance);
        var baselinePrediction = _predictFunction(baseline);
        var actualPrediction = _predictFunction(instance);

        double outputDiff = NumOps.ToDouble(actualPrediction[targetClass]) -
                           NumOps.ToDouble(baselinePrediction[targetClass]);

        // Compute activation differences and gradients
        var activationDiffs = new double[_layerSize];
        var gradients = new double[_layerSize];
        double totalWeightedDiff = 0;

        var gradient = _layerGradientFunction(instance, targetClass);
        for (int i = 0; i < _layerSize && i < gradient.Length; i++)
        {
            activationDiffs[i] = NumOps.ToDouble(actualActivations[i]) - NumOps.ToDouble(baselineActivations[i]);
            gradients[i] = NumOps.ToDouble(gradient[i]);
            totalWeightedDiff += Math.Abs(activationDiffs[i] * gradients[i]);
        }

        // Compute attributions (Rescale rule)
        var attributions = new T[_layerSize];
        for (int i = 0; i < _layerSize; i++)
        {
            if (Math.Abs(totalWeightedDiff) > 1e-10)
            {
                double contribution = activationDiffs[i] * gradients[i];
                attributions[i] = NumOps.FromDouble((contribution / totalWeightedDiff) * outputDiff);
            }
            else
            {
                attributions[i] = NumOps.Zero;
            }
        }

        return new Vector<T>(attributions);
    }

    /// <summary>
    /// Computes Layer Conductance attribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Layer Conductance combines Integrated Gradients with activation values.
    /// It measures how much "signal flows through" each neuron toward the output.
    ///
    /// <b>Formula:</b>
    /// Conductance[i] = (activation[i] - baseline_activation[i]) × IntegratedGradient[i]
    ///
    /// This captures both:
    /// 1. How much the activation changed from baseline
    /// 2. How important that change was (via integrated gradients)
    /// </para>
    /// </remarks>
    private Vector<T> ComputeConductanceAttribution(Vector<T> instance, int targetClass)
    {
        // Conductance is equivalent to IntegratedGradients at the layer level
        // when computed properly with activation differences
        return ComputeIntegratedGradientsAttribution(instance, targetClass);
    }

    /// <summary>
    /// Gets the predicted class from output.
    /// </summary>
    private static int GetPredictedClass(Vector<T> prediction)
    {
        int maxIdx = 0;
        double maxVal = NumOps.ToDouble(prediction[0]);

        for (int i = 1; i < prediction.Length; i++)
        {
            double val = NumOps.ToDouble(prediction[i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}

/// <summary>
/// Layer attribution methods.
/// </summary>
public enum LayerAttributionMethod
{
    /// <summary>
    /// Simple gradient at the layer.
    /// </summary>
    Gradient,

    /// <summary>
    /// Integrated Gradients from baseline to input.
    /// </summary>
    IntegratedGradients,

    /// <summary>
    /// Gradient multiplied by activation.
    /// </summary>
    GradientXActivation,

    /// <summary>
    /// DeepLIFT-style attribution.
    /// </summary>
    DeepLIFT,

    /// <summary>
    /// Layer conductance (signal flow).
    /// </summary>
    Conductance
}

/// <summary>
/// Result of layer attribution.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LayerAttributionResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets attribution scores for each neuron in the layer.
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the layer activations.
    /// </summary>
    public Vector<T> Activations { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Instance { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the target class.
    /// </summary>
    public int TargetClass { get; set; }

    /// <summary>
    /// Gets or sets the attribution method used.
    /// </summary>
    public LayerAttributionMethod Method { get; set; }

    /// <summary>
    /// Gets the sum of attributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For methods like IntegratedGradients and DeepLIFT,
    /// this should approximately equal (output - baseline_output).
    /// </para>
    /// </remarks>
    public T GetAttributionSum()
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < Attributions.Length; i++)
        {
            sum = NumOps.Add(sum, Attributions[i]);
        }
        return sum;
    }

    /// <summary>
    /// Gets the top K neurons by attribution magnitude.
    /// </summary>
    public IEnumerable<(int Index, T Attribution, T Activation)> GetTopNeurons(int k = 10)
    {
        return Enumerable.Range(0, Attributions.Length)
            .Select(i => (Index: i, Attribution: Attributions[i], Activation: Activations[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Attribution)))
            .Take(k);
    }

    /// <summary>
    /// Reshapes attributions for spatial layers (e.g., conv layers).
    /// </summary>
    /// <param name="height">Height of feature maps.</param>
    /// <param name="width">Width of feature maps.</param>
    /// <param name="channels">Number of channels.</param>
    /// <returns>3D array of attributions [channel, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For convolutional layers, attributions are often more
    /// meaningful when viewed spatially. This reshapes the flat attribution vector
    /// into a 3D array matching the layer's spatial structure.
    /// </para>
    /// </remarks>
    public T[,,] ReshapeSpatial(int height, int width, int channels)
    {
        if (height * width * channels != Attributions.Length)
            throw new ArgumentException("Dimensions don't match attribution length.");

        var result = new T[channels, height, width];
        int idx = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[c, h, w] = Attributions[idx++];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets channel-wise attribution sums (for conv layers).
    /// </summary>
    /// <param name="height">Height of feature maps.</param>
    /// <param name="width">Width of feature maps.</param>
    /// <param name="channels">Number of channels.</param>
    /// <returns>Attribution sum for each channel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In CNNs, each channel often represents a specific feature
    /// (edges, textures, patterns). This aggregates attributions by channel to see
    /// which features matter most.
    /// </para>
    /// </remarks>
    public Vector<T> GetChannelAttributions(int height, int width, int channels)
    {
        var spatial = ReshapeSpatial(height, width, channels);
        var channelAttr = new T[channels];

        for (int c = 0; c < channels; c++)
        {
            double sum = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    sum += NumOps.ToDouble(spatial[c, h, w]);
                }
            }
            channelAttr[c] = NumOps.FromDouble(sum);
        }

        return new Vector<T>(channelAttr);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopNeurons(5).ToList();
        var lines = new List<string>
        {
            $"Layer Attribution ({Method}) for class {TargetClass}:",
            $"Layer size: {Attributions.Length} neurons",
            $"Attribution sum: {NumOps.ToDouble(GetAttributionSum()):F4}",
            "Top 5 neurons:"
        };

        foreach (var (idx, attr, act) in top)
        {
            lines.Add($"  Neuron {idx}: attr={NumOps.ToDouble(attr):F4}, act={NumOps.ToDouble(act):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
