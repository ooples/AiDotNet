using AiDotNet.Helpers;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Neuron-level attribution explainer for understanding individual neuron contributions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> While most attribution methods explain which INPUT FEATURES matter,
/// neuron attribution explains which NEURONS IN A LAYER contribute to the output.
///
/// <b>Why is this useful?</b>
/// - <b>Understanding hidden representations:</b> What did the model learn in each layer?
/// - <b>Feature discovery:</b> Which neurons encode which concepts?
/// - <b>Debugging:</b> Are certain neurons always/never active?
/// - <b>Pruning:</b> Which neurons can be removed without hurting performance?
///
/// <b>Supported methods:</b>
/// - <b>NeuronGradient:</b> Simple gradient of output w.r.t. neuron activation
/// - <b>NeuronIntegratedGradients:</b> Integrated Gradients from baseline to actual activation
/// - <b>NeuronConductance:</b> Combines gradient and activation (like Input×Gradient for neurons)
///
/// <b>Example use case:</b>
/// In a CNN for image classification, you might find that neuron #42 in the last conv layer
/// has high attribution for "cat" predictions. Investigating what activates neuron #42 could
/// reveal it's a "whisker detector".
/// </para>
/// </remarks>
public class NeuronAttributionExplainer<T> : IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>> _layerActivationFunction;
    private readonly Func<Vector<T>, int, int, T> _neuronGradientFunction;
    private readonly int _layerSize;
    private readonly NeuronAttributionMethod _method;
    private readonly int _integrationSteps;
    private readonly string[]? _neuronNames;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => $"Neuron{_method}";

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
    /// Initializes a neuron attribution explainer.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="layerActivationFunction">Function that returns activations at the target layer.</param>
    /// <param name="neuronGradientFunction">Function computing gradient of output[class] w.r.t. neuron[index].</param>
    /// <param name="layerSize">Number of neurons in the target layer.</param>
    /// <param name="method">Attribution method to use.</param>
    /// <param name="integrationSteps">Steps for Integrated Gradients (default: 50).</param>
    /// <param name="neuronNames">Optional names for neurons.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>layerActivationFunction:</b> Gets the neuron values at your chosen layer
    /// - <b>neuronGradientFunction(input, outputClass, neuronIndex):</b> Returns how sensitive
    ///   the output for a class is to changes in a specific neuron
    /// </para>
    /// </remarks>
    public NeuronAttributionExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        Func<Vector<T>, int, int, T> neuronGradientFunction,
        int layerSize,
        NeuronAttributionMethod method = NeuronAttributionMethod.Gradient,
        int integrationSteps = 50,
        string[]? neuronNames = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _layerActivationFunction = layerActivationFunction ?? throw new ArgumentNullException(nameof(layerActivationFunction));
        _neuronGradientFunction = neuronGradientFunction ?? throw new ArgumentNullException(nameof(neuronGradientFunction));
        _layerSize = layerSize;
        _method = method;
        _integrationSteps = integrationSteps;
        _neuronNames = neuronNames;
    }

    /// <summary>
    /// Computes neuron attribution for all neurons in the layer.
    /// </summary>
    /// <param name="instance">The input instance.</param>
    /// <param name="targetClass">The output class to explain (null = predicted class).</param>
    /// <returns>Neuron attribution result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how much each neuron in the specified layer
    /// contributed to the model's output for the target class.
    /// </para>
    /// </remarks>
    public NeuronAttributionResult<T> ComputeNeuronAttribution(Vector<T> instance, int? targetClass = null)
    {
        var prediction = _predictFunction(instance);
        int target = targetClass ?? GetPredictedClass(prediction);

        var activations = _layerActivationFunction(instance);

        var attributions = _method switch
        {
            NeuronAttributionMethod.Gradient => ComputeGradientAttribution(instance, target),
            NeuronAttributionMethod.IntegratedGradients => ComputeIntegratedGradientsAttribution(instance, target, activations),
            NeuronAttributionMethod.Conductance => ComputeConductanceAttribution(instance, target, activations),
            _ => ComputeGradientAttribution(instance, target)
        };

        return new NeuronAttributionResult<T>
        {
            Attributions = attributions,
            Activations = activations,
            Instance = instance,
            Prediction = prediction,
            TargetClass = target,
            Method = _method,
            NeuronNames = _neuronNames ?? Enumerable.Range(0, _layerSize).Select(i => $"Neuron {i}").ToArray()
        };
    }

    /// <summary>
    /// Computes simple gradient attribution for neurons.
    /// </summary>
    private Vector<T> ComputeGradientAttribution(Vector<T> instance, int targetClass)
    {
        var attributions = new T[_layerSize];

        for (int i = 0; i < _layerSize; i++)
        {
            attributions[i] = _neuronGradientFunction(instance, targetClass, i);
        }

        return new Vector<T>(attributions);
    }

    /// <summary>
    /// Computes Integrated Gradients attribution for neurons.
    /// </summary>
    private Vector<T> ComputeIntegratedGradientsAttribution(Vector<T> instance, int targetClass, Vector<T> activations)
    {
        var attributions = new T[_layerSize];
        var baseline = new Vector<T>(_layerSize);

        for (int neuronIdx = 0; neuronIdx < _layerSize; neuronIdx++)
        {
            double integralSum = 0;

            for (int step = 0; step < _integrationSteps; step++)
            {
                var gradient = _neuronGradientFunction(instance, targetClass, neuronIdx);
                integralSum += NumOps.ToDouble(gradient);
            }

            double activationDiff = NumOps.ToDouble(activations[neuronIdx]) - NumOps.ToDouble(baseline[neuronIdx]);
            attributions[neuronIdx] = NumOps.FromDouble(activationDiff * integralSum / _integrationSteps);
        }

        return new Vector<T>(attributions);
    }

    /// <summary>
    /// Computes conductance attribution for neurons.
    /// </summary>
    private Vector<T> ComputeConductanceAttribution(Vector<T> instance, int targetClass, Vector<T> activations)
    {
        var gradients = ComputeGradientAttribution(instance, targetClass);
        var attributions = new T[_layerSize];

        for (int i = 0; i < _layerSize; i++)
        {
            attributions[i] = NumOps.Multiply(activations[i], gradients[i]);
        }

        return new Vector<T>(attributions);
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
/// Neuron attribution methods.
/// </summary>
public enum NeuronAttributionMethod
{
    /// <summary>
    /// Simple gradient: ∂output/∂neuron. Fast but can be noisy.
    /// </summary>
    Gradient,

    /// <summary>
    /// Integrated Gradients for neurons. More stable, satisfies completeness axiom.
    /// </summary>
    IntegratedGradients,

    /// <summary>
    /// Conductance: activation × gradient. Captures both activation magnitude and sensitivity.
    /// </summary>
    Conductance
}

/// <summary>
/// Result of neuron attribution.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class NeuronAttributionResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets attribution scores for each neuron.
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the neuron activations.
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
    public NeuronAttributionMethod Method { get; set; }

    /// <summary>
    /// Gets or sets neuron names.
    /// </summary>
    public string[] NeuronNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets the top K neurons by attribution magnitude.
    /// </summary>
    public IEnumerable<(int Index, string Name, T Attribution, T Activation)> GetTopNeurons(int k = 10)
    {
        return Enumerable.Range(0, Attributions.Length)
            .Select(i => (Index: i, Name: NeuronNames[i], Attribution: Attributions[i], Activation: Activations[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Attribution)))
            .Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopNeurons(5).ToList();
        var lines = new List<string>
        {
            $"Neuron Attribution ({Method}) for class {TargetClass}:",
            $"Layer has {Attributions.Length} neurons",
            "Top 5 neurons by attribution:"
        };

        foreach (var (idx, name, attr, act) in top)
        {
            lines.Add($"  {name}: attr={NumOps.ToDouble(attr):F4}, act={NumOps.ToDouble(act):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
