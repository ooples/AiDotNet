using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Input × Gradient attribution explainer - multiplies input values by their gradients.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Input × Gradient is one of the simplest gradient-based attribution methods.
/// It multiplies each input feature by its gradient to get an attribution score.
///
/// <b>Intuition:</b>
/// - Gradient tells you: "If I change this feature, how much does the output change?"
/// - But gradient alone doesn't consider the feature's current value
/// - Input × Gradient says: "The attribution is both HOW MUCH the feature matters AND what its value is"
///
/// <b>Formula:</b>
/// attribution[i] = input[i] × gradient[i]
///
/// <b>Why multiply by input?</b>
/// Consider a feature x with gradient g:
/// - If x = 0 and g = 100: The feature COULD matter a lot, but currently contributes nothing
/// - If x = 10 and g = 0.1: The feature has high value but low sensitivity
/// - x × g captures both aspects
///
/// <b>Comparison to other methods:</b>
/// - Simpler than Integrated Gradients (just one gradient computation)
/// - Less theoretically grounded than SHAP (doesn't satisfy Shapley axioms)
/// - Good as a quick baseline or sanity check
/// - Can have issues with saturation (gradients near zero even for important features)
///
/// <b>When to use:</b>
/// - Quick initial analysis
/// - As a baseline to compare against more sophisticated methods
/// - When computational resources are limited
/// - For debugging (if Input×Gradient and SHAP disagree dramatically, investigate)
/// </para>
/// </remarks>
public class InputXGradientExplainer<T> : ILocalExplainer<T, InputXGradientExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T>? _network;
    private readonly Func<Vector<T>, Vector<T>>? _predictFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _gradientFunction;
    private readonly int _numFeatures;
    private readonly string[]? _featureNames;
    private readonly bool _absoluteValue;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "InputXGradient";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes an Input × Gradient explainer with a neural network.
    /// </summary>
    /// <param name="network">The neural network to explain.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="absoluteValue">Whether to return absolute values (default: false).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor uses backpropagation to compute gradients.
    /// The network must support gradient computation.
    /// </para>
    /// </remarks>
    public InputXGradientExplainer(
        INeuralNetwork<T> network,
        int numFeatures,
        string[]? featureNames = null,
        bool absoluteValue = false)
    {
        Guard.NotNull(network);
        _network = network;
        _numFeatures = numFeatures;
        _featureNames = featureNames;
        _absoluteValue = absoluteValue;
    }

    /// <summary>
    /// Initializes an Input × Gradient explainer with custom functions.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="gradientFunction">Function that computes gradients w.r.t. input for a target class.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="absoluteValue">Whether to return absolute values (default: false).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have custom gradient computation.
    /// The gradientFunction should return ∂output[targetClass]/∂input.
    /// </para>
    /// </remarks>
    public InputXGradientExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, int, Vector<T>> gradientFunction,
        int numFeatures,
        string[]? featureNames = null,
        bool absoluteValue = false)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        Guard.NotNull(gradientFunction);
        _gradientFunction = gradientFunction;
        _numFeatures = numFeatures;
        _featureNames = featureNames;
        _absoluteValue = absoluteValue;
    }

    /// <summary>
    /// Computes Input × Gradient attribution for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>Attribution explanation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes attributions for the predicted class.
    /// Positive attribution means the feature pushed the prediction higher.
    /// Negative attribution means the feature pushed the prediction lower.
    /// </para>
    /// </remarks>
    public InputXGradientExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, null);
    }

    /// <summary>
    /// Computes Input × Gradient attribution for an input with a specific target class.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="targetClass">The class to explain (null = predicted class).</param>
    /// <returns>Attribution explanation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You can specify which class to explain. For example,
    /// in a cat/dog classifier, you can ask "why did the model think this was a cat?"
    /// even if the model actually predicted dog.
    /// </para>
    /// </remarks>
    public InputXGradientExplanation<T> Explain(Vector<T> instance, int? targetClass)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        // Get prediction
        var prediction = GetPrediction(instance);

        // Determine target class
        int target = targetClass ?? GetPredictedClass(prediction);

        // Compute gradient
        var gradient = ComputeGradient(instance, target);

        // Compute Input × Gradient
        var attributions = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
        {
            var product = NumOps.Multiply(instance[i], gradient[i]);
            attributions[i] = _absoluteValue
                ? NumOps.FromDouble(Math.Abs(NumOps.ToDouble(product)))
                : product;
        }

        return new InputXGradientExplanation<T>
        {
            Attributions = new Vector<T>(attributions),
            Gradients = gradient,
            Instance = instance,
            Prediction = prediction,
            TargetClass = target,
            FeatureNames = _featureNames ?? Enumerable.Range(0, _numFeatures).Select(i => $"Feature {i}").ToArray()
        };
    }

    /// <inheritdoc/>
    public InputXGradientExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new InputXGradientExplanation<T>[instances.Rows];

        if (_gpuHelper != null && _gpuHelper.IsGPUEnabled && instances.Rows > 1)
        {
            Parallel.For(0, instances.Rows, new ParallelOptions
            {
                MaxDegreeOfParallelism = _gpuHelper.MaxParallelism
            }, i =>
            {
                explanations[i] = Explain(instances.GetRow(i));
            });
        }
        else
        {
            for (int i = 0; i < instances.Rows; i++)
            {
                explanations[i] = Explain(instances.GetRow(i));
            }
        }

        return explanations;
    }

    /// <summary>
    /// Gets the model prediction for an input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the input through the model to get output probabilities
    /// or scores for each class.
    /// </para>
    /// </remarks>
    private Vector<T> GetPrediction(Vector<T> input)
    {
        if (_network is not null)
        {
            var tensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
            return _network.Predict(tensor).ToVector();
        }
        else if (_predictFunction is not null)
        {
            return _predictFunction(input);
        }

        throw new InvalidOperationException("No prediction method available.");
    }

    /// <summary>
    /// Gets the predicted class (argmax of output).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For classification, this returns the class with the highest
    /// output score/probability.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Computes gradient of output w.r.t. input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes ∂output[targetClass]/∂input using backpropagation.
    /// The gradient tells us how sensitive the output is to each input feature.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGradient(Vector<T> input, int targetClass)
    {
        if (_gradientFunction is not null)
        {
            return _gradientFunction(input, targetClass);
        }

        if (_network is not null)
        {
            var gradHelper = new InputGradientHelper<T>(_network);
            return gradHelper.ComputeGradient(input, targetClass);
        }

        throw new InvalidOperationException("No gradient computation method available.");
    }
}

/// <summary>
/// Result of Input × Gradient attribution.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InputXGradientExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the attribution scores (input × gradient) for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These scores indicate how much each feature contributed
    /// to the prediction. Positive = pushed prediction higher, negative = pushed lower.
    /// </para>
    /// </remarks>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the raw gradients for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient shows sensitivity - how much the output
    /// would change if we changed each input slightly.
    /// </para>
    /// </remarks>
    public Vector<T> Gradients { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Instance { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the target class being explained.
    /// </summary>
    public int TargetClass { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets attributions sorted by absolute magnitude.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns features ordered by importance (highest magnitude first).
    /// </para>
    /// </remarks>
    public List<(string Name, T Value, T Attribution, T Gradient)> GetSortedAttributions()
    {
        var result = new List<(string, T, T, T)>();
        for (int i = 0; i < Attributions.Length; i++)
        {
            result.Add((FeatureNames[i], Instance[i], Attributions[i], Gradients[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item3))).ToList();
    }

    /// <summary>
    /// Gets the sum of attributions (for completeness check).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike Integrated Gradients or SHAP, Input × Gradient
    /// does NOT guarantee that attributions sum to the output. This is a limitation
    /// of the method.
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
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetSortedAttributions().Take(5);
        var lines = new List<string>
        {
            $"Input × Gradient Attribution (class {TargetClass}):",
            "Top 5 features by attribution:"
        };

        foreach (var (name, value, attr, grad) in top)
        {
            lines.Add($"  {name}: value={NumOps.ToDouble(value):F4}, attr={NumOps.ToDouble(attr):F4}, grad={NumOps.ToDouble(grad):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
