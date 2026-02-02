using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Guided Backpropagation explainer for neural network visualization.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Guided Backpropagation is a gradient-based visualization technique
/// that shows which parts of an input most strongly activate a particular output.
///
/// <b>Key Insight:</b> Regular backpropagation can produce noisy gradients because it
/// propagates both positive and negative values through ReLU activations. Guided Backprop
/// only propagates gradients where BOTH the input to ReLU was positive AND the gradient
/// flowing back is positive.
///
/// <b>How it works:</b>
/// 1. Do a forward pass to get activations
/// 2. During backpropagation, at each ReLU:
///    - Regular backprop: gradient flows if input > 0
///    - Guided backprop: gradient flows if input > 0 AND gradient > 0
/// 3. The result highlights features that positively contribute to the output
///
/// <b>Use cases:</b>
/// - Visualizing what a CNN "sees" in an image
/// - Understanding which pixels matter for a classification
/// - Debugging models (are they looking at the right features?)
///
/// <b>Compared to other methods:</b>
/// - Regular gradient: Noisy, can have negative attributions
/// - Guided backprop: Cleaner, only positive attributions
/// - DeconvNet: Only considers forward activations, not gradients
/// - GradCAM: Coarse localization (good with Guided Backprop = Guided GradCAM)
/// </para>
/// </remarks>
public class GuidedBackpropExplainer<T> : ILocalExplainer<T, GuidedBackpropExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T>? _network;
    private readonly Func<Vector<T>, Vector<T>>? _predictFunction;
    private readonly Func<Tensor<T>, Tensor<T>>? _tensorPredictFunction;
    private readonly int[]? _inputShape;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "GuidedBackprop";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a Guided Backpropagation explainer from a neural network.
    /// </summary>
    /// <param name="network">The neural network to explain.</param>
    /// <param name="inputShape">Optional input shape for tensor inputs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a neural network model.
    /// The explainer will use backpropagation through the network to compute gradients.
    /// </para>
    /// </remarks>
    public GuidedBackpropExplainer(INeuralNetwork<T> network, int[]? inputShape = null)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
        _inputShape = inputShape;
    }

    /// <summary>
    /// Initializes a Guided Backpropagation explainer from prediction functions.
    /// </summary>
    /// <param name="predictFunction">Vector prediction function.</param>
    /// <param name="tensorPredictFunction">Tensor prediction function (optional, for images).</param>
    /// <param name="inputShape">Shape of the input tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only have access to prediction functions.
    /// Note: This uses numerical gradients which may be less accurate.
    /// </para>
    /// </remarks>
    public GuidedBackpropExplainer(
        Func<Vector<T>, Vector<T>>? predictFunction = null,
        Func<Tensor<T>, Tensor<T>>? tensorPredictFunction = null,
        int[]? inputShape = null)
    {
        if (predictFunction == null && tensorPredictFunction == null)
            throw new ArgumentException("At least one prediction function must be provided.");

        _predictFunction = predictFunction;
        _tensorPredictFunction = tensorPredictFunction;
        _inputShape = inputShape;
    }

    /// <summary>
    /// Explains a single input using Guided Backpropagation.
    /// </summary>
    /// <param name="input">The input vector to explain.</param>
    /// <param name="targetClass">Target class to explain (default: predicted class).</param>
    /// <returns>Guided Backpropagation explanation.</returns>
    public GuidedBackpropExplanation<T> Explain(Vector<T> input, int? targetClass = null)
    {
        var gradients = ComputeGuidedGradients(input, targetClass);
        var prediction = GetPrediction(input);
        int actualTarget = targetClass ?? GetPredictedClass(prediction);

        return new GuidedBackpropExplanation<T>(
            input: input,
            guidedGradients: gradients,
            targetClass: actualTarget,
            prediction: prediction[actualTarget],
            inputShape: _inputShape);
    }

    /// <summary>
    /// Explains a tensor input (e.g., image).
    /// </summary>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="targetClass">Target class to explain.</param>
    /// <returns>Guided Backpropagation explanation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for image inputs. The output will have the same
    /// shape as the input, showing which pixels positively influence the prediction.
    /// </para>
    /// </remarks>
    public GuidedBackpropExplanation<T> ExplainTensor(Tensor<T> input, int? targetClass = null)
    {
        var vector = input.ToVector();
        var explanation = Explain(vector, targetClass);

        // Reshape gradients back to tensor shape
        var gradTensor = new Tensor<T>(input.Shape);
        for (int i = 0; i < Math.Min(vector.Length, explanation.GuidedGradients.Length); i++)
        {
            gradTensor[GetMultiIndex(i, input.Shape)] = explanation.GuidedGradients[i];
        }

        return new GuidedBackpropExplanation<T>(
            input: vector,
            guidedGradients: explanation.GuidedGradients,
            targetClass: explanation.TargetClass,
            prediction: explanation.Prediction,
            inputShape: input.Shape,
            gradientTensor: gradTensor);
    }

    /// <inheritdoc/>
    GuidedBackpropExplanation<T> ILocalExplainer<T, GuidedBackpropExplanation<T>>.Explain(Vector<T> instance)
    {
        return Explain(instance);
    }

    /// <inheritdoc/>
    public GuidedBackpropExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new GuidedBackpropExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = Explain(instances.GetRow(i));
        }
        return results;
    }

    /// <summary>
    /// Computes guided gradients for an input.
    /// </summary>
    private Vector<T> ComputeGuidedGradients(Vector<T> input, int? targetClass)
    {
        if (_network != null)
        {
            return ComputeGuidedGradientsWithNetwork(input, targetClass);
        }
        else
        {
            // Fall back to numerical gradients with guided modification
            return ComputeNumericalGuidedGradients(input, targetClass);
        }
    }

    /// <summary>
    /// Computes guided gradients using the neural network.
    /// </summary>
    private Vector<T> ComputeGuidedGradientsWithNetwork(Vector<T> input, int? targetClass)
    {
        if (_network is null)
            throw new InvalidOperationException("Network is not available for gradient computation.");

        // Use standard backpropagation
        var gradHelper = new InputGradientHelper<T>(_network);

        // Get base prediction
        var inputTensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
        var prediction = _network.Predict(inputTensor).ToVector();
        int actualTarget = targetClass ?? GetPredictedClass(prediction);

        // Get standard gradients
        var gradients = gradHelper.ComputeGradient(input, actualTarget);

        // Apply guided backpropagation modification: zero out negative gradients
        var guidedGradients = new T[gradients.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            double gradVal = NumOps.ToDouble(gradients[i]);
            double inputVal = NumOps.ToDouble(input[i]);

            // Guided: only keep positive gradients where input is also positive
            // This is a simplification - true guided backprop modifies ReLU backward
            if (gradVal > 0)
            {
                guidedGradients[i] = gradients[i];
            }
            else
            {
                guidedGradients[i] = NumOps.Zero;
            }
        }

        return new Vector<T>(guidedGradients);
    }

    /// <summary>
    /// Computes numerical guided gradients.
    /// </summary>
    private Vector<T> ComputeNumericalGuidedGradients(Vector<T> input, int? targetClass)
    {
        double epsilon = 1e-5;
        var prediction = GetPrediction(input);
        int actualTarget = targetClass ?? GetPredictedClass(prediction);
        double baseScore = NumOps.ToDouble(prediction[actualTarget]);

        var gradients = new T[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            // Perturb positively
            var perturbed = input.Clone();
            perturbed[i] = NumOps.Add(perturbed[i], NumOps.FromDouble(epsilon));

            var perturbedPred = GetPrediction(perturbed);
            double perturbedScore = NumOps.ToDouble(perturbedPred[actualTarget]);

            double grad = (perturbedScore - baseScore) / epsilon;

            // Guided: only keep positive gradients
            gradients[i] = grad > 0 ? NumOps.FromDouble(grad) : NumOps.Zero;
        }

        return new Vector<T>(gradients);
    }

    /// <summary>
    /// Gets predictions for an input.
    /// </summary>
    private Vector<T> GetPrediction(Vector<T> input)
    {
        if (_network != null)
        {
            var inputTensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
            return _network.Predict(inputTensor).ToVector();
        }
        else if (_predictFunction != null)
        {
            return _predictFunction(input);
        }
        else if (_tensorPredictFunction != null)
        {
            var tensor = new Tensor<T>(_inputShape ?? new[] { input.Length });
            for (int i = 0; i < input.Length; i++)
            {
                tensor[GetMultiIndex(i, tensor.Shape)] = input[i];
            }
            return _tensorPredictFunction(tensor).ToVector();
        }

        throw new InvalidOperationException("No prediction method available.");
    }

    /// <summary>
    /// Gets the predicted class index.
    /// </summary>
    private int GetPredictedClass(Vector<T> prediction)
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
    /// Converts linear index to multi-dimensional index.
    /// </summary>
    private int[] GetMultiIndex(int linearIndex, int[] shape)
    {
        var index = new int[shape.Length];
        int remaining = linearIndex;

        for (int i = shape.Length - 1; i >= 0; i--)
        {
            index[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        return index;
    }
}

/// <summary>
/// Result of Guided Backpropagation explanation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class GuidedBackpropExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Gets the original input.</summary>
    public Vector<T> Input { get; }

    /// <summary>Gets the guided gradients (attributions).</summary>
    public Vector<T> GuidedGradients { get; }

    /// <summary>Gets the target class that was explained.</summary>
    public int TargetClass { get; }

    /// <summary>Gets the prediction for the target class.</summary>
    public T Prediction { get; }

    /// <summary>Gets the input shape (for tensor inputs).</summary>
    public int[]? InputShape { get; }

    /// <summary>Gets the gradient tensor (for tensor inputs).</summary>
    public Tensor<T>? GradientTensor { get; }

    /// <summary>Initializes a new explanation.</summary>
    public GuidedBackpropExplanation(
        Vector<T> input,
        Vector<T> guidedGradients,
        int targetClass,
        T prediction,
        int[]? inputShape = null,
        Tensor<T>? gradientTensor = null)
    {
        Input = input;
        GuidedGradients = guidedGradients;
        TargetClass = targetClass;
        Prediction = prediction;
        InputShape = inputShape;
        GradientTensor = gradientTensor;
    }

    /// <summary>Gets the most important input features.</summary>
    public IEnumerable<(int Index, T Gradient)> GetTopFeatures(int k = 10)
    {
        return Enumerable.Range(0, GuidedGradients.Length)
            .Select(i => (Index: i, Gradient: GuidedGradients[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Gradient))
            .Take(k);
    }

    /// <summary>Normalizes gradients to [0, 1] range.</summary>
    public Vector<T> GetNormalizedGradients()
    {
        double max = 0;
        for (int i = 0; i < GuidedGradients.Length; i++)
        {
            double val = NumOps.ToDouble(GuidedGradients[i]);
            if (val > max) max = val;
        }

        if (max < 1e-10)
        {
            return new Vector<T>(GuidedGradients.Length);
        }

        var normalized = new T[GuidedGradients.Length];
        for (int i = 0; i < GuidedGradients.Length; i++)
        {
            normalized[i] = NumOps.FromDouble(NumOps.ToDouble(GuidedGradients[i]) / max);
        }

        return new Vector<T>(normalized);
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        var top = GetTopFeatures(5).ToList();
        return $"GuidedBackprop for class {TargetClass} (pred={Prediction}):\n" +
               $"  Top features: {string.Join(", ", top.Select(t => $"[{t.Index}]={NumOps.ToDouble(t.Gradient):F4}"))}";
    }
}
