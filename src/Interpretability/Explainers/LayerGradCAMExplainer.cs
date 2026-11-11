using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Layer GradCAM (Gradient-weighted Class Activation Mapping) explainer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GradCAM produces a coarse localization map showing which regions
/// of an input (usually an image) are important for a prediction.
///
/// <b>How it works:</b>
/// 1. Get activations at a target layer (usually the last convolutional layer)
/// 2. Compute gradients of the target class with respect to these activations
/// 3. Global average pool the gradients to get importance weights for each channel
/// 4. Compute weighted combination of activation channels
/// 5. Apply ReLU to keep only positive influences
///
/// <b>Why GradCAM is useful:</b>
/// - Shows WHERE the model is looking, not just WHAT features matter
/// - Works with any CNN architecture
/// - Produces interpretable heatmaps
/// - Doesn't require architectural changes or retraining
///
/// <b>Layer choice:</b>
/// - Last conv layer: Best balance of semantic meaning and spatial detail
/// - Earlier layers: More spatial detail but less semantic meaning
/// - Later layers: More semantic but coarser resolution
///
/// <b>Limitations:</b>
/// - Resolution limited by the target layer's spatial dimensions
/// - May miss fine-grained details (use GuidedGradCAM for that)
/// </para>
/// </remarks>
public class LayerGradCAMExplainer<T> : ILocalExplainer<T, LayerGradCAMExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T>? _network;
    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>>? _layerActivationFunction;
    private readonly Func<Vector<T>, int, Vector<T>>? _layerGradientFunction;
    private readonly int _layerHeight;
    private readonly int _layerWidth;
    private readonly int _numChannels;
    private readonly int[]? _inputShape;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "LayerGradCAM";

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
    /// Initializes a LayerGradCAM explainer.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="layerActivationFunction">Function to get layer activations.</param>
    /// <param name="layerGradientFunction">Function to compute gradients w.r.t. layer.</param>
    /// <param name="layerHeight">Height of the layer's spatial output.</param>
    /// <param name="layerWidth">Width of the layer's spatial output.</param>
    /// <param name="numChannels">Number of channels in the layer.</param>
    /// <param name="inputShape">Shape of the original input (for upsampling).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>layerActivationFunction:</b> Returns activations at the target layer
    /// - <b>layerGradientFunction:</b> Returns gradients of output w.r.t. layer activations
    /// - <b>layerHeight/Width:</b> Spatial dimensions of the layer output
    /// - <b>numChannels:</b> Number of feature maps in the layer
    /// </para>
    /// </remarks>
    public LayerGradCAMExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        Func<Vector<T>, int, Vector<T>> layerGradientFunction,
        int layerHeight,
        int layerWidth,
        int numChannels,
        int[]? inputShape = null)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        Guard.NotNull(layerActivationFunction);
        _layerActivationFunction = layerActivationFunction;
        Guard.NotNull(layerGradientFunction);
        _layerGradientFunction = layerGradientFunction;
        _layerHeight = layerHeight;
        _layerWidth = layerWidth;
        _numChannels = numChannels;
        _inputShape = inputShape;
    }

    /// <summary>
    /// Initializes a LayerGradCAM explainer from a neural network.
    /// </summary>
    /// <param name="network">The neural network to explain.</param>
    /// <param name="targetLayerIndex">Index of the target layer.</param>
    /// <param name="layerHeight">Height of the layer's spatial output.</param>
    /// <param name="layerWidth">Width of the layer's spatial output.</param>
    /// <param name="numChannels">Number of channels in the layer.</param>
    /// <param name="inputShape">Shape of the original input.</param>
    public LayerGradCAMExplainer(
        INeuralNetwork<T> network,
        int targetLayerIndex,
        int layerHeight,
        int layerWidth,
        int numChannels,
        int[]? inputShape = null)
    {
        Guard.NotNull(network);
        _network = network;
        _layerHeight = layerHeight;
        _layerWidth = layerWidth;
        _numChannels = numChannels;
        _inputShape = inputShape;

        _predictFunction = input =>
        {
            var tensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
            return network.Predict(tensor).ToVector();
        };
    }

    /// <summary>
    /// Generates a GradCAM explanation for an input.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="targetClass">Target class (default: predicted class).</param>
    /// <returns>GradCAM explanation with activation map.</returns>
    public LayerGradCAMExplanation<T> Explain(Vector<T> input, int? targetClass = null)
    {
        // Get prediction
        var prediction = _predictFunction(input);
        int actualTarget = targetClass ?? GetPredictedClass(prediction);

        // Get layer activations
        var activations = GetLayerActivations(input);

        // Get gradients of target class w.r.t. layer activations
        var gradients = GetLayerGradients(input, actualTarget);

        // Compute GradCAM
        var gradcam = ComputeGradCAM(activations, gradients);

        // Upsample to input size if needed
        Tensor<T>? upsampledMap = null;
        if (_inputShape != null)
        {
            upsampledMap = UpsampleGradCAM(gradcam);
        }

        return new LayerGradCAMExplanation<T>(
            input: input,
            gradcamMap: gradcam,
            upsampledMap: upsampledMap,
            targetClass: actualTarget,
            prediction: prediction[actualTarget],
            layerHeight: _layerHeight,
            layerWidth: _layerWidth,
            inputShape: _inputShape);
    }

    /// <inheritdoc/>
    LayerGradCAMExplanation<T> ILocalExplainer<T, LayerGradCAMExplanation<T>>.Explain(Vector<T> instance)
    {
        return Explain(instance);
    }

    /// <inheritdoc/>
    public LayerGradCAMExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new LayerGradCAMExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = Explain(instances.GetRow(i));
        }
        return results;
    }

    /// <summary>
    /// Computes the GradCAM activation map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the core GradCAM computation:
    /// 1. Global average pool gradients to get channel importance weights
    /// 2. Weighted sum of activation channels
    /// 3. ReLU to keep only positive influences
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeGradCAM(Vector<T> activations, Vector<T> gradients)
    {
        // Reshape activations and gradients to [channels, height * width]
        // Assuming they're flattened in channel-major order (missing entries count as zero)
        var engine = AiDotNetEngine.Current;
        int spatialSize = _layerHeight * _layerWidth;
        int size = _numChannels * spatialSize;
        var gradientPlanes = new Tensor<T>(new[] { _numChannels, spatialSize }, ExplainerTensorOps<T>.PadOrTruncate(gradients, size));
        var activationPlanes = new Tensor<T>(new[] { _numChannels, spatialSize }, ExplainerTensorOps<T>.PadOrTruncate(activations, size));

        // Step 1: Global average pool gradients to get weights per channel
        var weights = engine.ReduceMean(gradientPlanes, new[] { 1 }, false).Reshape(new[] { 1, _numChannels });

        // Step 2: Weighted combination of activation maps
        var combined = engine.TensorMatMul(weights, activationPlanes);

        // Step 3: ReLU - only keep positive values
        return engine.ReLU(combined).Reshape(new[] { _layerHeight, _layerWidth }).ToMatrix();
    }

    /// <summary>
    /// Upsamples the GradCAM map to input size using bilinear interpolation.
    /// </summary>
    private Tensor<T> UpsampleGradCAM(Matrix<T> gradcam)
    {
        var inputShape = _inputShape ?? throw new InvalidOperationException("_inputShape has not been initialized.");
        int targetH = inputShape[^2];
        int targetW = inputShape[^1];

        // Bilinear interpolation
        return ExplainerTensorOps<T>.BilinearResize(Tensor<T>.FromMatrix(gradcam), targetH, targetW);
    }

    /// <summary>
    /// Gets layer activations.
    /// </summary>
    private Vector<T> GetLayerActivations(Vector<T> input)
    {
        if (_layerActivationFunction != null)
        {
            return _layerActivationFunction(input);
        }

        // Approximate with zeros if not available
        int size = _numChannels * _layerHeight * _layerWidth;
        return new Vector<T>(size);
    }

    /// <summary>
    /// Gets layer gradients.
    /// </summary>
    private Vector<T> GetLayerGradients(Vector<T> input, int targetClass)
    {
        if (_layerGradientFunction != null)
        {
            return _layerGradientFunction(input, targetClass);
        }

        // Compute numerical gradients
        return ComputeNumericalLayerGradients(input, targetClass);
    }

    /// <summary>
    /// Computes numerical gradients for the layer.
    /// </summary>
    private Vector<T> ComputeNumericalLayerGradients(Vector<T> input, int targetClass)
    {
        var engine = AiDotNetEngine.Current;
        T epsilon = NumOps.FromDouble(1e-5);
        var basePred = _predictFunction(input);
        T baseScore = basePred[targetClass];

        int size = _numChannels * _layerHeight * _layerWidth;
        int count = Math.Min(input.Length, size);

        // Approximate layer gradients via input perturbations
        // This is a rough approximation
        var perturbedPredictions = new List<Vector<T>>(count);
        for (int i = 0; i < count; i++)
        {
            var step = ExplainerTensorOps<T>.RangeIndicator(input.Length, i, i + 1).ToVector();
            var perturbed = engine.Add(input, engine.Multiply(step, epsilon));
            perturbedPredictions.Add(_predictFunction(perturbed));
        }

        if (count == 0)
        {
            return new Vector<T>(size);
        }

        var perturbedScores = engine.TensorSliceAxis(ExplainerTensorOps<T>.Stack(perturbedPredictions), 1, targetClass);
        var gradients = engine.TensorDivideScalar(engine.TensorSubtractScalar(perturbedScores, baseScore), epsilon);
        return ExplainerTensorOps<T>.PadOrTruncate(gradients.ToVector(), size);
    }

    /// <summary>
    /// Gets the predicted class.
    /// </summary>
    private int GetPredictedClass(Vector<T> prediction)
    {
        return ExplainerTensorOps<T>.ArgMax(prediction);
    }
}

/// <summary>
/// GradCAM explanation result.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class LayerGradCAMExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Gets the original input.</summary>
    public Vector<T> Input { get; }

    /// <summary>Gets the GradCAM activation map.</summary>
    public Matrix<T> GradCAMMap { get; }

    /// <summary>Gets the upsampled map (if available).</summary>
    public Tensor<T>? UpsampledMap { get; }

    /// <summary>Gets the target class.</summary>
    public int TargetClass { get; }

    /// <summary>Gets the prediction score.</summary>
    public T Prediction { get; }

    /// <summary>Gets the layer height.</summary>
    public int LayerHeight { get; }

    /// <summary>Gets the layer width.</summary>
    public int LayerWidth { get; }

    /// <summary>Gets the input shape.</summary>
    public int[]? InputShape { get; }

    /// <summary>Initializes a new LayerGradCAM explanation.</summary>
    public LayerGradCAMExplanation(
        Vector<T> input,
        Matrix<T> gradcamMap,
        Tensor<T>? upsampledMap,
        int targetClass,
        T prediction,
        int layerHeight,
        int layerWidth,
        int[]? inputShape = null)
    {
        Input = input;
        GradCAMMap = gradcamMap;
        UpsampledMap = upsampledMap;
        TargetClass = targetClass;
        Prediction = prediction;
        LayerHeight = layerHeight;
        LayerWidth = layerWidth;
        InputShape = inputShape;
    }

    /// <summary>Gets normalized GradCAM map (0-1 range).</summary>
    public Matrix<T> GetNormalizedMap()
    {
        var engine = AiDotNetEngine.Current;
        var map = Tensor<T>.FromMatrix(GradCAMMap);
        T max = map.Length > 0 ? engine.TensorMaxValue(engine.TensorMax(map, NumOps.Zero)) : NumOps.Zero;

        if (NumOps.GreaterThan(max, NumOps.FromDouble(1e-10)))
        {
            return engine.TensorDivideScalar(map, max).ToMatrix();
        }

        return new Matrix<T>(LayerHeight, LayerWidth);
    }

    /// <summary>Gets top activated regions.</summary>
    public IEnumerable<(int H, int W, T Activation)> GetTopRegions(int k = 5)
    {
        var regions = new List<(int H, int W, T Activation)>();

        for (int h = 0; h < LayerHeight; h++)
        {
            for (int w = 0; w < LayerWidth; w++)
            {
                regions.Add((h, w, GradCAMMap[h, w]));
            }
        }

        return regions
            .OrderByDescending(r => NumOps.ToDouble(r.Activation))
            .Take(k);
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        var top = GetTopRegions(3).ToList();
        return $"GradCAM for class {TargetClass} (pred={Prediction}):\n" +
               $"  Map size: {LayerHeight}x{LayerWidth}\n" +
               $"  Top regions: {string.Join(", ", top.Select(r => $"({r.H},{r.W})={r.Activation:F4}"))}";
    }
}
