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
        // Reshape activations and gradients to [channels, height, width]
        // Assuming they're flattened in channel-major order

        // Step 1: Global average pool gradients to get weights per channel
        var weights = new double[_numChannels];
        int spatialSize = _layerHeight * _layerWidth;

        for (int c = 0; c < _numChannels; c++)
        {
            double sum = 0;
            for (int s = 0; s < spatialSize; s++)
            {
                int idx = c * spatialSize + s;
                if (idx < gradients.Length)
                {
                    sum += NumOps.ToDouble(gradients[idx]);
                }
            }
            weights[c] = sum / spatialSize;
        }

        // Step 2: Weighted combination of activation maps
        var gradcam = new Matrix<T>(_layerHeight, _layerWidth);

        for (int h = 0; h < _layerHeight; h++)
        {
            for (int w = 0; w < _layerWidth; w++)
            {
                double value = 0;
                for (int c = 0; c < _numChannels; c++)
                {
                    int idx = c * spatialSize + h * _layerWidth + w;
                    if (idx < activations.Length)
                    {
                        value += weights[c] * NumOps.ToDouble(activations[idx]);
                    }
                }

                // Step 3: ReLU - only keep positive values
                gradcam[h, w] = NumOps.FromDouble(Math.Max(0, value));
            }
        }

        return gradcam;
    }

    /// <summary>
    /// Upsamples the GradCAM map to input size using bilinear interpolation.
    /// </summary>
    private Tensor<T> UpsampleGradCAM(Matrix<T> gradcam)
    {
        int targetH = _inputShape![^2];
        int targetW = _inputShape![^1];

        var upsampled = new Tensor<T>(new[] { targetH, targetW });

        for (int h = 0; h < targetH; h++)
        {
            for (int w = 0; w < targetW; w++)
            {
                // Compute source coordinates
                double srcH = (double)h / targetH * _layerHeight;
                double srcW = (double)w / targetW * _layerWidth;

                // Bilinear interpolation
                int h0 = Math.Min((int)Math.Floor(srcH), _layerHeight - 1);
                int h1 = Math.Min(h0 + 1, _layerHeight - 1);
                int w0 = Math.Min((int)Math.Floor(srcW), _layerWidth - 1);
                int w1 = Math.Min(w0 + 1, _layerWidth - 1);

                double dh = srcH - h0;
                double dw = srcW - w0;

                double v00 = NumOps.ToDouble(gradcam[h0, w0]);
                double v01 = NumOps.ToDouble(gradcam[h0, w1]);
                double v10 = NumOps.ToDouble(gradcam[h1, w0]);
                double v11 = NumOps.ToDouble(gradcam[h1, w1]);

                double value = v00 * (1 - dh) * (1 - dw) +
                              v01 * (1 - dh) * dw +
                              v10 * dh * (1 - dw) +
                              v11 * dh * dw;

                upsampled[h, w] = NumOps.FromDouble(value);
            }
        }

        return upsampled;
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
        double epsilon = 1e-5;
        var basePred = _predictFunction(input);
        double baseScore = NumOps.ToDouble(basePred[targetClass]);

        int size = _numChannels * _layerHeight * _layerWidth;
        var gradients = new T[size];

        // Approximate layer gradients via input perturbations
        // This is a rough approximation
        for (int i = 0; i < Math.Min(input.Length, size); i++)
        {
            var perturbed = input.Clone();
            perturbed[i] = NumOps.Add(perturbed[i], NumOps.FromDouble(epsilon));

            var perturbedPred = _predictFunction(perturbed);
            double perturbedScore = NumOps.ToDouble(perturbedPred[targetClass]);

            gradients[i] = NumOps.FromDouble((perturbedScore - baseScore) / epsilon);
        }

        return new Vector<T>(gradients);
    }

    /// <summary>
    /// Gets the predicted class.
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
        double max = 0;
        for (int h = 0; h < LayerHeight; h++)
        {
            for (int w = 0; w < LayerWidth; w++)
            {
                double val = NumOps.ToDouble(GradCAMMap[h, w]);
                if (val > max) max = val;
            }
        }

        var normalized = new Matrix<T>(LayerHeight, LayerWidth);
        if (max > 1e-10)
        {
            for (int h = 0; h < LayerHeight; h++)
            {
                for (int w = 0; w < LayerWidth; w++)
                {
                    normalized[h, w] = NumOps.FromDouble(
                        NumOps.ToDouble(GradCAMMap[h, w]) / max);
                }
            }
        }

        return normalized;
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
               $"  Top regions: {string.Join(", ", top.Select(r => $"({r.H},{r.W})={NumOps.ToDouble(r.Activation):F4}"))}";
    }
}
