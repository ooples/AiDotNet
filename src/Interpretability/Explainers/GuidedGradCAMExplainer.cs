using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Guided GradCAM explainer combining GuidedBackprop with GradCAM for high-resolution explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Guided GradCAM is the best of both worlds - it combines:
///
/// 1. <b>GradCAM:</b> Tells you WHERE to look (coarse localization)
/// 2. <b>GuidedBackprop:</b> Tells you WHAT to look for (fine-grained details)
///
/// <b>How it works:</b>
/// 1. Compute GradCAM map (coarse localization of important regions)
/// 2. Upsample GradCAM to input resolution
/// 3. Compute Guided Backpropagation (fine-grained pixel-level importance)
/// 4. Element-wise multiply: GuidedGradCAM = GuidedBackprop * upsampled_GradCAM
///
/// <b>Why this works:</b>
/// - GradCAM alone is too coarse (can't see fine details)
/// - GuidedBackprop alone highlights details everywhere (not localized)
/// - Multiplying them together keeps only important details in important regions
///
/// <b>Result:</b>
/// High-resolution, class-discriminative visualizations that show both WHERE
/// the model is looking AND WHAT features it sees there.
///
/// <b>Use cases:</b>
/// - Medical imaging (showing exactly what pixels indicate disease)
/// - Object detection debugging (why was this object detected?)
/// - Fine-grained classification (what distinguishes this bird species?)
/// </para>
/// </remarks>
public class GuidedGradCAMExplainer<T> : ILocalExplainer<T, GuidedGradCAMExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private readonly GuidedBackpropExplainer<T> _guidedBackpropExplainer;
    private readonly LayerGradCAMExplainer<T> _gradcamExplainer;
    private readonly int[] _inputShape;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "GuidedGradCAM";

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
        _guidedBackpropExplainer.SetGPUHelper(helper);
        _gradcamExplainer.SetGPUHelper(helper);
    }

    /// <summary>
    /// Initializes a Guided GradCAM explainer from component explainers.
    /// </summary>
    /// <param name="guidedBackpropExplainer">The Guided Backprop explainer.</param>
    /// <param name="gradcamExplainer">The GradCAM explainer.</param>
    /// <param name="inputShape">Shape of the input (e.g., [height, width] or [channels, height, width]).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines pre-configured explainers. Make sure both
    /// explainers are set up for the same model and target layer.
    /// </para>
    /// </remarks>
    public GuidedGradCAMExplainer(
        GuidedBackpropExplainer<T> guidedBackpropExplainer,
        LayerGradCAMExplainer<T> gradcamExplainer,
        int[] inputShape)
    {
        Guard.NotNull(guidedBackpropExplainer);
        _guidedBackpropExplainer = guidedBackpropExplainer;
        Guard.NotNull(gradcamExplainer);
        _gradcamExplainer = gradcamExplainer;
        Guard.NotNull(inputShape);
        _inputShape = inputShape;
    }

    /// <summary>
    /// Initializes a Guided GradCAM explainer from a neural network.
    /// </summary>
    /// <param name="network">The neural network to explain.</param>
    /// <param name="predictFunction">Prediction function.</param>
    /// <param name="layerActivationFunction">Function to get target layer activations.</param>
    /// <param name="layerGradientFunction">Function to compute gradients w.r.t. layer.</param>
    /// <param name="inputShape">Shape of the input.</param>
    /// <param name="layerHeight">Height of the target layer output.</param>
    /// <param name="layerWidth">Width of the target layer output.</param>
    /// <param name="numChannels">Number of channels in the target layer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a neural network and
    /// can provide the layer activation and gradient functions.
    /// </para>
    /// </remarks>
    public GuidedGradCAMExplainer(
        INeuralNetwork<T> network,
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        Func<Vector<T>, int, Vector<T>> layerGradientFunction,
        int[] inputShape,
        int layerHeight,
        int layerWidth,
        int numChannels)
    {
        Guard.NotNull(inputShape);
        _inputShape = inputShape;

        _guidedBackpropExplainer = new GuidedBackpropExplainer<T>(network, inputShape);
        _gradcamExplainer = new LayerGradCAMExplainer<T>(
            predictFunction,
            layerActivationFunction,
            layerGradientFunction,
            layerHeight,
            layerWidth,
            numChannels,
            inputShape);
    }

    /// <summary>
    /// Generates a Guided GradCAM explanation for an input.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="targetClass">Target class (default: predicted class).</param>
    /// <returns>Guided GradCAM explanation.</returns>
    public GuidedGradCAMExplanation<T> Explain(Vector<T> input, int? targetClass = null)
    {
        // Compute Guided Backprop
        var guidedBackprop = _guidedBackpropExplainer.Explain(input, targetClass);

        // Compute GradCAM
        var gradcam = _gradcamExplainer.Explain(input, guidedBackprop.TargetClass);

        // Get upsampled GradCAM or compute it
        Tensor<T> upsampledGradCAM;
        if (gradcam.UpsampledMap != null)
        {
            upsampledGradCAM = gradcam.UpsampledMap;
        }
        else
        {
            upsampledGradCAM = UpsampleGradCAM(gradcam.GradCAMMap);
        }

        // Element-wise multiplication: GuidedGradCAM = GuidedBackprop * upsampled_GradCAM
        var guidedGradcam = ComputeGuidedGradCAM(guidedBackprop.GuidedGradients, upsampledGradCAM);

        return new GuidedGradCAMExplanation<T>(
            input: input,
            guidedGradcam: guidedGradcam,
            guidedBackprop: guidedBackprop.GuidedGradients,
            gradcam: gradcam.GradCAMMap,
            upsampledGradcam: upsampledGradCAM,
            targetClass: guidedBackprop.TargetClass,
            prediction: guidedBackprop.Prediction,
            inputShape: _inputShape);
    }

    /// <summary>
    /// Explains a tensor input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="targetClass">Target class.</param>
    /// <returns>Guided GradCAM explanation.</returns>
    public GuidedGradCAMExplanation<T> ExplainTensor(Tensor<T> input, int? targetClass = null)
    {
        return Explain(input.ToVector(), targetClass);
    }

    /// <inheritdoc/>
    GuidedGradCAMExplanation<T> ILocalExplainer<T, GuidedGradCAMExplanation<T>>.Explain(Vector<T> instance)
    {
        return Explain(instance);
    }

    /// <inheritdoc/>
    public GuidedGradCAMExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new GuidedGradCAMExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = Explain(instances.GetRow(i));
        }
        return results;
    }

    /// <summary>
    /// Computes Guided GradCAM by element-wise multiplication.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the magic happens. By multiplying:
    /// - Guided Backprop (shows all important details)
    /// - GradCAM (shows important regions)
    ///
    /// We get details that are BOTH important AND in the right regions.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGuidedGradCAM(Vector<T> guidedBackprop, Tensor<T> upsampledGradCAM)
    {
        int n = guidedBackprop.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Flatten upsampled GradCAM to match guided backprop (entries past its end count as zero)
        var flatGradCAM = ExplainerTensorOps<T>.PadOrTruncate(upsampledGradCAM.ToVector(), n);

        // For multi-channel images, broadcast GradCAM across channels
        IEnumerable<int> sourceIndices = Enumerable.Range(0, n);
        if (_inputShape.Length == 3 && _inputShape[0] <= 4)
        {
            // CHW format - GradCAM is HxW, need to broadcast across C
            int spatialSize = _inputShape[1] * _inputShape[2];
            sourceIndices = sourceIndices.Select(i => i % spatialSize);
        }
        else if (_inputShape.Length == 3)
        {
            // HWC format
            int channels = _inputShape[2];
            sourceIndices = sourceIndices.Select(i => i / channels);
        }

        var engine = AiDotNetEngine.Current;
        var gradcamPerFeature = engine.TensorIndexSelect(flatGradCAM.AsTensor(), new Tensor<int>(sourceIndices.ToArray(), new[] { n }), 0);
        return engine.Multiply(guidedBackprop, gradcamPerFeature.ToVector());
    }

    /// <summary>
    /// Upsamples GradCAM to input resolution.
    /// </summary>
    private Tensor<T> UpsampleGradCAM(Matrix<T> gradcam)
    {
        int srcH = gradcam.Rows;
        int srcW = gradcam.Columns;

        int targetH, targetW;
        if (_inputShape.Length == 2)
        {
            targetH = _inputShape[0];
            targetW = _inputShape[1];
        }
        else if (_inputShape.Length == 3)
        {
            targetH = _inputShape[0] <= 4 ? _inputShape[1] : _inputShape[0];
            targetW = _inputShape[0] <= 4 ? _inputShape[2] : _inputShape[1];
        }
        else
        {
            targetH = srcH;
            targetW = srcW;
        }

        return ExplainerTensorOps<T>.BilinearResize(Tensor<T>.FromMatrix(gradcam), targetH, targetW);
    }
}

/// <summary>
/// Guided GradCAM explanation result.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class GuidedGradCAMExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Gets the original input.</summary>
    public Vector<T> Input { get; }

    /// <summary>Gets the Guided GradCAM result (main output).</summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is what you visualize. High values indicate
    /// pixels that are both important AND in important regions.
    /// </para>
    /// </remarks>
    public Vector<T> GuidedGradCAM { get; }

    /// <summary>Gets the Guided Backprop component.</summary>
    public Vector<T> GuidedBackprop { get; }

    /// <summary>Gets the GradCAM component.</summary>
    public Matrix<T> GradCAM { get; }

    /// <summary>Gets the upsampled GradCAM.</summary>
    public Tensor<T> UpsampledGradCAM { get; }

    /// <summary>Gets the target class.</summary>
    public int TargetClass { get; }

    /// <summary>Gets the prediction score.</summary>
    public T Prediction { get; }

    /// <summary>Gets the input shape.</summary>
    public int[] InputShape { get; }

    /// <summary>Initializes a new Guided GradCAM explanation.</summary>
    public GuidedGradCAMExplanation(
        Vector<T> input,
        Vector<T> guidedGradcam,
        Vector<T> guidedBackprop,
        Matrix<T> gradcam,
        Tensor<T> upsampledGradcam,
        int targetClass,
        T prediction,
        int[] inputShape)
    {
        Input = input;
        GuidedGradCAM = guidedGradcam;
        GuidedBackprop = guidedBackprop;
        GradCAM = gradcam;
        UpsampledGradCAM = upsampledGradcam;
        TargetClass = targetClass;
        Prediction = prediction;
        InputShape = inputShape;
    }

    /// <summary>Gets normalized Guided GradCAM (0-1 range).</summary>
    public Vector<T> GetNormalizedGuidedGradCAM()
    {
        var engine = AiDotNetEngine.Current;
        T max = GuidedGradCAM.Length > 0
            ? engine.TensorMaxValue(engine.TensorMax(GuidedGradCAM.AsTensor(), NumOps.Zero))
            : NumOps.Zero;

        if (NumOps.GreaterThan(max, ExplainerTensorOps<T>.Tolerance(1e-10)))
        {
            return engine.Divide(GuidedGradCAM, max);
        }

        return new Vector<T>(GuidedGradCAM.Length);
    }

    /// <summary>Gets the Guided GradCAM reshaped to input shape.</summary>
    public Tensor<T> GetGuidedGradCAMTensor()
    {
        int length = InputShape.Aggregate(1, (product, dim) => product * dim);
        return new Tensor<T>(InputShape, ExplainerTensorOps<T>.PadOrTruncate(GuidedGradCAM, length));
    }

    /// <summary>Gets top activated features.</summary>
    public IEnumerable<(int Index, T Activation)> GetTopFeatures(int k = 10)
    {
        return Enumerable.Range(0, GuidedGradCAM.Length)
            .Select(i => (Index: i, Activation: GuidedGradCAM[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Activation))
            .Take(k);
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        var top = GetTopFeatures(5).ToList();
        T maxVal = top.Count > 0 ? top[0].Activation : NumOps.Zero;

        return $"GuidedGradCAM for class {TargetClass} (pred={Prediction}):\n" +
               $"  Input shape: {string.Join("x", InputShape)}\n" +
               $"  Max activation: {maxVal:F4}\n" +
               $"  Top features: {string.Join(", ", top.Select(t => $"[{t.Index}]={t.Activation:F4}"))}";
    }
}
