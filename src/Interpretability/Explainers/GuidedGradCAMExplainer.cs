using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
        _guidedBackpropExplainer = guidedBackpropExplainer ?? throw new ArgumentNullException(nameof(guidedBackpropExplainer));
        _gradcamExplainer = gradcamExplainer ?? throw new ArgumentNullException(nameof(gradcamExplainer));
        _inputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
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
        _inputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));

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
        var result = new T[guidedBackprop.Length];

        // Flatten upsampled GradCAM to match guided backprop
        var flatGradCAM = upsampledGradCAM.ToVector();

        for (int i = 0; i < guidedBackprop.Length; i++)
        {
            double gb = NumOps.ToDouble(guidedBackprop[i]);
            double gc = i < flatGradCAM.Length ? NumOps.ToDouble(flatGradCAM[i]) : 0;

            // For multi-channel images, broadcast GradCAM across channels
            if (_inputShape.Length == 3 && _inputShape[0] <= 4)
            {
                // CHW format - GradCAM is HxW, need to broadcast across C
                int c = i / (_inputShape[1] * _inputShape[2]);
                int spatial = i % (_inputShape[1] * _inputShape[2]);
                if (spatial < flatGradCAM.Length)
                {
                    gc = NumOps.ToDouble(flatGradCAM[spatial]);
                }
            }
            else if (_inputShape.Length == 3)
            {
                // HWC format
                int spatial = i / _inputShape[2];
                if (spatial < flatGradCAM.Length)
                {
                    gc = NumOps.ToDouble(flatGradCAM[spatial]);
                }
            }

            result[i] = NumOps.FromDouble(gb * gc);
        }

        return new Vector<T>(result);
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

        var upsampled = new Tensor<T>(new[] { targetH, targetW });

        for (int h = 0; h < targetH; h++)
        {
            for (int w = 0; w < targetW; w++)
            {
                double srcHf = (double)h / targetH * srcH;
                double srcWf = (double)w / targetW * srcW;

                int h0 = Math.Min((int)Math.Floor(srcHf), srcH - 1);
                int h1 = Math.Min(h0 + 1, srcH - 1);
                int w0 = Math.Min((int)Math.Floor(srcWf), srcW - 1);
                int w1 = Math.Min(w0 + 1, srcW - 1);

                double dh = srcHf - h0;
                double dw = srcWf - w0;

                double value = NumOps.ToDouble(gradcam[h0, w0]) * (1 - dh) * (1 - dw) +
                              NumOps.ToDouble(gradcam[h0, w1]) * (1 - dh) * dw +
                              NumOps.ToDouble(gradcam[h1, w0]) * dh * (1 - dw) +
                              NumOps.ToDouble(gradcam[h1, w1]) * dh * dw;

                upsampled[h, w] = NumOps.FromDouble(value);
            }
        }

        return upsampled;
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
        double max = 0;
        for (int i = 0; i < GuidedGradCAM.Length; i++)
        {
            double val = NumOps.ToDouble(GuidedGradCAM[i]);
            if (val > max) max = val;
        }

        var normalized = new T[GuidedGradCAM.Length];
        if (max > 1e-10)
        {
            for (int i = 0; i < GuidedGradCAM.Length; i++)
            {
                normalized[i] = NumOps.FromDouble(NumOps.ToDouble(GuidedGradCAM[i]) / max);
            }
        }

        return new Vector<T>(normalized);
    }

    /// <summary>Gets the Guided GradCAM reshaped to input shape.</summary>
    public Tensor<T> GetGuidedGradCAMTensor()
    {
        var tensor = new Tensor<T>(InputShape);
        for (int i = 0; i < Math.Min(GuidedGradCAM.Length, tensor.Length); i++)
        {
            tensor[GetMultiIndex(i, InputShape)] = GuidedGradCAM[i];
        }
        return tensor;
    }

    /// <summary>Gets top activated features.</summary>
    public IEnumerable<(int Index, T Activation)> GetTopFeatures(int k = 10)
    {
        return Enumerable.Range(0, GuidedGradCAM.Length)
            .Select(i => (Index: i, Activation: GuidedGradCAM[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Activation))
            .Take(k);
    }

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

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        var top = GetTopFeatures(5).ToList();
        double maxVal = top.Count > 0 ? NumOps.ToDouble(top[0].Activation) : 0;

        return $"GuidedGradCAM for class {TargetClass} (pred={Prediction}):\n" +
               $"  Input shape: {string.Join("x", InputShape)}\n" +
               $"  Max activation: {maxVal:F4}\n" +
               $"  Top features: {string.Join(", ", top.Select(t => $"[{t.Index}]={NumOps.ToDouble(t.Activation):F4}"))}";
    }
}
