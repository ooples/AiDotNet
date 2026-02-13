using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Occlusion explainer for image and sequential data interpretation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Occlusion is a simple but powerful explanation technique.
/// The idea is: systematically hide different parts of the input and see how the
/// prediction changes.
///
/// <b>How it works:</b>
/// 1. Take your input (e.g., an image)
/// 2. Place a "patch" over one part of the input (occlude it)
/// 3. See how the model's prediction changes
/// 4. Move the patch and repeat
/// 5. The result is a map showing which regions matter most
///
/// <b>Intuition:</b> If covering a region causes the prediction to drop significantly,
/// that region was important for the prediction.
///
/// <b>Use cases:</b>
/// - Understanding which parts of an image a classifier looks at
/// - Finding what regions of a medical scan led to a diagnosis
/// - Debugging models that use spurious correlations (e.g., looking at background)
///
/// <b>Advantages:</b>
/// - Very simple and intuitive
/// - Model-agnostic (works with any model)
/// - Easy to visualize
///
/// <b>Disadvantages:</b>
/// - Can be slow (many forward passes required)
/// - May not capture feature interactions well
/// - Results depend on occlusion patch size
/// </para>
/// </remarks>
public class OcclusionExplainer<T> : ILocalExplainer<T, OcclusionExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Tensor<T>, Tensor<T>> _predictFunction;
    private readonly int[] _windowShape;
    private readonly int[] _strides;
    private readonly T _baselineValue;
    private readonly OcclusionShape _shape;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => "Occlusion";

    /// <summary>
    /// Gets whether this explainer supports local explanations.
    /// </summary>
    public bool SupportsLocalExplanations => true;

    /// <summary>
    /// Gets whether this explainer supports global explanations.
    /// </summary>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new Occlusion explainer.
    /// </summary>
    /// <param name="predictFunction">Function that takes a tensor and returns predictions.</param>
    /// <param name="windowShape">Shape of the occlusion window (e.g., [3, 3] for 3x3 patch).</param>
    /// <param name="strides">Step sizes for sliding the window (e.g., [1, 1]).</param>
    /// <param name="baselineValue">Value to use for occluded regions (default: 0).</param>
    /// <param name="shape">Shape of the occlusion patch (rectangular or circular).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>windowShape:</b> Size of the patch used to hide parts of the image
    ///   - Larger = faster but less precise
    ///   - Smaller = slower but more detailed
    /// - <b>strides:</b> How many pixels to move the patch each step
    ///   - [1,1] = move 1 pixel at a time (thorough but slow)
    ///   - Same as windowShape = non-overlapping patches (fast but coarse)
    /// - <b>baselineValue:</b> What to fill the hidden region with
    ///   - 0 = black patch
    ///   - Mean of image = gray patch
    ///   - Random noise = more robust but slower
    /// </para>
    /// </remarks>
    public OcclusionExplainer(
        Func<Tensor<T>, Tensor<T>> predictFunction,
        int[] windowShape,
        int[]? strides = null,
        T? baselineValue = default,
        OcclusionShape shape = OcclusionShape.Rectangular)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        Guard.NotNull(windowShape);
        _windowShape = windowShape;
        _strides = strides ?? windowShape; // Default: non-overlapping
        _baselineValue = baselineValue ?? NumOps.Zero;
        _shape = shape;
    }

    /// <summary>
    /// Initializes an Occlusion explainer for 2D images.
    /// </summary>
    /// <param name="predictFunction">Function that takes a tensor and returns predictions.</param>
    /// <param name="patchHeight">Height of the occlusion patch.</param>
    /// <param name="patchWidth">Width of the occlusion patch.</param>
    /// <param name="strideHeight">Vertical stride (default: same as patchHeight).</param>
    /// <param name="strideWidth">Horizontal stride (default: same as patchWidth).</param>
    /// <param name="baselineValue">Value for occluded pixels.</param>
    /// <param name="shape">Shape of the occlusion patch.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most common use case - occluding parts of images.
    /// A typical setup might be:
    /// - patchHeight/Width = 8 (8x8 pixel patches)
    /// - strideHeight/Width = 4 (move 4 pixels at a time for overlapping)
    /// - baselineValue = 0.5 (gray patch for normalized images)
    /// </para>
    /// </remarks>
    public static OcclusionExplainer<T> ForImages(
        Func<Tensor<T>, Tensor<T>> predictFunction,
        int patchHeight,
        int patchWidth,
        int? strideHeight = null,
        int? strideWidth = null,
        T? baselineValue = default,
        OcclusionShape shape = OcclusionShape.Rectangular)
    {
        return new OcclusionExplainer<T>(
            predictFunction,
            new[] { patchHeight, patchWidth },
            new[] { strideHeight ?? patchHeight, strideWidth ?? patchWidth },
            baselineValue,
            shape);
    }

    /// <summary>
    /// Explains a single input by computing occlusion sensitivity.
    /// </summary>
    /// <param name="input">The input tensor to explain (e.g., image).</param>
    /// <param name="targetClass">The target class to explain (default: predicted class).</param>
    /// <returns>Occlusion explanation with sensitivity map.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns a "sensitivity map" showing how important
    /// each region is. Brighter regions are more important.
    ///
    /// The map shows how much the target class prediction drops when each region
    /// is occluded. High values = important for prediction.
    /// </para>
    /// </remarks>
    public OcclusionExplanation<T> Explain(Tensor<T> input, int? targetClass = null)
    {
        // Get baseline prediction
        var basePrediction = _predictFunction(input);
        int actualTargetClass = targetClass ?? GetPredictedClass(basePrediction);
        T baseScore = basePrediction.ToVector()[actualTargetClass];

        // Determine output shape for sensitivity map
        var inputShape = input.Shape;
        var outputShape = ComputeOutputShape(inputShape);
        var sensitivityMap = new Tensor<T>(outputShape);

        // Generate all occlusion positions
        var positions = GenerateOcclusionPositions(inputShape);

        // Compute sensitivity for each position
        int idx = 0;
        foreach (var position in positions)
        {
            var occludedInput = ApplyOcclusion(input, position);
            var occludedPrediction = _predictFunction(occludedInput);
            T occludedScore = occludedPrediction.ToVector()[actualTargetClass];

            // Sensitivity = drop in score when occluded
            T sensitivity = NumOps.Subtract(baseScore, occludedScore);

            // Map back to output position
            SetSensitivityValue(sensitivityMap, idx, sensitivity);
            idx++;
        }

        return new OcclusionExplanation<T>(
            input: input,
            sensitivityMap: sensitivityMap,
            targetClass: actualTargetClass,
            basePrediction: baseScore,
            windowShape: _windowShape,
            strides: _strides);
    }

    /// <summary>
    /// Explains a batch of inputs.
    /// </summary>
    /// <param name="inputs">List of input tensors to explain.</param>
    /// <param name="targetClass">Target class to explain for all inputs.</param>
    /// <returns>List of occlusion explanations.</returns>
    public List<OcclusionExplanation<T>> ExplainBatch(List<Tensor<T>> inputs, int? targetClass = null)
    {
        return inputs.Select(input => Explain(input, targetClass)).ToList();
    }

    /// <summary>
    /// Applies occlusion at the given position.
    /// </summary>
    private Tensor<T> ApplyOcclusion(Tensor<T> input, int[] position)
    {
        var occluded = input.Clone();
        var shape = input.Shape;

        // Apply occlusion based on shape type
        if (_shape == OcclusionShape.Rectangular)
        {
            ApplyRectangularOcclusion(occluded, position);
        }
        else
        {
            ApplyCircularOcclusion(occluded, position);
        }

        return occluded;
    }

    /// <summary>
    /// Applies rectangular occlusion.
    /// </summary>
    private void ApplyRectangularOcclusion(Tensor<T> tensor, int[] position)
    {
        var shape = tensor.Shape;

        // For 2D (height, width)
        if (shape.Length == 2 && position.Length >= 2 && _windowShape.Length >= 2)
        {
            for (int h = 0; h < _windowShape[0]; h++)
            {
                for (int w = 0; w < _windowShape[1]; w++)
                {
                    int ph = position[0] + h;
                    int pw = position[1] + w;

                    if (ph >= 0 && ph < shape[0] && pw >= 0 && pw < shape[1])
                    {
                        tensor[ph, pw] = _baselineValue;
                    }
                }
            }
        }
        // For 3D (channels, height, width) or (height, width, channels)
        else if (shape.Length == 3)
        {
            int hDim = 0, wDim = 1, cDim = 2;
            // Assume HWC format; adjust for CHW if first dim is small
            if (shape[0] <= 4)
            {
                cDim = 0; hDim = 1; wDim = 2;
            }

            for (int h = 0; h < _windowShape[0]; h++)
            {
                for (int w = 0; w < _windowShape[1]; w++)
                {
                    int ph = position[0] + h;
                    int pw = position[1] + w;

                    if (ph >= 0 && ph < shape[hDim] && pw >= 0 && pw < shape[wDim])
                    {
                        for (int c = 0; c < shape[cDim]; c++)
                        {
                            if (cDim == 0)
                                tensor[c, ph, pw] = _baselineValue;
                            else
                                tensor[ph, pw, c] = _baselineValue;
                        }
                    }
                }
            }
        }
        // For 4D (batch, channels, height, width)
        else if (shape.Length == 4 && position.Length >= 2)
        {
            for (int h = 0; h < _windowShape[0]; h++)
            {
                for (int w = 0; w < _windowShape[1]; w++)
                {
                    int ph = position[0] + h;
                    int pw = position[1] + w;

                    if (ph >= 0 && ph < shape[2] && pw >= 0 && pw < shape[3])
                    {
                        for (int b = 0; b < shape[0]; b++)
                        {
                            for (int c = 0; c < shape[1]; c++)
                            {
                                tensor[b, c, ph, pw] = _baselineValue;
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Applies circular occlusion.
    /// </summary>
    private void ApplyCircularOcclusion(Tensor<T> tensor, int[] position)
    {
        var shape = tensor.Shape;

        // Compute center and radius
        double centerH = position[0] + _windowShape[0] / 2.0;
        double centerW = position[1] + _windowShape[1] / 2.0;
        double radiusH = _windowShape[0] / 2.0;
        double radiusW = _windowShape[1] / 2.0;

        // For 2D (height, width)
        if (shape.Length == 2)
        {
            for (int h = 0; h < shape[0]; h++)
            {
                for (int w = 0; w < shape[1]; w++)
                {
                    double dh = (h - centerH) / radiusH;
                    double dw = (w - centerW) / radiusW;
                    if (dh * dh + dw * dw <= 1)
                    {
                        tensor[h, w] = _baselineValue;
                    }
                }
            }
        }
        // For 3D
        else if (shape.Length == 3)
        {
            int hDim = 0, wDim = 1, cDim = 2;
            if (shape[0] <= 4) { cDim = 0; hDim = 1; wDim = 2; }

            for (int h = 0; h < shape[hDim]; h++)
            {
                for (int w = 0; w < shape[wDim]; w++)
                {
                    double dh = (h - centerH) / radiusH;
                    double dw = (w - centerW) / radiusW;
                    if (dh * dh + dw * dw <= 1)
                    {
                        for (int c = 0; c < shape[cDim]; c++)
                        {
                            if (cDim == 0)
                                tensor[c, h, w] = _baselineValue;
                            else
                                tensor[h, w, c] = _baselineValue;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates all occlusion positions.
    /// </summary>
    private List<int[]> GenerateOcclusionPositions(int[] inputShape)
    {
        var positions = new List<int[]>();

        // For 2D/3D images, slide over height and width
        int hDim = inputShape.Length == 2 ? 0 : (inputShape[0] <= 4 ? 1 : 0);
        int wDim = hDim + 1;
        int height = inputShape.Length > hDim ? inputShape[hDim] : 1;
        int width = inputShape.Length > wDim ? inputShape[wDim] : 1;

        for (int h = 0; h <= height - _windowShape[0]; h += _strides[0])
        {
            for (int w = 0; w <= width - _windowShape[Math.Min(1, _windowShape.Length - 1)]; w += _strides[Math.Min(1, _strides.Length - 1)])
            {
                positions.Add(new[] { h, w });
            }
        }

        return positions;
    }

    /// <summary>
    /// Computes the output shape for the sensitivity map.
    /// </summary>
    private int[] ComputeOutputShape(int[] inputShape)
    {
        int hDim = inputShape.Length == 2 ? 0 : (inputShape[0] <= 4 ? 1 : 0);
        int wDim = hDim + 1;
        int height = inputShape.Length > hDim ? inputShape[hDim] : 1;
        int width = inputShape.Length > wDim ? inputShape[wDim] : 1;

        int outH = (height - _windowShape[0]) / _strides[0] + 1;
        int outW = (width - _windowShape[Math.Min(1, _windowShape.Length - 1)]) / _strides[Math.Min(1, _strides.Length - 1)] + 1;

        return new[] { outH, outW };
    }

    /// <summary>
    /// Sets a sensitivity value in the map.
    /// </summary>
    private void SetSensitivityValue(Tensor<T> map, int linearIndex, T value)
    {
        int w = map.Shape[1];
        int h = linearIndex / w;
        int wPos = linearIndex % w;

        if (h < map.Shape[0] && wPos < map.Shape[1])
        {
            map[h, wPos] = value;
        }
    }

    /// <summary>
    /// Gets the predicted class from output.
    /// </summary>
    private int GetPredictedClass(Tensor<T> output)
    {
        var vec = output.ToVector();
        int maxIdx = 0;
        double maxVal = NumOps.ToDouble(vec[0]);

        for (int i = 1; i < vec.Length; i++)
        {
            double val = NumOps.ToDouble(vec[i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Explains a single vector input.
    /// </summary>
    public OcclusionExplanation<T> Explain(Vector<T> instance)
    {
        // Convert vector to tensor and explain
        var tensor = new Tensor<T>(new[] { 1, instance.Length });
        for (int i = 0; i < instance.Length; i++)
        {
            tensor[0, i] = instance[i];
        }
        return Explain(tensor);
    }

    /// <summary>
    /// Explains a batch of inputs.
    /// </summary>
    public OcclusionExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new OcclusionExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = Explain(instances.GetRow(i));
        }
        return results;
    }
}

/// <summary>
/// Shape of occlusion patches.
/// </summary>
public enum OcclusionShape
{
    /// <summary>
    /// Rectangular/square occlusion patches.
    /// </summary>
    Rectangular,

    /// <summary>
    /// Circular/elliptical occlusion patches.
    /// </summary>
    Circular
}

/// <summary>
/// Result of occlusion analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OcclusionExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the original input.
    /// </summary>
    public Tensor<T> Input { get; }

    /// <summary>
    /// Gets the sensitivity map showing importance of each region.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Higher values mean the region is more important
    /// for the target class prediction.
    /// </para>
    /// </remarks>
    public Tensor<T> SensitivityMap { get; }

    /// <summary>
    /// Gets the target class that was explained.
    /// </summary>
    public int TargetClass { get; }

    /// <summary>
    /// Gets the base prediction for the target class.
    /// </summary>
    public T BasePrediction { get; }

    /// <summary>
    /// Gets the window shape used for occlusion.
    /// </summary>
    public int[] WindowShape { get; }

    /// <summary>
    /// Gets the strides used for sliding.
    /// </summary>
    public int[] Strides { get; }

    /// <summary>
    /// Initializes a new occlusion explanation.
    /// </summary>
    public OcclusionExplanation(
        Tensor<T> input,
        Tensor<T> sensitivityMap,
        int targetClass,
        T basePrediction,
        int[] windowShape,
        int[] strides)
    {
        Input = input;
        SensitivityMap = sensitivityMap;
        TargetClass = targetClass;
        BasePrediction = basePrediction;
        WindowShape = windowShape;
        Strides = strides;
    }

    /// <summary>
    /// Gets the most important regions (highest sensitivity).
    /// </summary>
    /// <param name="k">Number of regions to return.</param>
    /// <returns>Positions and sensitivity values of top regions.</returns>
    public IEnumerable<(int[] Position, T Sensitivity)> GetMostImportantRegions(int k = 10)
    {
        var regions = new List<(int[] Position, T Sensitivity)>();

        for (int h = 0; h < SensitivityMap.Shape[0]; h++)
        {
            for (int w = 0; w < SensitivityMap.Shape[1]; w++)
            {
                regions.Add((new[] { h * Strides[0], w * Strides[Math.Min(1, Strides.Length - 1)] },
                            SensitivityMap[h, w]));
            }
        }

        return regions
            .OrderByDescending(r => NumOps.ToDouble(r.Sensitivity))
            .Take(k);
    }

    /// <summary>
    /// Upsamples the sensitivity map to match input size.
    /// </summary>
    /// <returns>Upsampled sensitivity map.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since the sensitivity map is smaller than the input
    /// (due to striding), this method scales it up for visualization.
    /// </para>
    /// </remarks>
    public Tensor<T> GetUpsampledMap()
    {
        // Simple nearest-neighbor upsampling
        int targetH = Input.Shape[Input.Shape.Length >= 3 ? (Input.Shape[0] <= 4 ? 1 : 0) : 0];
        int targetW = Input.Shape[Input.Shape.Length >= 3 ? (Input.Shape[0] <= 4 ? 2 : 1) : 1];

        var upsampled = new Tensor<T>(new[] { targetH, targetW });

        for (int h = 0; h < targetH; h++)
        {
            for (int w = 0; w < targetW; w++)
            {
                int srcH = Math.Min(h / Strides[0], SensitivityMap.Shape[0] - 1);
                int srcW = Math.Min(w / Strides[Math.Min(1, Strides.Length - 1)], SensitivityMap.Shape[1] - 1);
                upsampled[h, w] = SensitivityMap[srcH, srcW];
            }
        }

        return upsampled;
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetMostImportantRegions(3).ToList();
        double maxSens = NumOps.ToDouble(top.FirstOrDefault().Sensitivity);
        double minSens = double.MaxValue;

        for (int h = 0; h < SensitivityMap.Shape[0]; h++)
        {
            for (int w = 0; w < SensitivityMap.Shape[1]; w++)
            {
                double val = NumOps.ToDouble(SensitivityMap[h, w]);
                if (val < minSens) minSens = val;
            }
        }

        return $"Occlusion Analysis for class {TargetClass}:\n" +
               $"  Base prediction: {BasePrediction}\n" +
               $"  Window: {string.Join("x", WindowShape)}, Stride: {string.Join("x", Strides)}\n" +
               $"  Sensitivity range: [{minSens:F4}, {maxSens:F4}]\n" +
               $"  Top regions: {string.Join(", ", top.Select(r => $"({r.Position[0]},{r.Position[1]})={NumOps.ToDouble(r.Sensitivity):F4}"))}";
    }
}
