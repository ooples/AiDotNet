using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Interpretability.Helpers;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Gradient-weighted Class Activation Mapping (Grad-CAM) explainer for CNNs.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Grad-CAM creates visual explanations showing which parts of an image
/// were most important for a CNN's prediction. It produces a heatmap highlighting important regions.
///
/// How it works:
/// 1. Pass the image through the CNN
/// 2. Get the feature maps from a convolutional layer (typically the last one)
/// 3. Compute gradients of the target class score with respect to feature maps
/// 4. Weight each feature map by its average gradient (importance)
/// 5. Combine weighted feature maps and apply ReLU
/// 6. Resize the result to match input image size
///
/// Why Grad-CAM is useful:
/// - Visual and intuitive: shows a heatmap over the image
/// - Class-discriminative: different classes highlight different regions
/// - Works with any CNN architecture (VGG, ResNet, etc.)
/// - No modification to the model architecture needed
///
/// Example: For an image classified as "cat":
/// - The heatmap would highlight the cat's face, body, ears
/// - Areas like the background would have low activation
///
/// Grad-CAM++ is an improved version that handles multiple instances of the same object better.
/// </para>
/// </remarks>
public class GradCAMExplainer<T> : ILocalExplainer<T, GradCAMExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Tensor<T>, Tensor<T>> _predictFunction;
    private readonly Func<Tensor<T>, int, Tensor<T>>? _featureMapFunction;
    [Scratch]
    private readonly Func<Tensor<T>, int, int, Tensor<T>>? _gradientFunction;
    private readonly int[] _inputShape;
    private readonly int[] _featureMapShape;
    private readonly bool _useGradCAMPlusPlus;

    /// <inheritdoc/>
    public string MethodName => _useGradCAMPlusPlus ? "GradCAM++" : "GradCAM";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Grad-CAM explainer.
    /// </summary>
    /// <param name="predictFunction">Function that takes input tensor and returns class scores.</param>
    /// <param name="featureMapFunction">Function that returns feature maps from a conv layer.
    /// Takes (input, layerIndex) and returns feature maps [batch, channels, height, width].</param>
    /// <param name="gradientFunction">Function that computes gradients of class score w.r.t. feature maps.
    /// Takes (input, layerIndex, classIndex) and returns gradients.</param>
    /// <param name="inputShape">Shape of input tensor [height, width, channels] or [channels, height, width].</param>
    /// <param name="featureMapShape">Shape of feature maps [channels, height, width].</param>
    /// <param name="useGradCAMPlusPlus">Use Grad-CAM++ variant (default: false).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>featureMapFunction</b>: You need to extract intermediate CNN activations. Most frameworks
    ///   support this (e.g., using hooks in PyTorch or layer outputs in Keras).
    /// - <b>useGradCAMPlusPlus</b>: Use this when images have multiple instances of the same class.
    ///   Standard Grad-CAM might miss some of them.
    /// </para>
    /// </remarks>
    public GradCAMExplainer(
        Func<Tensor<T>, Tensor<T>> predictFunction,
        Func<Tensor<T>, int, Tensor<T>>? featureMapFunction,
        Func<Tensor<T>, int, int, Tensor<T>>? gradientFunction,
        int[] inputShape,
        int[] featureMapShape,
        bool useGradCAMPlusPlus = false)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        _featureMapFunction = featureMapFunction;
        _gradientFunction = gradientFunction;
        Guard.NotNull(inputShape);
        _inputShape = inputShape;
        Guard.NotNull(featureMapShape);
        _featureMapShape = featureMapShape;
        _useGradCAMPlusPlus = useGradCAMPlusPlus;
    }

    /// <summary>
    /// Computes Grad-CAM heatmap for an input image.
    /// </summary>
    /// <param name="instance">The input as a flattened vector.</param>
    /// <returns>Grad-CAM explanation with heatmap.</returns>
    public GradCAMExplanation<T> Explain(Vector<T> instance)
    {
        // Reshape vector to tensor
        int inputLength = _inputShape.Aggregate(1, (product, dim) => product * dim);
        var inputTensor = new Tensor<T>(_inputShape, ExplainerTensorOps<T>.PadOrTruncate(instance, inputLength));

        return ExplainTensor(inputTensor, targetClass: -1);
    }

    /// <summary>
    /// Computes Grad-CAM heatmap for an input tensor.
    /// </summary>
    /// <param name="input">The input tensor (image).</param>
    /// <param name="targetClass">Target class index. If -1, uses the predicted class.</param>
    /// <param name="layerIndex">Index of the conv layer to use (default: 0, typically last conv layer).</param>
    /// <returns>Grad-CAM explanation with heatmap.</returns>
    public GradCAMExplanation<T> ExplainTensor(Tensor<T> input, int targetClass = -1, int layerIndex = 0)
    {
        // Get predictions
        var predictions = _predictFunction(input);
        int numClasses = predictions.Shape.Length > 0 ? predictions.Shape[predictions.Shape.Length - 1] : 1;

        // Determine target class if not specified
        if (targetClass < 0)
        {
            targetClass = GetPredictedClass(predictions);
        }

        Tensor<T> heatmap;

        if (_featureMapFunction != null && _gradientFunction != null)
        {
            // Use actual feature maps and gradients
            var featureMaps = _featureMapFunction(input, layerIndex);
            var gradients = _gradientFunction(input, layerIndex, targetClass);

            heatmap = _useGradCAMPlusPlus
                ? ComputeGradCAMPlusPlus(featureMaps, gradients)
                : ComputeGradCAM(featureMaps, gradients);
        }
        else
        {
            // Create simulated heatmap using saliency approximation
            heatmap = ComputeSimulatedHeatmap(input, targetClass);
        }

        // Get target shape for upsampling
        int targetHeight = _inputShape.Length >= 2 ? _inputShape[_inputShape.Length - 2] : _inputShape[0];
        int targetWidth = _inputShape.Length >= 1 ? _inputShape[_inputShape.Length - 1] : _inputShape[0];

        // Upsample heatmap to input size
        var upsampledHeatmap = UpsampleHeatmap(heatmap, targetHeight, targetWidth);

        // Get class scores
        if (numClasses > predictions.Length)
            throw new InvalidOperationException(
                $"Prediction tensor has {predictions.Length} elements but expected {numClasses} class scores based on tensor shape.");

        var classScores = predictions.ToVector().GetSubVector(0, numClasses).ToArray();

        return new GradCAMExplanation<T>
        {
            Heatmap = upsampledHeatmap.ToMatrix(),
            OriginalHeatmap = heatmap.ToMatrix(),
            TargetClass = targetClass,
            ClassScores = classScores,
            InputShape = _inputShape,
            IsGradCAMPlusPlus = _useGradCAMPlusPlus
        };
    }

    /// <inheritdoc/>
    public GradCAMExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new GradCAMExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes standard Grad-CAM heatmap.
    /// </summary>
    private Tensor<T> ComputeGradCAM(Tensor<T> featureMaps, Tensor<T> gradients)
    {
        // Feature maps: [channels, height, width] or [batch, channels, height, width]
        int channels = _featureMapShape[0];
        int height = _featureMapShape.Length > 1 ? _featureMapShape[1] : 1;
        int width = _featureMapShape.Length > 2 ? _featureMapShape[2] : 1;
        var engine = AiDotNetEngine.Current;

        var grad = ChannelPlanes(gradients, channels, height * width);
        var feat = ChannelPlanes(featureMaps, channels, height * width);

        // Global average pooling of gradients to get channel weights
        var weights = engine.ReduceMean(grad, new[] { 1 }, false);

        return WeightedHeatmap(weights, feat, height, width);
    }

    /// <summary>
    /// The leading channels * pixels entries of a tensor (zero where it is shorter) as a [channels, pixels] tensor.
    /// </summary>
    private static Tensor<T> ChannelPlanes(Tensor<T> values, int channels, int pixels)
    {
        return new Tensor<T>(new[] { channels, pixels },
            ExplainerTensorOps<T>.PadOrTruncate(values.ToVector(), channels * pixels));
    }

    /// <summary>
    /// ReLU(sum_c weights_c * features_c) as a normalized [height, width] heatmap.
    /// </summary>
    private static Tensor<T> WeightedHeatmap(Tensor<T> weights, Tensor<T> features, int height, int width)
    {
        var engine = AiDotNetEngine.Current;

        // Weighted combination of feature maps, ReLU to keep only positive contributions
        var combined = engine.TensorMatMul(weights.Reshape(new[] { 1, weights.Length }), features);
        var heatmap = engine.ReLU(combined).Reshape(new[] { height, width });

        // Normalize
        return ExplainerTensorOps<T>.MinMaxNormalize(heatmap, 1e-10);
    }

    /// <summary>
    /// Computes Grad-CAM++ heatmap (better for multiple instances).
    /// </summary>
    private Tensor<T> ComputeGradCAMPlusPlus(Tensor<T> featureMaps, Tensor<T> gradients)
    {
        int channels = _featureMapShape[0];
        int height = _featureMapShape.Length > 1 ? _featureMapShape[1] : 1;
        int width = _featureMapShape.Length > 2 ? _featureMapShape[2] : 1;
        var engine = AiDotNetEngine.Current;

        var grad = ChannelPlanes(gradients, channels, height * width);
        var feat = ChannelPlanes(featureMaps, channels, height * width);

        // Grad-CAM++ uses pixel-wise weights instead of global average:
        // alpha = g^2 / (2 g^2 + f g^3 + 1e-8), weight_c = sum over pixels of alpha * ReLU(g)
        var grad2 = engine.TensorSquare(grad);
        var grad3 = engine.TensorMultiply(grad2, grad);
        var denominator = engine.TensorAddScalar(
            engine.TensorAdd(engine.TensorMultiplyScalar(grad2, NumOps.FromDouble(2)), engine.TensorMultiply(feat, grad3)),
            NumOps.FromDouble(1e-8));
        var alpha = engine.TensorDivide(grad2, denominator);
        var weights = engine.ReduceSum(engine.TensorMultiply(alpha, engine.ReLU(grad)), new[] { 1 }, false);

        return WeightedHeatmap(weights, feat, height, width);
    }

    /// <summary>
    /// Computes a simulated heatmap when gradient access is not available.
    /// Uses occlusion sensitivity as an approximation.
    /// </summary>
    private Tensor<T> ComputeSimulatedHeatmap(Tensor<T> input, int targetClass)
    {
        int inputLength = input.Length;
        int height = _inputShape.Length >= 2 ? _inputShape[_inputShape.Length - 2] : (int)Math.Sqrt(inputLength);
        int width = _inputShape.Length >= 1 ? _inputShape[_inputShape.Length - 1] : height;

        // Use a smaller resolution for efficiency
        int heatmapHeight = Math.Min(height, 14);
        int heatmapWidth = Math.Min(width, 14);
        if (heatmapHeight <= 0 || heatmapWidth <= 0)
        {
            return new Tensor<T>(new[] { Math.Max(0, heatmapHeight), Math.Max(0, heatmapWidth) });
        }

        var engine = AiDotNetEngine.Current;

        // Get baseline prediction
        var basePrediction = _predictFunction(input).ToVector();
        T baseScore = targetClass < basePrediction.Length ? basePrediction[targetClass] : NumOps.Zero;

        // Occlude different regions and measure importance
        int patchHeight = height / heatmapHeight;
        int patchWidth = width / heatmapWidth;
        var zeros = engine.TensorFullLike(input, NumOps.Zero);
        var occludedScores = new List<Vector<T>>(heatmapHeight * heatmapWidth);

        for (int i = 0; i < heatmapHeight; i++)
        {
            for (int j = 0; j < heatmapWidth; j++)
            {
                // Zero out the patch: flat entries h * width + w of the patch rows and columns
                int startH = i * patchHeight;
                int startW = j * patchWidth;
                var rows = ExplainerTensorOps<T>.RangeIndicator(height, startH, startH + patchHeight);
                var columns = ExplainerTensorOps<T>.RangeIndicator(width, startW, startW + patchWidth);
                var patch = ExplainerTensorOps<T>.PadOrTruncate(engine.TensorOuter(rows, columns).ToVector(), inputLength);
                var occluded = engine.TensorWhere(new Tensor<T>(input._shape, patch), zeros, input);

                // Get prediction with occlusion
                occludedScores.Add(ScoreOrZero(_predictFunction(occluded).ToVector(), targetClass));
            }
        }

        // Importance = drop in score when occluded
        var scores = ExplainerTensorOps<T>.Stack(occludedScores).Reshape(new[] { heatmapHeight, heatmapWidth });
        var heatmap = engine.ReLU(engine.TensorSubtract(engine.TensorFullLike(scores, baseScore), scores));

        return ExplainerTensorOps<T>.MinMaxNormalize(heatmap, 1e-10);
    }

    /// <summary>
    /// The one-entry vector holding the score at <paramref name="targetClass"/>, or zero when the prediction is shorter.
    /// </summary>
    private static Vector<T> ScoreOrZero(Vector<T> prediction, int targetClass)
    {
        return targetClass < prediction.Length ? prediction.GetSubVector(targetClass, 1) : new Vector<T>(1);
    }

    /// <summary>
    /// Gets the predicted class index.
    /// </summary>
    private int GetPredictedClass(Tensor<T> predictions)
    {
        return ExplainerTensorOps<T>.ArgMaxSkippingNaN(predictions.ToVector());
    }

    /// <summary>
    /// Upsamples heatmap to target size using bilinear interpolation.
    /// </summary>
    private static Tensor<T> UpsampleHeatmap(Tensor<T> heatmap, int targetHeight, int targetWidth)
    {
        return ExplainerTensorOps<T>.BilinearResize(heatmap, targetHeight, targetWidth);
    }
}

/// <summary>
/// Represents the result of a Grad-CAM analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GradCAMExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the upsampled heatmap (same size as input).
    /// Values are in [0, 1] where 1 = most important.
    /// </summary>
    public Matrix<T> Heatmap { get; set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets or sets the original (low-resolution) heatmap from feature maps.
    /// </summary>
    public Matrix<T> OriginalHeatmap { get; set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets or sets the target class that was explained.
    /// </summary>
    public int TargetClass { get; set; }

    /// <summary>
    /// Gets or sets the class scores from the model.
    /// </summary>
    public T[] ClassScores { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the input shape.
    /// </summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets whether Grad-CAM++ was used.
    /// </summary>
    public bool IsGradCAMPlusPlus { get; set; }

    /// <summary>
    /// Gets the heatmap value at a specific location (normalized coordinates).
    /// </summary>
    public T GetImportanceAt(double normalizedY, double normalizedX)
    {
        int height = Heatmap.Rows;
        int width = Heatmap.Columns;

        int y = Math.Min((int)(normalizedY * height), height - 1);
        int x = Math.Min((int)(normalizedX * width), width - 1);

        return Heatmap[Math.Max(0, y), Math.Max(0, x)];
    }

    /// <summary>
    /// Gets regions with importance above a threshold.
    /// </summary>
    public List<(int y, int x, T importance)> GetImportantRegions(double threshold = 0.5)
    {
        var engine = AiDotNetEngine.Current;
        int width = Heatmap.Columns;
        var values = Tensor<T>.FromMatrix(Heatmap).Reshape(new[] { Heatmap.Rows * width });
        var above = engine.TensorGreaterEqual(values, engine.TensorFullLike(values, NumOps.FromDouble(threshold)));
        if (engine.TensorCountNonzero(above) == 0)
        {
            return new List<(int y, int x, T importance)>();
        }

        // Row-major positions at or above the threshold, then ordered by descending importance
        // (the stable argsort keeps equal importances in row-major order, as OrderByDescending did).
        var nonzero = engine.TensorNonzero(above);
        var positions = nonzero.Reshape(new[] { nonzero.Shape[0] });
        var selected = engine.TensorIndexSelect(values, positions, 0);
        var order = engine.TensorArgsort(selected, 0, true);
        var orderedPositions = engine.TensorIndexSelect(positions, order, 0).AsSpan().ToArray();
        var orderedValues = engine.TensorIndexSelect(selected, order, 0).ToVector();
        return orderedPositions
            .Zip(orderedValues, (position, importance) => (y: position / width, x: position % width, importance))
            .ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        int height = Heatmap.Rows;
        int width = Heatmap.Columns;

        // Compute statistics
        var engine = AiDotNetEngine.Current;
        var map = Tensor<T>.FromMatrix(Heatmap);
        T maxImportance = NumOps.Zero;
        T avgImportance = NumOps.Zero;
        T percentHighImportance = NumOps.Zero;
        if (map.Length > 0)
        {
            maxImportance = engine.TensorMaxValue(engine.TensorMax(map, NumOps.Zero));
            avgImportance = engine.TensorMean(map);
            var aboveThreshold = engine.TensorGreaterThan(map, NumOps.FromDouble(0.5));
            percentHighImportance = NumOps.Multiply(NumOps.FromDouble(100), engine.TensorMean(aboveThreshold));
        }

        return $"{(IsGradCAMPlusPlus ? "Grad-CAM++" : "Grad-CAM")} Explanation:\n" +
               $"  Target class: {TargetClass}\n" +
               $"  Heatmap size: {height}x{width}\n" +
               $"  Max importance: {maxImportance:F4}\n" +
               $"  Avg importance: {avgImportance:F4}\n" +
               $"  High importance area: {percentHighImportance:F1}% (>0.5)";
    }
}
