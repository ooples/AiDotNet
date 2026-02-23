using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
        var inputTensor = new Tensor<T>(_inputShape);
        var dataSpan = inputTensor.Data.Span;
        for (int i = 0; i < instance.Length && i < dataSpan.Length; i++)
        {
            dataSpan[i] = instance[i];
        }

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

        T[,] heatmap;

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
        var predSpan = predictions.Data.Span;
        if (numClasses > predSpan.Length)
            throw new InvalidOperationException(
                $"Prediction tensor has {predSpan.Length} elements but expected {numClasses} class scores based on tensor shape.");

        var classScores = new T[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            classScores[i] = predSpan[i];
        }

        return new GradCAMExplanation<T>
        {
            Heatmap = upsampledHeatmap,
            OriginalHeatmap = heatmap,
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
    private T[,] ComputeGradCAM(Tensor<T> featureMaps, Tensor<T> gradients)
    {
        // Feature maps: [channels, height, width] or [batch, channels, height, width]
        int channels = _featureMapShape[0];
        int height = _featureMapShape.Length > 1 ? _featureMapShape[1] : 1;
        int width = _featureMapShape.Length > 2 ? _featureMapShape[2] : 1;

        var gradSpan = gradients.Data.Span;
        var featSpan = featureMaps.Data.Span;

        // Global average pooling of gradients to get channel weights
        var weights = new double[channels];

        for (int c = 0; c < channels; c++)
        {
            double sum = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < gradSpan.Length)
                    {
                        sum += NumOps.ToDouble(gradSpan[idx]);
                    }
                }
            }
            weights[c] = sum / (height * width);
        }

        // Weighted combination of feature maps
        var heatmap = new T[height, width];

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double value = 0;
                for (int c = 0; c < channels; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < featSpan.Length)
                    {
                        value += weights[c] * NumOps.ToDouble(featSpan[idx]);
                    }
                }
                // ReLU to keep only positive contributions
                heatmap[h, w] = NumOps.FromDouble(Math.Max(0, value));
            }
        }

        // Normalize
        NormalizeHeatmap(heatmap);

        return heatmap;
    }

    /// <summary>
    /// Computes Grad-CAM++ heatmap (better for multiple instances).
    /// </summary>
    private T[,] ComputeGradCAMPlusPlus(Tensor<T> featureMaps, Tensor<T> gradients)
    {
        int channels = _featureMapShape[0];
        int height = _featureMapShape.Length > 1 ? _featureMapShape[1] : 1;
        int width = _featureMapShape.Length > 2 ? _featureMapShape[2] : 1;

        var gradSpan = gradients.Data.Span;
        var featSpan = featureMaps.Data.Span;

        // Grad-CAM++ uses pixel-wise weights instead of global average
        var weights = new double[channels];

        for (int c = 0; c < channels; c++)
        {
            double sumAlpha = 0;
            double sumGrad = 0;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < gradSpan.Length && idx < featSpan.Length)
                    {
                        double grad = NumOps.ToDouble(gradSpan[idx]);
                        double feat = NumOps.ToDouble(featSpan[idx]);

                        // Alpha coefficient for Grad-CAM++
                        double grad2 = grad * grad;
                        double grad3 = grad2 * grad;
                        double alpha = grad2 / (2 * grad2 + feat * grad3 + 1e-8);

                        sumAlpha += alpha * Math.Max(0, grad);
                        sumGrad += grad;
                    }
                }
            }

            weights[c] = sumAlpha;
        }

        // Weighted combination
        var heatmap = new T[height, width];

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double value = 0;
                for (int c = 0; c < channels; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < featSpan.Length)
                    {
                        value += weights[c] * NumOps.ToDouble(featSpan[idx]);
                    }
                }
                heatmap[h, w] = NumOps.FromDouble(Math.Max(0, value));
            }
        }

        NormalizeHeatmap(heatmap);

        return heatmap;
    }

    /// <summary>
    /// Computes a simulated heatmap when gradient access is not available.
    /// Uses occlusion sensitivity as an approximation.
    /// </summary>
    private T[,] ComputeSimulatedHeatmap(Tensor<T> input, int targetClass)
    {
        var inputSpan = input.Data.Span;
        int height = _inputShape.Length >= 2 ? _inputShape[_inputShape.Length - 2] : (int)Math.Sqrt(inputSpan.Length);
        int width = _inputShape.Length >= 1 ? _inputShape[_inputShape.Length - 1] : height;

        // Use a smaller resolution for efficiency
        int heatmapHeight = Math.Min(height, 14);
        int heatmapWidth = Math.Min(width, 14);

        var heatmap = new T[heatmapHeight, heatmapWidth];

        // Get baseline prediction
        var basePred = _predictFunction(input);
        var basePredSpan = basePred.Data.Span;
        double baseScore = targetClass < basePredSpan.Length
            ? NumOps.ToDouble(basePredSpan[targetClass])
            : 0;

        // Occlude different regions and measure importance
        int patchHeight = height / heatmapHeight;
        int patchWidth = width / heatmapWidth;

        for (int i = 0; i < heatmapHeight; i++)
        {
            for (int j = 0; j < heatmapWidth; j++)
            {
                // Create occluded version
                var occluded = new Tensor<T>(input.Shape);
                var occSpan = occluded.Data.Span;
                inputSpan.CopyTo(occSpan);

                // Zero out the patch
                int startH = i * patchHeight;
                int startW = j * patchWidth;
                for (int h = startH; h < startH + patchHeight && h < height; h++)
                {
                    for (int w = startW; w < startW + patchWidth && w < width; w++)
                    {
                        int idx = h * width + w;
                        if (idx < occSpan.Length)
                        {
                            occSpan[idx] = NumOps.Zero;
                        }
                    }
                }

                // Get prediction with occlusion
                var occPred = _predictFunction(occluded);
                var occPredSpan = occPred.Data.Span;
                double occScore = targetClass < occPredSpan.Length
                    ? NumOps.ToDouble(occPredSpan[targetClass])
                    : 0;

                // Importance = drop in score when occluded
                heatmap[i, j] = NumOps.FromDouble(Math.Max(0, baseScore - occScore));
            }
        }

        NormalizeHeatmap(heatmap);

        return heatmap;
    }

    /// <summary>
    /// Gets the predicted class index.
    /// </summary>
    private int GetPredictedClass(Tensor<T> predictions)
    {
        int maxIdx = 0;
        double maxVal = double.MinValue;
        var predSpan = predictions.Data.Span;

        for (int i = 0; i < predSpan.Length; i++)
        {
            double val = NumOps.ToDouble(predSpan[i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Normalizes heatmap to [0, 1] range.
    /// </summary>
    private void NormalizeHeatmap(T[,] heatmap)
    {
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        int height = heatmap.GetLength(0);
        int width = heatmap.GetLength(1);

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double val = NumOps.ToDouble(heatmap[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }
        }

        double range = maxVal - minVal;
        if (range < 1e-10) range = 1;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double val = (NumOps.ToDouble(heatmap[i, j]) - minVal) / range;
                heatmap[i, j] = NumOps.FromDouble(val);
            }
        }
    }

    /// <summary>
    /// Upsamples heatmap to target size using bilinear interpolation.
    /// </summary>
    private T[,] UpsampleHeatmap(T[,] heatmap, int targetHeight, int targetWidth)
    {
        int srcHeight = heatmap.GetLength(0);
        int srcWidth = heatmap.GetLength(1);

        var result = new T[targetHeight, targetWidth];

        for (int i = 0; i < targetHeight; i++)
        {
            for (int j = 0; j < targetWidth; j++)
            {
                // Map to source coordinates
                double srcY = (double)i / targetHeight * srcHeight;
                double srcX = (double)j / targetWidth * srcWidth;

                // Bilinear interpolation
                int y0 = Math.Min((int)srcY, srcHeight - 1);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                int x0 = Math.Min((int)srcX, srcWidth - 1);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);

                double yFrac = srcY - y0;
                double xFrac = srcX - x0;

                double v00 = NumOps.ToDouble(heatmap[y0, x0]);
                double v01 = NumOps.ToDouble(heatmap[y0, x1]);
                double v10 = NumOps.ToDouble(heatmap[y1, x0]);
                double v11 = NumOps.ToDouble(heatmap[y1, x1]);

                double value = v00 * (1 - yFrac) * (1 - xFrac)
                             + v01 * (1 - yFrac) * xFrac
                             + v10 * yFrac * (1 - xFrac)
                             + v11 * yFrac * xFrac;

                result[i, j] = NumOps.FromDouble(value);
            }
        }

        return result;
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
    public T[,] Heatmap { get; set; } = new T[0, 0];

    /// <summary>
    /// Gets or sets the original (low-resolution) heatmap from feature maps.
    /// </summary>
    public T[,] OriginalHeatmap { get; set; } = new T[0, 0];

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
        int height = Heatmap.GetLength(0);
        int width = Heatmap.GetLength(1);

        int y = Math.Min((int)(normalizedY * height), height - 1);
        int x = Math.Min((int)(normalizedX * width), width - 1);

        return Heatmap[Math.Max(0, y), Math.Max(0, x)];
    }

    /// <summary>
    /// Gets regions with importance above a threshold.
    /// </summary>
    public List<(int y, int x, T importance)> GetImportantRegions(double threshold = 0.5)
    {
        var regions = new List<(int, int, T)>();
        int height = Heatmap.GetLength(0);
        int width = Heatmap.GetLength(1);

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (NumOps.ToDouble(Heatmap[i, j]) >= threshold)
                {
                    regions.Add((i, j, Heatmap[i, j]));
                }
            }
        }

        return regions.OrderByDescending(r => NumOps.ToDouble(r.Item3)).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        int height = Heatmap.GetLength(0);
        int width = Heatmap.GetLength(1);

        // Compute statistics
        double maxImportance = 0;
        double avgImportance = 0;
        int aboveThreshold = 0;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double val = NumOps.ToDouble(Heatmap[i, j]);
                maxImportance = Math.Max(maxImportance, val);
                avgImportance += val;
                if (val > 0.5) aboveThreshold++;
            }
        }
        avgImportance /= (height * width);
        double percentHighImportance = 100.0 * aboveThreshold / (height * width);

        return $"{(IsGradCAMPlusPlus ? "Grad-CAM++" : "Grad-CAM")} Explanation:\n" +
               $"  Target class: {TargetClass}\n" +
               $"  Heatmap size: {height}x{width}\n" +
               $"  Max importance: {maxImportance:F4}\n" +
               $"  Avg importance: {avgImportance:F4}\n" +
               $"  High importance area: {percentHighImportance:F1}% (>0.5)";
    }
}
