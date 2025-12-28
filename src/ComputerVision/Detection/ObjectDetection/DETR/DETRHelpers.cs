using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// Shared helper methods for DETR-family object detectors.
/// </summary>
internal static class DETRHelpers
{
    /// <summary>
    /// Flattens multi-scale features into a single sequence of tokens.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="features">List of feature tensors at different scales [batch, channels, height, width].</param>
    /// <param name="hiddenDim">The hidden dimension to use for the output.</param>
    /// <returns>
    /// A tuple containing:
    /// - flattened: The flattened tensor [batch, total_tokens, hidden_dim]
    /// - levelStarts: Starting index for each feature level
    /// - spatialShapes: Height and width for each feature level
    /// </returns>
    public static (Tensor<T> flattened, int[] levelStarts, int[][] spatialShapes) FlattenMultiScale<T>(
        List<Tensor<T>> features,
        int hiddenDim)
    {
        if (features is null || features.Count == 0)
        {
            throw new ArgumentException("Features list cannot be null or empty.", nameof(features));
        }

        int batch = features[0].Shape[0];

        // Validate consistent batch size across all features
        for (int i = 1; i < features.Count; i++)
        {
            if (features[i].Shape[0] != batch)
            {
                throw new ArgumentException(
                    $"All features must have the same batch size. Feature 0 has batch={batch}, feature {i} has batch={features[i].Shape[0]}.",
                    nameof(features));
            }
        }

        int totalTokens = 0;
        var spatialShapes = new int[features.Count][];
        var levelStarts = new int[features.Count];

        for (int i = 0; i < features.Count; i++)
        {
            int h = features[i].Shape[2];
            int w = features[i].Shape[3];
            spatialShapes[i] = new[] { h, w };
            levelStarts[i] = totalTokens;
            totalTokens += h * w;
        }

        var flattened = new Tensor<T>(new[] { batch, totalTokens, hiddenDim });

        int offset = 0;
        for (int level = 0; level < features.Count; level++)
        {
            var feat = features[level];
            int c = feat.Shape[1];
            int h = feat.Shape[2];
            int w = feat.Shape[3];

            for (int b = 0; b < batch; b++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int tokenIdx = offset + y * w + x;
                        for (int d = 0; d < c && d < hiddenDim; d++)
                        {
                            flattened[b, tokenIdx, d] = feat[b, d, y, x];
                        }
                    }
                }
            }
            offset += h * w;
        }

        return (flattened, levelStarts, spatialShapes);
    }

    /// <summary>
    /// Computes the GELU activation function.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>GELU activation output.</returns>
    public static double GELU(double x)
    {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <param name="numOps">Numeric operations provider.</param>
    /// <returns>Element-wise sum of the tensors.</returns>
    public static Tensor<T> AddTensors<T>(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        if (a is null)
        {
            throw new ArgumentNullException(nameof(a));
        }
        if (b is null)
        {
            throw new ArgumentNullException(nameof(b));
        }
        if (a.Shape.Length != b.Shape.Length || !a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException(
                $"Tensors must have the same shape. a.Shape=[{string.Join(",", a.Shape)}], b.Shape=[{string.Join(",", b.Shape)}].",
                nameof(b));
        }

        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.Add(a[i], b[i]);
        }
        return result;
    }
}
