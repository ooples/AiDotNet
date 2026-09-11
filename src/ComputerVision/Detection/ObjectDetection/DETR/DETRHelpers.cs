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
        for (int i = 1; i < features.Count; i++)
        {
            if (features[i].Shape[0] != batch)
            {
                throw new ArgumentException(
                    $"All features must have the same batch size. Feature 0 has batch={batch}, feature {i} has batch={features[i].Shape[0]}.",
                    nameof(features));
            }
        }

        var engine = AiDotNetEngine.Current;
        int totalTokens = 0;
        var spatialShapes = new int[features.Count][];
        var levelStarts = new int[features.Count];
        var levels = new Tensor<T>[features.Count];
        for (int i = 0; i < features.Count; i++)
        {
            int c = features[i].Shape[1];
            int h = features[i].Shape[2];
            int w = features[i].Shape[3];
            spatialShapes[i] = new[] { h, w };
            levelStarts[i] = totalTokens;
            totalTokens += h * w;

            // [B, C, H, W] -> [B, H*W, C], then fit the channel axis to hiddenDim: the first
            // min(C, hiddenDim) channels are kept and any shortfall is zero-filled. Engine ops, so the
            // backbone and neck below this point stay on the gradient tape.
            var tokens = CvTensorOps<T>.FlattenSpatial(features[i]);
            if (c > hiddenDim)
            {
                tokens = engine.TensorNarrow(tokens, 2, 0, hiddenDim);
            }
            else if (c < hiddenDim)
            {
                tokens = engine.TensorConcatenate(new[] { tokens, new Tensor<T>(new[] { batch, h * w, hiddenDim - c }) }, 2);
            }

            levels[i] = tokens;
        }

        var flattened = levels.Length == 1 ? levels[0] : engine.TensorConcatenate(levels, 1);
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
    public static Tensor<T> AddTensors<T>(Tensor<T> a, Tensor<T> b) => AiDotNetEngine.Current.TensorAdd(a, b);
}
