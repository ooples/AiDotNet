namespace AiDotNet.Models;

/// <summary>
/// Describes which dimensions of a model's input/output shapes are dynamic (variable at runtime).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Most machine learning models have fixed input sizes (e.g., always 784 features).
/// But some models can handle variable-sized inputs. For example, a model might accept any batch size,
/// or a sequence model might handle different sequence lengths. This class describes which dimensions
/// can vary and what their valid ranges are.
///
/// This follows the ONNX convention where -1 in a shape dimension means "variable at runtime".
/// For example, an input shape of [-1, 784] means "any number of samples, each with 784 features".
///
/// Industry standards:
/// - ONNX: Uses -1 for dynamic dimensions in shape arrays
/// - TensorFlow Serving: Uses None/null for dynamic dimensions
/// - TorchServe: Uses -1 for dynamic dimensions
/// </remarks>
public sealed class DynamicShapeInfo
{
    /// <summary>
    /// A shared instance representing no dynamic dimensions (all dimensions are fixed).
    /// </summary>
    public static readonly DynamicShapeInfo None = new();

    /// <summary>
    /// Gets the indices of input dimensions that are variable at runtime.
    /// For example, [0] means the first dimension (typically batch size) is dynamic.
    /// </summary>
    public int[] DynamicInputDimensions { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Gets the indices of output dimensions that are variable at runtime.
    /// </summary>
    public int[] DynamicOutputDimensions { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Gets the minimum allowed values for dynamic input dimensions.
    /// Array indices correspond to <see cref="DynamicInputDimensions"/>.
    /// A value of 0 means no minimum constraint.
    /// </summary>
    public int[] MinInputDimensions { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Gets the maximum allowed values for dynamic input dimensions.
    /// Array indices correspond to <see cref="DynamicInputDimensions"/>.
    /// A value of 0 means no maximum constraint (unlimited).
    /// </summary>
    public int[] MaxInputDimensions { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Gets whether the batch dimension (index 0) is dynamic.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Most serving systems need to know if a model can process
    /// different batch sizes. If this is true, the model supports batching natively.
    /// </remarks>
    public bool HasDynamicBatch => Array.IndexOf(DynamicInputDimensions, 0) >= 0;

    /// <summary>
    /// Gets whether any input dimensions are dynamic.
    /// </summary>
    public bool HasDynamicInput => DynamicInputDimensions.Length > 0;

    /// <summary>
    /// Gets whether any output dimensions are dynamic.
    /// </summary>
    public bool HasDynamicOutput => DynamicOutputDimensions.Length > 0;

    /// <summary>
    /// Validates a concrete shape against a template shape, respecting dynamic dimensions.
    /// </summary>
    /// <param name="concreteShape">The actual shape to validate (e.g., from user input).</param>
    /// <param name="templateShape">The expected shape from the model definition. Dynamic dimensions use -1.</param>
    /// <param name="dynamicDimIndices">Indices of dimensions that are dynamic (variable).</param>
    /// <param name="minDimensions">Optional minimum allowed values for dynamic dimensions.
    /// When null, defaults to <see cref="MinInputDimensions"/>.</param>
    /// <param name="maxDimensions">Optional maximum allowed values for dynamic dimensions.
    /// When null, defaults to <see cref="MaxInputDimensions"/>.</param>
    /// <returns>True if the concrete shape is compatible with the template.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> When a model has input shape [-1, 784], and you send data with shape [32, 784],
    /// this method checks that:
    /// - The number of dimensions matches (both have 2 dimensions)
    /// - Fixed dimensions match exactly (784 == 784)
    /// - Dynamic dimensions (-1) accept any positive value (32 > 0)
    /// - Min/max constraints are respected if configured
    ///
    /// When validating output shapes, pass the output-specific min/max arrays so that
    /// the input min/max constraints are not incorrectly applied.
    /// </remarks>
    public bool IsValidShape(int[] concreteShape, int[] templateShape, int[] dynamicDimIndices,
        int[]? minDimensions = null, int[]? maxDimensions = null)
    {
        if (concreteShape is null || templateShape is null || dynamicDimIndices is null)
        {
            return false;
        }

        if (concreteShape.Length != templateShape.Length)
        {
            return false;
        }

        // Pre-compute a lookup mask for O(1) per-dimension checks instead of O(n*m) linear scans.
        var dynamicLookup = new Dictionary<int, int>(dynamicDimIndices.Length);
        for (int d = 0; d < dynamicDimIndices.Length; d++)
        {
            dynamicLookup[dynamicDimIndices[d]] = d;
        }

        for (int i = 0; i < concreteShape.Length; i++)
        {
            bool hasDynamicIndex = dynamicLookup.TryGetValue(i, out int dynamicIndex);
            bool isDynamic = hasDynamicIndex || templateShape[i] == -1;

            if (isDynamic)
            {
                // Dynamic dimensions must be positive
                if (concreteShape[i] <= 0)
                {
                    return false;
                }

                // Check min/max constraints if available
                if (hasDynamicIndex)
                {
                    var minArray = minDimensions ?? MinInputDimensions;
                    var maxArray = maxDimensions ?? MaxInputDimensions;

                    if (dynamicIndex < minArray.Length && minArray[dynamicIndex] > 0)
                    {
                        if (concreteShape[i] < minArray[dynamicIndex])
                        {
                            return false;
                        }
                    }

                    if (dynamicIndex < maxArray.Length && maxArray[dynamicIndex] > 0)
                    {
                        if (concreteShape[i] > maxArray[dynamicIndex])
                        {
                            return false;
                        }
                    }
                }
            }
            else
            {
                // Fixed dimensions must match exactly
                if (concreteShape[i] != templateShape[i])
                {
                    return false;
                }
            }
        }

        return true;
    }
}
