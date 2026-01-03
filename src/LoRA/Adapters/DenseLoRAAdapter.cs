using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The DenseLoRAAdapter wraps Dense or FullyConnected layers and adds a LoRA layer in parallel.
/// During forward pass, both the base layer and LoRA layer process the input, and their outputs are
/// summed. The base layer's parameters can be frozen while only the LoRA parameters are trained.
/// </para>
/// <para><b>For Beginners:</b> This adapter lets you add LoRA to Dense or FullyConnected layers.
/// Think of it like adding a "correction layer" that learns what adjustments are needed:
///
/// - The base layer keeps its original weights (optionally frozen)
/// - The LoRA layer learns a small correction
/// - The final output is: original_output + lora_correction
///
/// This is incredibly useful for fine-tuning pre-trained models:
/// 1. Load a pre-trained model with Dense/FullyConnected layers
/// 2. Wrap those layers with DenseLoRAAdapter
/// 3. Freeze the base layers
/// 4. Train only the small LoRA corrections
/// 5. Achieve similar results with 100x fewer trainable parameters!
///
/// Example: If you have a dense layer with 1000x1000 weights, wrapping it with rank=8 LoRA
/// (frozen) reduces trainable parameters from 1,000,000 to just 16,000!
/// </para>
/// </remarks>
public class DenseLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Initializes a new Dense LoRA adapter wrapping an existing Dense or FullyConnected layer.
    /// </summary>
    /// <param name="baseLayer">The Dense or FullyConnected layer to adapt with LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the base layer doesn't have 1D input/output shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an adapter that adds LoRA to a Dense or FullyConnected layer.
    ///
    /// Parameters:
    /// - baseLayer: The Dense or FullyConnected layer you want to make more efficient to fine-tune
    /// - rank: How much compression (lower = fewer parameters, less flexibility)
    /// - alpha: How strong the LoRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true for efficiency)
    ///
    /// This adapter only works with layers that have 1D input/output shapes, which includes:
    /// - DenseLayer (standard fully connected layer)
    /// - FullyConnectedLayer (another name for the same thing)
    ///
    /// It validates that the base layer has compatible shapes before proceeding.
    /// </para>
    /// </remarks>
    public DenseLoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Validate base layer has single-dimensional input/output (specific to Dense layers)
        if (baseLayer.GetInputShape().Length != 1 || baseLayer.GetOutputShape().Length != 1)
        {
            throw new ArgumentException("DenseLoRAAdapter only supports layers with 1D input/output shapes (Dense/FullyConnected layers)", nameof(baseLayer));
        }
    }

    /// <summary>
    /// Merges the LoRA adaptation into the base layer and returns the merged Dense layer.
    /// </summary>
    /// <returns>A new DenseLayer with LoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method supports merging for both DenseLayer and FullyConnectedLayer base layers.
    /// The LoRA weights are computed and added directly to the base layer's weight matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRA adaptation to create a regular Dense layer.
    /// After training with LoRA, you can merge the adaptation into the original weights for:
    /// - Faster inference (no need to compute LoRA separately)
    /// - Simpler deployment (single layer instead of two)
    /// - Compatibility with systems that don't support LoRA
    ///
    /// Think of it like merging tracked changes in a document - you go from "original + changes"
    /// to a single updated version.
    ///
    /// The merging process:
    /// 1. Gets the LoRA weight matrix (computed from A and B matrices)
    /// 2. Adds these weights to the base layer's existing weights
    /// 3. Copies biases unchanged (LoRA doesn't modify biases)
    /// 4. Creates a new DenseLayer with the merged weights
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DenseLoRAAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters (works for both DenseLayer and FullyConnectedLayer)
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Both DenseLayer and FullyConnectedLayer store parameters as [weights..., biases...]
        // We need to add the LoRA weights to the base weights
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        // DenseLayer uses industry standard [inputSize, outputSize] convention and GetParameters()
        // returns them in row-major order (row by row), so:
        // - Parameter index i corresponds to W[i / outputSize, i % outputSize]
        // - loraWeights from MergeWeights() is also [inputSize, outputSize]
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / outputSize;
            int col = i % outputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }
}
