using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// DoRA (Weight-Decomposed LoRA) extends standard LoRA by decomposing pre-trained weights into
/// magnitude and direction components, then applying LoRA only to the direction component.
/// This decomposition leads to more stable training and better convergence compared to standard LoRA.
/// </para>
/// <para>
/// <b>Mathematical Formulation:</b>
/// Given pre-trained weights W, DoRA decomposes them as:
/// - W = m * d, where m is magnitude (scalar per neuron) and d is direction (unit vector)
/// - W' = m * normalize(W + LoRA_delta), the norm taken per output neuron
/// - LoRA_delta = (alpha/rank) * B * A
///
/// This ensures that LoRA adaptations primarily affect the direction of weights, not their magnitude,
/// which improves training stability and convergence.
/// </para>
/// <para>
/// <b>Research Context:</b>
/// DoRA was published in February 2024 and presented as an ICML 2024 Oral paper.
/// In experiments on LLaMA-7B, DoRA achieved +3.7% improvement over standard LoRA.
/// The key insight is that separating magnitude and direction allows more stable gradient flow
/// and better control over the adaptation process.
/// </para>
/// <para>
/// <b>For Beginners:</b> DoRA is an improved version of LoRA that works better in practice.
///
/// Think of neural network weights as arrows:
/// - Each arrow has a length (magnitude) and a direction
/// - Standard LoRA adjusts both length and direction at the same time
/// - DoRA separates them: it keeps the length fixed and only adjusts the direction
/// - This makes training more stable and gives better results
///
/// Why this matters:
/// - More stable training (fewer divergences and NaN errors)
/// - Better final performance (+3.7% on LLaMA-7B)
/// - Same parameter efficiency as standard LoRA
/// - Slightly more computation (due to normalization), but worth it for the stability
///
/// When to use DoRA over standard LoRA:
/// - When training stability is important (large models, complex tasks)
/// - When you want the best possible fine-tuning results
/// - When you have the computational budget for normalization overhead
/// - When adapting very large pre-trained models (LLMs, large vision models)
/// </para>
/// <para>
/// <b>Reference:</b>
/// "DoRA: Weight-Decomposed Low-Rank Adaptation"
/// ICML 2024 Oral
/// https://arxiv.org/abs/2402.09353
/// </para>
/// </remarks>
[AutoParameters]
public partial class DoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Magnitude component of the decomposed weights (scalar per output neuron).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The magnitude vector stores the L2 norm of each weight vector (one per output neuron).
    /// During forward pass, this magnitude is applied after normalizing the direction vectors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This stores the "strength" of each output neuron.
    /// When we decompose weights into magnitude and direction, this is the magnitude part.
    /// Each output neuron gets one magnitude value.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters,
        Shape = "GetOutputLayerShape()[0]")]
    private Tensor<T> _magnitude;

    /// <summary>
    /// Gradients for the magnitude component, computed during backpropagation.
    /// </summary>
    [Scratch]
    private Vector<T>? _magnitudeGradient;

    /// <summary>
    /// Initializes a new DoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with DoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes the DoRA adapter by:
    /// 1. Setting up the standard LoRA components (via base constructor)
    /// 2. Decomposing the base layer's initial weights into magnitude and direction
    /// 3. Initializing magnitude gradients
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a DoRA adapter around your existing layer.
    ///
    /// What happens during initialization:
    /// - The base class sets up standard LoRA (matrices A and B)
    /// - We then decompose the layer's weights into magnitude and direction
    /// - The magnitude starts as the actual magnitudes from the original weights
    /// - During training, both the LoRA matrices and the magnitudes will be updated
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to fine-tune efficiently
    /// - rank: How much compression for LoRA (lower = fewer parameters)
    /// - alpha: Scaling factor for LoRA contribution
    /// - freezeBaseLayer: Usually true - we only train LoRA + magnitude, not base weights
    /// </para>
    /// </remarks>
    public DoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Initialize magnitude from base layer weights
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        _magnitude = new Tensor<T>([outputSize]);

        // Decompose initial weights to get magnitude
        DecomposeWeights();

        // Parameter surfaces are generated from the magnitude declaration and the registered
        // base/LoRA children. No shadow flat vector is maintained here.
    }

    /// <summary>
    /// Decomposes the base layer's weights into magnitude and direction components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each output neuron, this method:
    /// 1. Extracts the weight vector (all connections to that neuron)
    /// 2. Computes the L2 norm (magnitude)
    /// 3. Stores the magnitude
    ///
    /// The direction is implicitly W/||W|| and doesn't need to be stored separately.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This splits weights into magnitude (length) and direction.
    ///
    /// Imagine each weight vector as an arrow:
    /// - Magnitude = how long the arrow is
    /// - Direction = which way the arrow points
    ///
    /// We store the magnitude separately so we can apply LoRA only to the direction.
    /// This is the key innovation of DoRA over standard LoRA.
    /// </para>
    /// </remarks>
    private void DecomposeWeights()
    {
        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int weightCount = inputSize * outputSize;

        // For each output neuron, the magnitude of its weight vector is the L2 norm of its row,
        // never zero (for numerical stability).
        Tensor<T> weights = Tensor<T>.FromMatrix(ExtractOutputMajorWeights(baseParams, outputSize, inputSize, weightCount));
        Tensor<T> rowNorms = Engine.Reshape(SafeRowNorms(weights), [outputSize]);
        rowNorms.AsSpan().CopyTo(_magnitude.AsWritableSpan());
    }

    /// <summary>
    /// Reads the leading weight block of a flat parameter vector as an [outputSize, inputSize]
    /// row-major matrix; positions past the end of the parameters are zero.
    /// </summary>
    private Matrix<T> ExtractOutputMajorWeights(Vector<T> baseParams, int outputSize, int inputSize, int weightCount)
    {
        int available = Math.Min(weightCount, baseParams.Length);
        Vector<T> flat = available == weightCount
            ? baseParams.GetSubVector(0, weightCount)
            : Vector<T>.Concatenate(baseParams.GetSubVector(0, available), new Vector<T>(weightCount - available));
        bool inputMajor = _baseLayer is DenseLayer<T>;
        var weights = Engine.Reshape(Tensor<T>.FromVector(flat), inputMajor ? [inputSize, outputSize] : [outputSize, inputSize]).ToMatrix();
        return inputMajor ? Engine.MatrixTranspose(weights) : weights;
    }

    /// <summary>
    /// Row L2 norms as an [rows, 1] tensor, with exact zeros replaced by 1e-8.
    /// </summary>
    private Tensor<T> SafeRowNorms(Tensor<T> matrix)
    {
        Tensor<T> norms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(matrix, matrix), [1], true));
        return Engine.TensorWhere(
            Engine.TensorEquals(norms, NumOps.Zero),
            Tensor<T>.CreateDefault(norms.Shape.ToArray(), NumOps.FromDouble(1e-8)),
            norms);
    }

    /// <summary>
    /// Recomposes weights from magnitude and direction components.
    /// </summary>
    /// <param name="direction">The normalized direction matrix.</param>
    /// <returns>The full weight matrix (magnitude * direction).</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs the full weight matrix by scaling each direction vector
    /// by its corresponding magnitude value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This puts magnitude and direction back together.
    ///
    /// After we've adjusted the direction with LoRA and have the magnitude stored separately,
    /// this combines them back into normal weights. Think of it as:
    /// - Take each direction vector (unit vector)
    /// - Scale it by its magnitude (scalar)
    /// - Result: the full weight vector
    ///
    /// This is used during forward pass to get the effective weights.
    /// </para>
    /// </remarks>
    private Matrix<T> RecomposeWeights(Matrix<T> direction)
    {
        int outputSize = direction.Rows;
        int inputSize = direction.Columns;

        // W[i, j] = m[i] * direction[i, j]
        Matrix<T> weights = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(direction), Engine.Reshape(_magnitude, [outputSize, 1])).ToMatrix();


        return weights;
    }

    /// <summary>
    /// The DoRA weights in [outputSize, inputSize] form: W' = m * (W0 + ΔW) / ||W0 + ΔW||, with the
    /// norm taken per output neuron (over the input axis) and ΔW = (alpha / rank) * (A * B)^T.
    /// </summary>
    /// <remarks>
    /// This is equation (5) of Liu et al. (2024) and the PEFT implementation: the LoRA update is added
    /// to the pre-trained weights W0 themselves, not to the unit direction W0 / ||W0||, and the sum is
    /// then renormalized and rescaled by the learned magnitude.
    /// </remarks>
    private Matrix<T> ComputeAdaptedWeights(Vector<T> baseParams, int outputSize, int inputSize, int weightCount)
    {
        Matrix<T> baseWeights = ExtractOutputMajorWeights(baseParams, outputSize, inputSize, weightCount);

        // MergeWeights() is [inputSize, outputSize]; its transpose lines up with the output-major W0.
        Matrix<T> adaptedWeights = Engine.MatrixAdd(baseWeights, Engine.MatrixTranspose(_loraLayer.MergeWeights()));
        return RecomposeWeights(NormalizeRows(adaptedWeights));
    }

    /// <summary>
    /// Normalizes a matrix row-wise (each row becomes a unit vector).
    /// </summary>
    /// <param name="matrix">The matrix to normalize.</param>
    /// <returns>Row-normalized matrix where each row has unit L2 norm.</returns>
    /// <remarks>
    /// <para>
    /// For each row (weight vector), this computes the L2 norm and divides all elements by it.
    /// This ensures each direction vector has unit length.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This makes each weight vector have length 1.
    ///
    /// When we separate magnitude and direction, the direction must be a unit vector
    /// (length = 1). This method ensures that by dividing each weight vector by its length.
    ///
    /// Example: vector [3, 4] has length 5, so normalized it becomes [0.6, 0.8]
    /// </para>
    /// </remarks>
    private Matrix<T> NormalizeRows(Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        // Divide each row by its L2 norm (exact zero norms become 1e-8 to avoid division by zero)
        Tensor<T> values = Tensor<T>.FromMatrix(matrix);
        Matrix<T> normalized = Engine.TensorDivide(
            values, Engine.TensorBroadcastTo(SafeRowNorms(values), [rows, cols])).ToMatrix();

        return normalized;
    }

    /// <summary>
    /// Performs the forward pass through DoRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output combining base layer with DoRA-adapted weights.</returns>
    /// <remarks>
    /// <para>
    /// The DoRA forward pass:
    /// 1. Gets base layer weights W
    /// 2. Adds the LoRA weight delta: V = W + ΔW
    /// 3. Takes the direction of the sum: d = V / ||V|| (per output neuron)
    /// 4. Recomposes weights: W' = m * d
    /// 5. Computes output: y = input @ W'^T
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where DoRA's magic happens during prediction.
    ///
    /// Step by step:
    /// 1. Get the original weights from the base layer
    /// 2. Split into magnitude (stored) and direction (computed)
    /// 3. Apply LoRA's correction to the direction (not the magnitude!)
    /// 4. Normalize the new direction to keep it as a unit vector
    /// 5. Multiply magnitude back in to get final weights
    /// 6. Use these adjusted weights to compute the output
    ///
    /// The key difference from standard LoRA:
    /// - Standard LoRA: output = base_output + lora_output
    /// - DoRA: output = input @ (m * normalize(W + lora_delta))^T
    ///
    /// DoRA's approach gives more stable training because we control magnitude separately.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Get base layer parameters and extract weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int weightCount = inputSize * outputSize;

        int batchSize = input.Shape[0];

        // W' = m * (W0 + ΔW) / ||W0 + ΔW|| (row-wise norm over the input axis)
        Matrix<T> finalWeights = ComputeAdaptedWeights(baseParams, outputSize, inputSize, weightCount);

        // Compute output: y = input @ W'^T
        // Matrix multiply: [batchSize, inputSize] @ [inputSize, outputSize]
        var output = Engine.TensorMatMul(
            Engine.Reshape(input, [batchSize, inputSize]),
            Tensor<T>.FromMatrix(Engine.MatrixTranspose(finalWeights)));
        int biasCount = Math.Max(0, Math.Min(outputSize, baseParams.Length - weightCount));
        if (biasCount == 0) return output;
        var bias = baseParams.GetSubVector(weightCount, biasCount);
        if (biasCount < outputSize) bias = Vector<T>.Concatenate(bias, new Vector<T>(outputSize - biasCount));
        return Engine.TensorAdd(output, Engine.TensorBroadcastTo(
            Engine.Reshape(Tensor<T>.FromVector(bias), [1, outputSize]), [batchSize, outputSize]));
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Update LoRA layer (always)
        _loraLayer.UpdateParameters(learningRate);

        // Update magnitude parameters
        if (_magnitudeGradient != null)
        {
            // m -= learningRate * dm, then keep the magnitude positive (floor at 1e-8)
            T floor = NumOps.FromDouble(1e-8);
            Tensor<T> updated = Engine.TensorSubtract(
                _magnitude, Tensor<T>.FromVector(Engine.Multiply(_magnitudeGradient, learningRate)));
            updated = Engine.TensorWhere(
                Engine.TensorLessThan(updated, floor), Tensor<T>.CreateDefault(updated.Shape.ToArray(), floor), updated);
            updated.AsSpan().CopyTo(_magnitude.AsWritableSpan());
        }

        // Update base layer (only if not frozen)
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

    }

    /// <summary>
    /// Merges the DoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with DoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported for merging.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a final layer with the DoRA adaptations baked in.
    /// The merged weights are: W' = m * normalize(W + LoRA_delta)
    /// where m is magnitude, W is the base weight matrix, and LoRA_delta is the LoRA contribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This "bakes in" your DoRA adaptation for deployment.
    ///
    /// After training with DoRA, you probably want to deploy a simpler model without
    /// all the DoRA machinery. This method creates that simpler model by:
    /// 1. Computing the final adapted direction (base + LoRA)
    /// 2. Normalizing the direction
    /// 3. Multiplying by magnitude to get final weights
    /// 4. Creating a new layer with these merged weights
    ///
    /// The result is a standard layer that behaves like your DoRA-adapted model
    /// but is faster to run because it doesn't need to do the decomposition at runtime.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // This is a simplified implementation that works with DenseLayer base
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers for merging");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Get base layer weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // W' = m * (W0 + ΔW) / ||W0 + ΔW||, the same weights the forward pass applies
        Matrix<T> finalWeights = ComputeAdaptedWeights(baseParams, outputSize, inputSize, weightCount);

        // Create merged parameters: merged weights (row-major) then biases unchanged
        var flatWeights = (_baseLayer is DenseLayer<T> ? Engine.MatrixTranspose(finalWeights) : finalWeights).ToRowVector();
        int biasCount = baseParams.Length - weightCount;
        Vector<T> mergedParams = biasCount > 0
            ? Vector<T>.Concatenate(flatWeights, baseParams.GetSubVector(weightCount, biasCount))
            : flatWeights;

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();
        _magnitudeGradient = null;
    }
}
