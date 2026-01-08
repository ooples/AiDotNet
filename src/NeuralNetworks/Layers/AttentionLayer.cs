using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an Attention Layer for focusing on relevant parts of input sequences.
/// </summary>
/// <remarks>
/// <para>
/// The Attention Layer is a mechanism that allows a neural network to focus on different parts of the input
/// sequence when producing each element of the output sequence. It computes a weighted sum of the input sequence,
/// where the weights (attention weights) are determined based on the relevance of each input element to the current output.
/// </para>
/// <para><b>For Beginners:</b> An Attention Layer helps the network focus on important parts of the input.
/// 
/// Think of it like reading a long document to answer a question:
/// - Instead of remembering every word, you focus on key sentences or phrases
/// - The attention mechanism does something similar for the neural network
/// - It helps the network decide which parts of the input are most relevant for the current task
/// 
/// Common applications include:
/// - Machine translation (focusing on relevant words when translating)
/// - Image captioning (focusing on relevant parts of an image when describing it)
/// - Speech recognition (focusing on important audio segments)
/// 
/// The key advantage is that it allows the network to handle long sequences more effectively
/// by focusing on the most relevant parts rather than trying to remember everything.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class AttentionLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// The weight tensor for the query transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a query representation.
    /// </para>
    /// <para><b>For Beginners:</b> This helps create a "question" from the current state.
    /// 
    /// Think of it as formulating what information we're looking for in the input sequence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wq;

    /// <summary>
    /// The weight tensor for the key transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a key representation.
    /// </para>
    /// <para><b>For Beginners:</b> This helps create "labels" for each part of the input sequence.
    /// 
    /// Think of it as creating a way to identify or index different parts of the input.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wk;

    /// <summary>
    /// The weight tensor for the value transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a value representation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual content we'll use from each part of the input.
    /// 
    /// Think of it as extracting the useful information from each part of the input sequence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wv;

    /// <summary>
    /// The weight tensor for the output projection (Wo).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor projects the attention output back to the input size, following the
    /// industry-standard transformer architecture from "Attention is All You Need".
    /// </para>
    /// <para><b>For Beginners:</b> This is the final projection that transforms the attention
    /// result back to the original embedding dimension, enabling residual connections.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wo;

    /// <summary>
    /// The size of the input features.
    /// </summary>
    private readonly int _inputSize;

    /// <summary>
    /// The size of the attention mechanism (typically smaller than the input size).
    /// </summary>
    private readonly int _attentionSize;

    /// <summary>
    /// The last input processed by the layer.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The cached query input from the last forward pass (for cross-attention backward).
    /// </summary>
    private Tensor<T>? _lastQueryInput;

    /// <summary>
    /// The cached key input from the last forward pass (for cross-attention backward).
    /// </summary>
    private Tensor<T>? _lastKeyInput;

    /// <summary>
    /// The cached value input from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastValueInput;

    /// <summary>
    /// Tracks whether the last input was originally 2D (and thus reshaped to 3D).
    /// </summary>
    private bool _inputWas2D = false;

    /// <summary>
    /// Stores the original input shape for restoring higher-rank tensor output.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The cached attention mask from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastMask;

    /// <summary>
    /// The last attention weights computed by the layer.
    /// </summary>
    private Tensor<T>? _lastAttentionWeights;

    /// <summary>
    /// Stores the last computed attention entropy for diagnostics.
    /// </summary>
    private T _lastAttentionEntropy;

    /// <summary>
    /// Tracks whether the last forward pass used cross-attention (separate Q and K/V sources).
    /// </summary>
    private bool _lastWasCrossAttention;

    /// <summary>
    /// Tracks whether the last forward pass used an attention mask.
    /// </summary>
    private bool _lastUsedMask;

    /// <summary>
    /// Cached attention output before output projection (Wo), used for backward pass.
    /// </summary>
    private Tensor<T>? _lastAttentionOutput;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (attention entropy regularization) during training.
    /// Default is false. Enable to prevent attention collapse.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for attention entropy regularization.
    /// Default is 0.01. Higher values encourage more uniform attention distributions.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property calculates the total number of trainable parameters in the Attention Layer,
    /// which includes all the weights for query, key, and value transformations.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer needs to learn.
    ///
    /// It counts all the weights in the four transformation matrices (Wq, Wk, Wv, Wo).
    /// A higher number means the layer can potentially learn more complex patterns,
    /// but also requires more data and time to train effectively.
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        (_attentionSize * _inputSize * 3) + (_inputSize * _attentionSize); // Wq, Wk, Wv + Wo

    /// <summary>
    /// Gradient of the weight tensor for the value transformation.
    /// </summary>
    private Tensor<T>? _dWv;

    /// <summary>
    /// Gradient of the weight tensor for the key transformation.
    /// </summary>
    private Tensor<T>? _dWk;

    /// <summary>
    /// Gradient of the weight tensor for the query transformation.
    /// </summary>
    private Tensor<T>? _dWq;

    /// <summary>
    /// Gradient of the weight tensor for the output projection.
    /// </summary>
    private Tensor<T>? _dWo;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates that the Attention Layer can be trained using backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you that the layer can learn and improve its performance over time.
    ///
    /// When this is true, it means the layer can adjust its internal weights based on the errors it makes,
    /// allowing it to get better at its task as it sees more data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the AttentionLayer class with scalar activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="activation">The activation function to use. If null, SoftmaxActivation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Attention Layer with scalar activation, allowing for element-wise application of the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Attention Layer with its initial values, using a scalar activation function.
    /// 
    /// The scalar activation means the same function is applied to each element independently.
    /// This is useful when you want to treat each attention score separately.
    /// </para>
    /// </remarks>
    public AttentionLayer(int inputSize, int attentionSize, IActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new SoftmaxActivation<T>()) // Output size = input size (with Wo projection)
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastAttentionEntropy = NumOps.Zero;

        _inputSize = inputSize;
        _attentionSize = attentionSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _attentionSize)));
        _Wq = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wk = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wv = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        // Output projection Wo: [inputSize, attentionSize] to project attention output back to input dimension
        _Wo = InitializeTensor(new[] { _inputSize, _attentionSize }, scale);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_Wq, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wv, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wo, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Initializes a new instance of the AttentionLayer class with vector activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="activation">The vector activation function to use. If null, SoftmaxActivation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Attention Layer with vector activation, allowing for operations on entire vectors or tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Attention Layer with its initial values, using a vector activation function.
    /// 
    /// The vector activation means the function is applied to the entire set of attention scores at once.
    /// This can be more efficient and allows for more complex interactions between attention scores.
    /// </para>
    /// </remarks>
    public AttentionLayer(int inputSize, int attentionSize, IVectorActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new SoftmaxActivation<T>()) // Output size = input size (with Wo projection)
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastAttentionEntropy = NumOps.Zero;

        _inputSize = inputSize;
        _attentionSize = attentionSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _attentionSize)));
        _Wq = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wk = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wv = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        // Output projection Wo: [inputSize, attentionSize] to project attention output back to input dimension
        _Wo = InitializeTensor(new[] { _inputSize, _attentionSize }, scale);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_Wq, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wv, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wo, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Initializes a tensor with random values scaled by a given factor.
    /// </summary>
    /// <param name="shape">The shape of the tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <returns>A new tensor with randomly initialized values.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor and fills it with random values from a uniform distribution,
    /// scaled by the provided factor. This helps in initializing the weights of the attention mechanism.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the starting values for the layer's internal weights.
    /// 
    /// The random initialization is important because it gives the network a starting point from which to learn.
    /// The scaling helps to ensure that these initial values are neither too large nor too small,
    /// which can affect how well the network learns.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeTensor(int[] shape, T scale)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }

        return tensor;
    }

    /// <summary>
    /// Performs the forward pass of the attention mechanism.
    /// </summary>
    /// <param name="input">The input tensor to the layer.</param>
    /// <returns>The output tensor after applying the attention mechanism.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core functionality of the attention mechanism. It transforms the input
    /// into query, key, and value representations, computes attention scores, applies scaling and activation,
    /// and produces the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the attention magic happens!
    /// 
    /// 1. The input is transformed into three different representations: Query (Q), Key (K), and Value (V).
    /// 2. Attention scores are computed by comparing Q and K.
    /// 3. These scores are scaled and activated (usually with softmax) to get attention weights.
    /// 4. The final output is produced by applying these weights to V.
    /// 
    /// This process allows the layer to focus on different parts of the input as needed.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Validate input tensor
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        // Handle any rank >= 2: last 2 dims are [Seq, InputSize], earlier dims are batch-like
        int rank = input.Shape.Length;
        _inputWas2D = rank == 2;
        Tensor<T> input3D;
        _originalInputShape = input.Shape;

        if (_inputWas2D)
        {
            // 2D input: [Batch, InputSize] -> [Batch, 1, InputSize]
            if (input.Shape[1] != _inputSize)
            {
                throw new ArgumentException(
                    $"AttentionLayer input size mismatch. Expected InputSize={_inputSize}, " +
                    $"but got {input.Shape[1]} in 2D shape [{input.Shape[0]}, {input.Shape[1]}].",
                    nameof(input));
            }
            int batchSize2D = input.Shape[0];
            input3D = input.Reshape(batchSize2D, 1, _inputSize);
        }
        else if (rank == 3)
        {
            // 3D input: standard attention format [Batch, Seq, InputSize]
            if (input.Shape[2] != _inputSize)
            {
                throw new ArgumentException(
                    $"AttentionLayer input size mismatch. Expected InputSize={_inputSize}, " +
                    $"but got {input.Shape[2]} in shape [{string.Join(", ", input.Shape)}].",
                    nameof(input));
            }
            input3D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            if (input.Shape[rank - 1] != _inputSize)
            {
                throw new ArgumentException(
                    $"AttentionLayer input size mismatch. Expected InputSize={_inputSize}, " +
                    $"but got {input.Shape[rank - 1]} in shape [{string.Join(", ", input.Shape)}].",
                    nameof(input));
            }
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            input3D = input.Reshape(flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]);
        }

        _lastInput = input;
        _lastQueryInput = input3D;
        _lastKeyInput = input3D;
        _lastWasCrossAttention = false;
        _lastUsedMask = false;
        _lastMask = null;

        int batchSize = input3D.Shape[0];
        int seqLen = input3D.Shape[1];

        // 1. Project Input to Q, K, V
        // Reshape input to 2D [Batch*Seq, InputSize] for efficient MatrixMultiply
        var inputFlat = input3D.Reshape(batchSize * seqLen, _inputSize);

        // Transpose weights to [InputSize, AttSize] using Engine 2D transpose
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);

        // Compute Projections: [B*S, In] @ [In, Att] -> [B*S, Att]
        var qProjected = Engine.TensorMatMul(inputFlat, wqTransposed);
        var kProjected = Engine.TensorMatMul(inputFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(inputFlat, wvTransposed);

        // Reshape back to [Batch, Seq, AttSize]
        var Q = qProjected.Reshape(batchSize, seqLen, _attentionSize);
        var K = kProjected.Reshape(batchSize, seqLen, _attentionSize);
        var V = vProjected.Reshape(batchSize, seqLen, _attentionSize);

        Tensor<T> attentionOutput;

        // Check if we can use the optimized Engine.ScaledDotProductAttention
        // (only when using default Softmax activation)
        if (VectorActivation is SoftmaxActivation<T>)
        {
            // 2-5. Use Engine.ScaledDotProductAttention for optimized computation
            // Reshape to 4D by adding head dimension: [B, S, A] -> [B, 1, S, A]
            var Q4D = Q.Reshape(batchSize, 1, seqLen, _attentionSize);
            var K4D = K.Reshape(batchSize, 1, seqLen, _attentionSize);
            var V4D = V.Reshape(batchSize, 1, seqLen, _attentionSize);

            var output4D = Engine.ScaledDotProductAttention(
                Q4D, K4D, V4D,
                mask: null,
                scale: 1.0 / Math.Sqrt(_attentionSize),
                out var attentionWeights4D);

            // Reshape attention weights back to 3D for caching: [B, 1, S, S] -> [B, S, S]
            _lastAttentionWeights = attentionWeights4D.Reshape(batchSize, seqLen, seqLen);

            // Reshape output back to 3D: [B, 1, S, A] -> [B, S, A]
            attentionOutput = output4D.Reshape(batchSize, seqLen, _attentionSize);
        }
        else
        {
            // Fallback to manual computation for custom activations
            // 2. Compute Attention Scores: Q @ K.T
            var KT = K.Transpose(new[] { 0, 2, 1 });
            var attentionScores = Engine.BatchMatMul(Q, KT);

            // 3. Scale
            T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(_attentionSize)));
            attentionScores = Engine.TensorMultiplyScalar(attentionScores, scaleValue);

            // 4. Apply activation (custom, not softmax)
            _lastAttentionWeights = ApplyActivation(attentionScores);

            // 5. Output: Weights @ V
            attentionOutput = Engine.BatchMatMul(_lastAttentionWeights, V);
        }
        _lastAttentionOutput = attentionOutput; // Cache for backward pass

        // 6. Output Projection: Apply Wo to project from attentionSize back to inputSize
        // Flatten for matmul: [B*S, A] @ [A, inputSize] -> [B*S, inputSize]
        var attnFlat = attentionOutput.Reshape(batchSize * seqLen, _attentionSize);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = projectedFlat.Reshape(batchSize, seqLen, _inputSize);

        // Restore original tensor shape
        if (_inputWas2D)
        {
            output = output.Reshape(batchSize, _inputSize);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 2] = seqLen;
            outputShape[_originalInputShape.Length - 1] = _inputSize;
            output = output.Reshape(outputShape);
        }

        return output;
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass for the attention mechanism.
    /// All computations stay on GPU - no CPU roundtrips.
    /// </summary>
    /// <param name="inputs">The input GPU tensors. Expects one tensor with shape [batch, seqLen, inputSize].</param>
    /// <returns>The output GPU tensor after applying the attention mechanism.</returns>
    /// <exception cref="ArgumentException">Thrown when no inputs provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when engine is not a DirectGpuTensorEngine.</exception>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape;

        // Handle 2D [Batch, InputSize] or 3D [Batch, Seq, InputSize] input
        int batchSize;
        int seqLen;
        IGpuTensor<T> input3D;

        if (shape.Length == 2)
        {
            _inputWas2D = true;
            batchSize = shape[0];
            seqLen = 1;
            input3D = gpuEngine.ReshapeGpu(input, [batchSize, 1, _inputSize]);
        }
        else if (shape.Length == 3)
        {
            _inputWas2D = false;
            batchSize = shape[0];
            seqLen = shape[1];
            input3D = input;
        }
        else
        {
            throw new ArgumentException(
                $"AttentionLayer expects 2D [B,I] or 3D [B,S,I] input, got {shape.Length}D.", nameof(inputs));
        }

        // Store for backward pass (download only if needed)
        _lastWasCrossAttention = false;
        _lastUsedMask = false;
        _lastMask = null;

        int flatBatchSeq = batchSize * seqLen;

        // Compute Q, K, V projections entirely on GPU using FusedLinearGpu
        // FusedLinearGpu: input @ weights.T + bias with optional activation
        // Reshape to 2D for matmul: [B*S, InputSize]
        var inputFlat = gpuEngine.ReshapeGpu(input3D, [flatBatchSeq, _inputSize]);

        // Q projection: [B*S, InputSize] @ [InputSize, AttentionSize] -> [B*S, AttentionSize]
        var qFlat = gpuEngine.FusedLinearGpu(inputFlat, Engine.TensorTranspose(_Wq), null, FusedActivationType.None);

        // K projection
        var kFlat = gpuEngine.FusedLinearGpu(inputFlat, Engine.TensorTranspose(_Wk), null, FusedActivationType.None);

        // V projection
        var vFlat = gpuEngine.FusedLinearGpu(inputFlat, Engine.TensorTranspose(_Wv), null, FusedActivationType.None);

        // Reshape to 4D for attention: [B, 1, S, AttentionSize] (single head)
        var Q4D = gpuEngine.ReshapeGpu(qFlat, [batchSize, 1, seqLen, _attentionSize]);
        var K4D = gpuEngine.ReshapeGpu(kFlat, [batchSize, 1, seqLen, _attentionSize]);
        var V4D = gpuEngine.ReshapeGpu(vFlat, [batchSize, 1, seqLen, _attentionSize]);

        // Compute scaled dot-product attention on GPU
        double scale = 1.0 / Math.Sqrt(_attentionSize);
        var attnOutput4D = gpuEngine.ScaledDotProductAttentionGpu(Q4D, K4D, V4D, scale, out var attnWeights4D);

        // Store attention weights for backward pass (lazy download)
        // We'll download only when needed in Backward()
        _lastAttentionWeights = null; // Will be computed from attnWeights4D if needed

        // Reshape attention output: [B, 1, S, AttentionSize] -> [B*S, AttentionSize]
        var attnFlat = gpuEngine.ReshapeGpu(attnOutput4D, [flatBatchSeq, _attentionSize]);

        // Output projection: [B*S, AttentionSize] @ [AttentionSize, InputSize] -> [B*S, InputSize]
        var outputFlat = gpuEngine.FusedLinearGpu(attnFlat, Engine.TensorTranspose(_Wo), null, FusedActivationType.None);

        // Dispose intermediate tensors to free GPU memory
        if (!ReferenceEquals(input3D, input)) ((IDisposable)input3D).Dispose();
        ((IDisposable)inputFlat).Dispose();
        ((IDisposable)qFlat).Dispose();
        ((IDisposable)kFlat).Dispose();
        ((IDisposable)vFlat).Dispose();
        ((IDisposable)Q4D).Dispose();
        ((IDisposable)K4D).Dispose();
        ((IDisposable)V4D).Dispose();
        ((IDisposable)attnOutput4D).Dispose();
        ((IDisposable)attnWeights4D).Dispose();
        ((IDisposable)attnFlat).Dispose();

        // Reshape to final output shape
        IGpuTensor<T> output;
        if (_inputWas2D)
        {
            output = gpuEngine.ReshapeGpu(outputFlat, [batchSize, _inputSize]);
            ((IDisposable)outputFlat).Dispose();
        }
        else
        {
            output = gpuEngine.ReshapeGpu(outputFlat, [batchSize, seqLen, _inputSize]);
            ((IDisposable)outputFlat).Dispose();
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass of the attention mechanism with multiple inputs.
    /// </summary>
    /// <param name="inputs">An array of input tensors. Based on the number of inputs:
    ///   - One input: Standard forward pass with just the input tensor
    ///   - Two inputs: The first tensor is the query input, the second is either the key/value input or an attention mask
    ///   - Three inputs: The first tensor is the query input, the second is the key/value input, and the third is the attention mask
    /// </param>
    /// <returns>The output tensor after applying the attention mechanism.</returns>
    /// <exception cref="ArgumentException">Thrown when the input array is empty.</exception>
    /// <remarks>
    /// <para>
    /// This method extends the attention mechanism to support multiple input tensors, which is useful
    /// for implementing cross-attention (as used in transformer decoder layers) and masked attention.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows the attention layer to handle more complex scenarios:
    /// 
    /// 1. With one input: It works just like the standard attention (self-attention)
    /// 2. With two inputs: It can either:
    ///    - Perform cross-attention (where query comes from one source, and key/value from another)
    ///    - Apply a mask to self-attention to control which parts of the input to focus on
    /// 3. With three inputs: It performs masked cross-attention, which combines both features above
    /// 
    /// These capabilities are essential for transformer architectures, especially decoder layers
    /// that need to attend to both their own outputs and the encoder's outputs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required for attention mechanism.");

        // Validate that no input tensor is null
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i] == null)
                throw new ArgumentNullException($"inputs[{i}]", $"Input tensor at index {i} cannot be null.");
        }

        // Case 1: Standard self-attention with a single input
        if (inputs.Length == 1)
        {
            return Forward(inputs[0]);
        }

        // Case 2: Either cross-attention or masked self-attention
        else if (inputs.Length == 2)
        {
            var primaryInput = inputs[0];
            var secondInput = inputs[1];

            // Detect mask vs cross-attention key/value tensor:
            // - Mask shape: [batch, query_len, key_len] - typically [batch, seqLen, seqLen] for self-attention
            //   Masks must be at least 3D to have (batch, query, key) dimensions
            // - Cross-attention K/V: [batch, seqLen, inputSize] (last dim == _inputSize)
            //   Must match the embedding dimension of this layer

            // A valid attention mask requires at least 3 dimensions
            bool looksLikeMask = secondInput.Rank >= 3 &&
                                 secondInput.Shape[secondInput.Rank - 1] == secondInput.Shape[secondInput.Rank - 2];

            // Cross-attention K/V must have embedding dimension matching this layer's input size
            bool looksLikeCrossAttn = secondInput.Rank >= 2 &&
                                      secondInput.Shape[secondInput.Rank - 1] == _inputSize;

            // Priority: If it looks like a mask (square last two dims, 3D+), treat as mask
            // Otherwise, if it has matching embedding dim, treat as cross-attention
            // Edge case: if seqLen == inputSize, prefer mask interpretation if square
            if (looksLikeMask)
            {
                // This appears to be masked self-attention
                return ForwardMaskedAttention(primaryInput, secondInput);
            }
            else if (looksLikeCrossAttn)
            {
                // This appears to be cross-attention (query from primaryInput, key/value from secondInput)
                return ForwardCrossAttention(primaryInput, secondInput, null);
            }
            else
            {
                throw new ArgumentException(
                    $"Second input tensor has ambiguous shape {string.Join("x", secondInput.Shape)}. " +
                    $"Expected either a mask [batch, queryLen, keyLen] or cross-attention K/V [batch, seqLen, {_inputSize}].");
            }
        }

        // Case 3: Masked cross-attention with three inputs
        else if (inputs.Length == 3)
        {
            var queryInput = inputs[0];
            var keyValueInput = inputs[1];
            var attentionMask = inputs[2];

            return ForwardCrossAttention(queryInput, keyValueInput, attentionMask);
        }

        // Unsupported number of inputs
        else
        {
            throw new ArgumentException($"Unsupported number of inputs ({inputs.Length}) for attention mechanism. Expected 1-3 inputs.");
        }
    }

    /// <summary>
    /// Performs masked self-attention, where query, key, and value all come from the same input,
    /// but with an attention mask applied.
    /// </summary>
    /// <param name="input">The input tensor for query, key, and value.</param>
    /// <param name="mask">The attention mask tensor.</param>
    /// <returns>The output tensor after applying masked self-attention.</returns>
    private Tensor<T> ForwardMaskedAttention(Tensor<T> input, Tensor<T> mask)
    {
        _lastInput = input;
        _lastMask = mask;

        // Handle 2D [Batch, InputSize] or 3D [Batch, Seq, InputSize] input
        _inputWas2D = input.Shape.Length == 2;
        Tensor<T> input3D = _inputWas2D
            ? input.Reshape(input.Shape[0], 1, _inputSize)
            : input;

        _lastQueryInput = input3D;
        _lastKeyInput = input3D;
        _lastValueInput = input3D;

        int batchSize = input3D.Shape[0];
        int seqLen = input3D.Shape[1];

        // 1. Project Input to Q, K, V
        var inputFlat = input3D.Reshape(batchSize * seqLen, _inputSize);
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);

        var qProjected = Engine.TensorMatMul(inputFlat, wqTransposed);
        var kProjected = Engine.TensorMatMul(inputFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(inputFlat, wvTransposed);

        var Q = qProjected.Reshape(batchSize, seqLen, _attentionSize);
        var K = kProjected.Reshape(batchSize, seqLen, _attentionSize);
        var V = vProjected.Reshape(batchSize, seqLen, _attentionSize);

        // 2. Compute Attention Scores: Q @ K.T
        var KT = K.Transpose(new[] { 0, 2, 1 });
        var attentionScores = Engine.BatchMatMul(Q, KT);

        // 3. Scale
        T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(_attentionSize)));
        attentionScores = Engine.TensorMultiplyScalar(attentionScores, scaleValue);

        // 4. Mask
        attentionScores = Engine.TensorAdd(attentionScores, mask);

        // 5. Softmax
        _lastAttentionWeights = ApplyActivation(attentionScores);

        // 6. Output: Weights @ V
        var attentionOutput = Engine.BatchMatMul(_lastAttentionWeights, V);
        _lastAttentionOutput = attentionOutput; // Cache for backward pass

        // 7. Output Projection: Apply Wo to project from attentionSize back to inputSize
        var attnFlat = attentionOutput.Reshape(batchSize * seqLen, _attentionSize);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = projectedFlat.Reshape(batchSize, seqLen, _inputSize);

        // If input was 2D, reshape output back to 2D
        if (_inputWas2D)
        {
            output = output.Reshape(batchSize, _inputSize);
        }

        return output;
    }

    /// <summary>
    /// Performs cross-attention, where query comes from one input and key/value come from another,
    /// optionally with an attention mask applied.
    /// </summary>
    /// <param name="queryInput">The input tensor for query.</param>
    /// <param name="keyValueInput">The input tensor for key and value.</param>
    /// <param name="mask">Optional attention mask tensor.</param>
    /// <returns>The output tensor after applying cross-attention.</returns>
    private Tensor<T> ForwardCrossAttention(Tensor<T> queryInput, Tensor<T> keyValueInput, Tensor<T>? mask)
    {
        _lastWasCrossAttention = true;
        _lastUsedMask = mask != null;
        _lastMask = mask;
        _lastInput = queryInput;

        // Handle 2D [Batch, InputSize] or 3D [Batch, Seq, InputSize] input
        _inputWas2D = queryInput.Shape.Length == 2;
        Tensor<T> query3D, keyValue3D;

        if (_inputWas2D)
        {
            int batchSize2D = queryInput.Shape[0];
            query3D = queryInput.Reshape(batchSize2D, 1, _inputSize);
            // KeyValue input should match query dimensionality
            if (keyValueInput.Shape.Length == 2)
            {
                keyValue3D = keyValueInput.Reshape(keyValueInput.Shape[0], 1, _inputSize);
            }
            else
            {
                keyValue3D = keyValueInput;
            }
        }
        else
        {
            query3D = queryInput;
            keyValue3D = keyValueInput;
        }

        _lastQueryInput = query3D;
        _lastKeyInput = keyValue3D;
        _lastValueInput = keyValue3D;

        int batchSize = query3D.Shape[0];
        int seqLenQ = query3D.Shape[1];
        int seqLenKV = keyValue3D.Shape[1];

        // Project Q from query input
        var queryFlat = query3D.Reshape(batchSize * seqLenQ, _inputSize);
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var qProjected = Engine.TensorMatMul(queryFlat, wqTransposed);
        var Q = qProjected.Reshape(batchSize, seqLenQ, _attentionSize);

        // Project K, V from key/value input
        var kvFlat = keyValue3D.Reshape(batchSize * seqLenKV, _inputSize);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);
        var kProjected = Engine.TensorMatMul(kvFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(kvFlat, wvTransposed);
        var K = kProjected.Reshape(batchSize, seqLenKV, _attentionSize);
        var V = vProjected.Reshape(batchSize, seqLenKV, _attentionSize);

        // Compute Scores: Q @ K.T
        var KT = K.Transpose(new[] { 0, 2, 1 });
        var attentionScores = Engine.BatchMatMul(Q, KT);

        // Scale
        T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(_attentionSize)));
        attentionScores = Engine.TensorMultiplyScalar(attentionScores, scaleValue);

        // Mask
        if (mask != null)
        {
            attentionScores = Engine.TensorAdd(attentionScores, mask);
        }

        _lastAttentionWeights = ApplyActivation(attentionScores);

        // Output: Weights @ V -> [B, seqLenQ, attentionSize]
        var attentionOutput = Engine.BatchMatMul(_lastAttentionWeights, V);
        _lastAttentionOutput = attentionOutput; // Cache for backward pass

        // Output Projection: Apply Wo to project from attentionSize back to inputSize
        var attnFlat = attentionOutput.Reshape(batchSize * seqLenQ, _attentionSize);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = projectedFlat.Reshape(batchSize, seqLenQ, _inputSize);

        // If input was 2D, reshape output back to 2D
        if (_inputWas2D)
        {
            output = output.Reshape(batchSize, _inputSize);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the attention mechanism.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backpropagation algorithm for the attention mechanism. It computes
    /// the gradients of the loss with respect to the layer's parameters and input.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the layer learns from its mistakes.
    ///
    /// The method takes the gradient of the error with respect to the layer's output and works backwards to figure out:
    /// 1. How much each weight contributed to the error (stored in _dWq, _dWk, _dWv)
    /// 2. How the input itself contributed to the error (the returned value)
    ///
    /// This information is then used to update the weights and improve the layer's performance.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Autodiff doesn't support cross-attention or masked attention yet
        if (UseAutodiff && !_lastWasCrossAttention && !_lastUsedMask)
            return BackwardViaAutodiff(outputGradient);
        else
            return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastAttentionWeights == null || _lastAttentionOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // First, backprop through output projection (Wo)
        // Forward: output = attentionOutput @ Wo.T
        // Backward: dWo = dOutput.T @ attentionOutput, dAttentionOutput = dOutput @ Wo

        // Flatten gradients and cached output for matmul
        int totalElements = 1;
        for (int i = 0; i < outputGradient.Shape.Length - 1; i++)
            totalElements *= outputGradient.Shape[i];
        var dOutputFlat = outputGradient.Reshape(totalElements, _inputSize);
        var attnOutputFlat = _lastAttentionOutput.Reshape(totalElements, _attentionSize);

        // Compute dWo: [inputSize, N] @ [N, attentionSize] = [inputSize, attentionSize]
        _dWo = Engine.TensorMatMul(Engine.TensorTranspose(dOutputFlat), attnOutputFlat);

        // Compute dAttentionOutput: [N, inputSize] @ [inputSize, attentionSize] = [N, attentionSize]
        var dAttnOutputFlat = Engine.TensorMatMul(dOutputFlat, _Wo);
        var dAttnOutput = dAttnOutputFlat.Reshape(_lastAttentionOutput.Shape);

        // Now backprop through the attention computation using dAttnOutput
        // Build permutation to transpose last two dimensions for any-rank tensors
        int attnRank = _lastAttentionWeights.Rank;
        int[] attnPerm = Enumerable.Range(0, attnRank).ToArray();
        if (attnRank >= 2)
        {
            (attnPerm[attnRank - 2], attnPerm[attnRank - 1]) = (attnPerm[attnRank - 1], attnPerm[attnRank - 2]);
        }
        var dV = _lastAttentionWeights.Transpose(attnPerm).Multiply(dAttnOutput);

        // Compute V with same shape as in forward pass - should be 3D [B, S, A]
        // Forward: inputFlat @ _Wv^T -> reshape to [B, S, A]
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];
        var inputFlat = _lastInput.Reshape([batchSize * seqLen, _inputSize]);
        var wvTransposed = Engine.TensorTranspose(_Wv);
        var vProjected = Engine.TensorMatMul(inputFlat, wvTransposed);
        var V = vProjected.Reshape([batchSize, seqLen, _attentionSize]);

        int vRank = V.Rank;
        int[] vPerm = Enumerable.Range(0, vRank).ToArray();
        if (vRank >= 2)
        {
            (vPerm[vRank - 2], vPerm[vRank - 1]) = (vPerm[vRank - 1], vPerm[vRank - 2]);
        }

        // dAttentionWeights = dAttnOutput @ V^T -> [B, S, S]
        var dAttentionWeights = Engine.BatchMatMul(dAttnOutput, V.Transpose(vPerm));

        // Softmax backward: dL/dz_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
        // where y is softmax output (attention weights), z is pre-softmax scores
        int sumAxis = _lastAttentionWeights.Rank - 1;

        // Compute weighted sum: sum_j(y_j * dL/dy_j)
        var weightedGrad = _lastAttentionWeights.ElementwiseMultiply(dAttentionWeights);
        var weightedSum = weightedGrad.SumOverAxis(sumAxis); // Shape reduces by 1 dimension

        // Broadcast sum back to original shape by expanding the dimension
        var broadcastShape = new int[_lastAttentionWeights.Rank];
        for (int i = 0; i < _lastAttentionWeights.Rank; i++)
        {
            broadcastShape[i] = i == sumAxis ? 1 : _lastAttentionWeights.Shape[i];
        }
        var sumBroadcast = weightedSum.Reshape(broadcastShape);

        // Create result tensor and fill with broadcast values
        var sumExpanded = new Tensor<T>(_lastAttentionWeights.Shape);
        var totalBatches = 1;
        for (int i = 0; i < sumAxis; i++) totalBatches *= _lastAttentionWeights.Shape[i];
        var lastDim = _lastAttentionWeights.Shape[sumAxis];

        for (int batch = 0; batch < totalBatches; batch++)
        {
            T sumVal = sumBroadcast.GetFlat(batch);
            for (int j = 0; j < lastDim; j++)
            {
                sumExpanded.SetFlat(batch * lastDim + j, sumVal);
            }
        }

        var dAttentionScores = _lastAttentionWeights.ElementwiseMultiply(
            dAttentionWeights.Subtract(sumExpanded)
        );

        var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(_Wk.Shape[_Wk.Shape.Length - 1]));
        T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, scaleFactor);
        dAttentionScores = dAttentionScores.Scale(scaleValue);

        // Recompute Q, K from stored input for proper backward
        // Forward: inputFlat @ Wq^T -> reshape to [B, S, A]
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var qProjected = Engine.TensorMatMul(inputFlat, wqTransposed);
        var kProjected = Engine.TensorMatMul(inputFlat, wkTransposed);
        var Q = qProjected.Reshape([batchSize, seqLen, _attentionSize]);
        var K = kProjected.Reshape([batchSize, seqLen, _attentionSize]);

        // Correct attention backward formulas:
        // scores = Q @ K^T, so: dQ = dScores @ K, dK = dScores^T @ Q
        var dQ = Engine.BatchMatMul(dAttentionScores, K);
        var dK = Engine.BatchMatMul(dAttentionScores.Transpose([0, 2, 1]), Q);

        // Flatten for weight gradient computation: [B*S, attentionSize]
        var dQFlat = dQ.Reshape([batchSize * seqLen, _attentionSize]);
        var dKFlat = dK.Reshape([batchSize * seqLen, _attentionSize]);
        var dVFlat = dV.Reshape([batchSize * seqLen, _attentionSize]);

        // Weight gradients: dWq = input^T @ dQ, etc.
        var inputFlatT = Engine.TensorTranspose(inputFlat);
        _dWq = Engine.TensorMatMul(inputFlatT, dQFlat);
        _dWk = Engine.TensorMatMul(inputFlatT, dKFlat);
        _dWv = Engine.TensorMatMul(inputFlatT, dVFlat);

        // Input gradient: dInput = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
        var dinputFromQ = Engine.TensorMatMul(dQFlat, _Wq);
        var dinputFromK = Engine.TensorMatMul(dKFlat, _Wk);
        var dinputFromV = Engine.TensorMatMul(dVFlat, _Wv);
        var dinputFlat = dinputFromQ.Add(dinputFromK).Add(dinputFromV);
        var dinput = dinputFlat.Reshape(_lastInput.Shape);

        return dinput;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with production-grade optimizations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// <b>Production-Ready Features:</b>
    /// <list type="bullet">
    /// <item>Full computation graph construction for Self and Cross Attention</item>
    /// <item>Supports masking via graph operations</item>
    /// <item>Uses Permute/Reshape/MatMul for correct gradient flow</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastQueryInput == null || _lastKeyInput == null || _lastValueInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        var qInput = Autodiff.TensorOperations<T>.Variable(_lastQueryInput, "query", true);
        var kInput = Autodiff.TensorOperations<T>.Variable(_lastKeyInput, "key", true);
        var vInput = Autodiff.TensorOperations<T>.Variable(_lastValueInput, "value", true);

        var wq = Autodiff.TensorOperations<T>.Variable(_Wq, "Wq", true);
        var wk = Autodiff.TensorOperations<T>.Variable(_Wk, "Wk", true);
        var wv = Autodiff.TensorOperations<T>.Variable(_Wv, "Wv", true);

        int batchSize = _lastQueryInput.Shape[0];
        int seqLenQ = _lastQueryInput.Shape[1];
        int seqLenKV = _lastKeyInput.Shape[1];

        // 2. Projections
        // Reshape inputs to 2D [B*S, E]
        var q2D = Autodiff.TensorOperations<T>.Reshape(qInput, batchSize * seqLenQ, _inputSize);
        var k2D = Autodiff.TensorOperations<T>.Reshape(kInput, batchSize * seqLenKV, _inputSize);
        var v2D = Autodiff.TensorOperations<T>.Reshape(vInput, batchSize * seqLenKV, _inputSize);

        // Transpose weights: [In, Att] -> [Att, In]
        var wqT = Autodiff.TensorOperations<T>.Transpose(wq);
        var wkT = Autodiff.TensorOperations<T>.Transpose(wk);
        var wvT = Autodiff.TensorOperations<T>.Transpose(wv);

        var qFlat = Autodiff.TensorOperations<T>.MatrixMultiply(q2D, wqT);
        var kFlat = Autodiff.TensorOperations<T>.MatrixMultiply(k2D, wkT);
        var vFlat = Autodiff.TensorOperations<T>.MatrixMultiply(v2D, wvT);

        // Reshape back: [B, S, Att]
        var Q = Autodiff.TensorOperations<T>.Reshape(qFlat, batchSize, seqLenQ, _attentionSize);
        var K = Autodiff.TensorOperations<T>.Reshape(kFlat, batchSize, seqLenKV, _attentionSize);
        var V = Autodiff.TensorOperations<T>.Reshape(vFlat, batchSize, seqLenKV, _attentionSize);

        // 3. Scores: Q @ K.T
        // Permute K: [B, S, A] -> [B, A, S]
        var KT = Autodiff.TensorOperations<T>.Permute(K, 0, 2, 1);
        // Use BatchMatrixMultiply for 3D tensors [B, S_Q, A] @ [B, A, S_KV] -> [B, S_Q, S_KV]
        var scores = Autodiff.TensorOperations<T>.BatchMatrixMultiply(Q, KT);

        // Scale
        T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(_attentionSize)));
        var scaleTensor = new Tensor<T>(new int[] { 1 });
        scaleTensor[0] = scaleValue;
        var scaleNode = Autodiff.TensorOperations<T>.Constant(scaleTensor, "scale");
        var scaledScores = Autodiff.TensorOperations<T>.ElementwiseMultiply(scores, scaleNode);

        // Mask
        if (_lastMask != null)
        {
            var maskNode = Autodiff.TensorOperations<T>.Constant(_lastMask, "mask");
            scaledScores = Autodiff.TensorOperations<T>.Add(scaledScores, maskNode);
        }

        // Softmax
        var attentionWeights = Autodiff.TensorOperations<T>.Softmax(scaledScores);

        // Output: Weights @ V
        // Use BatchMatrixMultiply for 3D tensors [B, S_Q, S_KV] @ [B, S_KV, A] -> [B, S_Q, A]
        var output = Autodiff.TensorOperations<T>.BatchMatrixMultiply(attentionWeights, V);

        // Gradient
        output.Gradient = outputGradient;
        output.Backward();

        // Store
        _dWq = wq.Gradient;
        _dWk = wk.Gradient;
        _dWv = wv.Gradient;

        return qInput.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }

    /// <summary>
    /// Creates a tensor filled with a scalar value.
    /// </summary>
    private Tensor<T> CreateScalarTensor(T value, int[] shape)
    {
        // === Vectorized tensor fill using IEngine (Phase B: US-GPU-015) ===
        var tensor = new Tensor<T>(shape);
        Engine.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>
    /// Updates the layer's parameters based on the computed gradients and a learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <remarks>
    /// <para>
    /// This method applies the computed gradients to the layer's weights, scaled by the learning rate.
    /// This is typically called after the backward pass to adjust the layer's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the layer actually improves its performance.
    /// 
    /// After figuring out how each weight contributed to the error (in the Backward method),
    /// this method adjusts those weights to reduce the error:
    /// - Weights that contributed to large errors are changed more.
    /// - The learning rate determines how big these changes are.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_dWq == null || _dWk == null || _dWv == null || _dWo == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _Wq = _Wq.Subtract(_dWq.Scale(learningRate));
        _Wk = _Wk.Subtract(_dWk.Scale(learningRate));
        _Wv = _Wv.Subtract(_dWv.Scale(learningRate));
        _Wo = _Wo.Subtract(_dWo.Scale(learningRate));

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_Wq);
        Engine.InvalidatePersistentTensor(_Wk);
        Engine.InvalidatePersistentTensor(_Wv);
        Engine.InvalidatePersistentTensor(_Wo);
    }

    /// <summary>
    /// Updates the layer's parameters with the provided values.
    /// </summary>
    /// <param name="parameters">A vector containing new parameter values.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the current values of the layer's weights with new values provided in the parameters vector.
    /// It's useful for setting the layer's state to a specific configuration, such as when loading a pre-trained model.
    /// </para>
    /// <para><b>For Beginners:</b> This allows you to directly set the layer's internal weights.
    /// 
    /// Instead of the layer learning these weights through training, you're providing them directly.
    /// This is often used when you want to use a pre-trained attention layer or set up the layer with specific initial values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // === Vectorized Parameter Updates (Phase B: US-GPU-015) ===
        int startIndex = 0;

        // Update Wq - slice and copy
        var wqParams = parameters.Slice(startIndex, _Wq.Length);
        _Wq = Tensor<T>.FromVector(wqParams).Reshape(_Wq.Shape);
        startIndex += _Wq.Length;

        // Update Wk - slice and copy
        var wkParams = parameters.Slice(startIndex, _Wk.Length);
        _Wk = Tensor<T>.FromVector(wkParams).Reshape(_Wk.Shape);
        startIndex += _Wk.Length;

        // Update Wv - slice and copy
        var wvParams = parameters.Slice(startIndex, _Wv.Length);
        _Wv = Tensor<T>.FromVector(wvParams).Reshape(_Wv.Shape);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_Wq);
        Engine.InvalidatePersistentTensor(_Wk);
        Engine.InvalidatePersistentTensor(_Wv);
    }

    /// <summary>
    /// Retrieves the current parameters of the layer.
    /// </summary>
    /// <returns>A vector containing all the parameters of the layer.</returns>
    /// <remarks>
    /// <para>
    /// This method collects all the weights of the attention layer (Wq, Wk, Wv) into a single vector.
    /// It's useful for operations that need to work with all the layer's parameters at once,
    /// such as certain optimization algorithms or when saving the model's state.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you all the layer's learned values in one list.
    /// 
    /// It's like taking a snapshot of everything the layer has learned.
    /// This can be useful for saving the layer's current state or for advanced training techniques.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // === Vectorized Parameter Extraction (Phase B: US-GPU-015) ===
        // Flatten each tensor to vector and concatenate
        var wqVec = _Wq.ToVector();
        var wkVec = _Wk.ToVector();
        var wvVec = _Wv.ToVector();

        return Vector<T>.Concatenate(Vector<T>.Concatenate(wqVec, wkVec), wvVec);
    }

    /// <summary>
    /// Computes the auxiliary loss for the AttentionLayer, which is attention entropy regularization.
    /// </summary>
    /// <returns>The attention entropy loss value.</returns>
    /// <remarks>
    /// <para>
    /// Attention entropy regularization prevents attention collapse by encouraging diverse attention patterns.
    /// It computes the entropy of the attention distribution: H = -Σ(p * log(p))
    /// Lower entropy means more focused (peaky) attention, higher entropy means more distributed attention.
    /// We negate the entropy to create a loss that penalizes low entropy (collapsed attention).
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a penalty when attention becomes too focused on just one or two positions.
    ///
    /// Attention entropy regularization:
    /// - Measures how "spread out" the attention weights are
    /// - Penalizes attention that collapses to a single position
    /// - Encourages the model to consider multiple relevant parts of the input
    /// - Prevents the model from ignoring potentially important information
    ///
    /// Why this is important:
    /// - Prevents attention heads from becoming redundant or degenerate
    /// - Improves model robustness and generalization
    /// - Encourages learning diverse attention patterns
    /// - Helps prevent overfitting to specific positions
    ///
    /// Think of it like ensuring a student reads the entire textbook rather than just memorizing one page.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastAttentionWeights == null)
        {
            // Reset diagnostics when disabled to avoid stale values
            _lastAttentionEntropy = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute entropy of attention weights: H = -Σ(p * log(p))
        // Use GPU-accelerated tensor operations for better performance
        T epsilon = NumOps.FromDouble(1e-10); // Small value to prevent log(0)

        // Clamp weights to prevent log(0) - use TensorMax with scalar
        var clampedWeights = Engine.TensorMax(_lastAttentionWeights, epsilon);

        // Compute p * log(p) using GPU-accelerated tensor operations
        var logWeights = Engine.TensorLog(clampedWeights);
        var pLogP = Engine.TensorMultiply(clampedWeights, logWeights);

        // Sum all terms: Σ(p * log(p)) using GPU-accelerated reduction
        T sumPLogP = Engine.TensorSum(pLogP);
        T entropy = NumOps.Negate(sumPLogP);

        // Average entropy over all attention weights
        entropy = NumericalStabilityHelper.SafeDiv(entropy, NumOps.FromDouble(_lastAttentionWeights.Length));

        // Store for diagnostics
        _lastAttentionEntropy = entropy;

        // Return weighted negative entropy as loss (we want to maximize entropy, so minimize -entropy)
        T negativeEntropy = NumOps.Negate(entropy);
        return NumOps.Multiply(AuxiliaryLossWeight, negativeEntropy);
    }

    /// <summary>
    /// Gets diagnostic information about the attention regularization.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about attention patterns.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into attention behavior, including:
    /// - Attention entropy (measure of distribution spread)
    /// - Whether regularization is enabled
    /// - Regularization weight
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to monitor attention pattern health.
    ///
    /// The diagnostics include:
    /// - Attention Entropy: How spread out the attention is (higher = more distributed)
    /// - Entropy Weight: How much the regularization influences training
    /// - Use Auxiliary Loss: Whether regularization is enabled
    ///
    /// These values help you:
    /// - Detect attention collapse (very low entropy)
    /// - Monitor attention diversity during training
    /// - Tune the entropy regularization weight
    /// - Ensure attention heads are learning different patterns
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "AttentionEntropy", _lastAttentionEntropy?.ToString() ?? "0" },
            { "EntropyWeight", AuxiliaryLossWeight?.ToString() ?? "0.01" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        // Add attention weight statistics if available
        if (_lastAttentionWeights != null)
        {
            // Calculate max attention weight (indicates peakiness)
            // Use GPU-accelerated TensorMaxValue for efficient reduction
            T maxWeight = Engine.TensorMaxValue(_lastAttentionWeights);
            diagnostics["MaxAttentionWeight"] = maxWeight?.ToString() ?? "0";
        }

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Resets the state of the attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the attention layer. It clears the last input
    /// and attention weights, effectively preparing the layer for a new sequence or episode.
    /// </para>
    /// <para><b>For Beginners:</b> This is like clearing the layer's short-term memory.
    ///
    /// In attention mechanisms, sometimes we want to start fresh, forgetting any previous inputs.
    /// This is especially useful when starting a new sequence or when you don't want the layer
    /// to consider past information anymore.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastAttentionWeights = null;
        _lastWasCrossAttention = false;
        _lastUsedMask = false;
    }

    /// <summary>
    /// Exports the attention layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the attention operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, inputSize]
    /// 2. Creates constant nodes for Query, Key, Value projection weights
    /// 3. Projects input to Q, K, V using matrix multiplication
    /// 4. Applies scaled dot-product attention: softmax((Q @ K^T) / sqrt(d_k)) @ V
    /// 5. Returns the attention output
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of attention for JIT.
    ///
    /// JIT compilation converts the attention mechanism into optimized native code.
    /// Attention allows the model to focus on relevant parts of the input by:
    /// - Creating Query (what we're looking for), Key (what we have), Value (what we return) projections
    /// - Computing similarity scores between Query and all Keys
    /// - Using softmax to convert scores to weights (focusing mechanism)
    /// - Applying these weights to Values to get focused output
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize matrix multiplications using BLAS libraries
    /// - Fuse softmax computation with scaling
    /// - Generate efficient memory layouts for cache utilization
    ///
    /// Attention is the core mechanism in Transformers and modern NLP models.
    /// JIT compilation provides 5-10x speedup by optimizing these operations.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer parameters are not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Initialize the layer first.");

        if (_Wq == null || _Wk == null || _Wv == null)
            throw new InvalidOperationException("Layer projection weights not initialized. Train or initialize the model first.");

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // AttentionLayer expects input shape: [inputSize]
        // For attention, we use: [batch, inputSize]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for projection weights
        var wqNode = TensorOperations<T>.Constant(_Wq, "Wq");
        var wkNode = TensorOperations<T>.Constant(_Wk, "Wk");
        var wvNode = TensorOperations<T>.Constant(_Wv, "Wv");

        // Project input to Query, Key, Value
        // Q = input @ Wq^T, K = input @ Wk^T, V = input @ Wv^T
        var wqT = TensorOperations<T>.Transpose(wqNode);
        var wkT = TensorOperations<T>.Transpose(wkNode);
        var wvT = TensorOperations<T>.Transpose(wvNode);

        var q = TensorOperations<T>.MatrixMultiply(inputNode, wqT);
        var k = TensorOperations<T>.MatrixMultiply(inputNode, wkT);
        var v = TensorOperations<T>.MatrixMultiply(inputNode, wvT);

        // Apply scaled dot-product attention
        var output = TensorOperations<T>.ScaledDotProductAttention(q, k, v);

        return output;
    }

    /// <summary>
    /// Gets whether this attention layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer parameters are initialized.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Query, Key, Value projection weights are initialized
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - The layer has been initialized with projection weight matrices (Wq, Wk, Wv)
    ///
    /// Attention layers require these projection matrices to transform the input into
    /// query, key, and value representations. Once initialized, JIT compilation can
    /// provide significant speedup (5-10x) by optimizing:
    /// - Matrix multiplications for projections
    /// - Attention score computation (Q @ K^T)
    /// - Softmax activation
    /// - Weighted sum of values (attention @ V)
    ///
    /// This is especially important for Transformers where attention is computed
    /// many times in each forward pass (multiple layers, multiple heads).
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Attention supports JIT if projection weights are initialized
            return _Wq != null && _Wk != null && _Wv != null;
        }
    }
}
