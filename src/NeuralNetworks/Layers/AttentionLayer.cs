#pragma warning disable CS0649, CS0414, CS0169
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
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
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "1, 4, 4", TestConstructorArgs = "4, (AiDotNet.Interfaces.IVectorActivationFunction<double>?)null")]
public partial class AttentionLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

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
    private int _inputSize;

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
        _attentionSize * _inputSize * 3 + _inputSize * _attentionSize; // Wq, Wk, Wv, Wo

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int wqLen = _Wq.Length;
        int wkLen = _Wk.Length;
        int wvLen = _Wv.Length;
        int idx = 0;

        // Create new mutable tensors to avoid immutable Engine tensor issue
        _Wq = new Tensor<T>(_Wq._shape);
        var wqSpan = _Wq.Data.Span;
        for (int i = 0; i < wqLen; i++) wqSpan[i] = parameters[idx++];

        _Wk = new Tensor<T>(_Wk._shape);
        var wkSpan = _Wk.Data.Span;
        for (int i = 0; i < wkLen; i++) wkSpan[i] = parameters[idx++];

        _Wv = new Tensor<T>(_Wv._shape);
        var wvSpan = _Wv.Data.Span;
        for (int i = 0; i < wvLen; i++) wvSpan[i] = parameters[idx++];

        int woLen = _Wo.Length;
        _Wo = new Tensor<T>(_Wo._shape);
        var woSpan = _Wo.Data.Span;
        for (int i = 0; i < woLen; i++) woSpan[i] = parameters[idx++];

        RegisterTrainableParameter(_Wq, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wv, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wo, PersistentTensorRole.Weights);
    }

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

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuQ;
    private Tensor<T>? _gpuK;
    private Tensor<T>? _gpuV;
    private Tensor<T>? _gpuAttnOutput;
    private Tensor<T>? _gpuAttnWeights;
    private int[]? _gpuInputShape;
    private int _gpuBatchSize;
    private int _gpuSeqLen;

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
    public AttentionLayer(int attentionSize, IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1 }, new[] { -1 }, activation ?? new SoftmaxActivation<T>())
    {
        if (attentionSize <= 0) throw new ArgumentOutOfRangeException(nameof(attentionSize));

        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastAttentionEntropy = NumOps.Zero;

        _inputSize = -1;
        _attentionSize = attentionSize;
        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.Eager;
        _Wq = new Tensor<T>([0, 0]);
        _Wk = new Tensor<T>([0, 0]);
        _Wv = new Tensor<T>([0, 0]);
        _Wo = new Tensor<T>([0, 0]);
    }

    /// <summary>
    /// Resolves input feature size on first forward and allocates Q/K/V/O weights.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException(
                $"AttentionLayer requires rank>=1 input; got rank {rank}.", nameof(input));

        _inputSize = input.Shape[rank - 1];
        _Wq = new Tensor<T>(new[] { _attentionSize, _inputSize });
        _Wk = new Tensor<T>(new[] { _attentionSize, _inputSize });
        _Wv = new Tensor<T>(new[] { _attentionSize, _inputSize });
        _Wo = new Tensor<T>(new[] { _inputSize, _attentionSize });
        InitializeLayerWeights(_Wq, _inputSize, _attentionSize);
        InitializeLayerWeights(_Wk, _inputSize, _attentionSize);
        InitializeLayerWeights(_Wv, _inputSize, _attentionSize);
        InitializeLayerWeights(_Wo, _attentionSize, _inputSize);
        RegisterTrainableParameter(_Wq, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wk, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wv, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wo, PersistentTensorRole.Weights);

        // Output last dim = input last dim (passthrough via Wo projection)
        var inputShape = input.Shape.ToArray();
        ResolveShapes(inputShape, inputShape);
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
    public AttentionLayer(int attentionSize, IVectorActivationFunction<T>? activation = null)
        : base(new[] { -1 }, new[] { -1 }, activation ?? new SoftmaxActivation<T>())
    {
        if (attentionSize <= 0) throw new ArgumentOutOfRangeException(nameof(attentionSize));

        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastAttentionEntropy = NumOps.Zero;

        _inputSize = -1;
        _attentionSize = attentionSize;
        _Wq = new Tensor<T>([0, 0]);
        _Wk = new Tensor<T>([0, 0]);
        _Wv = new Tensor<T>([0, 0]);
        _Wo = new Tensor<T>([0, 0]);
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
        double scaleD = NumOps.ToDouble(scale);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble((Random.NextDouble() - 0.5) * scaleD);
        }

        return tensor;
    }

    /// <summary>
    /// Declares named input ports for this multi-input layer.
    /// </summary>
    public override IReadOnlyList<LayerPort> InputPorts =>
    [
        new LayerPort("input", GetInputShape()),
        new LayerPort("context", GetInputShape(), Required: false)
    ];

    /// <summary>
    /// Named multi-input forward pass.
    /// </summary>
    public override Tensor<T> Forward(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (!inputs.TryGetValue("input", out var input) || input == null)
            throw new ArgumentException("AttentionLayer requires a non-null 'input'.", nameof(inputs));
        if (inputs.TryGetValue("context", out var context) && context != null)
            return Forward(input, context);
        return Forward(input);
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
        if (input != null) EnsureInitializedFromInput(input);
        // Validate input tensor
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        // Handle any rank >= 2: last 2 dims are [Seq, InputSize], earlier dims are batch-like
        int rank = input.Shape.Length;
        _inputWas2D = rank == 2;
        Tensor<T> input3D;
        _originalInputShape = input._shape;

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
            input3D = Engine.Reshape(input, [batchSize2D, 1, _inputSize]);
        }
        else if (rank == 3)
        {
            // 3D input: standard attention format [Batch, Seq, InputSize]
            if (input.Shape[2] != _inputSize)
            {
                throw new ArgumentException(
                    $"AttentionLayer input size mismatch. Expected InputSize={_inputSize}, " +
                    $"but got {input.Shape[2]} in shape [{string.Join(", ", input._shape)}].",
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
                    $"but got {input.Shape[rank - 1]} in shape [{string.Join(", ", input._shape)}].",
                    nameof(input));
            }
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            input3D = Engine.Reshape(input, [flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
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
        var inputFlat = Engine.Reshape(input3D, [batchSize * seqLen, _inputSize]);

        // Transpose weights to [InputSize, AttSize] using Engine 2D transpose
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);

        // Compute Projections: [B*S, In] @ [In, Att] -> [B*S, Att]
        var qProjected = Engine.TensorMatMul(inputFlat, wqTransposed);
        var kProjected = Engine.TensorMatMul(inputFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(inputFlat, wvTransposed);

        // Reshape back to [Batch, Seq, AttSize]
        var Q = Engine.Reshape(qProjected, [batchSize, seqLen, _attentionSize]);
        var K = Engine.Reshape(kProjected, [batchSize, seqLen, _attentionSize]);
        var V = Engine.Reshape(vProjected, [batchSize, seqLen, _attentionSize]);

        Tensor<T> attentionOutput;

        // Check if we can use the optimized Engine.ScaledDotProductAttention
        // (only when using default Softmax activation)
        if (VectorActivation is SoftmaxActivation<T>)
        {
            // 2-5. Use Engine.ScaledDotProductAttention for optimized computation
            // Reshape to 4D by adding head dimension: [B, S, A] -> [B, 1, S, A]
            var Q4D = Engine.Reshape(Q, [batchSize, 1, seqLen, _attentionSize]);
            var K4D = Engine.Reshape(K, [batchSize, 1, seqLen, _attentionSize]);
            var V4D = Engine.Reshape(V, [batchSize, 1, seqLen, _attentionSize]);

            var output4D = Engine.ScaledDotProductAttention(
                Q4D, K4D, V4D,
                mask: null,
                scale: 1.0 / Math.Sqrt(_attentionSize),
                out var attentionWeights4D);

            // Reshape attention weights back to 3D for caching: [B, 1, S, S] -> [B, S, S]
            _lastAttentionWeights = Engine.Reshape(attentionWeights4D, [batchSize, seqLen, seqLen]);

            // Reshape output back to 3D: [B, 1, S, A] -> [B, S, A]
            attentionOutput = Engine.Reshape(output4D, [batchSize, seqLen, _attentionSize]);
        }
        else
        {
            // Fallback to manual computation for custom activations
            // 2. Compute Attention Scores: Q @ K.T (per-batch, Engine.BatchMatMul has issues)
            var KT = Engine.TensorPermute(K, new[] { 0, 2, 1 });
            var attentionScores = TensorAllocator.Rent<T>([batchSize, seqLen, seqLen]);
            for (int b = 0; b < batchSize; b++)
            {
                var qB = Q.GetSliceAlongDimension(b, 0);
                var ktB = KT.GetSliceAlongDimension(b, 0);
                var scoresB = Engine.TensorMatMul(qB, ktB);
                for (int i = 0; i < scoresB.Length; i++)
                    attentionScores.SetFlat(b * seqLen * seqLen + i, scoresB.GetFlat(i));
            }

            // 3. Scale
            T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(_attentionSize)));
            attentionScores = Engine.TensorMultiplyScalar(attentionScores, scaleValue);

            // 4. Apply activation (custom, not softmax)
            _lastAttentionWeights = ApplyActivation(attentionScores);

            // 5. Output: Weights @ V (per-batch)
            attentionOutput = TensorAllocator.Rent<T>([batchSize, seqLen, _attentionSize]);
            for (int b = 0; b < batchSize; b++)
            {
                var wB = _lastAttentionWeights.GetSliceAlongDimension(b, 0);
                var vB = V.GetSliceAlongDimension(b, 0);
                var outB = Engine.TensorMatMul(wB, vB);
                for (int i = 0; i < outB.Length; i++)
                    attentionOutput.SetFlat(b * seqLen * _attentionSize + i, outB.GetFlat(i));
            }
        }
        _lastAttentionOutput = attentionOutput; // Cache for backward pass

        // 6. Output Projection: Apply Wo to project from attentionSize back to inputSize
        // Flatten for matmul: [B*S, A] @ [A, inputSize] -> [B*S, inputSize]
        var attnFlat = Engine.Reshape(attentionOutput, [batchSize * seqLen, _attentionSize]);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = Engine.Reshape(projectedFlat, [batchSize, seqLen, _inputSize]);

        // Restore original tensor shape
        if (_inputWas2D)
        {
            output = Engine.Reshape(output, [batchSize, _inputSize]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 2] = seqLen;
            outputShape[_originalInputShape.Length - 1] = _inputSize;
            output = Engine.Reshape(output, outputShape);
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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input._shape;

        // Handle 2D [Batch, InputSize] or 3D [Batch, Seq, InputSize] input
        int batchSize;
        int seqLen;
        Tensor<T> input3D;

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

        // Cache GPU tensors for backward pass during training
        if (IsTrainingMode)
        {
            // Cache the 3D input (don't dispose)
            _gpuInput = input3D;
            _gpuBatchSize = batchSize;
            _gpuSeqLen = seqLen;
            _gpuInputShape = input._shape;

            // Reshape Q, K, V back to 3D [B, S, A] for backward
            _gpuQ = gpuEngine.ReshapeGpu(qFlat, [batchSize, seqLen, _attentionSize]);
            _gpuK = gpuEngine.ReshapeGpu(kFlat, [batchSize, seqLen, _attentionSize]);
            _gpuV = gpuEngine.ReshapeGpu(vFlat, [batchSize, seqLen, _attentionSize]);

            // Reshape attention output and weights to 3D
            _gpuAttnOutput = gpuEngine.ReshapeGpu(attnOutput4D, [batchSize, seqLen, _attentionSize]);
            _gpuAttnWeights = gpuEngine.ReshapeGpu(attnWeights4D, [batchSize, seqLen, seqLen]);

            // Also cache CPU versions for CPU backward compatibility
            _lastInput = input;
            _lastQueryInput = input3D;
            _lastKeyInput = input3D;
            _lastValueInput = input3D;
            _lastAttentionOutput = _gpuAttnOutput;
            _lastAttentionWeights = _gpuAttnWeights;

            // Dispose tensors we don't need (but keep ones cached for backward)
            ((IDisposable)inputFlat).Dispose();
            ((IDisposable)Q4D).Dispose();
            ((IDisposable)K4D).Dispose();
            ((IDisposable)V4D).Dispose();
            ((IDisposable)attnOutput4D).Dispose();
            ((IDisposable)attnWeights4D).Dispose();
            ((IDisposable)attnFlat).Dispose();
        }
        else
        {
            // Dispose intermediate tensors to free GPU memory (inference mode)
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
        }

        // Reshape to final output shape
        Tensor<T> output;
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
                    $"Second input tensor has ambiguous shape {string.Join("x", secondInput._shape)}. " +
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
            ? Engine.Reshape(input, [input.Shape[0], 1, _inputSize])
            : input;

        _lastQueryInput = input3D;
        _lastKeyInput = input3D;
        _lastValueInput = input3D;

        int batchSize = input3D.Shape[0];
        int seqLen = input3D.Shape[1];

        // 1. Project Input to Q, K, V
        var inputFlat = Engine.Reshape(input3D, [batchSize * seqLen, _inputSize]);
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);

        var qProjected = Engine.TensorMatMul(inputFlat, wqTransposed);
        var kProjected = Engine.TensorMatMul(inputFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(inputFlat, wvTransposed);

        var Q = Engine.Reshape(qProjected, [batchSize, seqLen, _attentionSize]);
        var K = Engine.Reshape(kProjected, [batchSize, seqLen, _attentionSize]);
        var V = Engine.Reshape(vProjected, [batchSize, seqLen, _attentionSize]);

        // 2. Compute Attention Scores: Q @ K.T
        var KT = Engine.TensorPermute(K, new[] { 0, 2, 1 });
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
        var attnFlat = Engine.Reshape(attentionOutput, [batchSize * seqLen, _attentionSize]);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = Engine.Reshape(projectedFlat, [batchSize, seqLen, _inputSize]);

        // If input was 2D, reshape output back to 2D
        if (_inputWas2D)
        {
            output = Engine.Reshape(output, [batchSize, _inputSize]);
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
            query3D = Engine.Reshape(queryInput, [batchSize2D, 1, _inputSize]);
            // KeyValue input should match query dimensionality
            if (keyValueInput.Shape.Length == 2)
            {
                keyValue3D = Engine.Reshape(keyValueInput, [keyValueInput.Shape[0], 1, _inputSize]);
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
        var queryFlat = Engine.Reshape(query3D, [batchSize * seqLenQ, _inputSize]);
        var wqTransposed = Engine.TensorTranspose(_Wq);
        var qProjected = Engine.TensorMatMul(queryFlat, wqTransposed);
        var Q = Engine.Reshape(qProjected, [batchSize, seqLenQ, _attentionSize]);

        // Project K, V from key/value input
        var kvFlat = Engine.Reshape(keyValue3D, [batchSize * seqLenKV, _inputSize]);
        var wkTransposed = Engine.TensorTranspose(_Wk);
        var wvTransposed = Engine.TensorTranspose(_Wv);
        var kProjected = Engine.TensorMatMul(kvFlat, wkTransposed);
        var vProjected = Engine.TensorMatMul(kvFlat, wvTransposed);
        var K = Engine.Reshape(kProjected, [batchSize, seqLenKV, _attentionSize]);
        var V = Engine.Reshape(vProjected, [batchSize, seqLenKV, _attentionSize]);

        // Compute Scores: Q @ K.T
        var KT = Engine.TensorPermute(K, new[] { 0, 2, 1 });
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
        var attnFlat = Engine.Reshape(attentionOutput, [batchSize * seqLenQ, _attentionSize]);
        var woTransposed = Engine.TensorTranspose(_Wo);
        var projectedFlat = Engine.TensorMatMul(attnFlat, woTransposed);
        var output = Engine.Reshape(projectedFlat, [batchSize, seqLenQ, _inputSize]);

        // If input was 2D, reshape output back to 2D
        if (_inputWas2D)
        {
            output = Engine.Reshape(output, [batchSize, _inputSize]);
        }

        return output;
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
        _Wq = Tensor<T>.FromVector(wqParams).Reshape(_Wq._shape);
        startIndex += _Wq.Length;

        // Update Wk - slice and copy
        var wkParams = parameters.Slice(startIndex, _Wk.Length);
        _Wk = Tensor<T>.FromVector(wkParams).Reshape(_Wk._shape);
        startIndex += _Wk.Length;

        // Update Wv - slice and copy
        var wvParams = parameters.Slice(startIndex, _Wv.Length);
        _Wv = Tensor<T>.FromVector(wvParams).Reshape(_Wv._shape);

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

        var woVec = _Wo.ToVector();
        return Vector<T>.Concatenate(Vector<T>.Concatenate(wqVec, wkVec), Vector<T>.Concatenate(wvVec, woVec));
    }

    public override Vector<T> GetParameterGradients()
    {
        var gQ = _dWq != null ? _dWq.ToVector() : new Vector<T>(_Wq.Length);
        var gK = _dWk != null ? _dWk.ToVector() : new Vector<T>(_Wk.Length);
        var gV = _dWv != null ? _dWv.ToVector() : new Vector<T>(_Wv.Length);
        var gO = new Vector<T>(_Wo.Length); // Wo gradient not tracked separately yet
        return Vector<T>.Concatenate(Vector<T>.Concatenate(gQ, gK), Vector<T>.Concatenate(gV, gO));
    }

    public override void ClearGradients()
    {
        _dWq = null;
        _dWk = null;
        _dWv = null;
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

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuQ = null;
        _gpuK = null;
        _gpuV = null;
        _gpuAttnOutput = null;
        _gpuAttnWeights = null;
        _gpuInputShape = null;
    }
}
