using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Residual in Residual Dense Block (RRDB) - the core building block of ESRGAN and Real-ESRGAN generators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RRDB combines 3 Residual Dense Blocks with a global residual connection.
/// This is the architecture from the ESRGAN paper (Wang et al., 2018) that enables
/// training very deep networks for high-quality image super-resolution.
/// </para>
/// <para>
/// The architecture is:
/// <code>
/// input
///   ↓
/// ResidualDenseBlock 1 (local residual inside)
///   ↓
/// ResidualDenseBlock 2 (local residual inside)
///   ↓
/// ResidualDenseBlock 3 (local residual inside)
///   ↓
/// output = RDB3_output * residualScale + input  (global residual)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> RRDB is like a "super block" that contains 3 smaller blocks (RDBs).
///
/// The key insight is **residual-in-residual** learning:
/// - Each RDB has its own residual connection (local)
/// - The entire RRDB also has a residual connection (global)
///
/// This nested residual structure helps:
/// - Very deep networks train more easily
/// - Gradients flow better during backpropagation
/// - The network can learn fine details without losing coarse features
///
/// Real-ESRGAN typically uses 23 RRDB blocks, each containing 3 RDBs,
/// for a total of 69 residual dense blocks!
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
/// ECCV 2018 Workshops. https://arxiv.org/abs/1809.00219
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "4, 8, 8", TestConstructorArgs = "4, 4")]
public class RRDBLayer<T> : LayerBase<T>
{
    #region Fields

    /// <summary>
    /// The 3 Residual Dense Blocks that make up this RRDB.
    /// </summary>
    private readonly ResidualDenseBlock<T>[] _rdbBlocks;

    /// <summary>
    /// Global residual scaling factor. Default: 0.2 from the paper.
    /// </summary>
    private readonly double _residualScale;

    /// <summary>
    /// Number of feature channels.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Growth channels for each RDB.
    /// </summary>
    private readonly int _growthChannels;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from RDB3 for backpropagation.
    /// </summary>
    private Tensor<T>? _rdb3Output;

    /// <summary>
    /// GPU cached input tensor for backward pass.
    /// </summary>
    private Tensor<T>? _gpuLastInput;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the global residual scaling factor.
    /// </summary>
    public double ResidualScale => _residualScale;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    public int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the growth channels used in each RDB.
    /// </summary>
    public int GrowthChannels => _growthChannels;

    /// <inheritdoc />
    public override long ParameterCount => GetParameters().Length;
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new RRDB layer.
    /// </summary>
    /// <param name="numFeatures">Number of input/output feature channels. Default: 64 (from paper).</param>
    /// <param name="growthChannels">Growth channels for RDBs. Default: 32 (from paper).</param>
    /// <param name="inputHeight">Height of input feature maps.</param>
    /// <param name="inputWidth">Width of input feature maps.</param>
    /// <param name="residualScale">Global residual scaling factor. Default: 0.2 (from paper).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create an RRDB block for use in ESRGAN/Real-ESRGAN generators:
    /// <code>
    /// var rrdb = new RRDBLayer&lt;float&gt;(
    ///     numFeatures: 64,       // Main feature channels
    ///     growthChannels: 32,    // Growth rate per conv
    ///     inputHeight: 128,
    ///     inputWidth: 128,
    ///     residualScale: 0.2     // Global residual scaling
    /// );
    /// </code>
    ///
    /// The default parameters match the ESRGAN paper exactly.
    /// </para>
    /// </remarks>
    public RRDBLayer(
        int numFeatures = 64,
        int growthChannels = 32,
        double residualScale = 0.2)
        : base(
            [numFeatures, -1, -1],
            [numFeatures, -1, -1])
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be positive.");
        if (growthChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(growthChannels), "Growth channels must be positive.");
        if (residualScale < 0 || residualScale > 1)
            throw new ArgumentOutOfRangeException(nameof(residualScale), "Residual scale must be between 0 and 1.");

        _numFeatures = numFeatures;
        _growthChannels = growthChannels;
        _residualScale = residualScale;

        // Create 3 Residual Dense Blocks — they're lazy-spatial too, so
        // their own input H/W get filled in on first Forward.
        _rdbBlocks = new ResidualDenseBlock<T>[3];
        for (int i = 0; i < 3; i++)
        {
            _rdbBlocks[i] = new ResidualDenseBlock<T>(
                numFeatures: numFeatures,
                growthChannels: growthChannels,
                residualScale: residualScale);
            RegisterSubLayer(_rdbBlocks[i]);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves spatial dims from <c>input.Shape</c>, drives each inner
    /// RDB's lazy resolution by the same shape (channels stay constant),
    /// and locks the layer's input/output shapes.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inChannels, inH, inW;
        if (s.Length == 3) { inChannels = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inChannels = s[1]; inH = s[2]; inW = s[3]; }
        else
            throw new ArgumentException(
                $"RRDBLayer requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));
        if (inChannels != _numFeatures)
            throw new ArgumentException(
                $"RRDBLayer expected {_numFeatures} input channels, got {inChannels}.",
                nameof(input));

        foreach (var block in _rdbBlocks)
        {
            block.ResolveFromShape(new[] { _numFeatures, inH, inW });
            block.SetTrainingMode(IsTrainingMode);
        }

        ResolveShapes(
            new[] { _numFeatures, inH, inW },
            new[] { _numFeatures, inH, inW });

        // Replay any Deserialize-buffered parameters now that inner RDBs are resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int offset = 0;
            foreach (var rdb in _rdbBlocks)
            {
                int count = rdb.GetParameters().Length;
                rdb.SetParameters(pending.SubVector(offset, count));
                offset += count;
            }
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Forward pass: input -> RDB1 -> RDB2 -> RDB3 -> scale + input (global residual)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = input;

        // Pass through 3 Residual Dense Blocks sequentially
        var x = _rdbBlocks[0].Forward(input);
        x = _rdbBlocks[1].Forward(x);
        x = _rdbBlocks[2].Forward(x);

        _rdb3Output = x;

        // Global residual: output = RDB3_output * residualScale + input
        return AddResidual(x, input, _residualScale);
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after RRDB processing.</returns>
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

        // Mirror Forward()'s lazy-resolution gate. OnFirstForward
        // resolves _rdbBlocks' inner shape and replays buffered
        // _pendingParameters; without this guard a GPU-first execution
        // dispatches to inner RDBs whose internal weights are still 0×0,
        // and the GPU forward reads zero-length buffers (silent wrong
        // output) or crashes on a kernel arg-check.
        if (!IsShapeResolved) OnFirstForward(input);

        var shape = input._shape;

        // Cache input for backward pass
        if (IsTrainingMode)
        {
            _gpuLastInput = input;
        }

        // Store input buffer for global residual
        var inputBuffer = input.Buffer;

        // Pass through 3 Residual Dense Blocks sequentially on GPU
        var x = _rdbBlocks[0].ForwardGpu(input);
        x = _rdbBlocks[1].ForwardGpu(x);
        x = _rdbBlocks[2].ForwardGpu(x);

        // Global residual: output = RDB3_output * residualScale + input
        int outputSize = shape.Aggregate(1, (a, b) => a * b);

        var scaledBuffer = backend.AllocateBuffer(outputSize);
        backend.Scale(x.Buffer, scaledBuffer, (float)_residualScale, outputSize);

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.Add(scaledBuffer, inputBuffer, outputBuffer, outputSize);

        scaledBuffer.Dispose();

        return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    #endregion

    #region Backward Pass

    #endregion

    #region Helper Methods

    /// <summary>
    /// Adds residual with scaling: output = a * scale + b.
    /// </summary>
    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        var scaled = Engine.TensorMultiplyScalar(a, scaleT);
        return Engine.TensorAdd(scaled, b);
    }

    /// <summary>
    /// Scales a tensor by a factor.
    /// </summary>
    private Tensor<T> ScaleGradient(Tensor<T> gradient, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        return Engine.TensorMultiplyScalar(gradient, scaleT);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var rdb in _rdbBlocks)
        {
            rdb.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var rdb in _rdbBlocks)
        {
            var rdbParams = rdb.GetParameters();
            for (int i = 0; i < rdbParams.Length; i++)
            {
                allParams.Add(rdbParams[i]);
            }
        }
        return new Vector<T>([.. allParams]);
    }

    public override Vector<T> GetParameterGradients()
    {
        var gradVectors = _rdbBlocks.Select(r => r.GetParameterGradients()).ToArray();
        return Vector<T>.Concatenate(gradVectors);
    }

    public override void ClearGradients()
    {
        foreach (var rdb in _rdbBlocks)
            rdb.ClearGradients();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: each inner RDB's shape is unresolved. Buffer
        // and replay from OnFirstForward.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        int offset = 0;
        foreach (var rdb in _rdbBlocks)
        {
            int count = rdb.GetParameters().Length;
            rdb.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    private Vector<T>? _pendingParameters;

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _rdb3Output = null;
        _gpuLastInput = null;
        foreach (var rdb in _rdbBlocks)
        {
            rdb.ResetState();
        }
    }


    /// <summary>
    /// Scales a computation node by a scalar value using element-wise multiplication.
    /// </summary>
    private static ComputationNode<T> ScaleNode(ComputationNode<T> node, double scale, string name)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaleValue = numOps.FromDouble(scale);

        // Create a constant tensor filled with the scale value matching the input shape
        var scaleTensor = new Tensor<T>(node.Value._shape);
        for (int i = 0; i < scaleTensor.Length; i++)
        {
            scaleTensor.Data.Span[i] = scaleValue;
        }

        var scaleNode = TensorOperations<T>.Constant(scaleTensor, name);
        return TensorOperations<T>.ElementwiseMultiply(node, scaleNode);
    }

    #endregion

}
