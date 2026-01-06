using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
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
public class RRDBLayer<T> : LayerBase<T>, IChainableComputationGraph<T>
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
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all RDB blocks support JIT
            foreach (var rdb in _rdbBlocks)
            {
                if (!rdb.SupportsJitCompilation)
                    return false;
            }
            return true;
        }
    }

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
        int inputHeight = 64,
        int inputWidth = 64,
        double residualScale = 0.2)
        : base(
            [numFeatures, inputHeight, inputWidth],
            [numFeatures, inputHeight, inputWidth])
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

        // Create 3 Residual Dense Blocks
        _rdbBlocks = new ResidualDenseBlock<T>[3];
        for (int i = 0; i < 3; i++)
        {
            _rdbBlocks[i] = new ResidualDenseBlock<T>(
                numFeatures: numFeatures,
                growthChannels: growthChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                residualScale: residualScale); // Each RDB also uses the same residual scale
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
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        // RRDB chains 3 ResidualDenseBlocks with global residual - use CPU fallback for complex chained operations
        var cpuInput = inputs[0].ToTensor();
        var cpuOutput = Forward(cpuInput);
        return gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _rdb3Output == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Gradient through global residual:
        // d(x * scale + input) / dx = scale
        // d(x * scale + input) / dinput = 1
        var rdb3Gradient = ScaleGradient(outputGradient, _residualScale);
        var inputGradientFromResidual = outputGradient;

        // Backward through RDB3, RDB2, RDB1
        var grad = _rdbBlocks[2].Backward(rdb3Gradient);
        grad = _rdbBlocks[1].Backward(grad);
        grad = _rdbBlocks[0].Backward(grad);

        // Add gradient from global residual connection
        return AddTensors(grad, inputGradientFromResidual);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Adds residual with scaling: output = a * scale + b.
    /// </summary>
    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b, double scale)
    {
        var output = new Tensor<T>(a.Shape);
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < a.Length; i++)
        {
            output.Data[i] = NumOps.Add(
                NumOps.Multiply(a.Data[i], scaleT),
                b.Data[i]);
        }
        return output;
    }

    /// <summary>
    /// Scales a tensor by a factor.
    /// </summary>
    private Tensor<T> ScaleGradient(Tensor<T> gradient, double scale)
    {
        var output = new Tensor<T>(gradient.Shape);
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(gradient.Data[i], scaleT);
        }
        return output;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var output = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            output.Data[i] = NumOps.Add(a.Data[i], b.Data[i]);
        }
        return output;
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

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var rdb in _rdbBlocks)
        {
            int count = rdb.GetParameters().Length;
            rdb.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _rdb3Output = null;
        foreach (var rdb in _rdbBlocks)
        {
            rdb.ResetState();
        }
    }

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension [batch, channels, height, width]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // Pass through 3 Residual Dense Blocks sequentially
        var x = _rdbBlocks[0].BuildComputationGraph(inputNode, $"{namePrefix}rdb0_");
        x = _rdbBlocks[1].BuildComputationGraph(x, $"{namePrefix}rdb1_");
        x = _rdbBlocks[2].BuildComputationGraph(x, $"{namePrefix}rdb2_");

        // Global residual: output = RDB3_output * residualScale + input
        var scaledOutput = ScaleNode(x, _residualScale, $"{namePrefix}global_scale");
        return TensorOperations<T>.Add(scaledOutput, inputNode);
    }

    /// <summary>
    /// Scales a computation node by a scalar value using element-wise multiplication.
    /// </summary>
    private static ComputationNode<T> ScaleNode(ComputationNode<T> node, double scale, string name)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaleValue = numOps.FromDouble(scale);

        // Create a constant tensor filled with the scale value matching the input shape
        var scaleTensor = new Tensor<T>(node.Value.Shape);
        for (int i = 0; i < scaleTensor.Length; i++)
        {
            scaleTensor.Data[i] = scaleValue;
        }

        var scaleNode = TensorOperations<T>.Constant(scaleTensor, name);
        return TensorOperations<T>.ElementwiseMultiply(node, scaleNode);
    }

    #endregion
}
