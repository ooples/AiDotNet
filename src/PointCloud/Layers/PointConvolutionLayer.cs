using AiDotNet.ActivationFunctions;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PointCloud.Layers;

/// <summary>
/// Implements a convolution layer specifically designed for point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Unlike regular convolutions for images, point cloud convolutions work on unordered 3D points.
///
/// Key differences from image convolutions:
/// - Images have regular grid structure (pixels in rows/columns)
/// - Point clouds are unordered sets of 3D coordinates
/// - Must be invariant to point order (permutation invariant)
/// - Must handle varying number of points
///
/// This layer applies a shared per-point linear map (a 1x1 convolution over points):
/// output[p] = activation(W^T x[p] + b), learned weights that work regardless of point order.
///
/// Applications:
/// - Feature extraction from local 3D geometry
/// - Learning shape patterns in point clouds
/// - Building blocks for PointNet / DGCNN-style architectures
/// </remarks>
// Rank 2 [points, channels], the shape the base constructor declares - [0, inputChannels] in,
// [0, outputChannels] out - and the shape ForwardTraced computes: TensorMatMul(input, _weights) with
// _weights sized [inputChannels, outputChannels] only types against a rank-2 [N, In] input.
//
// The leading axis is Other rather than Length or Time on purpose. It counts POINTS, and this layer's own
// doc is explicit that a point cloud is an unordered set - "must be invariant to point order" - so calling
// it a sequence position would name it something it is not. Other is the escape hatch for exactly that:
// the axis is real and passes through, but it carries no shared role a downstream layer could check.
[TensorLayout(TensorAxis.Other, TensorAxis.Channels, Direction = TensorLayoutDirection.Input,
    Note = "Leading axis is the point count; a point cloud is unordered, so it takes no sequence role.")]
[TensorLayout(TensorAxis.Other, TensorAxis.Channels, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PointConvolutionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// A shared per-point 1x1 map: the point count is untouched and the channel width becomes
    /// <c>_outputChannels</c>. Read off <c>ForwardTraced</c>, whose
    /// <c>Engine.TensorMatMul(input, _weights)</c> against <c>_weights</c> of
    /// <c>[inputChannels, outputChannels]</c> produces <c>[N, Out]</c>, and off the base constructor's
    /// declared <c>[0, outputChannels]</c>.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputChannels)</c> reads the field rather than a literal, so a layer built with a
    /// different width reports that width. The point axis is <c>Same</c> because nothing here mixes points
    /// - that is precisely the permutation invariance the layer is built around, and a relation that
    /// resized it would contradict the layer's defining property.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _outputChannels <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outputChannels)),
        };
    }

    // Trainable parameters as registered Tensors so the autodiff tape trains them.
    // Not readonly: the restore path re-points them for the copy-on-write DeepCopy/Clone
    // path (which rebinds shared tensor storage into each layer), and Forward reads these fields
    // directly, so a clone that only rebinds the base registry — without updating these fields —
    // would keep its fresh random init and diverge from the original. The generated setter
    // rebinds a mutable field for exactly that reason, and unlike the hand-written pair it
    // replaced it also brings the base registry along instead of leaving it pointing at the
    // pre-clone tensors.
    [AiDotNet.Attributes.TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weights; // [inputChannels, outputChannels]
    [AiDotNet.Attributes.TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _biases;  // [outputChannels]

    /// <summary>
    /// Initializes a new instance of the PointConvolutionLayer class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels.</param>
    /// <param name="outputChannels">Number of output feature channels.</param>
    /// <param name="activation">Optional activation function to apply.</param>
    public PointConvolutionLayer(int inputChannels, int outputChannels, IActivationFunction<T>? activation = null)
        : base([0, inputChannels], [0, outputChannels], activation ?? new IdentityActivation<T>())
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;

        _weights = new Tensor<T>([inputChannels, outputChannels]);
        InitializeWeights();
        _biases = new Tensor<T>([outputChannels]); // zero-initialized

        // Register so GetTrainableParameters() exposes them and the tape optimizer's Step
        // updates the SAME tensor instances the Forward reads.
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>He initialization: weights ~ N(0, sqrt(2 / inputDim)).</summary>
    private void InitializeWeights()
    {
        var numOps = NumOps;
        // LayerBase assigns the architecture-scoped initialization seed before
        // this derived constructor runs. Using the process-shared RNG here
        // discarded that contract, so identically seeded DGCNN/PointNet models
        // still received different point-convolution weights depending on which
        // models had initialized earlier in the process.
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : Random;
        double stddev = Math.Sqrt(2.0 / _inputChannels);
        // Data.Span is a read-oriented projection and may materialize detached
        // storage for copy-on-write/view-backed tensors. Initialization must go
        // through the tensor's writable-storage contract or the registered live
        // parameter can remain all zeros (most visibly in seeded DGCNN builds).
        var span = _weights.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = numOps.FromDouble(random.NextGaussian(0, stddev));
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Tape-tracked per-point linear map: [N, In] @ [In, Out] + bias -> activation.
        // The prior implementation copied into a Matrix<T>, ran the non-differentiable
        // Engine.MatrixMultiply, and copied the result back through a scalar array into a
        // fresh Tensor — which SEVERED the autodiff tape. With no manual backward either,
        // the layer was never trained: point-cloud models that build on it (DGCNN
        // EdgeConv, PointNet) had frozen conv weights and could not learn. All ops below
        // are tape-tracked, so the gradient reaches the registered _weights / _biases and
        // flows on to the input.
        var matmul = Engine.TensorMatMul(input, _weights);                               // [N, Out]
        var biased = Engine.TensorAdd(matmul, Engine.Reshape(_biases, [1, _outputChannels]));
        return ApplyActivation(biased);
    }

    // UpdateParameters delegated straight to SetParameters. The base does that now.
    public override void ClearGradients()
    {
        // No-op: gradients live on the tape, not in a per-layer buffer.
    }

    public override void ResetState()
    {
    }

    public override bool SupportsTraining => true;
}
