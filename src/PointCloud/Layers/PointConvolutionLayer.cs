using AiDotNet.ActivationFunctions;
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
public class PointConvolutionLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;

    // Trainable parameters as registered Tensors so the autodiff tape trains them.
    // Not readonly: SetTrainableParameters re-points them for the copy-on-write DeepCopy/Clone
    // path (which rebinds shared tensor storage into each layer), and Forward reads these fields
    // directly, so a clone that only rebinds the base registry — without updating these fields —
    // would keep its fresh random init and diverge from the original.
    private Tensor<T> _weights; // [inputChannels, outputChannels]
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
        var random = Random;
        double stddev = Math.Sqrt(2.0 / _inputChannels);
        var span = _weights.Data.Span;
        for (int i = 0; i < span.Length; i++)
            span[i] = numOps.FromDouble(random.NextGaussian(0, stddev));
    }

    public override Tensor<T> Forward(Tensor<T> input)
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
        var biased = Engine.TensorBroadcastAdd(matmul, Engine.Reshape(_biases, [1, _outputChannels]));
        return ApplyActivation(biased);
    }

    /// <summary>
    /// Returns the field-backed trainable tensors so the tape optimizer, the parameter-count walk,
    /// and the copy-on-write clone all see the SAME instances the Forward reads. Overriding this
    /// (rather than relying on the base <c>_registeredTensors</c> list) keeps GetTrainableParameters
    /// consistent with <see cref="SetTrainableParameters"/> after a field re-point.
    /// </summary>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() => new[] { _weights, _biases };

    /// <summary>
    /// Re-points the field-backed weight/bias tensors to the supplied instances. The copy-on-write
    /// DeepCopy/Clone path shares each source tensor into its clone through this method; because
    /// <see cref="Forward"/> reads the <c>_weights</c>/<c>_biases</c> fields directly, they must be
    /// rebound here (the base only updates its private registry), or the clone diverges from the
    /// original (issue #1221 class).
    /// </summary>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != 2)
        {
            throw new ArgumentException($"Expected 2 parameter tensors (weights, biases), got {parameters.Count}.", nameof(parameters));
        }

        _weights = parameters[0];
        _biases = parameters[1];
    }

    public override Vector<T> GetParameters()
    {
        int total = (int)ParameterCount;
        var parameters = new Vector<T>(total);
        var w = _weights.Data.Span;
        var b = _biases.Data.Span;
        int idx = 0;
        for (int i = 0; i < w.Length; i++) parameters[idx++] = w[i];
        for (int i = 0; i < b.Length; i++) parameters[idx++] = b[i];
        return parameters;
    }

    public override void UpdateParameters(Vector<T> parameters) => SetParameters(parameters);

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException("Parameter vector length does not match layer parameter count.", nameof(parameters));
        }

        // Write in place so the registered tensor instances (which the tape trained and
        // the Forward reads) keep their identity. This MUST override SetParameters — the
        // Clone / DeepCopy / serialize round-trip distributes weights through SetParameters,
        // and the LayerBase default only stashes the vector in the Parameters field without
        // writing _weights / _biases, so a clone would keep its fresh random init and diverge
        // from the original (issue #1221 class).
        var w = _weights.Data.Span;
        var b = _biases.Data.Span;
        int idx = 0;
        for (int i = 0; i < w.Length; i++) w[i] = parameters[idx++];
        for (int i = 0; i < b.Length; i++) b[i] = parameters[idx++];
    }

    public override void UpdateParameters(T learningRate)
    {
        // No-op: _weights / _biases are registered trainable tensors updated through the
        // tape optimizer's Step (there is no manual per-layer gradient buffer). Retained
        // for the ILayer contract and legacy per-layer training drivers.
    }

    public override void ClearGradients()
    {
        // No-op: gradients live on the tape, not in a per-layer buffer.
    }

    public override void ResetState()
    {
    }

    public override long ParameterCount => (long)_inputChannels * _outputChannels + _outputChannels;

    public override bool SupportsTraining => true;
}
