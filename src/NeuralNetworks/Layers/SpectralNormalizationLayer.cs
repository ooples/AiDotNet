using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a spectral normalization layer that normalizes the weights of a layer by their spectral norm.
/// </summary>
/// <remarks>
/// <para>
/// Spectral normalization is a weight normalization technique that constrains the Lipschitz constant
/// of a neural network layer. It does this by dividing the weight matrix by its largest singular value
/// (spectral norm). This technique is particularly effective for stabilizing GAN training.
/// </para>
/// <para><b>For Beginners:</b> Spectral normalization keeps layer weights from getting too large.
///
/// Key benefits:
/// - Stabilizes GAN training by preventing extreme weight values
/// - Ensures the discriminator doesn't become too powerful too quickly
/// - Helps prevent mode collapse in GANs
/// - Computationally efficient compared to other normalization methods
///
/// How it works:
/// - Computes the largest singular value of the weight matrix
/// - Divides all weights by this value
/// - Keeps weights normalized throughout training
///
/// Reference: Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Regularization)]
[LayerTask(LayerTask.Regularization)]
[LayerProperty(IsTrainable = true, TestConstructorArgs = "new AiDotNet.NeuralNetworks.Layers.ReadoutLayer<double>(4, 8, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.IdentityActivation<double>())", TestInputShape = "1, 4")]
// A DECORATOR: this layer rescales the inner layer's WEIGHTS and then returns
// `_innerLayer.Forward(input)` verbatim (ForwardTraced), so it has no shape law of its own - it has the
// inner layer's. The constructor says the same thing, chaining
// `base(innerLayer.GetInputShape(), innerLayer.GetOutputShape())`.
//
// The layouts below exist only to NAME the input axes, because ShapeInference.NameAxes reads them from
// the TYPE and a relation the inner layer hands back refers to its sources by role. The ranks declared
// are the ones spectral normalization is actually applied at: it normalizes GetParameters(), so the
// layers worth wrapping are the weight-bearing ones - Dense/FullyConnected over [Batch, Features] and
// [Batch, Time, Features], and 2-D convolutions over [Batch, Channels, Height, Width]. Rank 3 is named
// the sequence way rather than the unbatched-conv way because only one naming per rank is permitted
// (ADNSHAPE001); an unbatched [C,H,W] convolution therefore DECLINES here rather than resolving, since
// its relations name Height and Width and this naming has neither. Declining is the honest outcome - a
// wrong extent would not be.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SpectralNormalizationLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// The underlying layer whose weights will be normalized.
    /// </summary>
    private readonly ILayer<T> _innerLayer;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Delegated, not restated. <c>ForwardTraced</c> returns <c>_innerLayer.Forward(input)</c> - both on
    /// the no-parameter early-out and on the normalized path - so whatever shape law the wrapped layer
    /// has is this layer's shape law exactly. Dividing weights by their largest singular value changes
    /// their VALUES, never their count or arrangement, so no axis moves.
    /// </para>
    /// <para>
    /// Only expressible because <c>OutputAxesFor</c> is an INSTANCE method: the answer depends on which
    /// layer this was constructed around. Wrap a convolution and the spatial axes come back windowed by
    /// that convolution's stride; wrap something that declares no contract and this returns null, which
    /// is the honest answer rather than a guess.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => (_innerLayer as IShapeContract)?.OutputAxesFor(inputRank);

    /// <summary>
    /// The left singular vector used for power iteration to compute the spectral norm.
    /// </summary>
    // A BUFFER, not scratch. Power iteration starts from a RANDOM vector and refines it, and the
    // spectral norm it converges to is what divides the weights -- so a layer that regenerates it
    // on load computes a different norm and predicts differently from the model that was saved.
    // Marked scratch, it was dropped from the checkpoint and the restored layer's output moved
    // from 0.664 to 0.355. This is the same reason PyTorch registers u and v as buffers.
    [AiDotNet.Attributes.Buffer]
    private Tensor<T>? _u;

    /// <summary>
    /// The right singular vector used for power iteration.
    /// </summary>
    [AiDotNet.Attributes.Buffer]
    private Tensor<T>? _v;

    /// <summary>
    /// The number of power iterations to perform when computing the spectral norm.
    /// </summary>
    private readonly int _powerIterations;

    /// <summary>
    /// Epsilon value for numerical stability.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Cached input from the last forward pass.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from the last forward pass.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Original weights stored during Forward, to be restored after Backward.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _originalParameters;

    /// <summary>
    /// Flag indicating that normalized weights are currently applied.
    /// </summary>
    private bool _normalizedWeightsApplied;

    public override bool SupportsTraining => _innerLayer.SupportsTraining;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// Delegates to the inner layer's capability.
    /// </summary>
    public override bool SupportsGpuTraining =>
        _innerLayer is LayerBase<T> innerBase && innerBase.SupportsGpuTraining;

    /// <summary>
    /// GPU-resident power iteration vectors.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _uGpu;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _vGpu;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpectralNormalizationLayer{T}"/> class.
    /// </summary>
    /// <param name="innerLayer">The layer whose weights will be spectrally normalized.</param>
    /// <param name="powerIterations">The number of power iterations to perform. Default is 1.</param>
    public SpectralNormalizationLayer(ILayer<T> innerLayer, int powerIterations = 1)
        : base(innerLayer.GetInputShape(), innerLayer.GetOutputShape())
    {
        _innerLayer = innerLayer;
        _powerIterations = powerIterations;
        _epsilon = NumOps.FromDouble(1e-12);

        // Built HERE rather than on the first forward, whenever the inner layer can already say how
        // many weights it has. A buffer is registered only if it is non-null, so leaving these until
        // the first forward meant a freshly constructed layer had no slot for them -- and a restore
        // therefore had nowhere to put the saved vectors and silently kept its own random ones. With
        // a single power iteration by default the norm depends heavily on where it starts, so that
        // is the difference between a reloaded model and the one that was saved.
        SeedPowerIteration();
    }

    /// <summary>
    /// Builds the iteration vectors and refines them once, so sigma is meaningful from the start.
    /// </summary>
    /// <remarks>
    /// Power iteration begins at a RANDOM vector, and u^T W v on a random pair is an arbitrary
    /// bilinear form -- near zero or negative as easily as not -- so dividing by it does not
    /// normalize anything. In eval the vectors are deliberately frozen, which means a layer used for
    /// inference before it ever trained divided its weights by that arbitrary number: the output
    /// swung between -1.99 and 3.15 across a serialize round trip purely on which random pair each
    /// instance drew. PyTorch leaves them random until the first training forward and inherits the
    /// same hole; seeding here closes it.
    /// </remarks>
    private void SeedPowerIteration()
    {
        if (_innerLayer is not LayerBase<T> innerBase) return;

        Tensor<T>? weight = null;
        try
        {
            var tensors = innerBase.GetTrainableParameters();
            for (int i = 0; i < tensors.Count; i++)
            {
                if (tensors[i] is { } candidate && candidate.Shape.Length >= 2) { weight = candidate; break; }
            }
        }
        catch (Exception)
        {
            // A lazy inner layer cannot be asked yet; the first forward seeds it instead.
            return;
        }

        if (weight is null || weight.Length == 0) return;

        int rows = weight.Shape[0];
        int cols = weight.Length / rows;
        EnsurePowerIterationVectors(rows, cols);
        RefinePowerIterationVectors(
            weight.Shape.Length == 2 ? weight : Engine.Reshape(weight, [rows, cols]), force: true);
    }

    /// <summary>Weights the inner layer holds, or zero while it cannot yet say.</summary>
    private int InnerWeightCount()
    {
        try
        {
            int paramCount = _innerLayer.GetParameters().Length;
            return paramCount == 0 ? 0 : paramCount - GetBiasCount(paramCount);
        }
        catch (Exception)
        {
            // A lazy inner layer that has not resolved cannot be asked yet; the first forward will
            // build the vectors as before.
            return 0;
        }
    }

    /// <summary>
    /// Normalizes a vector tensor in-place using Engine operations.
    /// </summary>
    private void NormalizeVector(ref Tensor<T> vector)
    {
        // === Vectorized L2 normalization using IEngine (Phase B: US-GPU-015) ===
        var squared = Engine.TensorMultiply(vector, vector);
        T sumSquared = Engine.TensorSum(squared);
        T norm = NumOps.Sqrt(sumSquared);
        T normPlusEps = NumOps.Add(norm, _epsilon);

        // Vectorized division by scalar
        vector = Engine.TensorDivideScalar(vector, normPlusEps);
    }

    /// <summary>
    /// Initializes or reinitializes the power iteration vectors when dimensions change.
    /// </summary>
    private void EnsurePowerIterationVectors(int rows, int cols)
    {
        if (_u is null || _v is null || _u.Shape[0] != rows || _v.Shape[0] != cols)
        {
            var u = Engine.TensorRandomUniformRange<T>([rows], NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
            var v = Engine.TensorRandomUniformRange<T>([cols], NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
            NormalizeVector(ref u);
            NormalizeVector(ref v);
            _u = u;
            _v = v;
        }
    }

    /// <summary>
    /// Estimates the number of bias parameters, if present.
    /// </summary>
    private int GetBiasCount(int paramCount)
    {
        if (OutputShape.Length == 0)
        {
            return 0;
        }

        int biasCount = OutputShape[0];
        if (biasCount <= 0 || biasCount >= paramCount)
        {
            return 0;
        }

        return biasCount;
    }

    /// <summary>
    /// Computes the spectral norm using power iteration with vectorized operations.
    /// </summary>
    private T ComputeSpectralNorm(Tensor<T> weights)
    {
        // weights shape: [outputSize, inputSize]
        int outputSize = weights.Shape[0];
        int inputSize = weights.Shape[1];

        var u = _u ?? throw new InvalidOperationException("Power iteration vector u has not been initialized.");
        var v = _v ?? throw new InvalidOperationException("Power iteration vector v has not been initialized.");

        // Power iteration using vectorized matrix operations
        for (int iter = 0; iter < _powerIterations; iter++)
        {
            // v = W^T @ u, then normalize
            // W^T shape: [inputSize, outputSize]
            var wT = Engine.TensorTranspose(weights);

            // Reshape u for matrix multiplication: [outputSize] -> [outputSize, 1]
            var uReshaped = u.Reshape(outputSize, 1);

            // v_new = W^T @ u: [inputSize, outputSize] @ [outputSize, 1] -> [inputSize, 1]
            var vNew = Engine.TensorMatMul(wT, uReshaped);
            v = vNew.Reshape(inputSize);
            NormalizeVector(ref v);

            // u = W @ v, then normalize
            // Reshape v for matrix multiplication: [inputSize] -> [inputSize, 1]
            var vReshaped = v.Reshape(inputSize, 1);

            // u_new = W @ v: [outputSize, inputSize] @ [inputSize, 1] -> [outputSize, 1]
            var uNew = Engine.TensorMatMul(weights, vReshaped);
            u = uNew.Reshape(outputSize);
            NormalizeVector(ref u);
        }

        // Only update u and v during training — inference should be deterministic
        if (IsTrainingMode)
        {
            _u = u;
            _v = v;
        }

        // Compute spectral norm: u^T @ W @ v
        var vReshaped2 = v.Reshape(inputSize, 1);
        var Wv = Engine.TensorMatMul(weights, vReshaped2).Reshape(outputSize);

        // Dot product u^T @ Wv using Engine.DotProduct
        T spectralNorm = Engine.DotProduct(u.ToVector(), Wv.ToVector());

        return spectralNorm;
    }

    /// <summary>
    /// Performs the forward pass through the layer with spectrally normalized weights.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        // The normalization runs on the inner layer's LIVE weight tensor and the quotient is bound
        // back in its place, which is how PyTorch's spectral_norm parametrization works. The previous
        // form copied the weights into a Vector, divided the numbers with NumOps and wrote them back
        // through SetParameters, so the tape never saw the division: sigma depends on W, and the
        // analytical gradient was missing that dependence entirely while finite differences measured
        // it. It also meant the layer mutated the module it wraps.
        if (_innerLayer is not LayerBase<T> innerBase)
        {
            var passthrough = _innerLayer.Forward(input);
            _lastOutput = passthrough;
            return passthrough;
        }

        // A lazy Dense/Convolution layer may expose a rank-two placeholder whose first dimension is
        // zero until it sees an input. Resolve it from the real input before inspecting the weight
        // shape; otherwise `weight.Length / weight.Shape[0]` divides by zero during the wrapper's
        // first forward (and therefore during clone verification too).
        if (!innerBase.IsShapeResolved)
        {
            var inputShape = input.Shape.ToArray();
            try
            {
                innerBase.ResolveFromShape(inputShape);
            }
            catch (Exception) when (inputShape.Length > 1)
            {
                try { innerBase.ResolveFromShape(inputShape.Skip(1).ToArray()); }
                catch (Exception) { /* The unnormalized first forward below can materialize it. */ }
            }
        }

        var tensors = innerBase.GetTrainableParameters();
        int weightIndex = -1;
        for (int i = 0; i < tensors.Count; i++)
        {
            if (tensors[i] is { } candidate && candidate.Shape.Length >= 2
                && candidate.Shape[0] > 0 && candidate.Length > 0)
            {
                weightIndex = i;
                break;
            }
        }

        if (weightIndex < 0)
        {
            var unnormalized = _innerLayer.Forward(input);
            _lastOutput = unnormalized;
            return unnormalized;
        }

        var weight = tensors[weightIndex];
        int rows = weight.Shape[0];
        int cols = weight.Length / rows;
        var matrix = weight.Shape.Length == 2 ? weight : Engine.Reshape(weight, [rows, cols]);

        EnsurePowerIterationVectors(rows, cols);

        // Power iteration refines u and v from the CURRENT weights and, per the paper and every
        // reference implementation, contributes no gradient of its own: it is an estimate of the
        // singular vectors, not a function being differentiated. Detaching keeps sigma's gradient to
        // the weight alone. Updated only while training, so inference is reproducible.
        RefinePowerIterationVectors(matrix);

        var u = _u ?? throw new InvalidOperationException("Power iteration vector u has not been initialized.");
        var v = _v ?? throw new InvalidOperationException("Power iteration vector v has not been initialized.");

        // sigma = u^T W v, built from the live weight so the tape carries d(sigma)/dW.
        var wv = Engine.TensorMatMul(matrix, Engine.Reshape(v, [cols, 1]));          // [rows, 1]
        var sigma = Engine.TensorMatMul(Engine.Reshape(u, [1, rows]), wv);           // [1, 1]

        var epsilon = new Tensor<T>([1, 1]);
        epsilon[0, 0] = _epsilon;
        var denominator = Engine.TensorAdd(sigma, epsilon);

        // TensorDivide broadcasts on its own since AiDotNet.Tensors #919, so the explicit
        // Broadcast* variant is the older spelling of the same operation.
        var normalizedMatrix = Engine.TensorDivide(matrix, denominator);
        var normalizedWeight = weight.Shape.Length == 2
            ? normalizedMatrix
            : Engine.Reshape(normalizedMatrix, weight.Shape.ToArray());

        var rebound = new Tensor<T>[tensors.Count];
        for (int i = 0; i < tensors.Count; i++) rebound[i] = tensors[i];
        rebound[weightIndex] = normalizedWeight;

        var originals = new Tensor<T>[tensors.Count];
        for (int i = 0; i < tensors.Count; i++) originals[i] = tensors[i];

        innerBase.SetTrainableParameters(rebound);
        try
        {
            _lastOutput = _innerLayer.Forward(input);
            return _lastOutput;
        }
        finally
        {
            // The wrapped layer keeps the weights it came with. Leaving the quotient bound would
            // divide them again on the next pass, which is how they used to decay pass over pass.
            innerBase.SetTrainableParameters(originals);
        }
    }

    /// <summary>Refines u and v from the current weights, outside the gradient graph.</summary>
    /// <param name="matrix">The weight matrix to iterate against.</param>
    /// <param name="force">Refine even outside training, used once at construction.</param>
    private void RefinePowerIterationVectors(Tensor<T> matrix, bool force = false)
    {
        if (!force && !IsTrainingMode) return;

        int rows = matrix.Shape[0];
        int cols = matrix.Shape[1];
        var u = _u;
        var v = _v;
        if (u is null || v is null) return;

        // Values only: a detached copy, so nothing here reaches the tape.
        var detached = Tensor<T>.FromVector(matrix.ToVector()).Reshape(rows, cols);
        var transposed = Engine.TensorTranspose(detached);

        for (int iteration = 0; iteration < _powerIterations; iteration++)
        {
            var next = Engine.TensorMatMul(transposed, u.Reshape(rows, 1)).Reshape(cols);
            NormalizeVector(ref next);
            v = next;

            var refreshed = Engine.TensorMatMul(detached, v.Reshape(cols, 1)).Reshape(rows);
            NormalizeVector(ref refreshed);
            u = refreshed;
        }

        _u = u;
        _v = v;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors with GPU-accelerated spectral normalization.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs spectral normalization using GPU-accelerated power iteration,
    /// keeping all computations on GPU for maximum performance.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Get weights from inner layer
        var parameters = _innerLayer.GetParameters();
        int paramCount = parameters.Length;

        if (paramCount == 0)
        {
            // No parameters to normalize, just forward through inner layer
            if (_innerLayer is LayerBase<T> innerBase)
            {
                return innerBase.ForwardGpu(input);
            }
            throw new InvalidOperationException("Inner layer does not support ForwardGpu.");
        }

        // Store original parameters to restore after forward
        _originalParameters = parameters.Clone();

        int biasCount = GetBiasCount(paramCount);
        int weightCount = paramCount - biasCount;

        // Reshape weight parameters into 2D matrix for spectral norm computation
        int rows = (int)Math.Ceiling(Math.Sqrt(weightCount));
        int cols = (weightCount + rows - 1) / rows;

        // Create weight tensor [rows, cols] with zero-padding if needed
        var weightsData = new float[rows * cols];
        for (int i = 0; i < weightCount; i++)
        {
            weightsData[i] = Convert.ToSingle(parameters[i]);
        }

        // Upload weights to GPU
        var weightsGpu = gpuEngine.UploadToGpu(new Tensor<T>(
            DirectGpuEngine.FromFloatArray<T>(weightsData), [rows, cols]), GpuTensorRole.Weight);

        // Initialize GPU power iteration vectors if needed
        EnsureGpuPowerIterationVectors(gpuEngine, rows, cols);

        // Run power iteration on GPU
        float spectralNorm = gpuEngine.PowerIterationGpu(
            weightsGpu, ref _uGpu!, ref _vGpu!, _powerIterations, Convert.ToSingle(_epsilon));

        // Normalize weight parameters by spectral norm
        var normalizedParams = new Vector<T>(paramCount);
        T normDivisor = NumOps.FromDouble(spectralNorm);
        for (int i = 0; i < weightCount; i++)
        {
            normalizedParams[i] = NumOps.Divide(parameters[i], normDivisor);
        }

        // Copy bias parameters unchanged
        for (int i = weightCount; i < paramCount; i++)
        {
            normalizedParams[i] = parameters[i];
        }

        _innerLayer.SetParameters(normalizedParams);
        _normalizedWeightsApplied = true;

        try
        {
            // Forward through inner layer with normalized weights
            if (_innerLayer is LayerBase<T> innerBase)
            {
                return innerBase.ForwardGpu(input);
            }
            throw new InvalidOperationException("Inner layer does not support ForwardGpu.");
        }
        finally
        {
            // Same reason as the traced path above: the inner weights must not stay normalized.
            RestoreOriginalWeights();
        }
    }

    /// <summary>
    /// Initializes or reinitializes the GPU power iteration vectors when dimensions change.
    /// </summary>
    private void EnsureGpuPowerIterationVectors(DirectGpuTensorEngine gpuEngine, int rows, int cols)
    {
        if (_uGpu is null || _vGpu is null || _uGpu.Shape[0] != rows || _vGpu.Shape[0] != cols)
        {
            // Create random normalized vectors on CPU, then upload to GPU
            var uData = new float[rows];
            var vData = new float[cols];
            var random = RandomHelper.CreateSecureRandom();

            // Initialize with random values
            float uNorm = 0, vNorm = 0;
            for (int i = 0; i < rows; i++)
            {
                uData[i] = (float)(random.NextDouble() * 2 - 1);
                uNorm += uData[i] * uData[i];
            }
            for (int i = 0; i < cols; i++)
            {
                vData[i] = (float)(random.NextDouble() * 2 - 1);
                vNorm += vData[i] * vData[i];
            }

            // Normalize
            uNorm = (float)Math.Sqrt(uNorm);
            vNorm = (float)Math.Sqrt(vNorm);
            for (int i = 0; i < rows; i++) uData[i] /= uNorm;
            for (int i = 0; i < cols; i++) vData[i] /= vNorm;

            // Upload to GPU
            _uGpu = gpuEngine.UploadToGpu(new Tensor<T>(
                DirectGpuEngine.FromFloatArray<T>(uData), [rows]), GpuTensorRole.Activation);
            _vGpu = gpuEngine.UploadToGpu(new Tensor<T>(
                DirectGpuEngine.FromFloatArray<T>(vData), [cols]), GpuTensorRole.Activation);
        }
    }

    /// <summary>
    /// Restores the original weights after Backward or on exception.
    /// </summary>
    private void RestoreOriginalWeights()
    {
        if (_normalizedWeightsApplied && _originalParameters != null)
        {
            _innerLayer.SetParameters(_originalParameters);
            _normalizedWeightsApplied = false;
            _originalParameters = null;
        }
    }

    /// <summary>
    /// Updates the parameters of the inner layer.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets the parameter gradients from the inner layer.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        return _innerLayer.GetParameterGradients();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        RestoreOriginalWeights();
        _innerLayer.ResetState();
    }

    /// <summary>
    /// GPU-resident parameter update using the provided optimizer configuration.
    /// Delegates to the inner layer's UpdateParametersGpu method.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        // Delegate to inner layer's GPU parameter update
        if (_innerLayer is LayerBase<T> innerBase && innerBase.SupportsGpuTraining)
        {
            innerBase.UpdateParametersGpu(config);
        }
        else
        {
            // Fall back to CPU parameter update if inner layer doesn't support GPU training
            throw new InvalidOperationException(
                $"Inner layer ({_innerLayer.GetType().Name}) does not support GPU-resident training. " +
                "Use UpdateParameters() for CPU-based parameter updates.");
        }
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _innerLayer.ClearGradients();
    }

    /// <summary>
    /// Persists the inner layer's type name + shape and the
    /// power-iteration count so DeserializationHelper can reconstruct
    /// the wrapped layer concretely. Issue #1239 wrapped-layer round-trip.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InnerLayerTypeName"] = _innerLayer.GetType().Name;
        metadata["InnerLayerInputShape"] = string.Join(",", _innerLayer.GetInputShape());
        metadata["InnerLayerOutputShape"] = string.Join(",", _innerLayer.GetOutputShape());
        metadata["PowerIterations"] = _powerIterations.ToString();
        return metadata;
    }
}
