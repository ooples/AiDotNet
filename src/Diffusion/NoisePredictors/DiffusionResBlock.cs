#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Implements a residual block per the DDPM (Ho et al. 2020) and Stable Diffusion (Rombach et al. 2022)
/// U-Net architecture with time embedding conditioning.
/// </summary>
/// <remarks>
/// <para>
/// The forward pass implements the following computation:
/// <code>
///   h = GroupNorm(x) → SiLU → Conv3x3              (first conv block)
///   h = h + time_mlp(time_embed)                     (time conditioning)
///   h = GroupNorm(h) → SiLU → Conv3x3              (second conv block)
///   out = h + skip_conv(x)                           (residual connection)
/// </code>
/// where <c>skip_conv</c> is a 1x1 convolution if <c>inChannels != outChannels</c>, otherwise identity.
/// </para>
/// <para>
/// Performance: all intermediate tensors use <see cref="TensorAllocator"/> for pooled allocation.
/// GroupNorm uses 32 groups (SD standard) with channels that aren't divisible by 32 falling back
/// to the largest divisor.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DiffusionResBlock<T> : LayerBase<T>
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _spatialSize;
    private readonly int _timeEmbedDim;

    // First conv block: GroupNorm → SiLU → Conv3x3
    private readonly GroupNormalizationLayer<T> _norm1;
    private readonly ConvolutionalLayer<T> _conv1;

    // Time embedding projection: Linear(timeEmbedDim → outChannels)
    private readonly DenseLayer<T> _timeMlp;

    // Second conv block: GroupNorm → SiLU → Conv3x3
    private readonly GroupNormalizationLayer<T> _norm2;
    private readonly ConvolutionalLayer<T> _conv2;

    // Skip connection: 1x1 conv if channels differ, null for identity
    private readonly ConvolutionalLayer<T>? _skipConv;

    private readonly SiLUActivation<T> _silu = new();

    // Cache for backward
    private Tensor<T>? _lastInput;
    private Tensor<T>? _preSiLU1;   // norm1 output (before SiLU)
    private Tensor<T>? _preSiLU2;   // norm2 output (before SiLU)
    private bool _lastForwardUsedTime; // whether last forward was time-conditioned
    private int[]? _originalInputShape;

    // Pre-allocated reusable buffers for the eval-mode fused fast path. The
    // unfused two-step (Forward GroupNorm → ApplySiLU) allocates two ~10 MB
    // tensors per ResBlock at SD 320×64×64 shapes; pooling these eliminates
    // ~40 MB of per-ResBlock allocation churn (≈22 ResBlocks per UNet
    // forward → ~880 MB removed from each Predict's GC pressure).
    private Tensor<T>? _preAllocatedNorm1Out;
    private Tensor<T>? _preAllocatedNorm2Out;

    private static bool ShapeEquals(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override long ParameterCount =>
        _norm1.ParameterCount + _conv1.ParameterCount +
        _timeMlp.ParameterCount +
        _norm2.ParameterCount + _conv2.ParameterCount +
        (_skipConv?.ParameterCount ?? 0);

    /// <summary>
    /// Creates a new diffusion residual block per the DDPM/Stable Diffusion paper.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="spatialSize">Spatial size (height = width) of feature maps at this level.</param>
    /// <param name="timeEmbedDim">Dimension of the time embedding vector. Default: 0 (no time conditioning).</param>
    /// <param name="numGroups">Number of groups for GroupNorm. Default: 32 (SD standard).</param>
    public DiffusionResBlock(
        int inChannels,
        int outChannels,
        int spatialSize,
        int timeEmbedDim = 0,
        int numGroups = 32)
        : base(
            [1, inChannels, spatialSize, spatialSize],
            [1, outChannels, spatialSize, spatialSize])
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _spatialSize = spatialSize;
        _timeEmbedDim = timeEmbedDim;

        // Compute actual group count: SD uses 32, but fall back to largest divisor if needed
        int groups1 = ComputeNumGroups(inChannels, numGroups);
        int groups2 = ComputeNumGroups(outChannels, numGroups);

        // First block: GroupNorm(in) → SiLU → Conv3x3(in→out). Pass the Lazy strategy
        // so kernel tensors stay unallocated until the first Forward() call — diffusion
        // U-Nets contain dozens of these blocks and eager allocation OOMs CI.
        _norm1 = new GroupNormalizationLayer<T>(groups1, inChannels);
        _conv1 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>(),
            initializationStrategy: InitializationStrategies<T>.Lazy);

        // Time embedding projection: projects time embed to outChannels for additive conditioning
        _timeMlp = new DenseLayer<T>(
            outChannels,
            (IActivationFunction<T>)new SiLUActivation<T>(),
            InitializationStrategies<T>.Lazy);

        // Second block: GroupNorm(out) → SiLU → Conv3x3(out→out)
        _norm2 = new GroupNormalizationLayer<T>(groups2, outChannels);
        _conv2 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>(),
            initializationStrategy: InitializationStrategies<T>.Lazy);

        // Skip connection: 1x1 conv if channels differ
        if (inChannels != outChannels)
        {
            _skipConv = new ConvolutionalLayer<T>(
                outputDepth: outChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>(),
                initializationStrategy: InitializationStrategies<T>.Lazy);
        }
    }

    /// <summary>
    /// Forward pass implementing the DDPM residual block.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input._shape;
        _lastInput = input;
        _lastForwardUsedTime = false;

        // First block: GroupNorm → SiLU → Conv3x3
        var h = _norm1.Forward(input);
        _preSiLU1 = h;
        h = ApplySiLU(h);
        h = _conv1.Forward(h);

        // Skip connection
        var residual = _skipConv is not null ? _skipConv.Forward(input) : input;

        // Second block: GroupNorm → SiLU → Conv3x3
        h = _norm2.Forward(h);
        _preSiLU2 = h;
        h = ApplySiLU(h);
        h = _conv2.Forward(h);

        // Add residual
        h = Engine.TensorAdd(h, residual);
        return h;
    }

    /// <summary>
    /// Forward pass with time embedding conditioning per the DDPM paper.
    /// </summary>
    /// <param name="input">Input tensor [B, C, H, W].</param>
    /// <param name="timeEmbed">Time embedding [B, timeEmbedDim].</param>
    /// <returns>Output tensor [B, outChannels, H, W].</returns>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> timeEmbed)
    {
        _originalInputShape = input._shape;
        _lastInput = input;
        _lastForwardUsedTime = true;

        // Eval-mode (no tape) can use the in-place Engine variants for the
        // two element-wise adds in this block (broadcast-add of timeProj and
        // residual-add). Each switched call eliminates one ~10 MB tensor
        // allocation per ResBlock at SD shapes (320×64×64 doubles), and a
        // CatVTON Predict has ~22 ResBlocks per UNet forward — so ~440 MB
        // of allocation drops per Predict, cutting Gen0/Gen1 GC frequency.
        // Tape-active mode keeps the allocating variants because the tape
        // backward needs the pre-add tensor to recover gradients.
        bool useInPlace = AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is null
                          || AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed;

        Tensor<T> h;
        if (useInPlace && input.Rank == 4)
        {
            // Fused fast path: GroupNorm + SiLU + Conv3x3 with the
            // GroupNorm+SiLU result going straight into a pooled buffer.
            // This replaces the 2-step (norm.Forward → ApplySiLU) which
            // allocated 2 fresh ~10 MB tensors at SD shapes (320×64×64
            // doubles). Lazy-allocate the buffer at the input shape —
            // GroupNorm preserves NCHW dims so the SD UNet ResBlock's
            // first-norm output is always input-shaped.
            if (_preAllocatedNorm1Out is null
                || !ShapeEquals(_preAllocatedNorm1Out._shape, input._shape))
            {
                _preAllocatedNorm1Out = new Tensor<T>(input._shape);
            }
            h = _preAllocatedNorm1Out;
            Engine.GroupNormSwishInto(
                h, input,
                _norm1.NumGroups, _norm1.GetGammaTensor(), _norm1.GetBetaTensor(),
                Convert.ToDouble(_norm1.GetEpsilon(), System.Globalization.CultureInfo.InvariantCulture));
            // _preSiLU1 is only consumed by Backward; eval-mode skips it.
            _preSiLU1 = null;
            h = _conv1.Forward(h);
        }
        else
        {
            // Tape-active path: keep the allocating Forward chain so
            // backward can recover the pre-SiLU tensor.
            h = _norm1.Forward(input);
            _preSiLU1 = h;
            h = ApplySiLU(h);
            h = _conv1.Forward(h);
        }

        // Time conditioning: project time embed and add to feature maps.
        // Every op must go through Engine so the gradient tape records the
        // graph — direct Tensor<T>.Reshape / BroadcastAdd calls bypass the
        // tape and snap the chain between norm1/conv1 and the rest of the
        // block, which is exactly what caused diffusion tape training to
        // return all-zero gradients on the UNet backbone.
        if (_timeEmbedDim > 0 && timeEmbed.Length > 0)
        {
            // Validate the timeEmbed contract before letting the lazy MLP resolve its
            // input feature dim from whatever shape happens to come in. _timeMlp is
            // a lazy DenseLayer constructed without an input-feature size, so the
            // FIRST call to its Forward() bakes the input dim from timeEmbed's last
            // axis. If that's wrong we'd silently mis-shape the entire block.
            if (timeEmbed.Shape.Length < 2 ||
                timeEmbed.Shape[timeEmbed.Shape.Length - 1] != _timeEmbedDim)
            {
                throw new ArgumentException(
                    $"timeEmbed must have rank >= 2 and last dim == _timeEmbedDim ({_timeEmbedDim}). " +
                    $"Got rank {timeEmbed.Shape.Length}, last dim {timeEmbed.Shape[timeEmbed.Shape.Length - 1]}.",
                    nameof(timeEmbed));
            }
            var timeProj = _timeMlp.Forward(timeEmbed);
            // Reshape from [B, outChannels] to [B, outChannels, 1, 1] for broadcasting
            if (timeProj.Shape.Length == 1)
            {
                timeProj = Engine.Reshape(timeProj, new[] { 1, _outChannels, 1, 1 });
            }
            else if (timeProj.Shape.Length == 2)
            {
                timeProj = Engine.Reshape(timeProj, new[] { timeProj.Shape[0], _outChannels, 1, 1 });
            }
            if (useInPlace)
                Engine.TensorBroadcastAddInPlace(h, timeProj);
            else
                h = Engine.TensorBroadcastAdd(h, timeProj);
        }

        // Skip connection. Identity case (no skipConv) is alloc-free; the
        // 1×1 conv case pools its own output via
        // ConvolutionalLayer._preAllocatedOutput so no extra work needed.
        var residual = _skipConv is not null ? _skipConv.Forward(input) : input;

        // Second block: GroupNorm → SiLU → Conv3x3 — eval-mode uses the
        // same fused-into-pooled-buffer fast path as norm1, sized to the
        // post-conv1 tensor.
        if (useInPlace && h.Rank == 4)
        {
            if (_preAllocatedNorm2Out is null
                || !ShapeEquals(_preAllocatedNorm2Out._shape, h._shape))
            {
                _preAllocatedNorm2Out = new Tensor<T>(h._shape);
            }
            var n2 = _preAllocatedNorm2Out;
            Engine.GroupNormSwishInto(
                n2, h,
                _norm2.NumGroups, _norm2.GetGammaTensor(), _norm2.GetBetaTensor(),
                Convert.ToDouble(_norm2.GetEpsilon(), System.Globalization.CultureInfo.InvariantCulture));
            _preSiLU2 = null;
            h = n2;
            h = _conv2.Forward(h);
        }
        else
        {
            h = _norm2.Forward(h);
            _preSiLU2 = h;
            h = ApplySiLU(h);
            h = _conv2.Forward(h);
        }

        // Residual add — in place when no tape is recording, allocating
        // otherwise (the tape backward needs the pre-add value).
        if (useInPlace && residual._shape.Length == h._shape.Length)
        {
            bool sameShape = true;
            for (int d = 0; d < h._shape.Length; d++)
                if (h._shape[d] != residual._shape[d]) { sameShape = false; break; }
            if (sameShape)
            {
                Engine.TensorAddInPlace(h, residual);
                return h;
            }
        }
        h = Engine.TensorAdd(h, residual);
        return h;
    }

    // Stores time embed gradient for collection by UNet backward
    private Tensor<T>? _timeEmbedGradient;

    /// <summary>
    /// Gets the accumulated time embedding gradient from the last backward pass.
    /// Called by UNetNoisePredictor to propagate gradients through the time MLP.
    /// </summary>
    internal Tensor<T>? GetTimeEmbedGradient() => _timeEmbedGradient;

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        return Engine.Swish(x);
    }

    /// <summary>
    /// SiLU/Swish backward: d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    /// = sigmoid(x) * (1 + x*(1-sigmoid(x)))
    /// </summary>
    private Tensor<T> ApplySiLUBackward(Tensor<T> preSiLU, Tensor<T> gradOutput)
    {
        var sig = Engine.TensorSigmoid(preSiLU);
        var oneMinusSig = Engine.TensorSubtract(
            Engine.TensorAddScalar(Engine.TensorMultiplyScalar(sig, NumOps.Zero), NumOps.One),
            sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(preSiLU, oneMinusSig);
        var deriv = Engine.TensorMultiply(sig,
            Engine.TensorAdd(
                Engine.TensorAddScalar(Engine.TensorMultiplyScalar(sig, NumOps.Zero), NumOps.One),
                xTimesOneMinusSig));
        return Engine.TensorMultiply(gradOutput, deriv);
    }

    /// <summary>
    /// Computes appropriate number of groups for GroupNorm.
    /// SD uses 32 groups, but we fall back to the largest divisor ≤ numGroups.
    /// </summary>
    private static int ComputeNumGroups(int channels, int targetGroups)
    {
        if (channels % targetGroups == 0)
            return targetGroups;

        // Find largest divisor ≤ targetGroups
        for (int g = targetGroups; g >= 1; g--)
        {
            if (channels % g == 0)
                return g;
        }
        return 1;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        AddParams(parameters, _norm1);
        AddParams(parameters, _conv1);
        AddParams(parameters, _timeMlp);
        AddParams(parameters, _norm2);
        AddParams(parameters, _conv2);
        if (_skipConv is not null)
            AddParams(parameters, _skipConv);
        return new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        SetParams(_norm1, parameters, ref idx);
        SetParams(_conv1, parameters, ref idx);
        SetParams(_timeMlp, parameters, ref idx);
        SetParams(_norm2, parameters, ref idx);
        SetParams(_conv2, parameters, ref idx);
        if (_skipConv is not null)
            SetParams(_skipConv, parameters, ref idx);
    }

    private static void AddParams(List<T> list, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++)
            list.Add(p[i]);
    }

    private static void SetParams(ILayer<T> layer, Vector<T> parameters, ref int idx)
    {
        var count = layer.GetParameters().Length;
        var sub = new Vector<T>(count);
        for (int i = 0; i < count && idx < parameters.Length; i++)
            sub[i] = parameters[idx++];
        layer.SetParameters(sub);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _conv1.UpdateParameters(learningRate);
        _timeMlp.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _skipConv?.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _originalInputShape = null;
        _norm1.ResetState();
        _conv1.ResetState();
        _timeMlp.ResetState();
        _norm2.ResetState();
        _conv2.ResetState();
        _skipConv?.ResetState();
    }
}
