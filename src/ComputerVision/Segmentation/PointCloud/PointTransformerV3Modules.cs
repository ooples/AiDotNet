using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Owns the parameter tensors of a Point Transformer V3 network and hands them to its modules by
/// slot, so a module never caches a tensor the owning layer may later rebind (clone, load,
/// copy-on-write sharing).
/// </summary>
internal sealed class PtContext<T>
{
    private readonly List<Tensor<T>> _created = new();
    private readonly List<PersistentTensorRole> _roles = new();
    private readonly List<(Tensor<T> Tensor, string Name)> _buffers = new();
    private readonly bool _truncatedNormalInit;

    internal PtContext(Random random, bool truncatedNormalInit)
    {
        Random = random ?? throw new ArgumentNullException(nameof(random));
        _truncatedNormalInit = truncatedNormalInit;
    }

    internal Random Random { get; }

    internal IEngine Engine => AiDotNetEngine.Current;

    internal INumericOperations<T> NumOps { get; } = MathHelper.GetNumericOperations<T>();

    /// <summary>The live parameter list of the owner; null while the network is being built.</summary>
    internal Func<IReadOnlyList<Tensor<T>>>? Live { get; set; }

    internal IReadOnlyList<Tensor<T>> Created => _created;

    internal IReadOnlyList<PersistentTensorRole> Roles => _roles;

    internal IReadOnlyList<(Tensor<T> Tensor, string Name)> Buffers => _buffers;

    internal Tensor<T> this[int slot] => Live is null ? _created[slot] : Live()[slot];

    internal int Add(Tensor<T> tensor, PersistentTensorRole role)
    {
        _created.Add(tensor);
        _roles.Add(role);
        return _created.Count - 1;
    }

    internal void AddBuffer(Tensor<T> tensor, string name) => _buffers.Add((tensor, name));

    internal int NextSeed() => Random.Next();

    /// <summary>
    /// PTv3 m1 keeps PyTorch's default <c>nn.Linear</c> / spconv initialisation, U(-1/sqrt(fan_in),
    /// 1/sqrt(fan_in)); the Sonata backbone (m2) and its heads use <c>trunc_normal_(std=0.02)</c>
    /// with zero biases.
    /// </summary>
    internal void InitializeWeight(Tensor<T> weight, int fanIn)
    {
        if (_truncatedNormalInit)
        {
            // trunc_normal_ cuts at +-2 absolute; at std 0.02 that is 100 sigma, so a plain normal is the same draw.
            Engine.TensorNormalInPlace(weight, 0.0, 0.02, NextSeed());
            return;
        }

        double bound = 1.0 / Math.Sqrt(Math.Max(1, fanIn));
        Engine.TensorUniformInPlace(weight, -bound, bound, NextSeed());
    }

    internal void InitializeBias(Tensor<T> bias, int fanIn)
    {
        if (_truncatedNormalInit) return;
        double bound = 1.0 / Math.Sqrt(Math.Max(1, fanIn));
        Engine.TensorUniformInPlace(bias, -bound, bound, NextSeed());
    }
}

/// <summary><c>nn.Linear</c>: <c>y = x W + b</c> with <c>W</c> stored <c>[in, out]</c>.</summary>
internal sealed class PtLinear<T>
{
    private readonly PtContext<T> _ctx;
    private readonly int _weight;
    private readonly int _bias;

    internal PtLinear(PtContext<T> ctx, int inputSize, int outputSize, bool bias = true, int? fanIn = null)
    {
        _ctx = ctx;
        InputSize = inputSize;
        OutputSize = outputSize;
        int fan = fanIn ?? inputSize;
        var w = new Tensor<T>(new[] { inputSize, outputSize });
        ctx.InitializeWeight(w, fan);
        _weight = ctx.Add(w, PersistentTensorRole.Weights);
        _bias = -1;
        if (bias)
        {
            var b = new Tensor<T>(new[] { outputSize });
            ctx.InitializeBias(b, fan);
            _bias = ctx.Add(b, PersistentTensorRole.Biases);
        }
    }

    internal int InputSize { get; }

    internal int OutputSize { get; }

    internal Tensor<T> Weight => _ctx[_weight];

    internal Tensor<T>? Bias => _bias >= 0 ? _ctx[_bias] : null;

    internal Tensor<T> Forward(Tensor<T> x) => _ctx.Engine.FusedLinear(x, Weight, Bias, FusedActivationType.None);
}

/// <summary>A per-point normalisation over the channel axis.</summary>
internal abstract class PtNorm<T>
{
    internal abstract Tensor<T> Forward(Tensor<T> x, bool training);
}

/// <summary><c>nn.LayerNorm(channels)</c>, eps 1e-5.</summary>
internal sealed class PtLayerNorm<T> : PtNorm<T>
{
    private readonly PtContext<T> _ctx;
    private readonly int _gamma;
    private readonly int _beta;

    internal PtLayerNorm(PtContext<T> ctx, int channels)
    {
        _ctx = ctx;
        var gamma = new Tensor<T>(new[] { channels });
        gamma.Fill(ctx.NumOps.One);
        _gamma = ctx.Add(gamma, PersistentTensorRole.NormalizationParams);
        _beta = ctx.Add(new Tensor<T>(new[] { channels }), PersistentTensorRole.NormalizationParams);
    }

    internal override Tensor<T> Forward(Tensor<T> x, bool training)
        => _ctx.Engine.TensorLayerNorm(x, _ctx[_gamma], _ctx[_beta], 1e-5);
}

/// <summary>
/// <c>nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)</c> over the points, as PTv3 m1 uses in its
/// stem, pooling and unpooling.
/// </summary>
internal sealed class PtBatchNorm<T> : PtNorm<T>
{
    private const double Epsilon = 1e-3;
    private const double Momentum = 0.01;
    private readonly PtContext<T> _ctx;
    private readonly int _gamma;
    private readonly int _beta;
    private readonly Tensor<T> _runningMean;
    private readonly Tensor<T> _runningVariance;

    internal PtBatchNorm(PtContext<T> ctx, int channels, string name)
    {
        _ctx = ctx;
        var gamma = new Tensor<T>(new[] { channels });
        gamma.Fill(ctx.NumOps.One);
        _gamma = ctx.Add(gamma, PersistentTensorRole.NormalizationParams);
        _beta = ctx.Add(new Tensor<T>(new[] { channels }), PersistentTensorRole.NormalizationParams);
        _runningMean = new Tensor<T>(new[] { channels });
        _runningVariance = new Tensor<T>(new[] { channels });
        _runningVariance.Fill(ctx.NumOps.One);
        ctx.AddBuffer(_runningMean, name + ".running_mean");
        ctx.AddBuffer(_runningVariance, name + ".running_var");
    }

    internal override Tensor<T> Forward(Tensor<T> x, bool training)
    {
        var engine = _ctx.Engine;
        var ops = _ctx.NumOps;
        int count = x.Shape[0];
        if (training && count > 1)
        {
            var output = engine.BatchNorm(x, _ctx[_gamma], _ctx[_beta], Epsilon, out var mean, out var variance);
            using (new NoGradScope<T>())
            {
                // PyTorch: running = (1 - momentum) * running + momentum * batch, with the unbiased variance.
                engine.TensorMultiplyScalarInPlace(_runningMean, ops.FromDouble(1.0 - Momentum));
                engine.TensorAddInPlace(_runningMean, engine.TensorMultiplyScalar(mean, ops.FromDouble(Momentum)));
                engine.TensorMultiplyScalarInPlace(_runningVariance, ops.FromDouble(1.0 - Momentum));
                double unbiased = Momentum * count / (count - 1.0);
                engine.TensorAddInPlace(_runningVariance, engine.TensorMultiplyScalar(variance, ops.FromDouble(unbiased)));
            }

            return output;
        }

        // Inference (and a single-point training batch, whose variance is undefined): running statistics.
        var shape = x.Shape.ToArray();
        var std = engine.TensorSqrt(engine.TensorAddScalar(_runningVariance, ops.FromDouble(Epsilon)));
        var scale = engine.TensorDivide(_ctx[_gamma], std);
        var shift = engine.TensorSubtract(_ctx[_beta], engine.TensorMultiply(_runningMean, scale));
        var scaled = engine.TensorMultiply(x, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { 1, shape[1] }), shape));
        return engine.TensorAdd(scaled, engine.TensorBroadcastTo(engine.Reshape(shift, new[] { 1, shape[1] }), shape));
    }
}

/// <summary>
/// spconv's <c>SubMConv3d</c>: a cubic sparse convolution evaluated only at occupied voxels, as a
/// gather of the <c>k^3</c> neighbours followed by one matrix multiply.
/// </summary>
internal sealed class PtSubmanifoldConv<T>
{
    private readonly PtContext<T> _ctx;
    private readonly PtLinear<T> _kernel;

    internal PtSubmanifoldConv(PtContext<T> ctx, int inputChannels, int outputChannels, int kernelSize, bool bias)
    {
        _ctx = ctx;
        KernelSize = kernelSize;
        InputChannels = inputChannels;
        int volume = kernelSize * kernelSize * kernelSize;
        _kernel = new PtLinear<T>(ctx, volume * inputChannels, outputChannels, bias, volume * inputChannels);
    }

    internal int KernelSize { get; }

    internal int InputChannels { get; }

    internal Tensor<T> Forward(PointCloudLevel<T> level, Tensor<T> x)
    {
        var engine = _ctx.Engine;
        int count = level.Count;
        int volume = KernelSize * KernelSize * KernelSize;
        var table = level.NeighbourTable(KernelSize);
        // Row `count` is the zero row empty voxels gather.
        var padded = engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { 1, InputChannels }) }, 0);
        var gathered = engine.TensorIndexSelect(padded, new Tensor<int>(table, new[] { table.Length }), 0);
        var patches = engine.Reshape(gathered, new[] { count, volume * InputChannels });
        return _kernel.Forward(patches);
    }
}

/// <summary>Shared tensor helpers for the PTv3 modules.</summary>
internal static class PtOps
{
    internal static Tensor<int> Indices(int[] values) => new(values, new[] { values.Length });

    /// <summary>
    /// timm's <c>DropPath</c> on a <c>[points, channels]</c> feature: Pointcept applies it to
    /// <c>point.feat</c>, so each POINT's residual branch is dropped with probability <paramref name="rate"/>
    /// and the kept ones are scaled by <c>1 / (1 - rate)</c>.
    /// </summary>
    internal static Tensor<T> DropPath<T>(PtContext<T> ctx, Tensor<T> x, double rate, bool training)
    {
        if (!training || rate <= 0.0) return x;
        var engine = ctx.Engine;
        double keep = 1.0 - rate;
        var shape = x.Shape.ToArray();
        var mask = new Tensor<T>(new[] { shape[0], 1 });
        engine.TensorBernoulliInPlace(mask, keep, ctx.NextSeed());
        var scaled = engine.TensorMultiplyScalar(mask, ctx.NumOps.FromDouble(1.0 / keep));
        return engine.TensorMultiply(x, engine.TensorBroadcastTo(scaled, shape));
    }

    /// <summary>Row-wise broadcast of a <c>[1, C]</c> or <c>[N, 1]</c> tensor to <paramref name="shape"/>.</summary>
    internal static Tensor<T> Broadcast<T>(IEngine engine, Tensor<T> x, int[] shape) => engine.TensorBroadcastTo(x, shape);
}
