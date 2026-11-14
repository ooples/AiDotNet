namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Sonata's <c>OnlineCluster</c> head: <c>Linear -&gt; GELU -&gt; Linear</c>, L2 normalisation, then a
/// weight-normalised bias-free projection onto unit-length prototypes (gain fixed to one).
/// </summary>
/// <remarks>
/// The weight-norm gain is frozen at one in Sonata, so each prototype is <c>v / ||v||</c> and the
/// output is the cosine similarity between the embedding and each prototype.
/// </remarks>
internal sealed class SonataPrototypeHead<T>
{
    private const double NormalizeEpsilon = 1e-12;
    private readonly PtContext<T> _ctx;
    private readonly PtLinear<T> _fc1;
    private readonly PtLinear<T> _fc2;
    private readonly int _prototypes;

    internal SonataPrototypeHead(PtContext<T> ctx, int inputChannels, int hiddenChannels, int embedChannels, int prototypes)
    {
        _ctx = ctx;
        _fc1 = new PtLinear<T>(ctx, inputChannels, hiddenChannels);
        _fc2 = new PtLinear<T>(ctx, hiddenChannels, embedChannels);
        var prototype = new Tensor<T>(new[] { embedChannels, prototypes });
        // nn.Linear's default init for the direction v; only its direction matters.
        ctx.InitializeWeight(prototype, embedChannels);
        _prototypes = ctx.Add(prototype, PersistentTensorRole.Weights);
        EmbedChannels = embedChannels;
        PrototypeCount = prototypes;
    }

    internal int EmbedChannels { get; }

    internal int PrototypeCount { get; }

    /// <summary>The L2-normalised embedding before the prototypes, for regularisers.</summary>
    internal Tensor<T> Embed(Tensor<T> features)
    {
        var engine = _ctx.Engine;
        var x = _fc2.Forward(engine.TensorGELU(_fc1.Forward(features)));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(Norm(x, 1), x.Shape.ToArray()));
    }

    /// <summary>Cosine similarity of each embedding with each prototype, <c>[points, prototypes]</c>.</summary>
    internal Tensor<T> Score(Tensor<T> embedding)
    {
        var engine = _ctx.Engine;
        var v = _ctx[_prototypes];
        var w = engine.TensorDivide(v, engine.TensorBroadcastTo(Norm(v, 0), v.Shape.ToArray()));
        return engine.TensorMatMul(embedding, w);
    }

    private Tensor<T> Norm(Tensor<T> x, int axis)
    {
        var engine = _ctx.Engine;
        return engine.TensorClampMin(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { axis }, true)),
            _ctx.NumOps.FromDouble(NormalizeEpsilon));
    }
}
