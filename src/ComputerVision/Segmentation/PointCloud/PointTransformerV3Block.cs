namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// One PTv3 block (Pointcept <c>Block</c>): xCPE, then pre-norm serialized attention and a pre-norm
/// MLP, each on a residual branch with drop path.
/// </summary>
/// <remarks>
/// <code>
/// x = x + LN(Linear(SubMConv3d_k3(x)))          // xCPE (Sec. 3.3), when enabled
/// x = x + DropPath(Attn(LN(x)))                 // serialized patch attention
/// x = x + DropPath(fc2(GELU(fc1(LN(x)))))       // MLP, hidden = mlp_ratio * C
/// </code>
/// With <c>pre_norm = false</c> the norms move after each residual sum, as in Pointcept.
/// </remarks>
internal sealed class PointTransformerV3Block<T>
{
    private readonly PtContext<T> _ctx;
    private readonly PtSubmanifoldConv<T>? _cpeConv;
    private readonly PtLinear<T>? _cpeLinear;
    private readonly PtLayerNorm<T>? _cpeNorm;
    private readonly PtLayerNorm<T> _norm1;
    private readonly PtLayerNorm<T> _norm2;
    private readonly PtLinear<T> _fc1;
    private readonly PtLinear<T> _fc2;
    private readonly double _dropPath;
    private readonly bool _preNorm;

    internal PointTransformerV3Block(PtContext<T> ctx, int channels, int numHeads, int patchSize, double mlpRatio,
        double dropPath, bool preNorm, int orderIndex, bool enableConditionalPositionalEncoding)
    {
        _ctx = ctx;
        _dropPath = dropPath;
        _preNorm = preNorm;
        OrderIndex = orderIndex;
        if (enableConditionalPositionalEncoding)
        {
            _cpeConv = new PtSubmanifoldConv<T>(ctx, channels, channels, 3, bias: true);
            _cpeLinear = new PtLinear<T>(ctx, channels, channels);
            _cpeNorm = new PtLayerNorm<T>(ctx, channels);
        }

        _norm1 = new PtLayerNorm<T>(ctx, channels);
        Attention = new SerializedPatchAttention<T>(ctx, channels, numHeads, patchSize);
        _norm2 = new PtLayerNorm<T>(ctx, channels);
        int hidden = (int)(channels * mlpRatio);
        _fc1 = new PtLinear<T>(ctx, channels, hidden);
        _fc2 = new PtLinear<T>(ctx, hidden, channels);
    }

    internal int OrderIndex { get; }

    internal SerializedPatchAttention<T> Attention { get; }

    internal Tensor<T> Forward(PointCloudLevel<T> level, Tensor<T> x, bool training)
    {
        var engine = _ctx.Engine;
        if (_cpeConv is not null && _cpeLinear is not null && _cpeNorm is not null)
        {
            var cpe = _cpeNorm.Forward(_cpeLinear.Forward(_cpeConv.Forward(level, x)), training);
            x = engine.TensorAdd(x, cpe);
        }

        var shortcut = x;
        var branch = _preNorm ? _norm1.Forward(x, training) : x;
        branch = PtOps.DropPath(_ctx, Attention.Forward(level, branch, OrderIndex), _dropPath, training);
        x = engine.TensorAdd(shortcut, branch);
        if (!_preNorm) x = _norm1.Forward(x, training);

        shortcut = x;
        branch = _preNorm ? _norm2.Forward(x, training) : x;
        branch = _fc2.Forward(engine.TensorGELU(_fc1.Forward(branch)));
        branch = PtOps.DropPath(_ctx, branch, _dropPath, training);
        x = engine.TensorAdd(shortcut, branch);
        if (!_preNorm) x = _norm2.Forward(x, training);
        return x;
    }
}
