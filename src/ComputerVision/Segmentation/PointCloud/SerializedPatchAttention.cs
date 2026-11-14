namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// PTv3's serialized patch attention (Sec. 3.2, Pointcept <c>SerializedAttention</c>): multi-head
/// self-attention restricted to patches of consecutive points in one serialized order.
/// </summary>
/// <remarks>
/// <para>
/// <c>qkv = Linear(C, 3C)</c> is applied per point, the rows are gathered patch by patch in serialized
/// order, scaled dot-product attention runs inside each patch with scale <c>(C / H)^-0.5</c>, the
/// outputs are scattered back to the points and projected by <c>Linear(C, C)</c>. Inside one patch
/// this is exactly ordinary multi-head attention, because attention is permutation-equivariant.
/// </para>
/// <para>
/// Every patch of the same length is one batched matrix multiply; the gathers and scatters are
/// <see cref="int"/> index tensors, so the whole layer is engine operations on <c>T</c> and
/// differentiable end to end.
/// </para>
/// </remarks>
internal sealed class SerializedPatchAttention<T>
{
    private readonly PtContext<T> _ctx;

    internal SerializedPatchAttention(PtContext<T> ctx, int channels, int numHeads, int patchSize, bool qkvBias = true)
    {
        if (numHeads <= 0 || channels % numHeads != 0)
            throw new ArgumentException($"Channels ({channels}) must be divisible by the head count ({numHeads}).", nameof(numHeads));
        _ctx = ctx;
        Channels = channels;
        NumHeads = numHeads;
        PatchSize = patchSize;
        Qkv = new PtLinear<T>(ctx, channels, channels * 3, qkvBias);
        Projection = new PtLinear<T>(ctx, channels, channels);
    }

    internal int Channels { get; }

    internal int NumHeads { get; }

    internal int PatchSize { get; }

    internal PtLinear<T> Qkv { get; }

    internal PtLinear<T> Projection { get; }

    internal Tensor<T> Forward(PointCloudLevel<T> level, Tensor<T> x, int orderIndex)
    {
        var engine = _ctx.Engine;
        int c = Channels, h = NumHeads, d = c / h;
        var scale = _ctx.NumOps.FromDouble(1.0 / Math.Sqrt(d));
        var qkv = Qkv.Forward(x);
        var plan = level.PatchPlan(orderIndex, PatchSize);

        var outputs = new Tensor<T>[plan.Groups.Count];
        for (int g = 0; g < plan.Groups.Count; g++)
        {
            var group = plan.Groups[g];
            int p = group.PatchCount, l = group.PatchLength;
            var rows = engine.TensorIndexSelect(qkv, PtOps.Indices(group.Gather), 0);
            var patches = engine.Reshape(rows, new[] { p, l, 3 * c });

            Tensor<T> Heads(int part)
            {
                var slice = engine.TensorSlice(patches, new[] { 0, 0, part * c }, new[] { p, l, c });
                var split = engine.Reshape(slice, new[] { p, l, h, d });
                return engine.Reshape(engine.TensorPermute(split, new[] { 0, 2, 1, 3 }), new[] { p * h, l, d });
            }

            var q = engine.TensorMultiplyScalar(Heads(0), scale);
            var k = Heads(1);
            var v = Heads(2);
            var scores = engine.TensorBatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 }));
            var attention = engine.TensorSoftmax(scores, -1);
            var mixed = engine.TensorBatchMatMul(attention, v);
            var merged = engine.TensorPermute(engine.Reshape(mixed, new[] { p, h, l, d }), new[] { 0, 2, 1, 3 });
            outputs[g] = engine.Reshape(merged, new[] { p * l, c });
        }

        var all = outputs.Length == 1 ? outputs[0] : engine.TensorConcatenate(outputs, 0);
        var perPoint = engine.TensorIndexSelect(all, PtOps.Indices(plan.Inverse), 0);
        return Projection.Forward(perPoint);
    }
}
