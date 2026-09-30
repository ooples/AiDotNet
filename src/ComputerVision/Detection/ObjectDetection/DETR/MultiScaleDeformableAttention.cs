using System.IO;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// Multi-scale deformable attention (Zhu et al., "Deformable DETR", ICLR 2021, Sec. 4.1), the attention
/// DINO's encoder and decoder cross-attention are built on.
/// </summary>
/// <remarks>
/// <para>
/// Each query attends to only <c>numLevels x numPoints</c> sampled locations per head instead of every
/// token: a linear layer on the query predicts a 2-D offset per sampled point, and those offsets are added
/// to the query's reference point. A second linear layer predicts one weight per sampled point, and the
/// weights are softmax-normalized over all levels and points of a head. The output is the weighted sum of
/// the bilinearly sampled values.
/// </para>
/// <para>
/// This mirrors the reference <c>MSDeformAttn</c> module and its pure-PyTorch core
/// <c>ms_deform_attn_core_pytorch</c>: sampling locations are normalized to [0, 1], mapped to
/// <c>2 * loc - 1</c>, and sampled with bilinear, zero-padded, <c>align_corners = false</c> GridSample,
/// one call per level over <c>[batch * heads, headDim, H, W]</c>. Every step is an engine op, so the
/// gradient reaches the offsets, the attention weights, the value projection and the reference points.
/// </para>
/// <para>
/// Initialization follows <c>MSDeformAttn._reset_parameters</c>: the offset weights are zero and their
/// bias places head <c>m</c>'s points on the ray at angle <c>2 pi m / heads</c>, point <c>p</c> at
/// distance <c>p + 1</c>. The attention-weight projection is zero, so the initial weights are uniform. The
/// value and output projections are Xavier-uniform with zero bias.
/// </para>
/// </remarks>
internal sealed class MultiScaleDeformableAttention<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _dModel;
    private readonly int _numLevels;
    private readonly int _numHeads;
    private readonly int _numPoints;
    private readonly int _headDim;

    private readonly Tensor<T> _offsetWeight;
    private readonly Tensor<T> _offsetBias;
    private readonly Tensor<T> _attentionWeight;
    private readonly Tensor<T> _attentionBias;
    private readonly Tensor<T> _valueWeight;
    private readonly Tensor<T> _valueBias;
    private readonly Tensor<T> _outputWeight;
    private readonly Tensor<T> _outputBias;

    /// <summary>Creates the attention with the reference defaults (d_model 256, 4 levels, 8 heads, 4 points).</summary>
    public MultiScaleDeformableAttention(int dModel = 256, int numLevels = 4, int numHeads = 8, int numPoints = 4)
    {
        if (dModel <= 0) throw new ArgumentOutOfRangeException(nameof(dModel), "dModel must be positive.");
        if (numLevels <= 0) throw new ArgumentOutOfRangeException(nameof(numLevels), "numLevels must be positive.");
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (numPoints <= 0) throw new ArgumentOutOfRangeException(nameof(numPoints), "numPoints must be positive.");
        if (dModel % numHeads != 0)
            throw new ArgumentException($"dModel ({dModel}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _numOps = MathHelper.GetNumericOperations<T>();
        _dModel = dModel;
        _numLevels = numLevels;
        _numHeads = numHeads;
        _numPoints = numPoints;
        _headDim = dModel / numHeads;

        int offsets = numHeads * numLevels * numPoints * 2;
        int weights = numHeads * numLevels * numPoints;
        _offsetWeight = new Tensor<T>(new[] { dModel, offsets });
        _offsetBias = new Tensor<T>(new[] { offsets });
        _attentionWeight = new Tensor<T>(new[] { dModel, weights });
        _attentionBias = new Tensor<T>(new[] { weights });
        _valueWeight = new Tensor<T>(new[] { dModel, dModel });
        _valueBias = new Tensor<T>(new[] { dModel });
        _outputWeight = new Tensor<T>(new[] { dModel, dModel });
        _outputBias = new Tensor<T>(new[] { dModel });

        for (int m = 0; m < numHeads; m++)
        {
            double theta = m * (2.0 * Math.PI / numHeads);
            double cx = Math.Cos(theta), cy = Math.Sin(theta);
            double scale = Math.Max(Math.Abs(cx), Math.Abs(cy));
            for (int l = 0; l < numLevels; l++)
            {
                for (int p = 0; p < numPoints; p++)
                {
                    int index = (((m * numLevels) + l) * numPoints + p) * 2;
                    _offsetBias[index] = _numOps.FromDouble(cx / scale * (p + 1));
                    _offsetBias[index + 1] = _numOps.FromDouble(cy / scale * (p + 1));
                }
            }
        }

        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        XavierUniform(_valueWeight, random);
        XavierUniform(_outputWeight, random);
    }

    /// <summary>Model width.</summary>
    public int DModel => _dModel;

    /// <summary>Number of feature levels sampled.</summary>
    public int NumLevels => _numLevels;

    // Test access for controlled-weight fixtures ([in, out] storage).
    internal Tensor<T> ValueWeight => _valueWeight;
    internal Tensor<T> AttentionWeight => _attentionWeight;
    internal Tensor<T> AttentionBias => _attentionBias;
    internal Tensor<T> OutputWeight => _outputWeight;
    internal Tensor<T> OutputBias => _outputBias;

    /// <summary>
    /// Runs the attention.
    /// </summary>
    /// <param name="query">Queries, <c>[batch, queries, dModel]</c>, with their positional embedding already added.</param>
    /// <param name="referencePoints">
    /// Reference points per query and level, normalized to [0, 1]: <c>[batch, queries, levels, 2]</c> as
    /// (x, y), or <c>[batch, queries, levels, 4]</c> as boxes (cx, cy, w, h). For boxes, offsets are scaled
    /// by half the box size divided by the number of points.
    /// </param>
    /// <param name="value">Flattened multi-scale features, <c>[batch, tokens, dModel]</c>.</param>
    /// <param name="spatialShapes">(height, width) of each level, in the order the tokens are flattened.</param>
    /// <param name="levelStarts">First token index of each level.</param>
    /// <returns><c>[batch, queries, dModel]</c>.</returns>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> referencePoints, Tensor<T> value, int[][] spatialShapes, int[] levelStarts)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (referencePoints is null) throw new ArgumentNullException(nameof(referencePoints));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (spatialShapes is null) throw new ArgumentNullException(nameof(spatialShapes));
        if (levelStarts is null) throw new ArgumentNullException(nameof(levelStarts));
        if (query.Rank != 3 || query.Shape[2] != _dModel)
            throw new ArgumentException($"query must be [batch, queries, {_dModel}].", nameof(query));
        if (value.Rank != 3 || value.Shape[2] != _dModel || value.Shape[0] != query.Shape[0])
            throw new ArgumentException($"value must be [batch, tokens, {_dModel}] with the query's batch.", nameof(value));
        if (spatialShapes.Length != _numLevels || levelStarts.Length != _numLevels)
            throw new ArgumentException($"Expected {_numLevels} levels of spatial shapes and starts.", nameof(spatialShapes));
        int batch = query.Shape[0], numQueries = query.Shape[1], tokens = value.Shape[1];
        int refDim = referencePoints.Rank == 4 ? referencePoints.Shape[3] : -1;
        if (referencePoints.Rank != 4 || referencePoints.Shape[0] != batch || referencePoints.Shape[1] != numQueries
            || referencePoints.Shape[2] != _numLevels || (refDim != 2 && refDim != 4))
            throw new ArgumentException($"referencePoints must be [batch, queries, {_numLevels}, 2 or 4].", nameof(referencePoints));
        int expectedTokens = 0;
        for (int l = 0; l < _numLevels; l++)
        {
            if (levelStarts[l] != expectedTokens)
                throw new ArgumentException($"Level {l} starts at {levelStarts[l]}; the flattened layout puts it at {expectedTokens}.", nameof(levelStarts));
            expectedTokens += spatialShapes[l][0] * spatialShapes[l][1];
        }
        if (expectedTokens != tokens)
            throw new ArgumentException($"The spatial shapes cover {expectedTokens} tokens; value has {tokens}.", nameof(value));

        var engine = AiDotNetEngine.Current;
        int heads = _numHeads, levels = _numLevels, points = _numPoints, headDim = _headDim;

        // value: [N, S, D] -> [N, S, M, Dh]
        var projectedValue = engine.Reshape(Linear(value, _valueWeight, _valueBias), new[] { batch, tokens, heads, headDim });

        // offsets: [N, Lq, M, L, P, 2]; weights: softmax over L*P per head -> [N, Lq, M, L, P]
        var offsets = engine.Reshape(Linear(query, _offsetWeight, _offsetBias), new[] { batch, numQueries, heads, levels, points, 2 });
        var logits = engine.Reshape(Linear(query, _attentionWeight, _attentionBias), new[] { batch, numQueries, heads, levels * points });
        var attention = engine.Reshape(engine.Softmax(logits, 3), new[] { batch, numQueries, heads, levels, points });

        var locationShape = new[] { batch, numQueries, heads, levels, points, 2 };
        Tensor<T> locations;
        if (refDim == 2)
        {
            // offset / (W, H) of its level
            var normalizer = new Tensor<T>(new[] { 1, 1, 1, levels, 1, 2 });
            for (int l = 0; l < levels; l++)
            {
                normalizer[(l * 2) + 0] = _numOps.FromDouble(1.0 / spatialShapes[l][1]);
                normalizer[(l * 2) + 1] = _numOps.FromDouble(1.0 / spatialShapes[l][0]);
            }

            var reference = engine.TensorBroadcastTo(
                engine.Reshape(referencePoints, new[] { batch, numQueries, 1, levels, 1, 2 }), locationShape);
            locations = engine.TensorAdd(reference, engine.TensorMultiply(offsets, engine.TensorBroadcastTo(normalizer, locationShape)));
        }
        else
        {
            // reference_xy + offset / P * reference_wh * 0.5
            var centers = engine.TensorSlice(referencePoints, new[] { 0, 0, 0, 0 }, new[] { batch, numQueries, levels, 2 });
            var sizes = engine.TensorSlice(referencePoints, new[] { 0, 0, 0, 2 }, new[] { batch, numQueries, levels, 2 });
            var center = engine.TensorBroadcastTo(engine.Reshape(centers, new[] { batch, numQueries, 1, levels, 1, 2 }), locationShape);
            var size = engine.TensorBroadcastTo(engine.Reshape(sizes, new[] { batch, numQueries, 1, levels, 1, 2 }), locationShape);
            locations = engine.TensorAdd(center,
                engine.TensorMultiply(offsets, engine.TensorMultiplyScalar(size, _numOps.FromDouble(0.5 / points))));
        }

        // [0, 1] -> GridSample's [-1, 1] (align_corners = false)
        var grids = engine.TensorAddScalar(engine.TensorMultiplyScalar(locations, _numOps.FromDouble(2.0)), _numOps.FromDouble(-1.0));

        var sampledLevels = new Tensor<T>[levels];
        for (int l = 0; l < levels; l++)
        {
            int h = spatialShapes[l][0], w = spatialShapes[l][1];
            // [N, H*W, M, Dh] -> [N, M, Dh, H*W] -> [N*M, Dh, H, W]
            var levelValue = engine.TensorSlice(projectedValue, new[] { 0, levelStarts[l], 0, 0 }, new[] { batch, h * w, heads, headDim });
            levelValue = engine.Reshape(engine.TensorPermute(levelValue, new[] { 0, 2, 3, 1 }), new[] { batch * heads, headDim, h, w });

            // [N, Lq, M, 1, P, 2] -> [N, M, Lq, P, 2] -> [N*M, Lq, P, 2]
            var levelGrid = engine.TensorSlice(grids, new[] { 0, 0, 0, l, 0, 0 }, new[] { batch, numQueries, heads, 1, points, 2 });
            levelGrid = engine.Reshape(
                engine.TensorPermute(engine.Reshape(levelGrid, new[] { batch, numQueries, heads, points, 2 }), new[] { 0, 2, 1, 3, 4 }),
                new[] { batch * heads, numQueries, points, 2 });

            sampledLevels[l] = engine.GridSample(levelValue, levelGrid); // [N*M, Dh, Lq, P]
        }

        // [N*M, Dh, Lq, L, P] -> [N*M, Dh, Lq, L*P]
        var sampled = engine.Reshape(engine.TensorStack(sampledLevels, 3), new[] { batch * heads, headDim, numQueries, levels * points });
        // weights: [N, Lq, M, L, P] -> [N, M, Lq, L, P] -> [N*M, 1, Lq, L*P]
        var weights = engine.Reshape(engine.TensorPermute(attention, new[] { 0, 2, 1, 3, 4 }), new[] { batch * heads, 1, numQueries, levels * points });
        var weighted = engine.TensorMultiply(sampled, engine.TensorBroadcastTo(weights, sampled._shape));
        var summed = engine.ReduceSum(weighted, new[] { 3 }, keepDims: false); // [N*M, Dh, Lq]

        // [N, M*Dh, Lq] -> [N, Lq, M*Dh]
        var output = engine.TensorPermute(engine.Reshape(summed, new[] { batch, heads * headDim, numQueries }), new[] { 0, 2, 1 });
        return Linear(output, _outputWeight, _outputBias);
    }

    /// <summary>Writes the configuration and parameters.</summary>
    public void WriteParameters(BinaryWriter writer)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        writer.Write(_dModel);
        writer.Write(_numLevels);
        writer.Write(_numHeads);
        writer.Write(_numPoints);
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) writer.Write(_numOps.ToDouble(tensor[i]));
    }

    /// <summary>Reads parameters written by <see cref="WriteParameters"/>, rejecting another configuration.</summary>
    public void ReadParameters(BinaryReader reader)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        int dModel = reader.ReadInt32(), levels = reader.ReadInt32(), heads = reader.ReadInt32(), points = reader.ReadInt32();
        if (dModel != _dModel || levels != _numLevels || heads != _numHeads || points != _numPoints)
            throw new InvalidOperationException(
                $"MultiScaleDeformableAttention configuration mismatch: expected d={_dModel}, levels={_numLevels}, heads={_numHeads}, " +
                $"points={_numPoints}; read d={dModel}, levels={levels}, heads={heads}, points={points}.");
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) tensor[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    /// <inheritdoc/>
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    /// <inheritdoc/>
    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return _offsetWeight;
        yield return _offsetBias;
        yield return _attentionWeight;
        yield return _attentionBias;
        yield return _valueWeight;
        yield return _valueBias;
        yield return _outputWeight;
        yield return _outputBias;
    }

    private static Tensor<T> Linear(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var engine = AiDotNetEngine.Current;
        int inFeatures = weight.Shape[0], outFeatures = weight.Shape[1];
        int rows = input.Length / inFeatures;
        var outShape = (int[])input._shape.Clone();
        outShape[outShape.Length - 1] = outFeatures;
        var product = engine.TensorMatMul(engine.Reshape(input, new[] { rows, inFeatures }), weight);
        var biased = engine.TensorAdd(product, engine.TensorBroadcastTo(engine.Reshape(bias, new[] { 1, outFeatures }), new[] { rows, outFeatures }));
        return engine.Reshape(biased, outShape);
    }

    private void XavierUniform(Tensor<T> weight, Random random)
    {
        double bound = Math.Sqrt(6.0 / (weight.Shape[0] + weight.Shape[1]));
        for (int i = 0; i < weight.Length; i++)
            weight[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
    }
}
