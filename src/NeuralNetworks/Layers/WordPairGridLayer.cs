using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// W2NER's word-pair relation classifier (Li et al. 2022, "Unified Named Entity Recognition as Word-Word Relation
/// Classification", section 3): conditional layer normalization builds a grid V_ij = gamma_ij * (h_j - mu) / sigma +
/// lambda_ij with gamma_ij and lambda_ij linear in h_i (Equations 1 and 2); distance and region embeddings are
/// concatenated and an MLP reduces the grid to dc channels (Equation 3); three depthwise 3x3 convolutions with dilation
/// rates 1, 2 and 3 and GELU, applied in sequence as the authors' code does, refine it (Equation 4); and a co-predictor sums an MLP over the refined grid with a biaffine
/// classifier over the word representations (Equations 5 to 9).
/// </summary>
/// <remarks>
/// <para><b>Shapes:</b> input <c>[B, S, D]</c> (or <c>[S, D]</c>) word representations; output <c>[B, S * S, R]</c>, the
/// relation scores of every word pair (row i, column j) at index <c>i * S + j</c>, R being None, NNW and one THW per
/// entity type.</para>
/// <para>Distances use W2NER's buckets (0, 1, 2-3, 4-7, ..., 256+, signed, with a separate bucket for the diagonal); regions
/// distinguish the lower triangle (diagonal included) from the upper.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 4, 6, 5, 3")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
public partial class WordPairGridLayer<T> : LayerBase<T>, IShapeContract
{
    private const int EmbeddingDim = 20;
    private static readonly int[] Dilations = { 1, 2, 3 };
    private readonly int _inputDim;
    private readonly int _channels;
    private readonly int _hidden;
    private readonly int _biaffineDim;
    private readonly int _relations;
    private readonly FullyConnectedLayer<T> _gain;
    private readonly FullyConnectedLayer<T> _shift;
    private readonly EmbeddingLayer<T> _distance;
    private readonly EmbeddingLayer<T> _region;
    private readonly FullyConnectedLayer<T> _reduce;
    private readonly BiasFreeLinearLayer<T>[] _dilated;
    private readonly FullyConnectedLayer<T> _predictorHidden;
    private readonly FullyConnectedLayer<T> _predictorOutput;
    private readonly FullyConnectedLayer<T> _subject;
    private readonly FullyConnectedLayer<T> _object;
    private readonly BiasFreeLinearLayer<T> _bilinear;
    private readonly FullyConnectedLayer<T> _additive;

    /// <summary>Creates the classifier.</summary>
    /// <param name="inputDim">Width of the word representations (d_h).</param>
    /// <param name="channels">Grid channels after the reducing MLP (Table 7: d_c of 64, 96 or 128).</param>
    /// <param name="hiddenDim">Hidden units of the MLP predictor.</param>
    /// <param name="biaffineDim">Width of the subject and object representations of the biaffine predictor.</param>
    /// <param name="numRelations">Relations: None, NNW and one THW per entity type.</param>
    public WordPairGridLayer([LayerState] int inputDim, [LayerState] int channels, [LayerState] int hiddenDim, [LayerState] int biaffineDim, [LayerState] int numRelations)
        : base(new[] { -1, -1, inputDim }, new[] { -1, -1, numRelations })
    {
        if (inputDim <= 0 || channels <= 0 || hiddenDim <= 0 || biaffineDim <= 0 || numRelations <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        _inputDim = inputDim; _channels = channels; _hidden = hiddenDim; _biaffineDim = biaffineDim; _relations = numRelations;
        var identity = (IActivationFunction<T>)new ActivationFunctions.IdentityActivation<T>();
        var gelu = (IActivationFunction<T>)new ActivationFunctions.GELUActivation<T>();
        _gain = new FullyConnectedLayer<T>(inputDim, inputDim, identity);
        _shift = new FullyConnectedLayer<T>(inputDim, inputDim, identity);
        _distance = new EmbeddingLayer<T>(20, EmbeddingDim);
        _region = new EmbeddingLayer<T>(3, EmbeddingDim);
        _reduce = new FullyConnectedLayer<T>(inputDim + 2 * EmbeddingDim, channels, gelu);
        _dilated = Dilations.Select(_ => new BiasFreeLinearLayer<T>(1, channels * 9)).ToArray();
        _predictorHidden = new FullyConnectedLayer<T>(3 * channels, hiddenDim, gelu);
        _predictorOutput = new FullyConnectedLayer<T>(hiddenDim, numRelations, identity);
        _subject = new FullyConnectedLayer<T>(inputDim, biaffineDim, gelu);
        _object = new FullyConnectedLayer<T>(inputDim, biaffineDim, gelu);
        _bilinear = new BiasFreeLinearLayer<T>(biaffineDim, numRelations * biaffineDim);
        _additive = new FullyConnectedLayer<T>(2 * biaffineDim, numRelations, identity);
        // CLN starts as plain layer normalization: gain 1 + W h, shift W h, with small W (Liu et al. 2021).
        RegisterSubLayer(_gain); RegisterSubLayer(_shift); RegisterSubLayer(_distance); RegisterSubLayer(_region); RegisterSubLayer(_reduce);
        foreach (var conv in _dilated) RegisterSubLayer(conv);
        RegisterSubLayer(_predictorHidden); RegisterSubLayer(_predictorOutput); RegisterSubLayer(_subject); RegisterSubLayer(_object);
        RegisterSubLayer(_bilinear); RegisterSubLayer(_additive);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3)) return null;
        var pairs = new OutputAxisContract(TensorAxis.Other, AxisRelation.Product(TensorAxis.Time, TensorAxis.Time));
        var relations = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_relations));
        return inputRank == 2 ? new[] { pairs, relations }
            : new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), pairs, relations };
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in GetSubLayers()) layer.ResetState();
    }

    // W2NER's distance bucket of i - j: 0, 1, 2-3, 4-7, ... 256+ for positive offsets, +9 for negative ones, 19 on the diagonal.
    private static int DistanceBucket(int offset)
    {
        if (offset == 0) return 19;
        int magnitude = Math.Abs(offset), bucket = magnitude == 1 ? 1 : magnitude < 4 ? 2 : magnitude < 8 ? 3 : magnitude < 16 ? 4
            : magnitude < 32 ? 5 : magnitude < 64 ? 6 : magnitude < 128 ? 7 : magnitude < 256 ? 8 : 9;
        return offset < 0 ? bucket + 9 : bucket;
    }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder, int count) =>
        Engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))), new[] { count });

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched) input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Shape.Length != 3 || input.Shape[2] != _inputDim) throw new ArgumentException($"WordPairGridLayer expects [B, S, {_inputDim}].", nameof(input));
        int b = input.Shape[0], s = input.Shape[1], d = _inputDim, c = _channels;
        var flat = Engine.Reshape(input, new[] { b * s, d });
        // Equations 1 and 2: normalize h_j, scale and shift by linear functions of h_i (gain starts at 1).
        var mean = Engine.ReduceMean(input, new[] { 2 }, keepDims: true);
        var centred = Engine.TensorSubtract(input, Engine.TensorBroadcastTo(mean, input.Shape.ToArray()));
        var std = Engine.TensorPower(Engine.TensorAddScalar(Engine.ReduceMean(Engine.TensorMultiply(centred, centred), new[] { 2 }, keepDims: true), NumOps.FromDouble(1e-12)), NumOps.FromDouble(0.5));
        var normalized = Engine.TensorDivide(centred, Engine.TensorBroadcastTo(std, input.Shape.ToArray()));
        var gain = Engine.TensorAddScalar(Engine.Reshape(_gain.Forward(flat), new[] { b, s, 1, d }), NumOps.One);
        var shift = Engine.Reshape(_shift.Forward(flat), new[] { b, s, 1, d });
        var grid = Engine.TensorAdd(Engine.TensorMultiply(Engine.TensorBroadcastTo(gain, new[] { b, s, s, d }), Engine.TensorBroadcastTo(Engine.Reshape(normalized, new[] { b, 1, s, d }), new[] { b, s, s, d })),
            Engine.TensorBroadcastTo(shift, new[] { b, s, s, d }));
        // Equation 3: distance and region embeddings, then the reducing MLP.
        var distanceIds = new Tensor<T>(new[] { 1, s * s });
        var regionIds = new Tensor<T>(new[] { 1, s * s });
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
            {
                distanceIds[0, i * s + j] = NumOps.FromDouble(DistanceBucket(i - j));
                regionIds[0, i * s + j] = NumOps.FromDouble(j <= i ? 2 : 1);
            }
        var distance = Engine.TensorBroadcastTo(Engine.Reshape(_distance.Forward(distanceIds), new[] { 1, s, s, EmbeddingDim }), new[] { b, s, s, EmbeddingDim });
        var region = Engine.TensorBroadcastTo(Engine.Reshape(_region.Forward(regionIds), new[] { 1, s, s, EmbeddingDim }), new[] { b, s, s, EmbeddingDim });
        var joined = Engine.Reshape(Engine.TensorConcatenate(new[] { grid, distance, region }, 3), new[] { b * s * s, d + 2 * EmbeddingDim });
        var reduced = Engine.Reshape(_reduce.Forward(joined), new[] { b, s, s, c });
        // Equation 4: depthwise 3x3 convolutions with dilations 1, 2 and 3, each followed by GELU, applied in sequence as the
        // reference's ConvolutionLayer does, and their outputs concatenated.
        var refined = new Tensor<T>[Dilations.Length];
        var current = reduced;
        for (int k = 0; k < Dilations.Length; k++)
        {
            int rate = Dilations[k];
            var padded = Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { b, rate, s, c }), current, new Tensor<T>(new[] { b, rate, s, c }) }, 1);
            padded = Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { b, s + 2 * rate, rate, c }), padded, new Tensor<T>(new[] { b, s + 2 * rate, rate, c }) }, 2);
            var weights = Engine.Reshape(Value(_dilated[k], c * 9), new[] { 9, c });
            Tensor<T>? sum = null;
            for (int dy = 0; dy < 3; dy++)
                for (int dx = 0; dx < 3; dx++)
                {
                    var shifted = Engine.TensorSlice(padded, new[] { 0, dy * rate, dx * rate, 0 }, new[] { b, s, s, c });
                    var tap = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorSlice(weights, new[] { dy * 3 + dx, 0 }, new[] { 1, c }), new[] { 1, 1, 1, c }), new[] { b, s, s, c });
                    var term = Engine.TensorMultiply(shifted, tap);
                    sum = sum is null ? term : Engine.TensorAdd(sum, term);
                }
            current = AiDotNet.Helpers.ActivationHelper.ExactGelu(sum ?? current, Engine);
            refined[k] = current;
        }
        var q = Engine.Reshape(Engine.TensorConcatenate(refined, 3), new[] { b * s * s, 3 * c });
        var mlpScores = Engine.Reshape(_predictorOutput.Forward(_predictorHidden.Forward(q)), new[] { b, s, s, _relations });
        // Equations 5 to 7: biaffine scores s_i^T U o_j + W [s_i; o_j] + b.
        var subject = Engine.Reshape(_subject.Forward(flat), new[] { b, s, _biaffineDim });
        var obj = Engine.Reshape(_object.Forward(flat), new[] { b, s, _biaffineDim });
        var su = Engine.Reshape(_bilinear.Forward(Engine.Reshape(subject, new[] { b * s, _biaffineDim })), new[] { b, s * _relations, _biaffineDim });
        var bilinear = Engine.BatchMatMul(su, Engine.TensorPermute(obj, new[] { 0, 2, 1 }));                                       // [B, S*R, S]
        bilinear = Engine.TensorPermute(Engine.Reshape(bilinear, new[] { b, s, _relations, s }), new[] { 0, 1, 3, 2 });               // [B, S, S, R]
        var pairs = Engine.Reshape(Engine.TensorConcatenate(new[]
        {
            Engine.TensorBroadcastTo(Engine.Reshape(subject, new[] { b, s, 1, _biaffineDim }), new[] { b, s, s, _biaffineDim }),
            Engine.TensorBroadcastTo(Engine.Reshape(obj, new[] { b, 1, s, _biaffineDim }), new[] { b, s, s, _biaffineDim }),
        }, 3), new[] { b * s * s, 2 * _biaffineDim });
        var biaffine = Engine.TensorAdd(bilinear, Engine.Reshape(_additive.Forward(pairs), new[] { b, s, s, _relations }));
        // Equation 9: the two predictors' scores are summed before the softmax.
        var output = Engine.Reshape(Engine.TensorAdd(mlpScores, biaffine), new[] { b, s * s, _relations });
        return unbatched ? Engine.Reshape(output, new[] { s * s, _relations }) : output;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputDim"] = _inputDim.ToString(ci);
        metadata["Channels"] = _channels.ToString(ci);
        metadata["HiddenDim"] = _hidden.ToString(ci);
        metadata["BiaffineDim"] = _biaffineDim.ToString(ci);
        metadata["NumRelations"] = _relations.ToString(ci);
        return metadata;
    }
}
