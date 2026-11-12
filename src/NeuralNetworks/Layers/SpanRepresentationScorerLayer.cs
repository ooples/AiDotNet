using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>How a <see cref="SpanRepresentationScorerLayer{T}"/> represents a span before classifying it.</summary>
public enum SpanRepresentationKind
{
    /// <summary>PURE (Zhong and Chen 2021, section 3.2): the start and end token states and a learned width embedding,
    /// classified by a two-layer feed-forward network.</summary>
    Endpoints,

    /// <summary>SpERT (Eberts and Ulges 2020, section 3.1): the max-pooled token states of the span, a learned width
    /// embedding and a sentence context vector, classified by a linear layer.</summary>
    MaxPool,
}

/// <summary>
/// Scores every span <c>(start, end)</c> of a sentence against every entity category from a span representation: the
/// endpoint representation of PURE or the max-pooled representation of SpERT, each with a learned width embedding.
/// </summary>
/// <remarks>
/// <para><b>Shapes:</b> input <c>[B, S, D]</c> (or <c>[S, D]</c>) token states; output <c>[B, S * S, C]</c>, the
/// flattened span grid with index <c>start * S + end</c>, the layout of <see cref="BiaffineSpanScorerLayer{T}"/>. Spans
/// with <c>end &lt; start</c> are scored too (as the empty width) and are expected to be masked by the targets.</para>
/// <para>Widths above <c>maxWidth</c> share the last width embedding. In SpERT's representation the first token's
/// state stands in for the sentence's <c>[CLS]</c> vector, since the input carries no separate classification
/// token.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "AiDotNet.NeuralNetworks.Layers.SpanRepresentationKind.Endpoints, 8, 4, 3, 6, 3")]
// The same layouts as BiaffineSpanScorerLayer: the output's middle axis is the flattened S x S span grid.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
public partial class SpanRepresentationScorerLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3)) return null;
        var spans = new OutputAxisContract(TensorAxis.Other, AxisRelation.Product(TensorAxis.Time, TensorAxis.Time));
        var categories = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_numCategories));
        return inputRank == 2 ? new[] { spans, categories }
            : new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), spans, categories };
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _widthEmbedding.ResetState();
        foreach (var layer in _classifier) layer.ResetState();
    }

    private readonly SpanRepresentationKind _kind;
    private readonly int _inputDim;
    private readonly int _widthDim;
    private readonly int _maxWidth;
    private readonly int _hiddenDim;
    private readonly int _numCategories;
    private readonly EmbeddingLayer<T> _widthEmbedding;
    private readonly FullyConnectedLayer<T>[] _classifier;

    /// <summary>Creates the scorer.</summary>
    /// <param name="kind">The span representation.</param>
    /// <param name="inputDim">Width of the token states.</param>
    /// <param name="widthDim">Width of the span-width embedding (PURE: 150; SpERT: 25).</param>
    /// <param name="maxWidth">Longest span width with its own embedding (PURE: 8; SpERT: 10).</param>
    /// <param name="hiddenDim">Hidden units of PURE's two-layer feed-forward network (150); unused by SpERT.</param>
    /// <param name="numCategories">Entity categories, the non-entity category included.</param>
    public SpanRepresentationScorerLayer(
        [LayerState] SpanRepresentationKind kind,
        [LayerState] int inputDim,
        [LayerState] int widthDim,
        [LayerState] int maxWidth,
        [LayerState] int hiddenDim,
        [LayerState] int numCategories)
        : base(new[] { -1, -1, inputDim }, new[] { -1, -1, numCategories })
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (widthDim <= 0) throw new ArgumentOutOfRangeException(nameof(widthDim));
        if (maxWidth <= 0) throw new ArgumentOutOfRangeException(nameof(maxWidth));
        if (numCategories <= 0) throw new ArgumentOutOfRangeException(nameof(numCategories));
        if (kind == SpanRepresentationKind.Endpoints && hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        _kind = kind;
        _inputDim = inputDim;
        _widthDim = widthDim;
        _maxWidth = maxWidth;
        _hiddenDim = hiddenDim;
        _numCategories = numCategories;
        _widthEmbedding = new EmbeddingLayer<T>(maxWidth + 1, widthDim);
        var identity = (IActivationFunction<T>)new ActivationFunctions.IdentityActivation<T>();
        var relu = (IActivationFunction<T>)new ActivationFunctions.ReLUActivation<T>();
        int spanDim = 2 * inputDim + widthDim;
        _classifier = kind == SpanRepresentationKind.Endpoints
            ? new[]
            {
                new FullyConnectedLayer<T>(spanDim, hiddenDim, relu),
                new FullyConnectedLayer<T>(hiddenDim, hiddenDim, relu),
                new FullyConnectedLayer<T>(hiddenDim, numCategories, identity),
            }
            : new[] { new FullyConnectedLayer<T>(spanDim, numCategories, identity) };
        RegisterSubLayer(_widthEmbedding);
        foreach (var layer in _classifier) RegisterSubLayer(layer);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched) input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Shape.Length != 3 || input.Shape[2] != _inputDim)
            throw new ArgumentException($"SpanRepresentationScorerLayer expects [B, S, {_inputDim}] token states.", nameof(input));
        int b = input.Shape[0], s = input.Shape[1], d = _inputDim;

        // The width embedding of every grid cell: min(end - start, maxWidth), and 0 below the diagonal.
        var widthIds = new Tensor<T>(new[] { 1, s * s });
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) widthIds[0, i * s + j] = NumOps.FromDouble(j >= i ? Math.Min(j - i, _maxWidth) : 0);
        var width = Engine.TensorBroadcastTo(Engine.Reshape(_widthEmbedding.Forward(widthIds), new[] { 1, s, s, _widthDim }), new[] { b, s, s, _widthDim });

        Tensor<T> first, second;
        if (_kind == SpanRepresentationKind.Endpoints)
        {
            first = Engine.TensorBroadcastTo(Engine.Reshape(input, new[] { b, s, 1, d }), new[] { b, s, s, d });    // start token
            second = Engine.TensorBroadcastTo(Engine.Reshape(input, new[] { b, 1, s, d }), new[] { b, s, s, d });   // end token
        }
        else
        {
            first = MaxPoolGrid(input, b, s, d);
            // SpERT's context vector: the sentence's first token stands in for [CLS].
            var context = Engine.TensorSlice(input, new[] { 0, 0, 0 }, new[] { b, 1, d });
            second = Engine.TensorBroadcastTo(Engine.Reshape(context, new[] { b, 1, 1, d }), new[] { b, s, s, d });
        }
        var spans = Engine.TensorConcatenate(new[] { first, second, width }, 3);
        var x = Engine.Reshape(spans, new[] { b * s * s, 2 * d + _widthDim });
        foreach (var layer in _classifier) x = layer.Forward(x);
        var output = Engine.Reshape(x, new[] { b, s * s, _numCategories });
        return unbatched ? Engine.Reshape(output, new[] { s * s, _numCategories }) : output;
    }

    // The max-pooled state of every span [start, end] (SpERT's f(e_i, ..., e_j)), [B, S, S, D] with zeros below the
    // diagonal. Width k's maxima come from width k - 1's and the token k positions on.
    private Tensor<T> MaxPoolGrid(Tensor<T> input, int b, int s, int d)
    {
        var diagonals = new Tensor<T>[s + 1];
        var current = input;   // width 0: [B, S, D]
        for (int k = 0; k < s; k++)
        {
            if (k > 0)
            {
                var previous = Engine.TensorSlice(current, new[] { 0, 0, 0 }, new[] { b, s - k, d });
                var next = Engine.TensorSlice(input, new[] { 0, k, 0 }, new[] { b, s - k, d });
                current = Engine.TensorMax(previous, next);
            }
            // Pad each diagonal to S rows so the grid can index them uniformly.
            diagonals[k] = k == 0 ? current : Engine.TensorConcatenate(new[] { current, new Tensor<T>(new[] { b, k, d }) }, 1);
        }
        diagonals[s] = new Tensor<T>(new[] { b, s, d });   // the empty span below the diagonal
        // Rows of [(S + 1) * S, B * D], so the gather is the engine's 2-D index select.
        var stacked = Engine.Reshape(Engine.TensorPermute(Engine.TensorConcatenate(diagonals, 1), new[] { 1, 0, 2 }), new[] { (s + 1) * s, b * d });
        var index = new int[s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) index[i * s + j] = (j >= i ? j - i : s) * s + i;
        var grid = Engine.TensorIndexSelect(stacked, new Tensor<int>(new[] { s * s }, new Vector<int>(index)), 0);
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(grid, new[] { s * s, b, d }), new[] { 1, 0, 2 }), new[] { b, s, s, d });
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Kind"] = _kind.ToString();
        metadata["InputDim"] = _inputDim.ToString(ci);
        metadata["WidthDim"] = _widthDim.ToString(ci);
        metadata["MaxWidth"] = _maxWidth.ToString(ci);
        metadata["HiddenDim"] = _hiddenDim.ToString(ci);
        metadata["NumCategories"] = _numCategories.ToString(ci);
        return metadata;
    }
}
