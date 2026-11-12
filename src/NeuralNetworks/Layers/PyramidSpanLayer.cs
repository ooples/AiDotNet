using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The Pyramid nested-NER decoder (Wang et al. 2020, "Pyramid: A Layered Model for Nested Named Entity Recognition",
/// section 3): L decoding layers, each a layer norm and a bidirectional LSTM followed by a width-2 convolution that merges
/// adjacent hidden states, so layer l's position t represents tokens t to t + l - 1 (Equations 5 and 6); an inverse
/// pyramid that reconstructs each lower layer from the concatenated normal and inverse hidden states with a zero-padded
/// width-2 convolution (Equations 7 and 8, zeros for the top inverse layer); and a linear layer scoring each span from the
/// two hidden states (Equation 9). The layer norms, LSTMs and convolutions are shared by all decoding layers, as the
/// paper's notation states.
/// </summary>
/// <remarks>
/// <para><b>Shapes:</b> input <c>[B, S, D]</c> (or <c>[S, D]</c>); output <c>[B, S * S, C]</c>, the span grid at index
/// <c>start * S + end</c> (the layout of <see cref="BiaffineSpanScorerLayer{T}"/>), with zero scores for spans longer than
/// L or with end before start. The paper's remedy for mentions longer than L (B and I tags on the top layer) is not
/// built; L defaults to the paper's 16. The bidirectional LSTMs sum their directions.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 3, 3")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
public partial class PyramidSpanLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dim;
    private readonly int _layers;
    private readonly int _classes;
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly BidirectionalLayer<T> _lstm;
    private readonly Conv1DLayer<T> _merge;
    private readonly LayerNormalizationLayer<T> _inverseNorm;
    private readonly BidirectionalLayer<T> _inverseLstm;
    private readonly Conv1DLayer<T> _reconstruct;
    private readonly FullyConnectedLayer<T> _logits;

    /// <summary>Creates the decoder.</summary>
    /// <param name="dim">Width of the token representations and hidden states (Table 2: 200).</param>
    /// <param name="numLayers">Decoding layers L, the longest span scored (section 3.1: 16).</param>
    /// <param name="numClasses">Span classes, the non-entity class included.</param>
    public PyramidSpanLayer([LayerState] int dim, [LayerState] int numLayers, [LayerState] int numClasses)
        : base(new[] { -1, -1, dim }, new[] { -1, -1, numClasses })
    {
        if (dim <= 0 || numLayers <= 0 || numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        _dim = dim; _layers = numLayers; _classes = numClasses;
        var identity = (IActivationFunction<T>)new ActivationFunctions.IdentityActivation<T>();
        _norm = new LayerNormalizationLayer<T>(dim);
        _lstm = new BidirectionalLayer<T>(new LSTMLayer<T>(dim), mergeMode: true, activationFunction: identity);
        _merge = new Conv1DLayer<T>(inputChannels: dim, outputChannels: dim, kernelSize: 2, dilation: 1, stride: 1, padding: 0, activation: identity);
        _inverseNorm = new LayerNormalizationLayer<T>(dim);
        _inverseLstm = new BidirectionalLayer<T>(new LSTMLayer<T>(dim), mergeMode: true, activationFunction: identity);
        _reconstruct = new Conv1DLayer<T>(inputChannels: 2 * dim, outputChannels: dim, kernelSize: 2, dilation: 1, stride: 1, padding: 1, activation: identity);
        _logits = new FullyConnectedLayer<T>(2 * dim, numClasses, identity);
        RegisterSubLayer(_norm); RegisterSubLayer(_lstm); RegisterSubLayer(_merge);
        RegisterSubLayer(_inverseNorm); RegisterSubLayer(_inverseLstm); RegisterSubLayer(_reconstruct); RegisterSubLayer(_logits);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3)) return null;
        var spans = new OutputAxisContract(TensorAxis.Other, AxisRelation.Product(TensorAxis.Time, TensorAxis.Time));
        var classes = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_classes));
        return inputRank == 2 ? new[] { spans, classes }
            : new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), spans, classes };
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in GetSubLayers()) layer.ResetState();
    }

    // A width-2 convolution over time of [B, T, C] features.
    private Tensor<T> Convolve(Conv1DLayer<T> conv, Tensor<T> x) =>
        Engine.TensorPermute(conv.Forward(Engine.TensorPermute(x, new[] { 0, 2, 1 })), new[] { 0, 2, 1 });

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched) input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Shape.Length != 3 || input.Shape[2] != _dim) throw new ArgumentException($"PyramidSpanLayer expects [B, S, {_dim}].", nameof(input));
        int b = input.Shape[0], s = input.Shape[1], layers = Math.Min(_layers, s);
        // The pyramid (Equations 5 and 6).
        var hidden = new Tensor<T>[layers];
        var x = input;
        for (int l = 0; l < layers; l++)
        {
            hidden[l] = _lstm.Forward(_norm.Forward(x));
            if (l + 1 < layers) x = Convolve(_merge, hidden[l]);
        }
        // The inverse pyramid (Equations 7 and 8), from zeros above the top layer.
        var inverse = new Tensor<T>[layers];
        inverse[layers - 1] = new Tensor<T>(hidden[layers - 1].Shape.ToArray());
        for (int l = layers - 1; l > 0; l--)
        {
            var reconstructed = Convolve(_reconstruct, Engine.TensorConcatenate(new[] { hidden[l], inverse[l] }, 2));
            inverse[l - 1] = _inverseLstm.Forward(_inverseNorm.Forward(reconstructed));
        }
        // Equation 9 for every layer, then each span (start t, length l + 1) takes layer l's position t.
        var rows = new List<Tensor<T>>();
        var offsets = new int[layers];
        int total = 0;
        for (int l = 0; l < layers; l++)
        {
            int length = s - l;
            var joined = Engine.Reshape(Engine.TensorConcatenate(new[] { hidden[l], inverse[l] }, 2), new[] { b * length, 2 * _dim });
            rows.Add(Engine.TensorPermute(Engine.Reshape(_logits.Forward(joined), new[] { b, length, _classes }), new[] { 1, 0, 2 }));
            offsets[l] = total;
            total += length;
        }
        rows.Add(new Tensor<T>(new[] { 1, b, _classes }));                                           // zeros for uncovered spans
        var table = Engine.Reshape(Engine.TensorConcatenate(rows.ToArray(), 0), new[] { total + 1, b * _classes });
        var index = new int[s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                index[i * s + j] = j >= i && j - i < layers ? offsets[j - i] + i : total;
        var grid = Engine.TensorIndexSelect(table, new Tensor<int>(new[] { s * s }, new Vector<int>(index)), 0);
        var output = Engine.TensorPermute(Engine.Reshape(grid, new[] { s * s, b, _classes }), new[] { 1, 0, 2 });
        return unbatched ? Engine.Reshape(output, new[] { s * s, _classes }) : output;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Dim"] = _dim.ToString(ci);
        metadata["NumLayers"] = _layers.ToString(ci);
        metadata["NumClasses"] = _classes.ToString(ci);
        return metadata;
    }
}
