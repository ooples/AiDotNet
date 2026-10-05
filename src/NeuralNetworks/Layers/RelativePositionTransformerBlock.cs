using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A post-normalization Transformer encoder layer with relative position representations (Shaw et al. 2018) and a
/// convolutional feed-forward network: Glow-TTS's text encoder layer, shared by Grad-TTS, Matcha-TTS and VITS.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Glow-TTS (Kim et al. 2020, §3.3, App. A.1): "we do not use absolute positional encoding but relative position
/// representations ... the maximum relative position of 4". Following the reference implementation
/// (<c>attentions.Encoder</c>, <c>attentions.MultiHeadAttention</c>, <c>attentions.FFN</c>):
/// </para>
/// <list type="bullet">
/// <item>attention logits <c>(q_i·k_j + q_i·r^K_{j−i}) / √d_k</c>, outputs <c>Σ_j p_ij (v_j + r^V_{j−i})</c>, with learned
/// relative embeddings <c>r^K, r^V</c> for offsets in <c>[−w, w]</c> shared by every head and zero beyond the window;
/// dropout on the attention probabilities;</item>
/// <item><c>x = LN(x + Dropout(Attention(x)))</c>, <c>x = LN(x + Dropout(FFN(x)))</c>, with
/// <c>FFN = Conv1D_k(Dropout(ReLU(Conv1D_k(x))))</c> and layer normalization over channels (ε = 1e-4).</item>
/// </list>
/// <para>With <c>rotary</c> set, positions enter through rotary embeddings instead (Su et al. 2021), applied to the first
/// half of each head's query and key channels with the half-split pairing of Matcha-TTS's encoder
/// (<c>RotaryPositionalEmbeddings(k_channels * 0.5)</c>; Mehta et al. 2024), and no relative tables are used.</para>
/// <para>With <c>positional</c> cleared the attention carries no position information at all, as the reference
/// <c>attentions.Encoder</c> with <c>window_size=None</c> that VITS2 places in its normalizing flows (Kong et al. 2023,
/// §2.3).</para>
/// <para>Input and output are <c>[time, hidden]</c> or <c>[batch, time, hidden]</c>.</para>
/// <para><b>For Beginners:</b> Self-attention that knows how far apart two positions are rather than where each sits in
/// the sentence, followed by a small convolution — a good fit for text, where nearby sounds matter most.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 6, 8", TestConstructorArgs = "8, 2, 16, 3, 0.0, 4, false")]
[ElementWiseShape(Note = "A residual Transformer layer; the shape is carried through.")]
[AutoParameters]
public partial class RelativePositionTransformerBlock<T> : LayerBase<T>
{
    private readonly int _hidden;
    private readonly int _heads;
    private readonly int _filter;
    private readonly int _kernelSize;
    private readonly double _dropoutRate;
    private readonly int _window;
    private readonly bool _rotary;
    private readonly bool _positional;

    [SubLayerInput("1, _hidden")]
    private readonly DenseLayer<T> _query;
    [SubLayerInput("1, _hidden")]
    private readonly DenseLayer<T> _key;
    [SubLayerInput("1, _hidden")]
    private readonly DenseLayer<T> _value;
    [SubLayerInput("1, _hidden")]
    private readonly DenseLayer<T> _output;
    [SubLayerInput("_hidden")]
    private readonly LayerNormalizationLayer<T> _attentionNorm;
    [SubLayerInput("1, _hidden, 1")]
    private readonly Conv1DLayer<T> _conv1;
    [SubLayerInput("1, _filter, 1")]
    private readonly Conv1DLayer<T> _conv2;
    [SubLayerInput("_hidden")]
    private readonly LayerNormalizationLayer<T> _ffnNorm;
    private readonly DropoutLayer<T>? _dropout;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T>? _relativeKeys;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T>? _relativeValues;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="hidden">Model width (192 in Glow-TTS).</param>
    /// <param name="heads">Attention heads (2).</param>
    /// <param name="filter">Inner width of the feed-forward convolutions (768).</param>
    /// <param name="kernelSize">Odd kernel of the feed-forward convolutions (3).</param>
    /// <param name="dropoutRate">Dropout (0.1).</param>
    /// <param name="window">Maximum relative position (4); unused with <paramref name="rotary"/> or without
    /// <paramref name="positional"/>.</param>
    /// <param name="rotary">Use rotary position embeddings on half of each head's channels instead of relative tables.</param>
    public RelativePositionTransformerBlock(
        [LayerState] int hidden,
        [LayerState] int heads,
        [LayerState] int filter,
        [LayerState] int kernelSize,
        [LayerState] double dropoutRate,
        [LayerState] int window,
        [LayerState] bool rotary = false,
        [LayerState] bool positional = true)
        : base(new[] { hidden }, new[] { hidden })
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (heads <= 0 || hidden % heads != 0) throw new ArgumentException($"Heads ({heads}) must divide the width ({hidden}).", nameof(heads));
        if (filter <= 0) throw new ArgumentOutOfRangeException(nameof(filter));
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd.");
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        if (window < 0) throw new ArgumentOutOfRangeException(nameof(window));
        if (rotary && hidden / heads < 4) throw new ArgumentException("Rotary embeddings need at least 4 channels per head.", nameof(heads));
        if (rotary && !positional) throw new ArgumentException("Rotary embeddings are a positional encoding.", nameof(positional));
        _hidden = hidden;
        _heads = heads;
        _filter = filter;
        _kernelSize = kernelSize;
        _dropoutRate = dropoutRate;
        _window = window;
        _rotary = rotary;
        _positional = positional;

        IActivationFunction<T> identity = new IdentityActivation<T>();
        _query = new DenseLayer<T>(hidden, identity);
        _key = new DenseLayer<T>(hidden, identity);
        _value = new DenseLayer<T>(hidden, identity);
        _output = new DenseLayer<T>(hidden, identity);
        _attentionNorm = new LayerNormalizationLayer<T>(hidden, 1e-4);
        _conv1 = new Conv1DLayer<T>(inputChannels: hidden, outputChannels: filter, kernelSize: kernelSize, activation: new ReLUActivation<T>());
        _conv2 = new Conv1DLayer<T>(inputChannels: filter, outputChannels: hidden, kernelSize: kernelSize);
        _ffnNorm = new LayerNormalizationLayer<T>(hidden, 1e-4);
        if (dropoutRate > 0) _dropout = new DropoutLayer<T>(dropoutRate);

        RegisterSubLayer(_query);
        RegisterSubLayer(_key);
        RegisterSubLayer(_value);
        RegisterSubLayer(_output);
        RegisterSubLayer(_attentionNorm);
        RegisterSubLayer(_conv1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_ffnNorm);
        if (_dropout is not null) RegisterSubLayer(_dropout);
        if (rotary || !positional) return;

        // N(0, d_k^-1/2) as in the reference implementation; one table shared by all heads.
        int dk = hidden / heads, span = 2 * window + 1;
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        double std = Math.Pow(dk, -0.5);
        _relativeKeys = new Tensor<T>(new[] { span, dk });
        _relativeValues = new Tensor<T>(new[] { span, dk });
        foreach (var table in new[] { _relativeKeys, _relativeValues })
            for (int i = 0; i < table.Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                table[i] = NumOps.FromDouble(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            }
        RegisterTrainableParameter(_relativeKeys, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_relativeValues, PersistentTensorRole.Weights);
    }

    internal DenseLayer<T> QueryProjection => _query;
    internal DenseLayer<T> KeyProjection => _key;
    internal DenseLayer<T> ValueProjection => _value;
    internal DenseLayer<T> OutputProjection => _output;
    internal Tensor<T> RelativeKeys => _relativeKeys ?? throw new InvalidOperationException("Only a relative-position block has relative tables.");
    internal Tensor<T> RelativeValues => _relativeValues ?? throw new InvalidOperationException("Only a relative-position block has relative tables.");

    /// <summary>The self-attention sublayer alone on one sequence <c>[time, hidden]</c>, before its residual and norm.</summary>
    internal Tensor<T> SelfAttention(Tensor<T> x) => Attention(x);

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _hidden)
            throw new ArgumentException($"Expected [time, {_hidden}] or [batch, time, {_hidden}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        if (input.Rank == 2)
            return Run(input);
        int batch = input.Shape[0], time = input.Shape[1];
        var outputs = new Tensor<T>[batch];
        for (int b = 0; b < batch; b++)
            outputs[b] = Engine.Reshape(Run(Engine.Reshape(Engine.TensorSlice(input, new[] { b, 0, 0 }, new[] { 1, time, _hidden }),
                new[] { time, _hidden })), new[] { 1, time, _hidden });
        return batch == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
    }

    // One sequence, [time, hidden].
    private Tensor<T> Run(Tensor<T> x)
    {
        var attended = Attention(x);
        if (_dropout is not null) attended = _dropout.Forward(attended);
        x = _attentionNorm.Forward(Engine.TensorAdd(x, attended));

        var channelsFirst = Engine.Reshape(Engine.TensorTranspose(x), new[] { 1, _hidden, x.Shape[0] });
        var hidden = _conv1.Forward(channelsFirst);
        if (_dropout is not null) hidden = _dropout.Forward(hidden);
        var ffn = Engine.TensorTranspose(Engine.Reshape(_conv2.Forward(hidden), new[] { _hidden, x.Shape[0] }));
        if (_dropout is not null) ffn = _dropout.Forward(ffn);
        return _ffnNorm.Forward(Engine.TensorAdd(x, ffn));
    }

    private Tensor<T> Attention(Tensor<T> x)
    {
        int length = x.Shape[0], dk = _hidden / _heads, span = 2 * _window + 1;
        var q = _query.Forward(x);
        var k = _key.Forward(x);
        var v = _value.Forward(x);
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(dk));

        // Index maps between [length, span] relative tables and [length, length] absolute matrices; zero outside the window.
        var relToAbs = new Tensor<int>(new[] { length * length });
        var relToAbsMask = new Tensor<T>(new[] { length, length });
        var absToRel = new Tensor<int>(new[] { length * span });
        var absToRelMask = new Tensor<T>(new[] { length, span });
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                int r = j - i + _window;
                bool inside = r >= 0 && r < span;
                relToAbs[i * length + j] = inside ? i * span + r : 0;
                relToAbsMask[i, j] = inside ? NumOps.One : NumOps.Zero;
            }
            for (int r = 0; r < span; r++)
            {
                int j = i + r - _window;
                bool inside = j >= 0 && j < length;
                absToRel[i * span + r] = inside ? i * length + j : 0;
                absToRelMask[i, r] = inside ? NumOps.One : NumOps.Zero;
            }
        }

        var heads = new Tensor<T>[_heads];
        for (int h = 0; h < _heads; h++)
        {
            var qh = Engine.TensorSlice(q, new[] { 0, h * dk }, new[] { length, dk });
            var kh = Engine.TensorSlice(k, new[] { 0, h * dk }, new[] { length, dk });
            var vh = Engine.TensorSlice(v, new[] { 0, h * dk }, new[] { length, dk });
            if (_rotary || !_positional)
            {
                if (_rotary)
                {
                    qh = Rotate(qh);
                    kh = Rotate(kh);
                }
                var rotaryProbabilities = Engine.TensorSoftmax(Engine.TensorMultiplyScalar(Engine.TensorMatMul(qh, Engine.TensorTranspose(kh)), scale), axis: 1);
                if (_dropout is not null) rotaryProbabilities = _dropout.Forward(rotaryProbabilities);
                heads[h] = Engine.TensorMatMul(rotaryProbabilities, vh);
                continue;
            }

            var content = Engine.TensorMatMul(qh, Engine.TensorTranspose(kh));                       // [L, L]
            var relative = Engine.TensorMatMul(qh, Engine.TensorTranspose(_relativeKeys!));           // [L, span]
            var relativeAbs = Engine.Reshape(
                Engine.TensorIndexSelect(Engine.Reshape(relative, new[] { length * span, 1 }), relToAbs, 0), new[] { length, length });
            var logits = Engine.TensorMultiplyScalar(Engine.TensorAdd(content, Engine.TensorMultiply(relativeAbs, relToAbsMask)), scale);
            var probabilities = Engine.TensorSoftmax(logits, axis: 1);
            if (_dropout is not null) probabilities = _dropout.Forward(probabilities);

            var output = Engine.TensorMatMul(probabilities, vh);                                    // [L, dk]
            var relativeWeights = Engine.TensorMultiply(Engine.Reshape(
                Engine.TensorIndexSelect(Engine.Reshape(probabilities, new[] { length * length, 1 }), absToRel, 0), new[] { length, span }),
                absToRelMask);
            heads[h] = Engine.TensorAdd(output, Engine.TensorMatMul(relativeWeights, _relativeValues!));
        }
        var concatenated = _heads == 1 ? heads[0] : Engine.TensorConcatenate(heads, 1);
        return _output.Forward(concatenated);
    }

    /// <summary>
    /// Rotary position embedding of <paramref name="x"/> <c>[time, dk]</c> on its first <c>d = ⌊dk / 2⌋</c> channels:
    /// <c>x' = x ⊙ cos(tθ) + rotate_half(x) ⊙ sin(tθ)</c> with <c>θ_i = 10000^(−2i/d)</c> repeated over both halves and
    /// <c>rotate_half(x) = [−x_{d/2..d}, x_{0..d/2}]</c>; the remaining channels pass through.
    /// </summary>
    private Tensor<T> Rotate(Tensor<T> x)
    {
        int length = x.Shape[0], dk = x.Shape[1], d = dk / 2, half = d / 2;
        if (half == 0) return x;
        var cos = new Tensor<T>(new[] { length, d });
        var sin = new Tensor<T>(new[] { length, d });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < d; i++)
            {
                double theta = Math.Pow(10000.0, -2.0 * (i % half) / d);
                cos[t, i] = NumOps.FromDouble(Math.Cos(t * theta));
                sin[t, i] = NumOps.FromDouble(Math.Sin(t * theta));
            }
        var rope = Engine.TensorSlice(x, new[] { 0, 0 }, new[] { length, d });
        var rotatedHalf = Engine.TensorConcatenate(new[]
        {
            Engine.TensorNegate(Engine.TensorSlice(rope, new[] { 0, half }, new[] { length, d - half })),
            Engine.TensorSlice(rope, new[] { 0, 0 }, new[] { length, half }),
        }, 1);
        var rotated = Engine.TensorAdd(Engine.TensorMultiply(rope, cos), Engine.TensorMultiply(rotatedHalf, sin));
        return d == dk ? rotated : Engine.TensorConcatenate(new[] { rotated, Engine.TensorSlice(x, new[] { 0, d }, new[] { length, dk - d }) }, 1);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Hidden"] = _hidden.ToString(inv);
        metadata["Heads"] = _heads.ToString(inv);
        metadata["Filter"] = _filter.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        metadata["Window"] = _window.ToString(inv);
        metadata["Rotary"] = _rotary.ToString(inv);
        metadata["Positional"] = _positional.ToString(inv);
        return metadata;
    }
}
