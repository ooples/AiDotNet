using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// The shape of an X-codec2 codec (Ye et al. 2025, "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based
/// Speech Synthesis", §2.2; reference: the xcodec2 0.1.5 package). The defaults are the released 16 kHz model
/// (<c>HKUSTAudio/xcodec2</c>): 50 tokens a second from one 65,536-entry FSQ codebook.
/// </summary>
internal sealed record XCodec2Configuration
{
    /// <summary>The sampling rate (16000).</summary>
    public int SampleRate { get; init; } = 16000;

    /// <summary>The acoustic encoder's first width (48), doubled by each downsampling block.</summary>
    public int EncoderDim { get; init; } = 48;

    /// <summary>The acoustic encoder's strides (2, 2, 4, 4, 5): a hop of 320 samples.</summary>
    public int[] EncoderRates { get; init; } = { 2, 2, 4, 4, 5 };

    /// <summary>The residual units' dilations in each encoder block (1, 3, 9).</summary>
    public int[] Dilations { get; init; } = { 1, 3, 9 };

    /// <summary>The acoustic encoder's output width (1024).</summary>
    public int AcousticDim { get; init; } = 1024;

    /// <summary>The semantic encoder's width (1024), which maps the Wav2Vec2-BERT hidden state to it.</summary>
    public int SemanticDim { get; init; } = 1024;

    /// <summary>The Wav2Vec2-BERT hidden state the semantic encoder reads (16: the 16th layer's output).</summary>
    public int SemanticLayer { get; init; } = 16;

    /// <summary>The FSQ levels per code dimension (eight 4s: 65,536 codes).</summary>
    public int[] FsqLevels { get; init; } = { 4, 4, 4, 4, 4, 4, 4, 4 };

    /// <summary>The decoder's width (1024).</summary>
    public int DecoderDim { get; init; } = 1024;

    /// <summary>The decoder's Transformer blocks (12).</summary>
    public int DecoderDepth { get; init; } = 12;

    /// <summary>The decoder's attention heads (16).</summary>
    public int DecoderHeads { get; init; } = 16;

    /// <summary>The semantic encoder (w2v-BERT 2.0).</summary>
    public Wav2Vec2BertConfiguration Semantic { get; init; } = new();

    /// <summary>The hop in samples (the product of the encoder strides, 320).</summary>
    public int HopLength => EncoderRates.Aggregate(1, (a, b) => a * b);

    /// <summary>The codebook size (the product of the FSQ levels).</summary>
    public int CodebookSize => FsqLevels.Aggregate(1, (a, b) => a * b);

    /// <summary>The fused width the quantizer reads (semantic + acoustic).</summary>
    public int FusedDim => SemanticDim + AcousticDim;
}

/// <summary>
/// BigVGAN's SnakeBeta with log-scale parameters, as X-codec2 uses it: <c>x + 1/(e^β + 1e-9) · sin²(e^α x)</c> with a
/// frequency α and a magnitude β per channel, both starting at 0 (reference <c>activations.SnakeBeta(alpha_logscale=True)</c>).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 3, 5", TestConstructorArgs = "3")]
[ElementWiseShape(Note = "A per-channel activation; the shape is carried through.")]
[AutoParameters]
internal sealed partial class XCodec2SnakeBetaLayer<T> : LayerBase<T>
{
    private readonly int _channels;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logAlpha;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logBeta;

    public override bool SupportsTraining => true;

    public XCodec2SnakeBetaLayer([LayerState] int channels)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _logAlpha = new Tensor<T>(new[] { channels });
        _logBeta = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_logAlpha, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_logBeta, PersistentTensorRole.Weights);
    }

    internal int Channels => _channels;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2];
        Tensor<T> PerChannel(Tensor<T> values) => Engine.TensorTile(Engine.Reshape(Engine.TensorExp(values), new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        var sine = Engine.TensorSin(Engine.TensorMultiply(input, PerChannel(_logAlpha)));
        var periodic = Engine.TensorDivide(Engine.TensorMultiply(sine, sine), Engine.TensorAddScalar(PerChannel(_logBeta), NumOps.FromDouble(1e-9)));
        return Engine.TensorAdd(input, periodic);
    }

    /// <summary>Loads the log-scale <c>alpha</c> and <c>beta</c> (one per channel).</summary>
    internal void Load(double[] alpha, double[] beta)
    {
        if (alpha.Length != _channels || beta.Length != _channels) throw new ArgumentException($"Expected {_channels} values each.");
        for (int c = 0; c < _channels; c++)
        {
            _logAlpha[c] = NumOps.FromDouble(alpha[c]);
            _logBeta[c] = NumOps.FromDouble(beta[c]);
        }
        Engine.InvalidatePersistentTensor(_logAlpha);
        Engine.InvalidatePersistentTensor(_logBeta);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}

/// <summary>An anti-aliased SnakeBeta (reference <c>Activation1d(SnakeBeta(...))</c>) with its checkpoint name.</summary>
internal sealed class XCodec2Activation<T>
{
    private readonly AntiAliasedSnake<T> _filtered;

    public XCodec2Activation(IEngine engine, List<LayerBase<T>> owner, int channels)
    {
        Snake = new XCodec2SnakeBetaLayer<T>(channels);
        _filtered = new AntiAliasedSnake<T>(engine, owner, Snake);
    }

    public XCodec2SnakeBetaLayer<T> Snake { get; }

    public Tensor<T> Forward(Tensor<T> x) => _filtered.Forward(x);
}

/// <summary>
/// X-codec2's residual unit (reference <c>ResidualUnit</c>): anti-aliased SnakeBeta → a dilated k = 7 weight-normalized
/// convolution → anti-aliased SnakeBeta → a 1×1 convolution, added to the input.
/// </summary>
internal sealed class XCodec2ResidualUnit<T>
{
    private readonly IEngine _engine;
    private readonly XCodec2Activation<T> _act1, _act2;
    private readonly NormedConv1DLayer<T> _conv1, _conv2;

    public XCodec2ResidualUnit(IEngine engine, int dim, int dilation, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _act1 = new XCodec2Activation<T>(engine, owner, dim);
        _conv1 = XCodec2.WeightNormConv(owner, dim, dim, 7, 1, dilation, 3 * dilation);
        _act2 = new XCodec2Activation<T>(engine, owner, dim);
        _conv2 = XCodec2.WeightNormConv(owner, dim, dim, 1, 1, 1, 0);
    }

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.block.0.act", _act1.Snake);
        yield return ($"{prefix}.block.1", _conv1);
        yield return ($"{prefix}.block.2.act", _act2.Snake);
        yield return ($"{prefix}.block.3", _conv2);
    }

    public Tensor<T> Forward(Tensor<T> x) => _engine.TensorAdd(x, _conv2.Forward(_act2.Forward(_conv1.Forward(_act1.Forward(x)))));
}

/// <summary>
/// X-codec2's acoustic encoder (reference <c>CodecEncoder_Transformer</c>, whose Transformer is commented out): a k = 7
/// convolution to <c>ngf</c>; per stride s three residual units at half the new width (dilations 1, 3, 9), anti-aliased
/// SnakeBeta and a strided convolution (kernel 2s, padding ⌊s/2⌋ + s mod 2) doubling it; then anti-aliased SnakeBeta and
/// a k = 3 convolution to the acoustic width. All convolutions are weight-normalized.
/// </summary>
internal sealed class XCodec2AcousticEncoder<T>
{
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(XCodec2ResidualUnit<T>[] Units, XCodec2Activation<T> Activation, NormedConv1DLayer<T> Down)> _blocks = new();
    private readonly XCodec2Activation<T> _finalActivation;

    public XCodec2AcousticEncoder(IEngine engine, XCodec2Configuration c, List<LayerBase<T>> owner)
    {
        int d = c.EncoderDim;
        _first = XCodec2.WeightNormConv(owner, 1, d, 7, 1, 1, 3);
        foreach (int s in c.EncoderRates)
        {
            var units = c.Dilations.Select(dilation => new XCodec2ResidualUnit<T>(engine, d, dilation, owner)).ToArray();
            var activation = new XCodec2Activation<T>(engine, owner, d);
            var down = XCodec2.WeightNormConv(owner, d, 2 * d, 2 * s, s, 1, s / 2 + s % 2);
            _blocks.Add((units, activation, down));
            d *= 2;
        }
        _finalActivation = new XCodec2Activation<T>(engine, owner, d);
        _last = XCodec2.WeightNormConv(owner, d, c.AcousticDim, 3, 1, 1, 1);
    }

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.conv_blocks.0", _first);
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (units, activation, down) = _blocks[i];
            string block = $"{prefix}.conv_blocks.{i + 1}.block";
            for (int u = 0; u < units.Length; u++)
                foreach (var named in units[u].Named($"{block}.{u}")) yield return named;
            yield return ($"{block}.{units.Length}.act", activation.Snake);
            yield return ($"{block}.{units.Length + 1}", down);
        }
        yield return ($"{prefix}.conv_final_block.0.act", _finalActivation.Snake);
        yield return ($"{prefix}.conv_final_block.1", _last);
    }

    /// <summary>The acoustic features <c>[1, acoustic, frames]</c> of a waveform <c>[1, 1, samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (units, activation, down) in _blocks)
        {
            foreach (var unit in units) x = unit.Forward(x);
            x = down.Forward(activation.Forward(x));
        }
        return _last.Forward(_finalActivation.Forward(x));
    }
}

/// <summary>
/// X-codec2's semantic encoder (reference <c>SemanticEncoder</c>): a k = 3 convolution, a residual pair of ReLU and k = 3
/// convolutions, and a k = 3 convolution. The reference's first ReLU acts in place on the convolution's output, so the
/// skip carries the rectified value: <c>out = final(conv₂(relu(conv₁(r))) + r)</c> with <c>r = relu(initial(x))</c>.
/// </summary>
internal sealed class XCodec2SemanticEncoder<T>
{
    private readonly IEngine _engine;
    private readonly NormedConv1DLayer<T> _initial, _conv1, _conv2, _final;

    public XCodec2SemanticEncoder(IEngine engine, int input, int width, int output, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _initial = XCodec2.Conv(owner, input, width, 3, 1, false);
        _conv1 = XCodec2.Conv(owner, width, width, 3, 1, true);
        _conv2 = XCodec2.Conv(owner, width, width, 3, 1, true);
        _final = XCodec2.Conv(owner, width, output, 3, 1, false);
    }

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.initial_conv", _initial);
        yield return ($"{prefix}.residual_blocks.1", _conv1);
        yield return ($"{prefix}.residual_blocks.3", _conv2);
        yield return ($"{prefix}.final_conv", _final);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var r = _engine.ReLU(_initial.Forward(x));
        return _final.Forward(_engine.TensorAdd(_conv2.Forward(_engine.ReLU(_conv1.Forward(r))), r));
    }
}

/// <summary>
/// The decoder backbone's residual block (reference <c>ResnetBlock</c> of <c>codec_decoder_vocos</c>): group norm (32
/// groups, ε = 1e-6), swish, a k = 3 convolution, group norm, swish, dropout (inactive at inference), a k = 3
/// convolution, added to the input.
/// </summary>
internal sealed class XCodec2ResnetBlock<T>
{
    private readonly IEngine _engine;
    private readonly GroupNormalizationLayer<T> _norm1, _norm2;
    private readonly NormedConv1DLayer<T> _conv1, _conv2;

    public XCodec2ResnetBlock(IEngine engine, int channels, List<LayerBase<T>> owner)
    {
        _engine = engine;
        owner.Add(_norm1 = new GroupNormalizationLayer<T>(32, channels, 1e-6));
        _conv1 = XCodec2.Conv(owner, channels, channels, 3, 1, true);
        owner.Add(_norm2 = new GroupNormalizationLayer<T>(32, channels, 1e-6));
        _conv2 = XCodec2.Conv(owner, channels, channels, 3, 1, true);
    }

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.norm1", _norm1);
        yield return ($"{prefix}.conv1", _conv1);
        yield return ($"{prefix}.norm2", _norm2);
        yield return ($"{prefix}.conv2", _conv2);
    }

    private Tensor<T> Norm(GroupNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var y = norm.Forward(_engine.Reshape(x, new[] { 1, c, t, 1 }));
        y = _engine.Reshape(y, new[] { 1, c, t });
        return _engine.TensorMultiply(y, _engine.Sigmoid(y));
    }

    public Tensor<T> Forward(Tensor<T> x) => _engine.TensorAdd(x, _conv2.Forward(Norm(_norm2, _conv1.Forward(Norm(_norm1, x)))));
}

/// <summary>
/// The decoder's Transformer block (reference <c>bs_roformer5.TransformerBlock</c>): pre-RMS-norm (ε = 1e-6)
/// bidirectional multi-head attention through a fused bias-free QKV projection, and a pre-RMS-norm SiLU MLP of four times
/// the width, each added to its input.
/// </summary>
/// <remarks>The reference also applies torchtune's rotary embedding to the queries and keys, but on tensors laid out
/// <c>[batch, heads, time, dim]</c> where torchtune reads axis 1 as the sequence: each head's queries and keys get the same
/// rotation whatever their time step, and a rotation applied to both sides of a dot product cancels. The scores are those
/// of attention with no position encoding, so none is applied here.</remarks>
internal sealed class XCodec2TransformerBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _dim, _heads;
    private readonly RMSNormalizationLayer<T> _attentionNorm, _feedForwardNorm;
    private readonly BiasFreeLinearLayer<T> _qkv, _output, _fc1, _fc2;

    public XCodec2TransformerBlock(IEngine engine, int dim, int heads, List<LayerBase<T>> owner)
    {
        if (dim % heads != 0) throw new ArgumentException($"The width {dim} must divide into {heads} heads.");
        _engine = engine;
        _dim = dim;
        _heads = heads;
        owner.Add(_attentionNorm = new RMSNormalizationLayer<T>(dim, 1e-6));
        owner.Add(_feedForwardNorm = new RMSNormalizationLayer<T>(dim, 1e-6));
        owner.Add(_qkv = new BiasFreeLinearLayer<T>(dim, 3 * dim));
        owner.Add(_output = new BiasFreeLinearLayer<T>(dim, dim));
        owner.Add(_fc1 = new BiasFreeLinearLayer<T>(dim, 4 * dim));
        owner.Add(_fc2 = new BiasFreeLinearLayer<T>(4 * dim, dim));
    }

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.att_norm", _attentionNorm);
        yield return ($"{prefix}.ffn_norm", _feedForwardNorm);
        yield return ($"{prefix}.att.c_attn", _qkv);
        yield return ($"{prefix}.att.c_proj", _output);
        yield return ($"{prefix}.mlp.fc1", _fc1);
        yield return ($"{prefix}.mlp.fc2", _fc2);
    }

    private Tensor<T> Heads(Tensor<T> qkv, int part, int length)
    {
        int e = _dim / _heads;
        var slice = _engine.TensorSlice(qkv, new[] { 0, part * _dim }, new[] { length, _dim });
        return _engine.TensorPermute(_engine.Reshape(slice, new[] { length, _heads, e }), new[] { 1, 0, 2 });
    }

    /// <summary>The block's output <c>[time, dim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        int length = x.Shape[0], e = _dim / _heads;
        var qkv = _qkv.Forward(_attentionNorm.Forward(x));
        var q = Heads(qkv, 0, length);
        var k = Heads(qkv, 1, length);
        var v = Heads(qkv, 2, length);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            MathHelper.GetNumericOperations<T>().FromDouble(1.0 / Math.Sqrt(e)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        x = _engine.TensorAdd(x, _output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, _dim })));
        var hidden = _fc1.Forward(_feedForwardNorm.Forward(x));
        return _engine.TensorAdd(x, _fc2.Forward(_engine.TensorMultiply(hidden, _engine.Sigmoid(hidden))));
    }
}

/// <summary>
/// X-codec2's decoder (reference <c>CodecDecoderVocos</c>, Vocos with a Transformer backbone; paper §2.2): a k = 7
/// convolution, two residual blocks, the Transformer blocks, two residual blocks and a layer norm (ε = 1e-6), then the
/// ISTFT head: a linear map to n_fft + 2 values per frame, split into log-magnitude and phase, magnitude
/// <c>min(exp(m), 100)</c>, and the inverse STFT with "same" padding (n_fft = 4 · hop, Hann window), which gives hop
/// samples a frame.
/// </summary>
internal sealed class XCodec2Decoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _fft;
    private readonly NormedConv1DLayer<T> _embed;
    private readonly XCodec2ResnetBlock<T>[] _prior, _post;
    private readonly XCodec2TransformerBlock<T>[] _transformers;
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly DenseLayer<T> _head;
    private readonly InverseStft<T> _istft;

    public XCodec2Decoder(IEngine engine, XCodec2Configuration c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        int dim = c.DecoderDim;
        _fft = 4 * c.HopLength;
        _embed = XCodec2.Conv(owner, dim, dim, 7, 3, true);
        _prior = new[] { new XCodec2ResnetBlock<T>(engine, dim, owner), new XCodec2ResnetBlock<T>(engine, dim, owner) };
        _transformers = Enumerable.Range(0, c.DecoderDepth).Select(_ => new XCodec2TransformerBlock<T>(engine, dim, c.DecoderHeads, owner)).ToArray();
        _post = new[] { new XCodec2ResnetBlock<T>(engine, dim, owner), new XCodec2ResnetBlock<T>(engine, dim, owner) };
        owner.Add(_finalNorm = new LayerNormalizationLayer<T>(dim, 1e-6));
        owner.Add(_head = new DenseLayer<T>(_fft + 2, (IActivationFunction<T>)new IdentityActivation<T>()));
        _istft = new InverseStft<T>(engine, _fft, c.HopLength, _fft, samePadding: true);
        Dim = dim;
    }

    public int Dim { get; }

    /// <summary>The ISTFT's window (the checkpoint's <c>head.istft.window</c> buffer).</summary>
    internal InverseStft<T> Istft => _istft;

    /// <summary>The window length.</summary>
    internal int WindowLength => _fft;

    /// <summary>Runs the head once so its dense layer takes its input width.</summary>
    internal void Materialize() => _head.Forward(new Tensor<T>(new[] { 1, Dim }));

    internal IEnumerable<(string Name, object Module)> Named(string prefix)
    {
        yield return ($"{prefix}.backbone.embed", _embed);
        for (int i = 0; i < _prior.Length; i++)
            foreach (var named in _prior[i].Named($"{prefix}.backbone.prior_net.{i}")) yield return named;
        for (int i = 0; i < _transformers.Length; i++)
            foreach (var named in _transformers[i].Named($"{prefix}.backbone.transformers.{i}")) yield return named;
        for (int i = 0; i < _post.Length; i++)
            foreach (var named in _post[i].Named($"{prefix}.backbone.post_net.{i}")) yield return named;
        yield return ($"{prefix}.backbone.final_layer_norm", _finalNorm);
        yield return ($"{prefix}.head.out", _head);
    }

    /// <summary>The waveform <c>[frames · hop]</c> of features <c>[frames, dim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        int frames = features.Shape[0];
        Tensor<T> Channels(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });
        Tensor<T> Rows(Tensor<T> channels) => _engine.TensorTranspose(_engine.Reshape(channels, new[] { channels.Shape[1], channels.Shape[2] }));
        var x = _embed.Forward(Channels(features));
        foreach (var block in _prior) x = block.Forward(x);
        var rows = Rows(x);
        foreach (var block in _transformers) rows = block.Forward(rows);
        x = Channels(rows);
        foreach (var block in _post) x = block.Forward(x);
        var h = _head.Forward(_finalNorm.Forward(Rows(x)));                                                  // [frames, n_fft + 2]
        int bins = _fft / 2 + 1;
        var spectrum = Channels(h);
        var m = _engine.TensorSlice(spectrum, new[] { 0, 0, 0 }, new[] { 1, bins, frames });
        var p = _engine.TensorSlice(spectrum, new[] { 0, bins, 0 }, new[] { 1, bins, frames });
        // min(exp(m), 100) = 100 − relu(100 − exp(m))
        var hundred = NumOps.FromDouble(100);
        var magnitude = _engine.TensorAddScalar(_engine.TensorNegate(_engine.ReLU(_engine.TensorAddScalar(_engine.TensorNegate(_engine.TensorExp(m)), hundred))), hundred);
        return _istft.Forward(magnitude, p);
    }
}

/// <summary>
/// X-codec2 (Ye et al. 2025, §2.2): a Wav2Vec2-BERT semantic encoder (its 16th hidden state through a small convolutional
/// encoder) and a convolutional acoustic encoder, concatenated and projected (<c>fc_prior</c>), quantized by one
/// finite-scalar quantizer (residual FSQ with one stage; levels 4⁸ = 65,536 codes; projections to and from 8 values), then
/// projected (<c>fc_post_a</c>) into the Vocos-style decoder. The released model makes 50 codes a second at 16 kHz.
/// </summary>
/// <remarks>
/// <para>Encoding follows the xcodec2 package's <c>encode_code</c>: the waveform is right-padded by
/// <c>320 − (samples mod 320)</c> zeros (a full hop when it is already whole), its features are taken with 160 more zeros
/// at each end, and the codes are the FSQ indices of the fused features. FSQ bounds each value as
/// <c>tanh(z + atanh(0.5/h)) · h − 0.5</c> with <c>h = 1.5015</c>; residual FSQ bounds before its stage and the stage
/// bounds again before rounding (half to even), and the index is Σ (level + 2) · 4^d. The bound's constants are taken in
/// single precision, as the package computes them from its integer level tensor.</para>
/// <para>Decoding (<c>decode_code</c>) maps each index to its levels, the levels to codes in [−1, 0.5], projects them
/// out, applies <c>fc_post_a</c> and runs the decoder.</para>
/// </remarks>
internal sealed class XCodec2<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers;
    private readonly Wav2Vec2BertFeatures<T> _features;
    private readonly XCodec2SemanticEncoder<T> _semanticEncoder;
    private readonly XCodec2AcousticEncoder<T> _acoustic;
    private readonly DenseLayer<T> _prior, _projectIn, _projectOut, _post;
    private readonly XCodec2Decoder<T> _decoder;
    private readonly double _halfLevel, _shift;

    public XCodec2(IEngine engine, XCodec2Configuration configuration, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (configuration.FsqLevels.Any(l => l != 4))
            throw new ArgumentException("X-codec2's quantizer has four levels per dimension.", nameof(configuration));
        if (configuration.SemanticLayer < 0 || configuration.SemanticLayer > configuration.Semantic.Layers)
            throw new ArgumentException("The semantic layer must be one of the encoder's hidden states.", nameof(configuration));
        _layers = owner;
        DenseLayer<T> Linear(int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            owner.Add(layer);
            return layer;
        }
        _features = new Wav2Vec2BertFeatures<T>(engine, configuration.Semantic);
        Semantic = new Wav2Vec2BertNetwork<T>(engine, owner, configuration.Semantic, configuration.SemanticLayer);
        _semanticEncoder = new XCodec2SemanticEncoder<T>(engine, configuration.Semantic.HiddenSize, configuration.SemanticDim, configuration.SemanticDim, owner);
        _acoustic = new XCodec2AcousticEncoder<T>(engine, configuration, owner);
        _prior = Linear(configuration.FusedDim);
        _projectIn = Linear(configuration.FsqLevels.Length);
        _projectOut = Linear(configuration.FusedDim);
        _post = Linear(configuration.DecoderDim);
        _decoder = new XCodec2Decoder<T>(engine, configuration, owner);
        // FSQ.bound's constants as the package forms them in float32: h = (levels − 1)(1 + 1e-3)/2, atanh(0.5 / h).
        float half = 3f * 1.001f / 2f;
        _halfLevel = half;
        _shift = (float)Math.Log((1 + 0.5f / half) / (1 - 0.5f / half)) / 2;
        foreach (var layer in owner) layer.SetTrainingMode(false);
        Materialize();
    }

    public XCodec2Configuration Configuration { get; }

    /// <summary>The semantic encoder (only the layers up to <see cref="XCodec2Configuration.SemanticLayer"/>).</summary>
    public Wav2Vec2BertNetwork<T> Semantic { get; }

    /// <summary>Samples per code.</summary>
    public int HopLength => Configuration.HopLength;

    // The dense layers take their input width on first use; run each once so they can be loaded and counted.
    private void Materialize()
    {
        using var _ = new NoGradScope<T>();
        _prior.Forward(new Tensor<T>(new[] { 1, Configuration.FusedDim }));
        _projectIn.Forward(new Tensor<T>(new[] { 1, Configuration.FusedDim }));
        _projectOut.Forward(new Tensor<T>(new[] { 1, Configuration.FsqLevels.Length }));
        _post.Forward(new Tensor<T>(new[] { 1, Configuration.FusedDim }));
        Semantic.Projection.Forward(new Tensor<T>(new[] { 1, Configuration.Semantic.FeatureInputDim }));
        _decoder.Materialize();
    }

    /// <summary>The padded waveform <c>encode_code</c> encodes: <c>320 − (samples mod 320)</c> zeros appended.</summary>
    public double[] PadForEncoding(IReadOnlyList<double> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int hop = HopLength;
        var padded = new double[audio.Count + (hop - audio.Count % hop)];
        for (int i = 0; i < audio.Count; i++) padded[i] = audio[i];
        return padded;
    }

    /// <summary>The semantic features <c>encode_code</c> reads for a padded waveform: the w2v-BERT features of the
    /// waveform with half a hop of zeros at each end.</summary>
    public Tensor<T> Features(IReadOnlyList<double> padded)
    {
        int half = HopLength / 2;
        var framed = new double[padded.Count + 2 * half];
        for (int i = 0; i < padded.Count; i++) framed[half + i] = padded[i];
        return _features.Forward(framed);
    }

    /// <summary>The codes of a 16 kHz waveform (<c>encode_code</c>).</summary>
    public int[] Encode(IReadOnlyList<double> audio)
    {
        var padded = PadForEncoding(audio);
        return Encode(padded, Features(padded));
    }

    /// <summary>The codes of an already padded waveform given its w2v-BERT features <c>[rows, 160]</c>.</summary>
    public int[] Encode(IReadOnlyList<double> padded, Tensor<T> features)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var hidden = Semantic.HiddenStates(features)[Configuration.SemanticLayer];                         // [rows, hs]
        var semantic = _semanticEncoder.Forward(_engine.Reshape(_engine.TensorTranspose(hidden), new[] { 1, hidden.Shape[1], hidden.Shape[0] }));
        var wave = new Tensor<T>(new[] { 1, 1, padded.Count });
        for (int i = 0; i < padded.Count; i++) wave[0, 0, i] = Ops.FromDouble(padded[i]);
        var acoustic = _acoustic.Forward(wave);
        int frames = acoustic.Shape[2];
        if (semantic.Shape[2] != frames)
            throw new InvalidOperationException($"The semantic encoder made {semantic.Shape[2]} frames and the acoustic encoder {frames}.");
        var fused = _engine.TensorConcatenate(new[] { semantic, acoustic }, 1);
        var z = _projectIn.Forward(_prior.Forward(_engine.TensorTranspose(_engine.Reshape(fused, new[] { Configuration.FusedDim, frames }))));
        int dims = Configuration.FsqLevels.Length;
        var codes = new int[frames];
        for (int t = 0; t < frames; t++)
        {
            int index = 0, basis = 1;
            for (int d = 0; d < dims; d++)
            {
                double bounded = Bound(Bound(Ops.ToDouble(z[t, d])));
                int level = (int)Math.Round(bounded, MidpointRounding.ToEven) + 2;
                index += level * basis;
                basis *= 4;
            }
            codes[t] = index;
        }
        return codes;
    }

    private double Bound(double z) => Math.Tanh(z + _shift) * _halfLevel - 0.5;

    /// <summary>The decoder's input <c>[frames, decoder]</c> for codes (<c>get_output_from_indices</c>, then
    /// <c>fc_post_a</c>).</summary>
    public Tensor<T> Embed(IReadOnlyList<int> codes)
    {
        if (codes is null || codes.Count == 0) throw new ArgumentException("There are no codes.", nameof(codes));
        int dims = Configuration.FsqLevels.Length;
        var values = new Tensor<T>(new[] { codes.Count, dims });
        for (int t = 0; t < codes.Count; t++)
        {
            int code = codes[t];
            if (code < 0 || code >= Configuration.CodebookSize)
                throw new ArgumentOutOfRangeException(nameof(codes), code, $"Codes lie in [0, {Configuration.CodebookSize}).");
            for (int d = 0; d < dims; d++, code /= 4) values[t, d] = Ops.FromDouble((code % 4 - 2) / 2.0);
        }
        return _post.Forward(_projectOut.Forward(values));
    }

    /// <summary>The 16 kHz waveform of codes (<c>decode_code</c>), <c>codes · hop</c> samples.</summary>
    public double[] Decode(IReadOnlyList<int> codes)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var audio = _decoder.Forward(Embed(codes));
        var result = new double[audio.Length];
        for (int i = 0; i < result.Length; i++) result[i] = Ops.ToDouble(audio[i]);
        return result;
    }

    /// <summary>Every module outside the semantic network under the checkpoint's names.</summary>
    internal IEnumerable<(string Name, object Module)> Named()
    {
        foreach (var named in _semanticEncoder.Named("SemanticEncoder_module")) yield return named;
        foreach (var named in _acoustic.Named("CodecEnc")) yield return named;
        yield return ("fc_prior", _prior);
        yield return ("generator.quantizer.project_in", _projectIn);
        yield return ("generator.quantizer.project_out", _projectOut);
        yield return ("fc_post_a", _post);
        foreach (var named in _decoder.Named("generator")) yield return named;
    }

    /// <summary>
    /// Loads an X-codec2 state dict (the released <c>model.safetensors</c>): the semantic network under
    /// <c>semantic_model.</c>, weight-normalized convolutions as <c>weight_g</c>/<c>weight_v</c> or
    /// <c>parametrizations.weight.original0/1</c>, SnakeBeta parameters as <c>act.alpha</c> and <c>act.beta</c> (the
    /// package's module names the magnitude <c>bias</c>, which transformers' loader renames it to), and the ISTFT's window
    /// buffer when present. The anti-aliasing filters stored as buffers are recomputed, not read. Returns the names it
    /// read.
    /// </summary>
    public HashSet<string> Load(Func<string, double[]?> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[]? Read(string name, int count, bool optional = false)
        {
            var values = read(name);
            if (values is null)
            {
                if (optional) return null;
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            }
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        Semantic.LoadTorchWeights((name, shape) => Read(name, shape.Aggregate(1, (a, b) => a * b))!, "semantic_model.");
        // The ISTFT window is a buffer PyTorch computed in single precision; use the stored values when present.
        var window = Read("generator.head.istft.window", _decoder.WindowLength, optional: true);
        if (window is not null) _decoder.Istft.LoadWindow(window);
        foreach (var (name, module) in Named())
        {
            switch (module)
            {
                case NormedConv1DLayer<T> conv:
                    var shape = conv.TorchWeightShape;
                    int count = shape.Aggregate(1, (a, b) => a * b);
                    if (conv.Normalization == ConvolutionNormalization.Weight)
                    {
                        var direction = Read($"{name}.parametrizations.weight.original1", count, optional: true);
                        var gain = direction is null ? Read($"{name}.weight_g", shape[0])! : Read($"{name}.parametrizations.weight.original0", shape[0])!;
                        direction ??= Read($"{name}.weight_v", count)!;
                        conv.LoadTorchWeights(direction, gain, Read($"{name}.bias", conv.OutputChannels)!);
                    }
                    else
                    {
                        var bias = conv.HasBias ? Read($"{name}.bias", conv.OutputChannels)! : new double[conv.OutputChannels];
                        conv.LoadTorchWeights(Read($"{name}.weight", count)!, null, bias);
                    }
                    break;
                case XCodec2SnakeBetaLayer<T> snake:
                    var alpha = Read($"{name}.alpha", snake.Channels)!;
                    var beta = Read($"{name}.beta", snake.Channels, optional: true) ?? Read($"{name}.bias", snake.Channels)!;
                    snake.Load(alpha, beta);
                    break;
                case GroupNormalizationLayer<T> group:
                    XCodec2.LoadNorm(_engine, group.GetGammaTensor(), group.GetBetaTensor(), Read($"{name}.weight", group.GetGammaTensor().Length)!,
                        Read($"{name}.bias", group.GetBetaTensor().Length)!);
                    break;
                case LayerNormalizationLayer<T> norm:
                    int width = norm.GetGammaTensor().Length;
                    TorchParameters.LayerNorm(_engine, norm, Read($"{name}.weight", width)!, Read($"{name}.bias", width)!);
                    break;
                case RMSNormalizationLayer<T> rms:
                    var gamma = rms.GetGammaTensor();
                    var weight = Read($"{name}.weight", gamma.Length)!;
                    for (int i = 0; i < weight.Length; i++) gamma[i] = Ops.FromDouble(weight[i]);
                    _engine.InvalidatePersistentTensor(gamma);
                    break;
                case BiasFreeLinearLayer<T> linear:
                    TorchParameters.Linear(linear, linear.InputSize, linear.OutputSize, Read($"{name}.weight", linear.InputSize * linear.OutputSize)!);
                    break;
                case DenseLayer<T> dense:
                    int inputs = dense.GetWeights().Shape[0], outputs = dense.GetWeights().Shape[1];
                    TorchParameters.Linear(_engine, dense, inputs, outputs, Read($"{name}.weight", inputs * outputs)!, Read($"{name}.bias", outputs)!);
                    break;
                default:
                    throw new InvalidOperationException($"No loader for {module.GetType().Name} ('{name}').");
            }
        }
        return used;
    }
}

/// <summary>Construction helpers for <see cref="XCodec2{T}"/>.</summary>
internal static class XCodec2
{
    internal static NormedConv1DLayer<T> WeightNormConv<T>(List<LayerBase<T>> owner, int input, int output, int kernel, int stride, int dilation, int padding)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, false, ConvolutionNormalization.Weight);
        owner.Add(conv);
        return conv;
    }

    internal static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> owner, int input, int output, int kernel, int padding, bool bias)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, 1, 1, padding, false, ConvolutionNormalization.None, useBias: bias);
        owner.Add(conv);
        return conv;
    }

    internal static void LoadNorm<T>(IEngine engine, Tensor<T> gamma, Tensor<T> beta, double[] weight, double[] bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < weight.Length; i++)
        {
            gamma[i] = ops.FromDouble(weight[i]);
            beta[i] = ops.FromDouble(bias[i]);
        }
        engine.InvalidatePersistentTensor(gamma);
        engine.InvalidatePersistentTensor(beta);
    }
}
