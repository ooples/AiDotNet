using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Codecs;

/// <summary>SEANet's padding helpers (reference <c>modules/conv.py</c>).</summary>
internal static class Seanet
{
    /// <summary>Pads <c>[1, C, T]</c> (reference <c>pad1d</c>: reflect on an input shorter than the pad first extends it
    /// with zeros on the right).</summary>
    internal static Tensor<T> Pad<T>(IEngine engine, Tensor<T> x, int left, int right, bool reflect)
    {
        if (left == 0 && right == 0) return x;
        if (!reflect) return AiDotNet.TextToSpeech.Vocoders.VocoderOps.ZeroPad(engine, x, left, right);
        int length = x.Shape[2], maxPad = Math.Max(left, right), extraPad = 0;
        if (length <= maxPad)
        {
            extraPad = maxPad - length + 1;
            x = AiDotNet.TextToSpeech.Vocoders.VocoderOps.ZeroPad(engine, x, 0, extraPad);
        }
        var padded = AiDotNet.TextToSpeech.Vocoders.VocoderOps.ReflectPad(engine, x, left, right);
        return extraPad == 0 ? padded
            : engine.TensorSlice(padded, new[] { 0, 0, 0 }, new[] { 1, padded.Shape[1], padded.Shape[2] - extraPad });
    }

    /// <summary>Applies a one-group normalization over channels and time to <c>[1, C, T]</c> (the layer reads rank 4).</summary>
    internal static Tensor<T> Normalize<T>(IEngine engine, GroupNormalizationLayer<T>? norm, Tensor<T> y)
    {
        if (norm is null) return y;
        var normalized = norm.Forward(engine.Reshape(y, new[] { 1, y.Shape[1], y.Shape[2], 1 }));
        return engine.Reshape(normalized, new[] { 1, y.Shape[1], y.Shape[2] });
    }

    /// <summary>The extra right padding that keeps the last frame whole (reference <c>get_extra_padding_for_conv1d</c>).</summary>
    internal static int ExtraPadding(int length, int kernel, int stride, int paddingTotal)
    {
        double frames = (length - kernel + paddingTotal) / (double)stride + 1;
        int ideal = ((int)Math.Ceiling(frames) - 1) * stride + (kernel - paddingTotal);
        return ideal - length;
    }
}

/// <summary>
/// A streamable 1-D convolution (EnCodec, Défossez et al. 2022, §3.1; reference <c>modules/conv.py</c> <c>SConv1d</c>):
/// the input is padded so that the output has ⌈T / stride⌉ frames — the total padding k_eff − stride split evenly
/// (non-causal, the extra sample on the left) or entirely on the left (causal), plus the extra right padding that keeps
/// the last frame whole — in reflect or zero mode, then convolved without padding.
/// </summary>
internal sealed class StreamableConv1d<T>
{
    private readonly IEngine _engine;
    private readonly int _kernel, _stride, _dilation;
    private readonly bool _causal, _reflect;

    public StreamableConv1d(IEngine engine, int input, int output, int kernel, int stride, int dilation, bool causal, bool reflect,
        ConvolutionNormalization normalization, List<LayerBase<T>> owner, int groups = 1, bool bias = true, bool timeGroupNorm = false)
    {
        _engine = engine;
        _kernel = kernel;
        _stride = stride;
        _dilation = dilation;
        _causal = causal;
        _reflect = reflect;
        Conv = new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, groups, 0, false, normalization, bias);
        owner.Add(Conv);
        if (timeGroupNorm)
        {
            Norm = new GroupNormalizationLayer<T>(1, output);
            owner.Add(Norm);
        }
    }

    /// <summary>The layer normalization over channels and time that follows the convolution, if any.</summary>
    public GroupNormalizationLayer<T>? Norm { get; }

    public NormedConv1DLayer<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int effective = (_kernel - 1) * _dilation + 1;
        int total = effective - _stride;
        int extra = Seanet.ExtraPadding(x.Shape[2], effective, _stride, total);
        int left, right;
        if (_causal)
        {
            left = total;
            right = extra;
        }
        else
        {
            int r = total / 2;
            left = total - r;
            right = r + extra;
        }
        return Seanet.Normalize(_engine, Norm, Conv.Forward(Seanet.Pad(_engine, x, left, right, _reflect)));
    }
}

/// <summary>
/// A streamable transposed convolution (reference <c>SConvTranspose1d</c>): the transposed convolution without padding,
/// then k − stride samples trimmed — split evenly (the extra one on the left) or, causally, by the trim-right ratio.
/// </summary>
internal sealed class StreamableConvTranspose1d<T>
{
    private readonly IEngine _engine;
    private readonly int _kernel, _stride;
    private readonly bool _causal;
    private readonly double _trimRightRatio;

    public StreamableConvTranspose1d(IEngine engine, int input, int output, int kernel, int stride, bool causal, double trimRightRatio,
        ConvolutionNormalization normalization, List<LayerBase<T>> owner, bool timeGroupNorm = false)
    {
        _engine = engine;
        _kernel = kernel;
        _stride = stride;
        _causal = causal;
        _trimRightRatio = trimRightRatio;
        Conv = new NormedConv1DLayer<T>(input, output, kernel, stride, 1, 1, 0, true, normalization);
        owner.Add(Conv);
        if (timeGroupNorm)
        {
            Norm = new GroupNormalizationLayer<T>(1, output);
            owner.Add(Norm);
        }
    }

    /// <summary>The layer normalization over channels and time that follows the convolution, if any.</summary>
    public GroupNormalizationLayer<T>? Norm { get; }

    public NormedConv1DLayer<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Seanet.Normalize(_engine, Norm, Conv.Forward(x));
        int total = _kernel - _stride, right, left;
        if (_causal)
        {
            right = (int)Math.Ceiling(total * _trimRightRatio);
            left = total - right;
        }
        else
        {
            right = total / 2;
            left = total - right;
        }
        return _engine.TensorSlice(y, new[] { 0, 0, left }, new[] { 1, y.Shape[1], y.Shape[2] - left - right });
    }
}

/// <summary>
/// SEANet's LSTM block (reference <c>SLSTM</c>): a stack of LSTM layers over time whose output is added to its input.
/// Bidirectional (SpeechTokenizer's encoder, as <c>nn.LSTM(dim, dim, layers, bidirectional=True)</c>) runs each direction at
/// the full width, feeds later layers both directions' outputs, and adds the input repeated twice to the 2·dim output.
/// </summary>
internal sealed class SeanetLstm<T>
{
    private readonly IEngine _engine;
    private readonly List<(LSTMCellLayer<T> Forward, LSTMCellLayer<T>? Backward)> _layers = new();
    private readonly int _dim;

    public SeanetLstm(IEngine engine, int dim, int layers, bool bidirectional, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _dim = dim;
        _bidirectional = bidirectional;
        for (int l = 0; l < layers; l++)
        {
            int input = l > 0 && bidirectional ? 2 * dim : dim;
            var forward = new LSTMCellLayer<T>(input, dim);
            owner.Add(forward);
            LSTMCellLayer<T>? backward = null;
            if (bidirectional)
            {
                backward = new LSTMCellLayer<T>(input, dim);
                owner.Add(backward);
            }
            _layers.Add((forward, backward));
        }
    }

    private readonly bool _bidirectional;

    /// <summary>The output channels: 2·dim when bidirectional, dim otherwise.</summary>
    public int OutputDim => _bidirectional ? 2 * _dim : _dim;

    /// <summary>The backward-direction cells by layer (the reference's <c>lstm.*_l{k}_reverse</c>), when bidirectional.</summary>
    internal IReadOnlyList<LSTMCellLayer<T>?> BackwardCells => _layers.Select(l => l.Backward).ToList();

    /// <summary>The forward-direction cells by layer (the reference's <c>lstm.*_l{k}</c>).</summary>
    internal IReadOnlyList<LSTMCellLayer<T>> ForwardCells => _layers.Select(l => l.Forward).ToList();

    // One direction over the rows [T, in] of x, returning [T, hidden].
    private Tensor<T> Run(LSTMCellLayer<T> cell, Tensor<T> rows, bool reverse)
    {
        int t = rows.Shape[0], hidden = cell.HiddenSize;
        var state = new Tensor<T>(new[] { 1, 2 * hidden });                                          // [h; c], zeros
        var outputs = new Tensor<T>[t];
        for (int s = 0; s < t; s++)
        {
            int i = reverse ? t - 1 - s : s;
            var x = _engine.TensorSlice(rows, new[] { i, 0 }, new[] { 1, rows.Shape[1] });
            state = cell.Forward(x, state);
            outputs[i] = cell.SplitState(state).Hidden;
        }
        return _engine.TensorConcatenate(outputs, 0);
    }

    /// <summary>The LSTM stack's output plus its input (repeated twice when bidirectional), <c>[1, OutputDim, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        int t = x.Shape[2];
        var rows = _engine.TensorTranspose(_engine.Reshape(x, new[] { _dim, t }));                     // [T, dim]
        var y = rows;
        foreach (var (forward, backward) in _layers)
        {
            var f = Run(forward, y, reverse: false);
            y = backward is null ? f : _engine.TensorConcatenate(new[] { f, Run(backward, y, reverse: true) }, 1);
        }
        var skip = _bidirectional ? _engine.TensorConcatenate(new[] { rows, rows }, 1) : rows;
        var sum = _engine.TensorAdd(y, skip);
        return _engine.Reshape(_engine.TensorTranspose(sum), new[] { 1, OutputDim, t });
    }
}

/// <summary>The SEANet configuration EnCodec, SoundStream-style codecs and SpeechTokenizer share.</summary>
internal sealed class SeanetConfig
{
    public int Channels = 1, Dimension = 128, Filters = 32, ResidualLayers = 1;
    public int[] Ratios = { 8, 5, 4, 2 };
    public double EluAlpha = 1.0;
    public ConvolutionNormalization Normalization = ConvolutionNormalization.Weight;
    public int KernelSize = 7, LastKernelSize = 7, DilationBase = 2;
    /// <summary>The encoder's last kernel when it differs from the decoder's (SoundStream: 3, then 7).</summary>
    public int? EncoderLastKernelSize;
    /// <summary>The residual unit's two kernels: the paper's "two convolutions with kernel size 3" (EnCodec §3.1);
    /// the official checkpoints use (3, 1).</summary>
    public int[] ResidualKernelSizes = { 3, 3 };
    public bool Causal, Reflect = true, TrueSkip, BidirectionalLstm;
    /// <summary>Whether each convolution is followed by a layer normalization over channels and time (the non-streamable
    /// model; reference <c>time_group_norm</c>, GroupNorm with one group).</summary>
    public bool TimeGroupNorm;
    public int Compress = 2, LstmLayers = 2;
    public double TrimRightRatio = 1.0;
}

/// <summary>
/// SEANet's residual block (reference <c>SEANetResnetBlock</c>): ELU → conv(k₀, dilation) to dim / compress → ELU →
/// conv(k₁) back to dim, plus a shortcut — the identity, or (EnCodec's default) a 1×1 convolution.
/// </summary>
internal sealed class SeanetResnetBlock<T>
{
    private readonly IEngine _engine;
    private readonly double _alpha;
    private readonly StreamableConv1d<T>[] _block;
    private readonly StreamableConv1d<T>? _shortcut;

    public SeanetResnetBlock(IEngine engine, SeanetConfig c, int dim, int dilation, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _alpha = c.EluAlpha;
        int hidden = dim / c.Compress;
        _block = new[]
        {
            new StreamableConv1d<T>(engine, dim, hidden, c.ResidualKernelSizes[0], 1, dilation, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm),
            new StreamableConv1d<T>(engine, hidden, dim, c.ResidualKernelSizes[1], 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm),
        };
        if (!c.TrueSkip) _shortcut = new StreamableConv1d<T>(engine, dim, dim, 1, 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
    }

    /// <summary>The block's convolutions under the reference's names (<c>block.1</c>, <c>block.3</c>, <c>shortcut</c>).</summary>
    internal IEnumerable<(string Name, StreamableConv1d<T> Conv)> NamedConvolutions(string prefix)
    {
        yield return ($"{prefix}.block.1", _block[0]);
        yield return ($"{prefix}.block.3", _block[1]);
        if (_shortcut is not null) yield return ($"{prefix}.shortcut", _shortcut);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = x;
        foreach (var conv in _block) y = conv.Forward(SeanetOps.Elu(_engine, y, _alpha));
        return _engine.TensorAdd(_shortcut is null ? x : _shortcut.Forward(x), y);
    }
}

internal static class SeanetOps
{
    /// <summary>ELU: x for x > 0, α(eˣ − 1) otherwise, as relu(x) − α·relu(1 − eˣ).</summary>
    public static Tensor<T> Elu<T>(IEngine engine, Tensor<T> x, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var negative = engine.ReLU(engine.TensorAddScalar(engine.TensorNegate(engine.TensorExp(x)), ops.One));        // relu(1 − eˣ)
        return engine.TensorSubtract(engine.ReLU(x), engine.TensorMultiplyScalar(negative, ops.FromDouble(alpha)));
    }
}

/// <summary>
/// SEANet's encoder (EnCodec §3.1, Fig. 1; reference <c>SEANetEncoder</c>): a k=7 convolution to the base width, then per
/// downsampling ratio (applied in reverse order) the residual blocks and ELU → a strided convolution (kernel 2·ratio)
/// doubling the width, then the LSTM block, ELU and a k=7 convolution to the latent dimension.
/// </summary>
internal sealed class SeanetEncoder<T>
{
    private readonly IEngine _engine;
    private readonly SeanetConfig _c;
    private readonly StreamableConv1d<T> _first;
    private readonly List<(List<SeanetResnetBlock<T>> Blocks, StreamableConv1d<T> Down)> _stages = new();
    private readonly SeanetLstm<T>? _lstm;
    private readonly StreamableConv1d<T> _last;

    public SeanetEncoder(IEngine engine, SeanetConfig c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _c = c;
        int mult = 1;
        _first = new StreamableConv1d<T>(engine, c.Channels, mult * c.Filters, c.KernelSize, 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
        for (int r = c.Ratios.Length - 1; r >= 0; r--)
        {
            int ratio = c.Ratios[r];
            var blocks = new List<SeanetResnetBlock<T>>();
            for (int j = 0; j < c.ResidualLayers; j++)
                blocks.Add(new SeanetResnetBlock<T>(engine, c, mult * c.Filters, (int)Math.Pow(c.DilationBase, j), owner));
            var down = new StreamableConv1d<T>(engine, mult * c.Filters, mult * c.Filters * 2, ratio * 2, ratio, 1, c.Causal, c.Reflect,
                c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
            _stages.Add((blocks, down));
            mult *= 2;
        }
        if (c.LstmLayers > 0) _lstm = new SeanetLstm<T>(engine, mult * c.Filters, c.LstmLayers, c.BidirectionalLstm, owner);
        // A bidirectional LSTM doubles the channels the last convolution reads (reference: mult *= 2 if bidirectional).
        int lastInput = _lstm?.OutputDim ?? mult * c.Filters;
        _last = new StreamableConv1d<T>(engine, lastInput, c.Dimension, c.EncoderLastKernelSize ?? c.LastKernelSize, 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
    }

    /// <summary>The convolutions under the reference's <c>encoder.layers.{i}</c> names: the first convolution, per stage the
    /// residual blocks, an ELU and the downsampling convolution, then the LSTM, an ELU and the last convolution.</summary>
    internal IEnumerable<(string Name, StreamableConv1d<T> Conv)> NamedConvolutions(string prefix)
    {
        int i = 0;
        yield return ($"{prefix}.{i++}", _first);
        foreach (var (blocks, down) in _stages)
        {
            foreach (var block in blocks)
                foreach (var named in block.NamedConvolutions($"{prefix}.{i++}")) yield return named;
            i++;                                                                                        // ELU
            yield return ($"{prefix}.{i++}", down);
        }
        if (_lstm is not null) i++;
        yield return ($"{prefix}.{i + 1}", _last);                                                      // ELU, then the convolution
    }

    /// <summary>The LSTM block's reference name and module.</summary>
    internal (string Name, SeanetLstm<T> Lstm)? NamedLstm(string prefix)
        => _lstm is null ? null : ($"{prefix}.{1 + _stages.Sum(s => s.Blocks.Count + 2)}", _lstm);

    /// <summary>The latent <c>[1, dimension, frames]</c> of audio <c>[1, channels, samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (blocks, down) in _stages)
        {
            foreach (var block in blocks) x = block.Forward(x);
            x = down.Forward(SeanetOps.Elu(_engine, x, _c.EluAlpha));
        }
        if (_lstm is not null) x = _lstm.Forward(x);
        return _last.Forward(SeanetOps.Elu(_engine, x, _c.EluAlpha));
    }
}

/// <summary>
/// SEANet's decoder (reference <c>SEANetDecoder</c>): a k=7 convolution from the latent to 2^stages × the base width, the
/// LSTM block, then per upsampling ratio ELU → a transposed convolution (kernel 2·ratio) halving the width and the
/// residual blocks, then ELU and a k=7 convolution to the audio channels.
/// </summary>
internal sealed class SeanetDecoder<T>
{
    private readonly IEngine _engine;
    private readonly SeanetConfig _c;
    private readonly StreamableConv1d<T> _first;
    private readonly SeanetLstm<T>? _lstm;
    private readonly List<(StreamableConvTranspose1d<T> Up, List<SeanetResnetBlock<T>> Blocks)> _stages = new();
    private readonly StreamableConv1d<T> _last;

    public SeanetDecoder(IEngine engine, SeanetConfig c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _c = c;
        int mult = 1 << c.Ratios.Length;
        _first = new StreamableConv1d<T>(engine, c.Dimension, mult * c.Filters, c.KernelSize, 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
        if (c.LstmLayers > 0) _lstm = new SeanetLstm<T>(engine, mult * c.Filters, c.LstmLayers, false, owner);
        foreach (int ratio in c.Ratios)
        {
            var up = new StreamableConvTranspose1d<T>(engine, mult * c.Filters, mult * c.Filters / 2, ratio * 2, ratio, c.Causal, c.TrimRightRatio,
                c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
            var blocks = new List<SeanetResnetBlock<T>>();
            for (int j = 0; j < c.ResidualLayers; j++)
                blocks.Add(new SeanetResnetBlock<T>(engine, c, mult * c.Filters / 2, (int)Math.Pow(c.DilationBase, j), owner));
            _stages.Add((up, blocks));
            mult /= 2;
        }
        _last = new StreamableConv1d<T>(engine, c.Filters, c.Channels, c.LastKernelSize, 1, 1, c.Causal, c.Reflect, c.Normalization, owner, timeGroupNorm: c.TimeGroupNorm);
    }

    /// <summary>The convolutions under the reference's <c>decoder.layers.{i}</c> names: the first convolution, the LSTM, per
    /// stage an ELU, the transposed convolution and the residual blocks, then an ELU and the last convolution.</summary>
    internal IEnumerable<(string Name, StreamableConv1d<T> Conv)> NamedConvolutions(string prefix)
    {
        int i = 0;
        yield return ($"{prefix}.{i++}", _first);
        if (_lstm is not null) i++;
        foreach (var (_, blocks) in _stages)
        {
            i += 2;                                                                                     // ELU, transposed convolution
            foreach (var block in blocks)
                foreach (var named in block.NamedConvolutions($"{prefix}.{i++}")) yield return named;
        }
        yield return ($"{prefix}.{i + 1}", _last);
    }

    /// <summary>The transposed convolutions under their reference names.</summary>
    internal IEnumerable<(string Name, StreamableConvTranspose1d<T> Conv)> NamedTransposedConvolutions(string prefix)
    {
        int i = _lstm is null ? 1 : 2;
        foreach (var (up, blocks) in _stages)
        {
            yield return ($"{prefix}.{i + 1}", up);
            i += 2 + blocks.Count;
        }
    }

    /// <summary>The LSTM block's reference name and module.</summary>
    internal (string Name, SeanetLstm<T> Lstm)? NamedLstm(string prefix) => _lstm is null ? null : ($"{prefix}.1", _lstm);

    /// <summary>The audio <c>[1, channels, frames · hop]</c> of a latent <c>[1, dimension, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> z)
    {
        var x = _first.Forward(z);
        if (_lstm is not null) x = _lstm.Forward(x);
        foreach (var (up, blocks) in _stages)
        {
            x = up.Forward(SeanetOps.Elu(_engine, x, _c.EluAlpha));
            foreach (var block in blocks) x = block.Forward(x);
        }
        return _last.Forward(SeanetOps.Elu(_engine, x, _c.EluAlpha));
    }
}
