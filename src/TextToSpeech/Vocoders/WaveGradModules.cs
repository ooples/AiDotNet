using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveGrad's noise predictor (Chen et al. 2021, §2, Fig. 3–6, App. A; reference lmnt-com/wavegrad <c>model.py</c>):
/// a 5×1 convolution and downsampling blocks (DBlocks) over the noisy waveform, a FiLM module per resolution that
/// mixes each DBlock output with a sinusoidal encoding of the noise level, and a 3×1 convolution and upsampling blocks
/// (UBlocks) over the mel spectrogram, modulated feature-wise by the FiLM outputs, then a 3×1 output convolution.
/// </summary>
/// <remarks>
/// <para>UBlock (Fig. 4): a nearest-neighbour upsampled 1×1 shortcut plus LReLU → upsample → 3×1 conv → γ⊙·+β → LReLU
/// → 3×1 conv; then a second residual of two (γ⊙·+β → LReLU → 3×1 conv) steps. DBlock (Fig. 5): a 1×1 shortcut
/// downsampled, plus downsample → three (LReLU → 3×1 conv) steps with dilations 1, 2, 4. Downsampling is a strided
/// convolution (App. A) whose kernel equals its stride, so each output sees exactly its own input window. FiLM (Fig. 6):
/// 3×1 conv → LReLU → + PE(C·√ᾱ) → 3×1 conv split into the shift β and the scale γ.</para>
/// <para>Initialization: orthogonal weights and zero biases in the UBlocks, DBlocks and the outer convolutions
/// (App. A, reference <c>Conv1d</c>); Xavier-uniform weights and zero biases in FiLM (reference).</para>
/// </remarks>
internal sealed class WaveGradNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Random _random;
    private readonly double _slope;
    private readonly double _levelScale;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _waveIn;
    private readonly NormedConv1DLayer<T> _melIn;
    private readonly NormedConv1DLayer<T> _out;
    private readonly List<DBlock> _down = new();
    private readonly List<Film> _films = new();
    private readonly List<UBlock> _up = new();

    private sealed record DBlock(int Factor, NormedConv1DLayer<T> Shortcut, NormedConv1DLayer<T>? ShortcutDown,
        NormedConv1DLayer<T>? MainDown, NormedConv1DLayer<T>[] Convs);

    private sealed record Film(int Channels, NormedConv1DLayer<T> Input, NormedConv1DLayer<T> Output);

    private sealed record UBlock(int Factor, int FilmIndex, NormedConv1DLayer<T> Shortcut, NormedConv1DLayer<T>[] Convs);

    /// <param name="engine">The tensor engine.</param>
    /// <param name="initialization">The source of the initial weights.</param>
    /// <param name="melChannels">Mel bands of the conditioning spectrogram (128).</param>
    /// <param name="melProjection">Channels of the 3×1 convolution over the mel spectrogram (768).</param>
    /// <param name="waveChannels">Channels of the 5×1 convolution over the noisy waveform (32).</param>
    /// <param name="factors">The UBlocks' upsampling factors in order (5, 5, 3, 2, 2).</param>
    /// <param name="channels">The UBlocks' output channels in order (512, 512, 256, 128, 128).</param>
    /// <param name="dilations">The four dilations of each UBlock.</param>
    /// <param name="repeatBlocks">Whether every UBlock and DBlock is followed by one of the same width that does not
    /// resample (WaveGrad Large).</param>
    /// <param name="slope">The leaky-ReLU slope (0.2).</param>
    /// <param name="levelScale">The linear scale C of the noise level in the positional encoding (5000).</param>
    public WaveGradNetwork(IEngine engine, Random initialization, int melChannels, int melProjection, int waveChannels,
        int[] factors, int[] channels, int[][] dilations, bool repeatBlocks, double slope, double levelScale)
    {
        if (factors.Length != channels.Length || factors.Length != dilations.Length || factors.Length < 1)
            throw new ArgumentException("Each UBlock needs a factor, a channel count and four dilations.");
        foreach (var d in dilations)
            if (d.Length != 4)
                throw new ArgumentException("A UBlock has four dilated convolutions.");
        _engine = engine;
        _random = initialization;
        _slope = slope;
        _levelScale = levelScale;
        int k = factors.Length;

        _waveIn = Orthogonal(1, waveChannels, 5, 1, 1, 2);
        // DBlocks mirror the UBlocks after the first: DBlock i (from the waveform side) undoes UBlock k − 1 − i.
        var filmInputs = new List<int> { waveChannels };
        int width = waveChannels;
        for (int i = 0; i < k - 1; i++)
        {
            int u = k - 1 - i;
            AddDBlock(width, channels[u], factors[u]);
            width = channels[u];
            if (repeatBlocks) AddDBlock(width, width, 1);
            filmInputs.Add(width);
        }
        // FiLM i reads DBlock output i and modulates the UBlock whose output has that resolution.
        for (int i = 0; i < k; i++)
            _films.Add(new Film(channels[k - 1 - i], Xavier(filmInputs[i], filmInputs[i], 3, 1),
                Xavier(filmInputs[i], 2 * channels[k - 1 - i], 3, 1)));

        _melIn = Orthogonal(melChannels, melProjection, 3, 1, 1, 1);
        width = melProjection;
        for (int j = 0; j < k; j++)
        {
            AddUBlock(width, channels[j], factors[j], dilations[j], k - 1 - j);
            width = channels[j];
            if (repeatBlocks) AddUBlock(width, width, 1, dilations[j], k - 1 - j);
        }
        _out = Orthogonal(width, 1, 3, 1, 1, 1);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private void AddDBlock(int input, int output, int factor)
    {
        var shortcut = Orthogonal(input, output, 1, 1, 1, 0);
        var shortcutDown = factor > 1 ? Orthogonal(output, output, factor, factor, 1, 0) : null;
        var mainDown = factor > 1 ? Orthogonal(input, input, factor, factor, 1, 0) : null;
        var convs = new[] { Orthogonal(input, output, 3, 1, 1, 1), Orthogonal(output, output, 3, 1, 2, 2), Orthogonal(output, output, 3, 1, 4, 4) };
        _down.Add(new DBlock(factor, shortcut, shortcutDown, mainDown, convs));
    }

    private void AddUBlock(int input, int output, int factor, int[] d, int film)
    {
        var shortcut = Orthogonal(input, output, 1, 1, 1, 0);
        var convs = new[]
        {
            Orthogonal(input, output, 3, 1, d[0], d[0]), Orthogonal(output, output, 3, 1, d[1], d[1]),
            Orthogonal(output, output, 3, 1, d[2], d[2]), Orthogonal(output, output, 3, 1, d[3], d[3]),
        };
        _up.Add(new UBlock(factor, film, shortcut, convs));
    }

    private NormedConv1DLayer<T> Conv(int input, int output, int kernel, int stride, int dilation, int padding)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, false, ConvolutionNormalization.None);
        _layers.Add(conv);
        return conv;
    }

    // torch.nn.init.orthogonal_ on the [out, in·kernel] weight, zero bias.
    private NormedConv1DLayer<T> Orthogonal(int input, int output, int kernel, int stride, int dilation, int padding)
    {
        var conv = Conv(input, output, kernel, stride, dilation, padding);
        var values = OrthogonalMatrix(_random, output, input * kernel);
        int next = 0;
        conv.Reinitialize(() => values[next++]);
        return conv;
    }

    // torch.nn.init.xavier_uniform_: U(±√(6 / (fan_in + fan_out))), zero bias.
    private NormedConv1DLayer<T> Xavier(int input, int output, int kernel, int padding)
    {
        var conv = Conv(input, output, kernel, 1, 1, padding);
        double bound = Math.Sqrt(6.0 / (input * kernel + output * kernel));
        conv.Reinitialize(() => (2 * _random.NextDouble() - 1) * bound);
        return conv;
    }

    /// <summary>A row-major <paramref name="rows"/>×<paramref name="cols"/> matrix with orthonormal rows or columns
    /// (whichever are fewer), as <c>torch.nn.init.orthogonal_</c>: the Q of a Gaussian matrix's QR decomposition with
    /// R's diagonal made positive.</summary>
    internal static double[] OrthogonalMatrix(Random random, int rows, int cols)
    {
        // Orthonormalize the shorter side's vectors by modified Gram–Schmidt, which yields R with a positive diagonal.
        bool transpose = rows < cols;
        int n = transpose ? cols : rows, m = transpose ? rows : cols;          // an n×m matrix with n ≥ m
        var q = new double[m][];
        for (int j = 0; j < m; j++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                v[i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
            for (int p = 0; p < j; p++)
            {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += q[p][i] * v[i];
                for (int i = 0; i < n; i++) v[i] -= dot * q[p][i];
            }
            double norm = 0;
            for (int i = 0; i < n; i++) norm += v[i] * v[i];
            norm = Math.Sqrt(norm);
            for (int i = 0; i < n; i++) v[i] /= norm;
            q[j] = v;
        }
        var result = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r * cols + c] = transpose ? q[r][c] : q[c][r];
        return result;
    }

    private Tensor<T> Lrelu(Tensor<T> x) => VocoderOps.LeakyRelu(_engine, x, _slope);

    // Nearest-neighbour upsampling: each step repeated factor times.
    private Tensor<T> Repeat(Tensor<T> x, int factor)
    {
        if (factor == 1) return x;
        int c = x.Shape[1], t = x.Shape[2];
        var tiled = _engine.TensorTile(_engine.Reshape(x, new[] { 1, c, t, 1 }), new[] { 1, 1, 1, factor });
        return _engine.Reshape(tiled, new[] { 1, c, t * factor });
    }

    // The positional encoding [1, channels, 1] of the scaled noise level: sin and cos of C·√ᾱ · 10^(−4i/(channels/2)).
    private Tensor<T> Encoding(int channels, double level)
    {
        int count = channels / 2;
        var e = new Tensor<T>(new[] { 1, channels, 1 });
        for (int i = 0; i < count; i++)
        {
            double v = _levelScale * level * Math.Exp(-Math.Log(1e4) * i / count);
            e[0, i, 0] = NumOps.FromDouble(Math.Sin(v));
            e[0, count + i, 0] = NumOps.FromDouble(Math.Cos(v));
        }
        return e;
    }

    private Tensor<T> Affine(Tensor<T> x, (Tensor<T> Shift, Tensor<T> Scale) film)
        => _engine.TensorAdd(film.Shift, _engine.TensorMultiply(film.Scale, x));

    /// <summary>ε_θ <c>[1, 1, samples]</c> for the noisy waveform <c>[1, 1, samples]</c> at noise level √ᾱ
    /// <paramref name="level"/> given the mel spectrogram <c>[1, mel, frames]</c> (samples = frames · hop).</summary>
    public Tensor<T> Forward(Tensor<T> noisy, double level, Tensor<T> mel)
    {
        var x = _waveIn.Forward(noisy);
        var resolutions = new List<Tensor<T>> { x };
        bool repeated = _down.Count > _films.Count - 1;
        for (int i = 0; i < _down.Count; i++)
        {
            var block = _down[i];
            var shortcut = block.Shortcut.Forward(x);
            if (block.ShortcutDown is not null) shortcut = block.ShortcutDown.Forward(shortcut);
            var y = block.MainDown is not null ? block.MainDown.Forward(x) : x;
            foreach (var conv in block.Convs) y = conv.Forward(Lrelu(y));
            x = _engine.TensorAdd(y, shortcut);
            if (!repeated || i % 2 == 1) resolutions.Add(x);
        }
        var films = new List<(Tensor<T> Shift, Tensor<T> Scale)>();
        for (int i = 0; i < _films.Count; i++)
        {
            var f = _films[i];
            var input = resolutions[i];
            var h = Lrelu(f.Input.Forward(input));
            h = _engine.TensorAdd(h, _engine.TensorTile(Encoding(h.Shape[1], level), new[] { 1, 1, h.Shape[2] }));
            var o = f.Output.Forward(h);
            int t = o.Shape[2];
            films.Add((_engine.TensorSlice(o, new[] { 0, 0, 0 }, new[] { 1, f.Channels, t }),
                _engine.TensorSlice(o, new[] { 0, f.Channels, 0 }, new[] { 1, f.Channels, t })));
        }
        x = _melIn.Forward(mel);
        foreach (var block in _up)
        {
            var film = films[block.FilmIndex];
            var shortcut = block.Shortcut.Forward(Repeat(x, block.Factor));
            var y = block.Convs[0].Forward(Repeat(Lrelu(x), block.Factor));
            y = block.Convs[1].Forward(Lrelu(Affine(y, film)));
            x = _engine.TensorAdd(shortcut, y);
            y = block.Convs[2].Forward(Lrelu(Affine(x, film)));
            y = block.Convs[3].Forward(Lrelu(Affine(y, film)));
            x = _engine.TensorAdd(x, y);
        }
        return _out.Forward(x);
    }
}
