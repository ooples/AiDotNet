using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes of SpiritLM's F0 VQ-VAE (the release's <c>vqvae_f0_quantizer/config.yaml</c>).</summary>
internal sealed record SpiritLmPitchQuantizerConfiguration
{
    /// <summary>Input channels: the F0 and the voicing.</summary>
    public int InputChannels { get; init; } = 2;

    /// <summary>The latent width (<c>emb_width</c>, <c>output_emb_width</c>).</summary>
    public int LatentDim { get; init; } = 128;

    /// <summary>The convolution width (<c>width</c>).</summary>
    public int Width { get; init; } = 32;

    /// <summary>Residual blocks per resolution (<c>depth</c>), dilated 1, 3, 9, 27 (<c>dilation_growth_rate</c> 3).</summary>
    public int Depth { get; init; } = 4;

    public int DilationGrowth { get; init; } = 3;

    /// <summary>Stride-2 downsamplings (<c>downs_t</c>): 4, so a code covers 16 frames of 5 ms.</summary>
    public int Downsamplings { get; init; } = 4;

    /// <summary>Codebook entries (<c>l_bins</c>).</summary>
    public int Codes { get; init; } = 64;

    /// <summary>The codebook's EMA decay (<c>mu</c>).</summary>
    public double Decay { get; init; } = 0.99;

    /// <summary>The commitment loss weight (<c>lambda_commit</c>).</summary>
    public double CommitWeight { get; init; } = 0.02;

    /// <summary>The voicing channel's weight in the reconstruction loss (<c>vuv_scale</c>).</summary>
    public double VoicingScale { get; init; } = 2.0;
}

/// <summary>
/// SpiritLM Expressive's pitch tokenizer (Nguyen et al. 2024 §3.2, after Polyak et al. 2021): a VQ-VAE over the F0 and
/// voicing at 200 frames a second (the release's <c>VQVAE</c>, Jukebox's encoder, bottleneck and decoder), 64 codes, one
/// per 16 frames: 12.5 pitch tokens a second.
/// </summary>
/// <remarks>
/// <para>Encoder: four stride-2 convolutions (kernel 4, padding 1), each followed by a residual stack of four blocks
/// (ReLU, a dilated 3-tap convolution, ReLU, a 1×1 convolution, added to the input), then a 3-tap convolution to the latent
/// width. Each latent frame's code is its nearest codebook row (squared distance, the first on a tie).</para>
/// <para>Decoder: a 3-tap convolution to the width, four residual stacks each followed by a stride-2 transposed
/// convolution (the last to the latent width), and a 3-tap convolution to the two channels.</para>
/// <para>Training (<see cref="TrainingLoss"/>): Jukebox's straight-through bottleneck with the commitment loss
/// <c>‖sg(e) − z‖² / numel</c>, the codebook moved by exponential moving averages of the assigned latents (codes used less
/// than once on average restarted from random latents), and the reconstruction loss the configuration names: L1 against
/// the input, the voicing channel weighted by <see cref="SpiritLmPitchQuantizerConfiguration.VoicingScale"/>.</para>
/// </remarks>
internal sealed class SpiritLmPitchQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private double[]? _codeSum;
    private double[]? _codeCount;

    internal sealed record ResidualBlock(NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Pointwise);

    public SpiritLmPitchQuantizer(IEngine engine, List<LayerBase<T>> layers, SpiritLmPitchQuantizerConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int dilation, int padding, bool transposed = false)
        {
            var conv = Own(new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, dilation, 1, padding, transposed, ConvolutionNormalization.None));
            // torch's default: U(±1/√fan_in), fan_in from weight.size(1) · kernel (outputs for a transposed convolution).
            int fanIn = (transposed ? outputs : inputs) * kernel;
            var weight = new double[inputs * outputs * kernel];
            for (int i = 0; i < weight.Length; i++) weight[i] = Uniform(fanIn);
            var bias = new double[outputs];
            for (int i = 0; i < bias.Length; i++) bias[i] = Uniform(fanIn);
            conv.LoadTorchWeights(weight, null, bias);
            return conv;
        }
        List<ResidualBlock> Stack()
        {
            var blocks = new List<ResidualBlock>();
            for (int d = 0, dilation = 1; d < c.Depth; d++, dilation *= c.DilationGrowth)
                blocks.Add(new ResidualBlock(Conv(c.Width, c.Width, 3, 1, dilation, dilation), Conv(c.Width, c.Width, 1, 1, 1, 0)));
            return blocks;
        }
        for (int i = 0; i < c.Downsamplings; i++)
            EncoderBlocks.Add((Conv(i == 0 ? c.InputChannels : c.Width, c.Width, 4, 2, 1, 1), Stack()));
        EncoderOut = Conv(c.Width, c.LatentDim, 3, 1, 1, 1);
        DecoderIn = Conv(c.LatentDim, c.Width, 3, 1, 1, 1);
        for (int i = 0; i < c.Downsamplings; i++)
            DecoderBlocks.Add((Stack(), Conv(c.Width, i == c.Downsamplings - 1 ? c.LatentDim : c.Width, 4, 2, 1, 1, transposed: true)));
        DecoderOut = Conv(c.LatentDim, c.InputChannels, 3, 1, 1, 1);
        Codebook = new double[c.Codes, c.LatentDim];
    }

    public SpiritLmPitchQuantizerConfiguration Configuration { get; }
    public List<(NormedConv1DLayer<T> Down, List<ResidualBlock> Stack)> EncoderBlocks { get; } = new();
    public NormedConv1DLayer<T> EncoderOut { get; }
    public NormedConv1DLayer<T> DecoderIn { get; }
    public List<(List<ResidualBlock> Stack, NormedConv1DLayer<T> Up)> DecoderBlocks { get; } = new();
    public NormedConv1DLayer<T> DecoderOut { get; }

    /// <summary>The codebook <c>[codes, latent]</c> (the bottleneck's <c>k</c> buffer).</summary>
    public double[,] Codebook { get; private set; }

    /// <summary>Whether the codebook holds trained or loaded entries (Jukebox initializes it from the first batch).</summary>
    public bool CodebookInitialized { get; private set; }

    private Tensor<T> Residual(List<ResidualBlock> stack, Tensor<T> x)
    {
        foreach (var block in stack)
            x = _engine.TensorAdd(x, block.Pointwise.Forward(_engine.ReLU(block.Dilated.Forward(_engine.ReLU(x)))));
        return x;
    }

    /// <summary>The latents <c>[1, latent, frames / 16]</c> of an input <c>[1, 2, frames]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> x)
    {
        foreach (var (down, stack) in EncoderBlocks) x = Residual(stack, down.Forward(x));
        return EncoderOut.Forward(x);
    }

    /// <summary>The reconstruction <c>[1, 2, frames]</c> of latents <c>[1, latent, frames / 16]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> z)
    {
        var x = DecoderIn.Forward(z);
        foreach (var (stack, up) in DecoderBlocks) x = up.Forward(Residual(stack, x));
        return DecoderOut.Forward(x);
    }

    /// <summary>The nearest code of each latent frame (<c>‖x‖² − 2x·k + ‖k‖²</c>, the first minimum).</summary>
    public int[] Nearest(Tensor<T> latents)
    {
        int width = latents.Shape[1], frames = latents.Shape[2], codes = Codebook.GetLength(0);
        var norms = new double[codes];
        for (int k = 0; k < codes; k++)
            for (int i = 0; i < width; i++) norms[k] += Codebook[k, i] * Codebook[k, i];
        var result = new int[frames];
        for (int f = 0; f < frames; f++)
        {
            double xx = 0;
            for (int i = 0; i < width; i++) { double v = NumOps.ToDouble(latents[0, i, f]); xx += v * v; }
            double best = double.PositiveInfinity;
            for (int k = 0; k < codes; k++)
            {
                double dot = 0;
                for (int i = 0; i < width; i++) dot += NumOps.ToDouble(latents[0, i, f]) * Codebook[k, i];
                double distance = xx - 2 * dot + norms[k];
                if (distance < best) { best = distance; result[f] = k; }
            }
        }
        return result;
    }

    /// <summary>The decoded reconstruction <c>[1, 2, frames]</c> of an input through its nearest codes.</summary>
    public Tensor<T> Reconstruct(Tensor<T> x)
    {
        var z = Encode(x);
        var codes = Nearest(z);
        int width = z.Shape[1], frames = z.Shape[2];
        var quantized = new Tensor<T>(new[] { 1, width, frames });
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < width; i++) quantized[0, i, f] = NumOps.FromDouble(Codebook[codes[f], i]);
        return Decode(quantized);
    }

    /// <summary>The pitch tokens of an F0 track and its voicing at 200 frames a second.</summary>
    public int[] Codes(IReadOnlyList<double> f0, IReadOnlyList<double> voiced) => Nearest(Encode(Input(f0, voiced)));

    /// <summary>The input <c>[1, 2, frames]</c>: the F0 and the voicing.</summary>
    public Tensor<T> Input(IReadOnlyList<double> f0, IReadOnlyList<double> voiced)
    {
        if (f0.Count != voiced.Count) throw new ArgumentException("The F0 and voicing tracks differ in length.");
        if (f0.Count < 1 << Configuration.Downsamplings) throw new ArgumentException($"{f0.Count} frames make no pitch token.", nameof(f0));
        var x = new Tensor<T>(new[] { 1, 2, f0.Count });
        for (int t = 0; t < f0.Count; t++)
        {
            x[0, 0, t] = NumOps.FromDouble(f0[t]);
            x[0, 1, t] = NumOps.FromDouble(voiced[t]);
        }
        return x;
    }

    /// <summary>
    /// The training loss of an input <c>[1, 2, frames]</c> (frames a multiple of 16): the weighted L1 reconstruction plus
    /// the commitment loss, through the straight-through bottleneck; the codebook moves by its moving averages
    /// (<paramref name="random"/> draws restarts).
    /// </summary>
    public Tensor<T> TrainingLoss(Tensor<T> x, Random random)
    {
        var c = Configuration;
        var z = Encode(x);
        int width = z.Shape[1], frames = z.Shape[2];
        if (!CodebookInitialized) InitializeCodebook(z, random);
        var codes = Nearest(z);
        var quantized = new Tensor<T>(new[] { 1, width, frames });
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < width; i++) quantized[0, i, f] = NumOps.FromDouble(Codebook[codes[f], i]);
        UpdateCodebook(z, codes, random);
        var difference = _engine.TensorSubtract(quantized, z);
        var commit = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), null, false),
            NumOps.FromDouble(1.0 / difference.Length));
        // z + sg(e − z): the decoder's input carries e forward and the gradient to z.
        var straight = _engine.TensorAdd(z, new Tensor<T>(difference.Shape.ToArray(), difference.ToVector()));
        var reconstruction = Decode(straight);
        int length = Math.Min(reconstruction.Shape[2], x.Shape[2]);
        var weights = new Tensor<T>(new[] { 1, 2, length });
        for (int t = 0; t < length; t++)
        {
            weights[0, 0, t] = NumOps.One;
            weights[0, 1, t] = NumOps.FromDouble(c.VoicingScale);
        }
        var error = _engine.TensorMultiply(_engine.TensorAbs(_engine.TensorSubtract(
            _engine.TensorSlice(reconstruction, new[] { 0, 0, 0 }, new[] { 1, 2, length }),
            _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, 2, length }))), weights);
        var recon = _engine.TensorMultiplyScalar(_engine.ReduceSum(error, null, false), NumOps.FromDouble(1.0 / (2.0 * length)));
        return _engine.TensorAdd(recon, _engine.TensorMultiplyScalar(commit, NumOps.FromDouble(c.CommitWeight)));
    }

    // The latents as rows [frames, width].
    private static double[][] Rows(Tensor<T> z)
    {
        int width = z.Shape[1], frames = z.Shape[2];
        var rows = new double[frames][];
        for (int f = 0; f < frames; f++)
        {
            rows[f] = new double[width];
            for (int i = 0; i < width; i++) rows[f][i] = NumOps.ToDouble(z[0, i, f]);
        }
        return rows;
    }

    // Jukebox's _tile: too few latents are repeated, with noise of std 0.01/√width, to cover the codebook.
    private double[][] Tile(double[][] rows, Random random)
    {
        int codes = Configuration.Codes, width = Configuration.LatentDim;
        if (rows.Length >= codes) return rows;
        int repeats = (codes + rows.Length - 1) / rows.Length;
        double std = 0.01 / Math.Sqrt(width);
        var tiled = new double[rows.Length * repeats][];
        for (int r = 0; r < tiled.Length; r++)
        {
            tiled[r] = new double[width];
            for (int i = 0; i < width; i++)
                tiled[r][i] = rows[r % rows.Length][i] + std * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        }
        return tiled;
    }

    private double[][] RandomRows(double[][] rows, Random random)
    {
        var tiled = Tile(rows, random);
        var order = Enumerable.Range(0, tiled.Length).ToArray();
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order.Take(Configuration.Codes).Select(i => tiled[i]).ToArray();
    }

    private void InitializeCodebook(Tensor<T> z, Random random)
    {
        var chosen = RandomRows(Rows(z), random);
        int codes = Configuration.Codes, width = Configuration.LatentDim;
        Codebook = new double[codes, width];
        _codeSum = new double[codes * width];
        _codeCount = new double[codes];
        for (int k = 0; k < codes; k++)
        {
            _codeCount[k] = 1;
            for (int i = 0; i < width; i++) Codebook[k, i] = _codeSum[k * width + i] = chosen[k][i];
        }
        CodebookInitialized = true;
    }

    // Jukebox's update_k: k_sum and k_elem by EMA, codes with k_elem under 1 restarted from random latents.
    private void UpdateCodebook(Tensor<T> z, int[] codes, Random random)
    {
        int count = Configuration.Codes, width = Configuration.LatentDim;
        double mu = Configuration.Decay;
        var rows = Rows(z);
        if (_codeSum is null || _codeCount is null)
        {
            // A loaded codebook continues from itself, one use per code (restore_k).
            _codeSum = new double[count * width];
            _codeCount = new double[count];
            for (int k = 0; k < count; k++)
            {
                _codeCount[k] = 1;
                for (int i = 0; i < width; i++) _codeSum[k * width + i] = Codebook[k, i];
            }
        }
        var batchSum = new double[count * width];
        var batchCount = new double[count];
        for (int f = 0; f < rows.Length; f++)
        {
            batchCount[codes[f]]++;
            for (int i = 0; i < width; i++) batchSum[codes[f] * width + i] += rows[f][i];
        }
        var restarts = RandomRows(rows, random);
        for (int k = 0; k < count; k++)
        {
            _codeCount[k] = mu * _codeCount[k] + (1 - mu) * batchCount[k];
            bool used = _codeCount[k] >= 1.0;
            for (int i = 0; i < width; i++)
            {
                _codeSum[k * width + i] = mu * _codeSum[k * width + i] + (1 - mu) * batchSum[k * width + i];
                Codebook[k, i] = used ? _codeSum[k * width + i] / _codeCount[k] : restarts[k][i];
            }
        }
    }

    /// <summary>Declares the codebook and its moving averages as model state: they move outside the optimizer.</summary>
    public void DeclareState(AiDotNet.Models.ModelStateRegistry<T> state, string name)
    {
        int count = Configuration.Codes, width = Configuration.LatentDim;
        state.DeclareDoubleVector(name + ".Codebook", () =>
        {
            var flat = new Vector<double>(count * width + 1);
            for (int k = 0; k < count; k++)
                for (int i = 0; i < width; i++) flat[k * width + i] = Codebook[k, i];
            flat[count * width] = CodebookInitialized ? 1 : 0;
            return flat;
        }, flat =>
        {
            if (flat is null) return;
            if (flat.Length != count * width + 1) throw new InvalidDataException($"State '{name}.Codebook' holds {flat.Length} values.");
            var codebook = new double[count, width];
            for (int k = 0; k < count; k++)
                for (int i = 0; i < width; i++) codebook[k, i] = flat[k * width + i];
            Codebook = codebook;
            CodebookInitialized = flat[count * width] != 0;
        });
        state.DeclareDoubleVector(name + ".Averages", () =>
        {
            if (_codeSum is null || _codeCount is null) return null;
            var flat = new Vector<double>(_codeSum.Length + _codeCount.Length);
            for (int i = 0; i < _codeSum.Length; i++) flat[i] = _codeSum[i];
            for (int i = 0; i < _codeCount.Length; i++) flat[_codeSum.Length + i] = _codeCount[i];
            return flat;
        }, flat =>
        {
            if (flat is null) { _codeSum = null; _codeCount = null; return; }
            _codeSum = new double[count * width];
            _codeCount = new double[count];
            for (int i = 0; i < _codeSum.Length; i++) _codeSum[i] = flat[i];
            for (int i = 0; i < _codeCount.Length; i++) _codeCount[i] = flat[_codeSum.Length + i];
        });
    }

    /// <summary>Loads the release's <c>model.pt</c> <c>model</c> state (<c>encoder.level_blocks.0.model</c>,
    /// <c>vq.level_blocks.0.k</c>, <c>decoder.level_blocks.0.model</c>, <c>decoder.out</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        double[] Read(string name) => take(name) ?? throw new InvalidDataException($"The F0 VQ-VAE checkpoint has no tensor '{name}'.");
        void Conv(NormedConv1DLayer<T> conv, string name) => conv.LoadTorchWeights(Read(name + ".weight"), null, Read(name + ".bias"));
        void Stack(List<ResidualBlock> stack, string name)
        {
            for (int d = 0; d < stack.Count; d++)
            {
                Conv(stack[d].Dilated, $"{name}.model.{d}.model.1");
                Conv(stack[d].Pointwise, $"{name}.model.{d}.model.3");
            }
        }
        const string encoder = "encoder.level_blocks.0.model";
        for (int i = 0; i < EncoderBlocks.Count; i++)
        {
            Conv(EncoderBlocks[i].Down, $"{encoder}.{i}.0");
            Stack(EncoderBlocks[i].Stack, $"{encoder}.{i}.1");
        }
        Conv(EncoderOut, $"{encoder}.{EncoderBlocks.Count}");
        const string decoder = "decoder.level_blocks.0.model";
        Conv(DecoderIn, $"{decoder}.0");
        for (int i = 0; i < DecoderBlocks.Count; i++)
        {
            Stack(DecoderBlocks[i].Stack, $"{decoder}.{i + 1}.0");
            Conv(DecoderBlocks[i].Up, $"{decoder}.{i + 1}.1");
        }
        Conv(DecoderOut, "decoder.out");
        var k = Read("vq.level_blocks.0.k");
        int codes = Configuration.Codes, width = Configuration.LatentDim;
        if (k.Length != codes * width) throw new InvalidDataException($"The codebook holds {k.Length} values, not {codes} x {width}.");
        Codebook = new double[codes, width];
        for (int c = 0; c < codes; c++)
            for (int i = 0; i < width; i++) Codebook[c, i] = k[c * width + i];
        CodebookInitialized = true;
        _codeSum = null;
        _codeCount = null;
    }
}
