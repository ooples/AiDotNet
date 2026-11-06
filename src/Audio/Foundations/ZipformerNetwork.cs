using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>A Zipformer transducer's shape (icefall's LibriSpeech <c>zipformer</c> recipe arguments).</summary>
internal sealed record ZipformerConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int NumMels { get; init; } = 80;
    public int[] NumLayers { get; init; } = { 2, 2, 3, 4, 3, 2 };
    public int[] Downsampling { get; init; } = { 1, 2, 4, 8, 4, 2 };
    public int[] FeedForwardDims { get; init; } = { 512, 768, 1024, 1536, 1024, 768 };
    public int[] NumHeads { get; init; } = { 4, 4, 4, 8, 4, 4 };
    public int[] EncoderDims { get; init; } = { 192, 256, 384, 512, 384, 256 };
    public int[] ConvKernels { get; init; } = { 31, 31, 15, 15, 15, 31 };
    public int QueryHeadDim { get; init; } = 32;
    public int ValueHeadDim { get; init; } = 12;
    public int PosHeadDim { get; init; } = 4;
    public int PosDim { get; init; } = 48;
    public int[] EmbedChannels { get; init; } = { 8, 32, 128 };
    public int DecoderDim { get; init; } = 512;
    public int JoinerDim { get; init; } = 512;
    public int ContextSize { get; init; } = 2;
    public int VocabSize { get; init; } = 500;
    /// <summary>Encoder frames each streaming chunk attends within (reduced to each stack's frame rate); zero for none.</summary>
    public int ChunkSize { get; init; }
    /// <summary>Encoder frames of left context a streaming chunk also attends to; negative for the whole past.</summary>
    public int LeftContextFrames { get; init; } = -1;
    /// <summary>Whether the convolution modules see only past frames (left padding), for streaming.</summary>
    public bool Causal { get; init; }
    public int FrequencyMasks { get; init; } = 2;
    public int FrequencyMaskWidth { get; init; } = 27;
    public int TimeMasks { get; init; } = 10;
    public double TimeMaskRatio { get; init; } = 0.05;
}

/// <summary>
/// Zipformer (Yao et al. 2024, "Zipformer: A faster and better encoder for automatic speech recognition") as icefall's
/// reference implementation computes it in inference: Conv-Embed (three 2-D convolutions with SwooshR, a ConvNeXt layer,
/// a linear layer and BiasNorm, halving the 100 Hz frame rate), six stacks at 50, 25, 12.5, 6.25, 12.5 and 25 Hz with
/// learnable-weight downsampling, repeat upsampling and a bypass combining each stack's input and output, the stack
/// outputs assembled to the largest width and downsampled to 25 Hz (section 3.1). Each Zipformer block (section 3.2,
/// Figure 2) computes one set of relative-position attention weights and reuses it in non-linear attention and two
/// self-attention modules, around three feed-forward and two convolution modules, BiasNorm (section 3.3) and two bypass
/// modules, with SwooshL and SwooshR activations (section 3.4). The decoder is icefall's stateless prediction network
/// (an embedding and a grouped convolution over the last two labels) and the joiner adds projected encodings before
/// tanh and its output layer.
/// </summary>
/// <remarks>
/// The training-only regularizers of the reference (balancers and whitening, which only alter gradients; layer and
/// module skipping; the bypass-scale floor schedule) are not part of the forward function and are not applied. The
/// relative positional encoding is the compact one of the reference (log-compressed, arctan-squashed offsets).
/// </remarks>
internal sealed class ZipformerNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ZipformerConfiguration _o;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _outputDim;

    private BiasFreeLinearLayer<T> Values(Func<LayerBase<T>, LayerBase<T>> own, int count, double fill)
    {
        var holder = new BiasFreeLinearLayer<T>(1, count);
        own(holder);
        if (!double.IsNaN(fill))
            for (int i = 0; i < count; i++) holder.Weights[0, i] = NumOps.FromDouble(fill);
        return holder;
    }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder) =>
        _engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))), new[] { holder.Weights.Shape[1] });

    internal sealed class Convolution
    {
        public Convolution(DenseLayer<T> input, DepthwiseConv1DLayer<T> depthwise, DenseLayer<T> output) { Input = input; Depthwise = depthwise; Output = output; }
        public DenseLayer<T> Input { get; }
        public DepthwiseConv1DLayer<T> Depthwise { get; }
        public DenseLayer<T> Output { get; }
    }

    internal sealed class FeedForward
    {
        public FeedForward(DenseLayer<T> input, DenseLayer<T> output) { Input = input; Output = output; }
        public DenseLayer<T> Input { get; }
        public DenseLayer<T> Output { get; }
    }

    internal sealed class Block
    {
        public Block(DenseLayer<T> attentionIn, BiasFreeLinearLayer<T> positionProjection, DenseLayer<T> attention1In, DenseLayer<T> attention1Out,
            DenseLayer<T> attention2In, DenseLayer<T> attention2Out, FeedForward ff1, FeedForward ff2, FeedForward ff3, DenseLayer<T> nonlinIn,
            DenseLayer<T> nonlinOut, Convolution conv1, Convolution conv2, BiasFreeLinearLayer<T> normBias, BiasFreeLinearLayer<T> normLogScale,
            BiasFreeLinearLayer<T> bypassMid, BiasFreeLinearLayer<T> bypass)
        {
            AttentionIn = attentionIn; PositionProjection = positionProjection; Attention1In = attention1In; Attention1Out = attention1Out;
            Attention2In = attention2In; Attention2Out = attention2Out; Ff1 = ff1; Ff2 = ff2; Ff3 = ff3; NonlinIn = nonlinIn; NonlinOut = nonlinOut;
            Conv1 = conv1; Conv2 = conv2; NormBias = normBias; NormLogScale = normLogScale; BypassMid = bypassMid; Bypass = bypass;
        }

        public DenseLayer<T> AttentionIn { get; }
        public BiasFreeLinearLayer<T> PositionProjection { get; }
        public DenseLayer<T> Attention1In { get; }
        public DenseLayer<T> Attention1Out { get; }
        public DenseLayer<T> Attention2In { get; }
        public DenseLayer<T> Attention2Out { get; }
        public FeedForward Ff1 { get; }
        public FeedForward Ff2 { get; }
        public FeedForward Ff3 { get; }
        public DenseLayer<T> NonlinIn { get; }
        public DenseLayer<T> NonlinOut { get; }
        public Convolution Conv1 { get; }
        public Convolution Conv2 { get; }
        public BiasFreeLinearLayer<T> NormBias { get; }
        public BiasFreeLinearLayer<T> NormLogScale { get; }
        public BiasFreeLinearLayer<T> BypassMid { get; }
        public BiasFreeLinearLayer<T> Bypass { get; }
    }

    internal sealed class Stack
    {
        public Stack(int dim, int downsampling, int heads, BiasFreeLinearLayer<T>? downsampleWeights, BiasFreeLinearLayer<T>? outCombiner)
        {
            Dim = dim; Downsampling = downsampling; Heads = heads; DownsampleWeights = downsampleWeights; OutCombiner = outCombiner;
        }

        public int Dim { get; }
        public int Downsampling { get; }
        public int Heads { get; }
        public List<Block> Blocks { get; } = new();
        public BiasFreeLinearLayer<T>? DownsampleWeights { get; }
        public BiasFreeLinearLayer<T>? OutCombiner { get; }
    }

    public ZipformerNetwork(IEngine engine, List<LayerBase<T>> layers, ZipformerConfiguration options)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _o = options ?? throw new ArgumentNullException(nameof(options));
        int stacks = _o.EncoderDims.Length;
        if (_o.NumLayers.Length != stacks || _o.Downsampling.Length != stacks || _o.FeedForwardDims.Length != stacks || _o.NumHeads.Length != stacks || _o.ConvKernels.Length != stacks)
            throw new ArgumentException("Every per-stack setting needs one entry per stack.", nameof(options));
        _outputDim = _o.EncoderDims.Max();
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.NumMels, nFft: 400, hopLength: 160, logMel: true, logOffset: 1e-6);
        LayerBase<T> Own(LayerBase<T> layer) { layers.Add(layer); return layer; }
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        // Conv-Embed (icefall Conv2dSubsampling): 3x3 convolutions with paddings (0, 1), 0 and 0 and strides 1, 2 and (1, 2).
        int c1 = _o.EmbedChannels[0], c2 = _o.EmbedChannels[1], c3 = _o.EmbedChannels[2];
        EmbedConv1 = Add(new Conv2DLayer<T>(1, c1, 3, 3, 1, 1, 0, 1));
        EmbedConv2 = Add(new Conv2DLayer<T>(c1, c2, 3, 3, 2, 2, 0, 0));
        EmbedConv3 = Add(new Conv2DLayer<T>(c2, c3, 3, 3, 1, 2, 0, 0));
        ConvNeXtDepthwisePointwise = Add(new DepthwiseSeparableConvolutionalLayer<T>(3 * c3, 7, stride: 1, padding: 3, activation: identity));
        ConvNeXtProject = Add(new Conv2DLayer<T>(3 * c3, c3, 1, 1, 1, 1, 0, 0));
        int width = (((_o.NumMels - 1) / 2) - 1) / 2;
        EmbedOut = Add(new DenseLayer<T>(_o.EncoderDims[0], identity));
        EmbedNormBias = Values(Own, _o.EncoderDims[0], double.NaN);
        EmbedNormLogScale = Values(Own, 1, 1.0);
        _embedWidth = width * c3;
        for (int s = 0; s < stacks; s++)
        {
            int d = _o.EncoderDims[s], heads = _o.NumHeads[s], ff = _o.FeedForwardDims[s], k = _o.ConvKernels[s], ds = _o.Downsampling[s];
            var stack = new Stack(d, ds, heads, ds > 1 ? Values(Own, ds, 0.0) : null, ds > 1 ? Values(Own, d, 0.5) : null);
            for (int l = 0; l < _o.NumLayers[s]; l++)
            {
                FeedForward NewFf(int units) => new(Add(new DenseLayer<T>(units, identity)), Add(new DenseLayer<T>(d, identity)));
                Convolution NewConv() => Remember(new(Add(new DenseLayer<T>(2 * d, identity)),
                    Add(new DepthwiseConv1DLayer<T>(d, k, multiplier: 1, stride: 1, padding: _o.Causal ? 0 : k / 2, activation: identity)), Add(new DenseLayer<T>(d, identity))), k);
                stack.Blocks.Add(new Block(
                    Add(new DenseLayer<T>((2 * _o.QueryHeadDim + _o.PosHeadDim) * heads, identity)), Add(new BiasFreeLinearLayer<T>(_o.PosDim, heads * _o.PosHeadDim)),
                    Add(new DenseLayer<T>(heads * _o.ValueHeadDim, identity)), Add(new DenseLayer<T>(d, identity)),
                    Add(new DenseLayer<T>(heads * _o.ValueHeadDim, identity)), Add(new DenseLayer<T>(d, identity)),
                    NewFf(ff * 3 / 4), NewFf(ff), NewFf(ff * 5 / 4),
                    Add(new DenseLayer<T>(3 * (3 * d / 4), identity)), Add(new DenseLayer<T>(d, identity)), NewConv(), NewConv(),
                    Values(Own, d, double.NaN), Values(Own, 1, 1.0), Values(Own, d, 0.5), Values(Own, d, 0.5)));
            }
            Stacks.Add(stack);
        }
        OutputDownsampleWeights = Values(Own, 2, 0.0);
        DecoderEmbedding = Add(new EmbeddingLayer<T>(_o.VocabSize, _o.DecoderDim));
        if (_o.ContextSize > 1) DecoderConv = Values(Own, _o.DecoderDim * 4 * _o.ContextSize, double.NaN);
        JoinerEncoder = Add(new DenseLayer<T>(_o.JoinerDim, identity));
        JoinerDecoder = Add(new DenseLayer<T>(_o.JoinerDim, identity));
        JoinerOutput = Add(new DenseLayer<T>(_o.VocabSize, identity));
        // BiasNorm biases start at N(0, 1e-4), the grouped decoder convolution like PyTorch's default (U(+-1/sqrt(fan_in))).
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(17);
        void Fill(BiasFreeLinearLayer<T> holder, double scale)
        {
            for (int i = 0; i < holder.Weights.Shape[1]; i++) holder.Weights[0, i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * scale);
        }
        Fill(EmbedNormBias, 1e-4);
        foreach (var stack in Stacks) foreach (var block in stack.Blocks) Fill(block.NormBias, 1e-4);
        if (DecoderConv is not null) Fill(DecoderConv, 1.0 / Math.Sqrt(4.0 * _o.ContextSize));
    }

    private readonly int _embedWidth;

    private Convolution Remember(Convolution conv, int kernel)
    {
        _kernels[conv] = kernel;
        return conv;
    }

    public Conv2DLayer<T> EmbedConv1 { get; }
    public Conv2DLayer<T> EmbedConv2 { get; }
    public Conv2DLayer<T> EmbedConv3 { get; }
    public DepthwiseSeparableConvolutionalLayer<T> ConvNeXtDepthwisePointwise { get; }
    public Conv2DLayer<T> ConvNeXtProject { get; }
    public DenseLayer<T> EmbedOut { get; }
    public BiasFreeLinearLayer<T> EmbedNormBias { get; }
    public BiasFreeLinearLayer<T> EmbedNormLogScale { get; }
    public List<Stack> Stacks { get; } = new();
    public BiasFreeLinearLayer<T> OutputDownsampleWeights { get; }
    public EmbeddingLayer<T> DecoderEmbedding { get; }
    public BiasFreeLinearLayer<T>? DecoderConv { get; }
    public DenseLayer<T> JoinerEncoder { get; }
    public DenseLayer<T> JoinerDecoder { get; }
    public DenseLayer<T> JoinerOutput { get; }

    public ZipformerConfiguration Configuration => _o;
    public int OutputDim => _outputDim;

    // ---------------------------------------------------------------- activations and norms

    // SwooshL(x) = log(1 + exp(x - 4)) - 0.08x - 0.035; SwooshR(x) = log(1 + exp(x - 1)) - 0.08x - 0.313261687 (section 3.4).
    private Tensor<T> Swoosh(Tensor<T> x, double offset, double constant) =>
        _engine.TensorAddScalar(_engine.TensorSubtract(_engine.Softplus(_engine.TensorAddScalar(x, NumOps.FromDouble(-offset))), _engine.TensorMultiplyScalar(x, NumOps.FromDouble(0.08))), NumOps.FromDouble(-constant));

    private Tensor<T> SwooshL(Tensor<T> x) => Swoosh(x, 4.0, 0.035);

    private Tensor<T> SwooshR(Tensor<T> x) => Swoosh(x, 1.0, 0.313261687);

    // BiasNorm (section 3.3): x / sqrt(mean((x - b)^2)) * exp(log_scale), over the last axis.
    private Tensor<T> BiasNorm(Tensor<T> x, BiasFreeLinearLayer<T> bias, BiasFreeLinearLayer<T> logScale)
    {
        int n = x.Shape[0], d = x.Shape[1];
        var centred = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(_engine.Reshape(Value(bias), new[] { 1, d }), new[] { n, d }));
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(_engine.ReduceMean(_engine.TensorMultiply(centred, centred), new[] { 1 }, keepDims: true), NumOps.FromDouble(1e-20)), NumOps.FromDouble(-0.5));
        var scale = _engine.TensorMultiply(inverse, _engine.TensorBroadcastTo(_engine.Reshape(_engine.TensorExp(Value(logScale)), new[] { 1, 1 }), new[] { n, 1 }));
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(scale, new[] { n, d }));
    }

    // Bypass: orig + (x - orig) * c, with the channel-wise scale c.
    private Tensor<T> Bypass(Tensor<T> original, Tensor<T> x, BiasFreeLinearLayer<T> scale)
    {
        int n = x.Shape[0], d = x.Shape[1];
        return _engine.TensorAdd(original, _engine.TensorMultiply(_engine.TensorSubtract(x, original), _engine.TensorBroadcastTo(_engine.Reshape(Value(scale), new[] { 1, d }), new[] { n, d })));
    }

    // ---------------------------------------------------------------- features and Conv-Embed

    /// <summary>Log-mel features <c>[frames, bins]</c>, each bin normalized over the utterance.</summary>
    public double[,] Features(IReadOnlyList<double> waveform)
    {
        var samples = new Tensor<T>(new[] { waveform.Count });
        for (int i = 0; i < waveform.Count; i++) samples[i] = NumOps.FromDouble(waveform[i]);
        var mel = _mel.Forward(samples);
        int frames = mel.Shape[0], bins = mel.Shape[1];
        var features = new double[frames, bins];
        for (int b = 0; b < bins; b++)
        {
            double mean = 0, square = 0;
            for (int t = 0; t < frames; t++) mean += NumOps.ToDouble(mel[t, b]);
            mean /= frames;
            for (int t = 0; t < frames; t++) { double v = NumOps.ToDouble(mel[t, b]) - mean; square += v * v; }
            double std = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++) features[t, b] = std > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - mean) / std : 0.0;
        }
        return features;
    }

    /// <summary>SpecAugment (Park et al. 2019): frequency and time masks set to zero.</summary>
    public double[,] SpecAugment(double[,] features, Random random)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var masked = (double[,])features.Clone();
        for (int m = 0; m < _o.FrequencyMasks; m++)
        {
            int width = random.Next(Math.Min(_o.FrequencyMaskWidth, bins) + 1), start = random.Next(Math.Max(1, bins - width + 1));
            for (int t = 0; t < frames; t++)
                for (int b = start; b < Math.Min(bins, start + width); b++) masked[t, b] = 0.0;
        }
        int maxLength = (int)(_o.TimeMaskRatio * frames);
        for (int m = 0; m < _o.TimeMasks && maxLength > 0; m++)
        {
            int length = random.Next(maxLength + 1), start = random.Next(Math.Max(1, frames - length + 1));
            for (int t = start; t < Math.Min(frames, start + length); t++)
                for (int b = 0; b < bins; b++) masked[t, b] = 0.0;
        }
        return masked;
    }

    private Tensor<T> ConvEmbed(double[,] features)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var x = new Tensor<T>(new[] { 1, 1, frames, bins });
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) x[0, 0, t, b] = NumOps.FromDouble(features[t, b]);
        x = SwooshR(EmbedConv1.Forward(x));
        x = SwooshR(EmbedConv2.Forward(x));
        x = SwooshR(EmbedConv3.Forward(x));
        // ConvNeXt: depthwise 7x7, pointwise to 384, SwooshL, pointwise back to 128, residual.
        x = _engine.TensorAdd(x, ConvNeXtProject.Forward(SwooshL(ConvNeXtDepthwisePointwise.Forward(x))));
        int c = x.Shape[1], t2 = x.Shape[2], f = x.Shape[3];
        if (t2 <= 0) throw new ArgumentException("The audio is too short for Conv-Embed.");
        var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { t2, c * f });
        return BiasNorm(EmbedOut.Forward(steps), EmbedNormBias, EmbedNormLogScale);
    }

    // ---------------------------------------------------------------- the block

    // The compact relative positional encoding of offsets -(T-1)..(T-1): [2T - 1, posDim].
    private Tensor<T> PositionalEncoding(int length)
    {
        int d = _o.PosDim, rows = 2 * length - 1;
        var pe = new Tensor<T>(new[] { rows, d });
        double compression = Math.Sqrt(d), lengthScale = d / (2.0 * Math.PI);
        for (int r = 0; r < rows; r++)
        {
            double x = r - (length - 1);
            double compressed = compression * Math.Sign(x) * (Math.Log(Math.Abs(x) + compression) - Math.Log(compression));
            double angle = Math.Atan(compressed / lengthScale);
            for (int i = 0; i < d / 2; i++)
            {
                pe[r, 2 * i] = NumOps.FromDouble(Math.Cos(angle * (i + 1)));
                pe[r, 2 * i + 1] = NumOps.FromDouble(Math.Sin(angle * (i + 1)));
            }
            pe[r, d - 1] = NumOps.One;
        }
        return pe;
    }

    // RelPositionMultiheadAttentionWeights: softmax(q k^T + p P_{j - i}) per head, [H, T, T].
    private Tensor<T> AttentionWeights(Block b, Tensor<T> x, Tensor<T> positions, int heads, Tensor<T>? mask)
    {
        int t = x.Shape[0], qd = _o.QueryHeadDim, pd = _o.PosHeadDim;
        var projected = b.AttentionIn.Forward(x);
        Tensor<T> Heads(int start, int width) =>
            _engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(projected, new[] { 0, start }, new[] { t, width * heads }), new[] { t, heads, width }), new[] { 1, 0, 2 });
        var q = Heads(0, qd);
        var k = Heads(qd * heads, qd);
        var p = Heads(2 * qd * heads, pd);
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                         // [H, T, T]
        int n = 2 * t - 1;
        var pos = _engine.TensorPermute(_engine.Reshape(b.PositionProjection.Forward(positions), new[] { n, heads, pd }), new[] { 1, 2, 0 });   // [H, pd, n]
        var posScores = _engine.Reshape(_engine.BatchMatMul(p, pos), new[] { heads * t * n, 1 });                  // [H, T, n]
        var index = new int[heads * t * t];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++) index[(h * t + i) * t + j] = (h * t + i) * n + (t - 1 - i + j);
        var relative = _engine.Reshape(_engine.TensorIndexSelect(posScores, new Tensor<int>(new[] { index.Length }, new Vector<int>(index)), 0), new[] { heads, t, t });
        scores = _engine.TensorAdd(scores, relative);
        if (mask is not null) scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(_engine.Reshape(mask, new[] { 1, t, t }), new[] { heads, t, t }));
        return _engine.Softmax(scores, axis: 2);
    }

    private Tensor<T> SelfAttention(DenseLayer<T> input, DenseLayer<T> output, Tensor<T> x, Tensor<T> weights, int heads)
    {
        int t = x.Shape[0], vd = _o.ValueHeadDim;
        var v = _engine.TensorPermute(_engine.Reshape(input.Forward(x), new[] { t, heads, vd }), new[] { 1, 0, 2 });
        var attended = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { t, heads * vd });
        return output.Forward(attended);
    }

    private Tensor<T> FeedForwardModule(FeedForward ff, Tensor<T> x) => ff.Output.Forward(SwooshL(ff.Input.Forward(x)));

    // NonlinAttention: out(A * attention(tanh(B) * C)) with the first head's weights.
    private Tensor<T> NonlinearAttention(Block b, Tensor<T> x, Tensor<T> weights)
    {
        int t = x.Shape[0], hidden = b.NonlinIn.GetOutputShape()[^1] / 3;
        var projected = b.NonlinIn.Forward(x);
        var s = _engine.Tanh(_engine.TensorSlice(projected, new[] { 0, 0 }, new[] { t, hidden }));
        var c = _engine.TensorSlice(projected, new[] { 0, hidden }, new[] { t, hidden });
        var a = _engine.TensorSlice(projected, new[] { 0, 2 * hidden }, new[] { t, hidden });
        var head = _engine.Reshape(_engine.TensorSlice(weights, new[] { 0, 0, 0 }, new[] { 1, t, t }), new[] { t, t });
        var aggregated = _engine.TensorMatMul(head, _engine.TensorMultiply(c, s));
        return b.NonlinOut.Forward(_engine.TensorMultiply(aggregated, a));
    }

    // ConvolutionModule: GLU-style gate, depthwise convolution over time, SwooshR, output projection.
    private Tensor<T> ConvolutionModule(Convolution conv, Tensor<T> x)
    {
        int t = x.Shape[0], d = x.Shape[1];
        var projected = conv.Input.Forward(x);
        var gated = _engine.TensorMultiply(_engine.TensorSlice(projected, new[] { 0, 0 }, new[] { t, d }), _engine.Sigmoid(_engine.TensorSlice(projected, new[] { 0, d }, new[] { t, d })));
        var channels = _engine.TensorPermute(_engine.Reshape(gated, new[] { 1, t, d }), new[] { 0, 2, 1 });
        if (_o.Causal) channels = _engine.TensorConstantPad(channels, new[] { KernelOf(conv) - 1, 0 }, NumOps.Zero);
        var convolved = conv.Depthwise.Forward(channels);
        var back = _engine.Reshape(_engine.TensorPermute(convolved, new[] { 0, 2, 1 }), new[] { t, d });
        return conv.Output.Forward(SwooshR(back));
    }

    private Tensor<T> RunBlock(Block b, Tensor<T> x, Tensor<T> positions, int heads, Tensor<T>? mask)
    {
        var original = x;
        var weights = AttentionWeights(b, x, positions, heads, mask);
        x = _engine.TensorAdd(x, FeedForwardModule(b.Ff1, x));
        x = _engine.TensorAdd(x, NonlinearAttention(b, x, weights));
        x = _engine.TensorAdd(x, SelfAttention(b.Attention1In, b.Attention1Out, x, weights, heads));
        x = _engine.TensorAdd(x, ConvolutionModule(b.Conv1, x));
        x = _engine.TensorAdd(x, FeedForwardModule(b.Ff2, x));
        x = Bypass(original, x, b.BypassMid);
        x = _engine.TensorAdd(x, SelfAttention(b.Attention2In, b.Attention2Out, x, weights, heads));
        x = _engine.TensorAdd(x, ConvolutionModule(b.Conv2, x));
        x = _engine.TensorAdd(x, FeedForwardModule(b.Ff3, x));
        x = BiasNorm(x, b.NormBias, b.NormLogScale);
        return Bypass(original, x, b.Bypass);
    }

    // SimpleDownsample: pad by repeating the last frame, then a softmax-weighted sum of each group of frames.
    private Tensor<T> Downsample(Tensor<T> x, int factor, BiasFreeLinearLayer<T> weights)
    {
        int t = x.Shape[0], d = x.Shape[1], reduced = (t + factor - 1) / factor, pad = reduced * factor - t;
        if (pad > 0)
        {
            var last = _engine.TensorSlice(x, new[] { t - 1, 0 }, new[] { 1, d });
            x = _engine.TensorConcatenate(new[] { x, _engine.TensorBroadcastTo(last, new[] { pad, d }) }, 0);
        }
        var w = _engine.Softmax(_engine.Reshape(Value(weights), new[] { 1, factor }), axis: 1);
        var grouped = _engine.Reshape(x, new[] { reduced, factor, d });
        return _engine.ReduceSum(_engine.TensorMultiply(grouped, _engine.TensorBroadcastTo(_engine.Reshape(w, new[] { 1, factor, 1 }), new[] { reduced, factor, d })), new[] { 1 }, keepDims: false);
    }

    private Tensor<T> Upsample(Tensor<T> x, int factor, int length)
    {
        int t = x.Shape[0], d = x.Shape[1];
        var repeated = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(x, new[] { t, 1, d }), new[] { t, factor, d }), new[] { t * factor, d });
        return _engine.TensorSlice(repeated, new[] { 0, 0 }, new[] { length, d });
    }

    private Tensor<T> ConvertChannels(Tensor<T> x, int channels)
    {
        int t = x.Shape[0], d = x.Shape[1];
        if (channels <= d) return _engine.TensorSlice(x, new[] { 0, 0 }, new[] { t, channels });
        return _engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { t, channels - d }) }, 1);
    }

    // Chunk-wise attention for streaming: frames attend within their chunk (chunks reduced to the stack's frame rate).
    private readonly Dictionary<Convolution, int> _kernels = new();

    private int KernelOf(Convolution conv) => _kernels[conv];

    // Chunk-wise attention for streaming (icefall's _get_attn_mask): a frame attends to its own chunk and to the chunks
    // whose start lies within the left context before its chunk, at the stack's frame rate.
    private Tensor<T>? ChunkMask(int length, int downsampling)
    {
        if (_o.ChunkSize <= 0) return null;
        int chunk = Math.Max(1, _o.ChunkSize / downsampling);
        int left = _o.LeftContextFrames < 0 ? int.MaxValue : _o.LeftContextFrames / downsampling;
        var mask = new Tensor<T>(new[] { length, length });
        var blocked = NumOps.FromDouble(-1000.0);
        for (int i = 0; i < length; i++)
        {
            int start = i / chunk * chunk;
            for (int j = 0; j < length; j++)
                if (j / chunk > i / chunk || (left != int.MaxValue && j < start - left)) mask[i, j] = blocked;
        }
        return mask;
    }

    /// <summary>The encoder output <c>[T, OutputDim]</c> at 25 Hz.</summary>
    public Tensor<T> Encode(double[,] features, bool training)
    {
        var x = ConvEmbed(features);
        var outputs = new List<Tensor<T>>();
        foreach (var stack in Stacks)
        {
            x = ConvertChannels(x, stack.Dim);
            var original = x;
            var h = stack.DownsampleWeights is null ? x : Downsample(x, stack.Downsampling, stack.DownsampleWeights);
            var positions = PositionalEncoding(h.Shape[0]);
            var mask = ChunkMask(h.Shape[0], stack.Downsampling);
            foreach (var block in stack.Blocks) h = RunBlock(block, h, positions, stack.Heads, mask);
            if (stack.OutCombiner is not null) h = Bypass(original, Upsample(h, stack.Downsampling, original.Shape[0]), stack.OutCombiner);
            x = h;
            outputs.Add(x);
        }
        // _get_full_dim_output: the last stack's output, extended with the wider channels of earlier stacks.
        var pieces = new List<Tensor<T>> { outputs[outputs.Count - 1] };
        int current = Stacks[Stacks.Count - 1].Dim;
        for (int s = Stacks.Count - 2; s >= 0; s--)
        {
            int d = Stacks[s].Dim;
            if (d <= current) continue;
            pieces.Add(_engine.TensorSlice(outputs[s], new[] { 0, current }, new[] { outputs[s].Shape[0], d - current }));
            current = d;
        }
        var full = _engine.TensorConcatenate(pieces.ToArray(), 1);
        return Downsample(full, 2, OutputDownsampleWeights);
    }

    // ---------------------------------------------------------------- decoder and joiner

    /// <summary>The stateless decoder's output <c>[U + 1, DecoderDim]</c> for the label history: the blank, then each label;
    /// each position reads the last <c>ContextSize</c> labels through a grouped convolution and ReLU.</summary>
    public Tensor<T> Predict(IReadOnlyList<int> labels)
    {
        int n = labels.Count + 1, d = _o.DecoderDim, context = _o.ContextSize;
        var history = new Tensor<T>(new[] { 1, n });
        for (int u = 0; u < labels.Count; u++) history[0, u + 1] = NumOps.FromDouble(labels[u]);
        var embedded = _engine.Reshape(DecoderEmbedding.Forward(history), new[] { n, d });
        if (DecoderConv is null) return embedded;
        // Left pad with context - 1 zero frames, then each output channel o of group g reads the 4 channels of g at the
        // last `context` positions: weights [G, 4 inputs * context, 4 outputs].
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { context - 1, d }), embedded }, 0);
        var windows = new Tensor<T>[context];
        for (int k = 0; k < context; k++) windows[k] = _engine.TensorSlice(padded, new[] { k, 0 }, new[] { n, d });
        int groups = d / 4;
        var stacked = _engine.TensorConcatenate(windows.Select(w => _engine.Reshape(w, new[] { n, groups, 4, 1 })).ToArray(), 3);   // [n, G, 4, context]
        var inputs = _engine.TensorPermute(_engine.Reshape(stacked, new[] { n, groups, 4 * context }), new[] { 1, 0, 2 });           // [G, n, 4 * context]
        var weights = _engine.Reshape(Value(DecoderConv), new[] { groups, 4 * context, 4 });
        var output = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(inputs, weights), new[] { 1, 0, 2 }), new[] { n, d });
        return _engine.ReLU(output);
    }

    /// <summary>The joiner's log-probabilities <c>[T, U + 1, VocabSize]</c>: log softmax(W tanh(W_enc f_t + W_dec g_u)).</summary>
    public Tensor<T> Joint(Tensor<T> encoded, Tensor<T> predicted)
    {
        int frames = encoded.Shape[0], width = predicted.Shape[0], j = _o.JoinerDim;
        var f = _engine.Reshape(JoinerEncoder.Forward(encoded), new[] { frames, 1, j });
        var g = _engine.Reshape(JoinerDecoder.Forward(predicted), new[] { 1, width, j });
        var hidden = _engine.Tanh(_engine.TensorAdd(_engine.TensorBroadcastTo(f, new[] { frames, width, j }), _engine.TensorBroadcastTo(g, new[] { frames, width, j })));
        var logits = JoinerOutput.Forward(_engine.Reshape(hidden, new[] { frames * width, j }));
        return _engine.TensorLogSoftmax(_engine.Reshape(logits, new[] { frames, width, _o.VocabSize }), axis: -1);
    }

    /// <summary>The transducer loss of a label sequence (icefall trains with its pruned approximation of this loss).</summary>
    public Tensor<T> TransducerLoss(Tensor<T> encoded, IReadOnlyList<int> labels) =>
        Foundations.TransducerLoss.Compute(_engine, Joint(encoded, Predict(labels)), labels, blank: 0);

    /// <summary>Greedy transducer decoding, at most <paramref name="maxSymbolsPerFrame"/> labels per frame.</summary>
    public List<int> GreedyTransducer(Tensor<T> encoded, int maxSymbolsPerFrame = 10)
    {
        var labels = new List<int>();
        int frames = encoded.Shape[0], d = encoded.Shape[1];
        for (int t = 0; t < frames; t++)
        {
            var frame = _engine.TensorSlice(encoded, new[] { t, 0 }, new[] { 1, d });
            for (int s = 0; s < maxSymbolsPerFrame; s++)
            {
                var g = Predict(labels);
                var last = _engine.TensorSlice(g, new[] { labels.Count, 0 }, new[] { 1, _o.DecoderDim });
                var logProbs = Joint(frame, last);
                int best = 0;
                for (int k = 1; k < _o.VocabSize; k++)
                    if (NumOps.GreaterThan(logProbs[0, 0, k], logProbs[0, 0, best])) best = k;
                if (best == 0) break;
                labels.Add(best);
            }
        }
        return labels;
    }

    /// <summary>Greedy CTC-style collapse of per-frame distributions (used to read a frame-level target).</summary>
    public static List<int> GreedyCtc(Tensor<T> logits) => ConformerAsrNetwork<T>.GreedyCtc(logits);
}
