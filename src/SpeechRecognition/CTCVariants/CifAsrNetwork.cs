using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.SpeechRecognition.CTCVariants;

/// <summary>The sizes and loss weights of a CIF-based recognizer (Dong and Xu 2020, section 4).</summary>
internal sealed record CifAsrConfiguration
{
    /// <summary>Filterbank bands; deltas and delta-deltas are stacked as channels (40).</summary>
    public int Mels { get; init; } = 40;
    /// <summary>The front-end convolutions' filters (64).</summary>
    public int Filters { get; init; } = 64;
    public int Width { get; init; } = 640;
    public int Heads { get; init; } = 4;
    public int FeedForward { get; init; } = 2560;
    /// <summary>SANs at each temporal resolution of the encoder (n = 5).</summary>
    public int SansPerResolution { get; init; } = 5;
    /// <summary>Temporal resolutions; a width-2 pooling separates each (3, which with the strided convolution encodes at
    /// 1/8 of the frame rate).</summary>
    public int Resolutions { get; init; } = 3;
    /// <summary>The weight predictor's convolution window (3).</summary>
    public int PredictorWindow { get; init; } = 3;
    /// <summary>The non-autoregressive decoder's SANs (2).</summary>
    public int DecoderLayers { get; init; } = 2;
    /// <summary>Output labels: blank 0, end of sentence 1 and pad 2 included.</summary>
    public int VocabSize { get; init; } = 5230;
    public int EndToken { get; init; } = 1;
    public int PadToken { get; init; } = 2;
    /// <summary>The residual weight above which inference fires once more at the tail (0.5).</summary>
    public double TailThreshold { get; init; } = 0.5;
    public double LabelSmoothing { get; init; } = 0.2;
    /// <summary>λ1, the encoder CTC loss's weight.</summary>
    public double CtcWeight { get; init; } = 0.5;
    /// <summary>λ2, the quantity loss's weight.</summary>
    public double QuantityWeight { get; init; } = 1.0;
}

/// <summary>
/// The CIF-based recognizer of Dong and Xu (2020): the self-attention aligner's encoder, a convolutional weight predictor,
/// continuous integrate-and-fire and the non-autoregressive SAN decoder.
/// </summary>
/// <remarks>
/// <para>Encoder (SAA, Dong et al. 2019, section 4; Extended-RNA, Dong et al. 2018, section 3): 40 filterbanks with deltas
/// and delta-deltas as three channels, a 3×3 convolution with 64 filters striding 2 in time followed by layer
/// normalization, a multiplicative unit (four 3×3 convolutions, each layer-normalized, making gates g1–g3 and an update u;
/// MU(I) = g1 ⊙ tanh(g2 ⊙ I + g3 ⊙ u + b)), a linear projection of each flattened frame to the model width, and n SANs at
/// each of three temporal resolutions separated by max pooling of width 2. A SAN is X1 = LN(X), X2 = LN(MHA(X1) + X1),
/// SAN(X) = FFN(X2) + X2, its attention scores biased by −log(1 + a) for positions a apart (no positional encoding).
/// Normalization in the layer norms is over each frame's channels and frequencies.</para>
/// <para>The weight predictor is a 1-D convolution over a window of encoder frames with d filters, layer normalization,
/// ReLU, and a one-unit sigmoid layer. CIF (threshold 1) integrates the frames into one embedding per label; training
/// scales the weights to sum to the target length, and inference fires once more when the residual weight exceeds 0.5.
/// The decoder passes the embeddings through SANs and a projection to the labels. The loss is the label-smoothed
/// cross-entropy on the targets ending with &lt;EOS&gt;, plus λ1 times the encoder's CTC loss and λ2 times the quantity
/// loss |Σα − S|.</para>
/// </remarks>
internal sealed class CifAsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly CifAsrConfiguration _o;
    private readonly KaldiFbank<T> _fbank;

    internal sealed record San(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public CifAsrNetwork(IEngine engine, List<LayerBase<T>> layers, CifAsrConfiguration configuration)
    {
        _engine = engine;
        _o = configuration;
        int d = _o.Width, c = _o.Filters, frame = c * _o.Mels;
        if (d % _o.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        if (_o.Resolutions < 1 || _o.SansPerResolution < 1 || _o.DecoderLayers < 1)
            throw new ArgumentException("The encoder and the decoder need at least one SAN.", nameof(configuration));
        _fbank = new KaldiFbank<T>(engine, 16000, _o.Mels, WindowType.Hamming);
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, identity));
        LayerNormalizationLayer<T> Norm(int width) => Own(new LayerNormalizationLayer<T>(width));
        Conv2DLayer<T> Conv(int inputs, int stride) => Own(new Conv2DLayer<T>(inputs, c, 3, 3, stride, 1, 1, 1));
        San NewSan() => new San(Norm(d), Linear(d), Linear(d), Linear(d), Linear(d), Norm(d), Linear(_o.FeedForward), Linear(d));

        FrontConvolution = Conv(3, 2);
        FrontNorm = Norm(frame);
        for (int i = 0; i < 4; i++) { MuConvolutions.Add(Conv(c, 1)); MuNorms.Add(Norm(frame)); }
        MuBias = Own(new EmbeddingLayer<T>(1, c));
        Projection = Linear(d);
        for (int i = 0; i < _o.Resolutions * _o.SansPerResolution; i++) EncoderSans.Add(NewSan());
        CtcOutput = Linear(_o.VocabSize);
        PredictorConvolution = Own(new NormedConv1DLayer<T>(d, d, _o.PredictorWindow, 1, 1, 1, 0, false, ConvolutionNormalization.None, useBias: true));
        PredictorNorm = Norm(d);
        PredictorOutput = Linear(1);
        for (int i = 0; i < _o.DecoderLayers; i++) DecoderSans.Add(NewSan());
        Output = Linear(_o.VocabSize);
    }

    public CifAsrConfiguration Configuration => _o;
    public Conv2DLayer<T> FrontConvolution { get; }
    public LayerNormalizationLayer<T> FrontNorm { get; }
    public List<Conv2DLayer<T>> MuConvolutions { get; } = new();
    public List<LayerNormalizationLayer<T>> MuNorms { get; } = new();
    public EmbeddingLayer<T> MuBias { get; }
    public DenseLayer<T> Projection { get; }
    public List<San> EncoderSans { get; } = new();
    public DenseLayer<T> CtcOutput { get; }
    public NormedConv1DLayer<T> PredictorConvolution { get; }
    public LayerNormalizationLayer<T> PredictorNorm { get; }
    public DenseLayer<T> PredictorOutput { get; }
    public List<San> DecoderSans { get; } = new();
    public DenseLayer<T> Output { get; }

    /// <summary>The input features <c>[1, 3, frames, mels]</c> of a 16 kHz waveform: log filterbanks, their deltas and
    /// delta-deltas (Kaldi's window of 2), each band normalized to zero mean and unit variance over the utterance.</summary>
    public Tensor<T> Features(Tensor<T> waveform)
    {
        var fbank = _fbank.Forward(waveform);
        int frames = fbank.Shape[0], mels = _o.Mels;
        if (frames == 0) throw new ArgumentException("The waveform is shorter than one analysis window.", nameof(waveform));
        var channels = new double[3][,];
        channels[0] = new double[frames, mels];
        for (int t = 0; t < frames; t++) for (int m = 0; m < mels; m++) channels[0][t, m] = NumOps.ToDouble(fbank[t, m]);
        channels[1] = Deltas(channels[0]);
        channels[2] = Deltas(channels[1]);
        var features = new Tensor<T>(new[] { 1, 3, frames, mels });
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < mels; m++)
            {
                double mean = 0, square = 0;
                for (int t = 0; t < frames; t++) { mean += channels[k][t, m]; square += channels[k][t, m] * channels[k][t, m]; }
                mean /= frames;
                double deviation = Math.Sqrt(Math.Max(square / frames - mean * mean, 0)) + 1e-5;
                for (int t = 0; t < frames; t++) features[0, k, t, m] = NumOps.FromDouble((channels[k][t, m] - mean) / deviation);
            }
        return features;
    }

    // Kaldi's deltas with a window of 2: Σ n (x[t+n] − x[t−n]) / (2 Σ n²), the edges repeated.
    private static double[,] Deltas(double[,] x)
    {
        int frames = x.GetLength(0), bands = x.GetLength(1);
        var d = new double[frames, bands];
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < bands; m++)
            {
                double sum = 0;
                for (int n = 1; n <= 2; n++) sum += n * (x[Math.Min(frames - 1, t + n), m] - x[Math.Max(0, t - n), m]);
                d[t, m] = sum / 10.0;
            }
        return d;
    }

    // [1, C, T, F] normalized over each frame's channels and frequencies.
    private Tensor<T> FrameNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2], f = x.Shape[3];
        var frames = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { c, t, f }), new[] { 1, 0, 2 }), new[] { t, c * f });
        return _engine.Reshape(_engine.TensorPermute(_engine.Reshape(norm.Forward(frames), new[] { t, c, f }), new[] { 1, 0, 2 }), new[] { 1, c, t, f });
    }

    /// <summary>The encoder's output <c>[U, width]</c> for the features <c>[1, 3, frames, mels]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> features)
    {
        var x = FrameNorm(FrontNorm, FrontConvolution.Forward(features));
        int c = x.Shape[1], t = x.Shape[2], f = x.Shape[3];
        Tensor<T> Gate(int i) => FrameNorm(MuNorms[i], MuConvolutions[i].Forward(x));
        var ids = new Tensor<T>(new[] { 1, 1 });
        var bias = _engine.TensorTile(_engine.Reshape(MuBias.Forward(ids), new[] { 1, c, 1, 1 }), new[] { 1, 1, t, f });
        var inner = _engine.TensorAdd(_engine.TensorAdd(_engine.TensorMultiply(_engine.Sigmoid(Gate(1)), x),
            _engine.TensorMultiply(_engine.Sigmoid(Gate(2)), _engine.Tanh(Gate(3)))), bias);
        x = _engine.TensorMultiply(_engine.Sigmoid(Gate(0)), _engine.Tanh(inner));
        var h = Projection.Forward(_engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { c, t, f }), new[] { 1, 0, 2 }), new[] { t, c * f }));
        for (int i = 0; i < EncoderSans.Count; i++)
        {
            if (i > 0 && i % _o.SansPerResolution == 0) h = Pool(h);
            h = Run(EncoderSans[i], h);
        }
        return h;
    }

    // Max pooling of width 2 over time, the last frame repeated when the length is odd.
    private Tensor<T> Pool(Tensor<T> x)
    {
        int t = x.Shape[0], d = x.Shape[1];
        if (t == 1) return x;
        if (t % 2 == 1) x = _engine.TensorConcatenate(new[] { x, _engine.TensorSlice(x, new[] { t - 1, 0 }, new[] { 1, d }) }, 0);
        int half = x.Shape[0] / 2;
        var pairs = _engine.Reshape(x, new[] { half, 2 * d });
        var even = _engine.TensorSlice(pairs, new[] { 0, 0 }, new[] { half, d });
        var odd = _engine.TensorSlice(pairs, new[] { 0, d }, new[] { half, d });
        // max(a, b) = (a + b + |a − b|) / 2
        var difference = _engine.TensorSubtract(even, odd);
        var magnitude = _engine.TensorAdd(_engine.ReLU(difference), _engine.ReLU(_engine.TensorNegate(difference)));
        return _engine.TensorMultiplyScalar(_engine.TensorAdd(_engine.TensorAdd(even, odd), magnitude), NumOps.FromDouble(0.5));
    }

    // SAA's SAN (Equations 1-6) with the proximity bias −log(1 + a).
    private Tensor<T> Run(San b, Tensor<T> x)
    {
        int n = x.Shape[0], heads = _o.Heads, dk = _o.Width / heads;
        var x1 = b.Norm1.Forward(x);
        Tensor<T> Heads(Tensor<T> y) => _engine.TensorPermute(_engine.Reshape(y, new[] { n, heads, dk }), new[] { 1, 0, 2 });
        var proximity = new Tensor<T>(new[] { heads, n, n });
        for (int h = 0; h < heads; h++) for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) proximity[h, i, j] = NumOps.FromDouble(-Math.Log(1 + Math.Abs(i - j)));
        var scores = _engine.TensorAdd(_engine.BatchMatMul(_engine.TensorMultiplyScalar(Heads(b.Query.Forward(x1)), NumOps.FromDouble(Math.Pow(dk, -0.5))),
            _engine.TensorPermute(Heads(b.Key.Forward(x1)), new[] { 0, 2, 1 })), proximity);
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(b.Value.Forward(x1)));
        var y1 = b.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { n, _o.Width }));
        var x2 = b.Norm2.Forward(_engine.TensorAdd(y1, x1));
        return _engine.TensorAdd(b.FeedForward2.Forward(_engine.ReLU(b.FeedForward1.Forward(x2))), x2);
    }

    /// <summary>The predictor's weights α <c>[1, U]</c> for the encoder output <c>[U, width]</c>.</summary>
    public Tensor<T> Alphas(Tensor<T> encoded)
    {
        int left = (_o.PredictorWindow - 1) / 2;
        var convolved = SanmDecoder<T>.Convolve(_engine, PredictorConvolution, encoded, left, _o.PredictorWindow - 1 - left);
        var alphas = _engine.Sigmoid(PredictorOutput.Forward(_engine.ReLU(PredictorNorm.Forward(convolved))));
        return _engine.Reshape(alphas, new[] { 1, encoded.Shape[0] });
    }

    /// <summary>The decoder's logits <c>[n, VocabSize]</c> for integrated embeddings <c>[n, width]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> embeddings)
    {
        var x = embeddings;
        foreach (var b in DecoderSans) x = Run(b, x);
        return Output.Forward(x);
    }

    /// <summary>Greedy decoding: CIF with the tail rule, then the most likely label per embedding up to &lt;EOS&gt;, without
    /// blank and pad.</summary>
    public List<int> Greedy(Tensor<T> encoded, int maxTokens)
    {
        var tokens = new List<int>();
        var alphas = Alphas(encoded);
        double total = 0;
        for (int i = 0; i < alphas.Length; i++) total += NumOps.ToDouble(alphas.GetFlat(i));
        int fires = (int)Math.Floor(total + 1e-9);
        if (total - fires > _o.TailThreshold) fires++;
        if (fires == 0) return tokens;
        var logits = Decode(ParaformerNetwork<T>.Integrate(_engine, encoded, alphas, fires));
        for (int n = 0; n < logits.Shape[0] && tokens.Count < maxTokens; n++)
        {
            int best = 0;
            for (int v = 1; v < logits.Shape[1]; v++) if (NumOps.GreaterThan(logits[n, v], logits[n, best])) best = v;
            if (best == _o.EndToken) break;
            if (best != 0 && best != _o.PadToken) tokens.Add(best);
        }
        return tokens;
    }

    /// <summary>The training loss of one utterance (Equation 1): L_CE + λ1 L_CTC + λ2 L_QUA, the cross-entropy over the
    /// labels followed by &lt;EOS&gt; on the scaled CIF embeddings.</summary>
    public Tensor<T> Loss(Tensor<T> encoded, IReadOnlyList<int> labels)
    {
        var targets = labels.Concat(new[] { _o.EndToken }).ToArray();
        int n = targets.Length, u = encoded.Shape[0];
        var alphas = Alphas(encoded);
        var sum = _engine.ReduceSum(alphas, new[] { 0, 1 }, keepDims: true);
        var count = new Tensor<T>(new[] { 1, 1 });
        count[0, 0] = NumOps.FromDouble(n);
        var difference = _engine.TensorSubtract(sum, count);
        var quantity = _engine.TensorAdd(_engine.ReLU(difference), _engine.ReLU(_engine.TensorNegate(difference)));
        var scaled = _engine.TensorMultiplyScalar(_engine.TensorDivide(alphas, _engine.TensorTile(sum, new[] { 1, u })), NumOps.FromDouble(n));
        var logProbabilities = _engine.TensorLogSoftmax(Decode(ParaformerNetwork<T>.Integrate(_engine, encoded, scaled, n)), axis: -1);
        var total = _engine.TensorAdd(ParaformerNetwork<T>.LabelSmoothedLoss(_engine, logProbabilities, targets, _o.LabelSmoothing, 1),
            _engine.TensorMultiplyScalar(quantity, NumOps.FromDouble(_o.QuantityWeight)));
        if (_o.CtcWeight > 0 && labels.Count > 0)
        {
            var ctc = _engine.TensorCTCLoss(_engine.Reshape(_engine.TensorLogSoftmax(CtcOutput.Forward(encoded), axis: -1), new[] { u, 1, _o.VocabSize }),
                new Tensor<int>(labels.ToArray(), new[] { labels.Count }), new[] { u }, new[] { labels.Count }, 0);
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(_engine.Reshape(ctc, new[] { 1, 1 }), NumOps.FromDouble(_o.CtcWeight)));
        }
        return _engine.Reshape(total, new[] { 1 });
    }
}
