using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>An Emformer RNN-T's shape (torchaudio's <c>emformer_rnnt_model</c> arguments).</summary>
internal sealed record EmformerRnntConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int NumMels { get; init; } = 80;
    public int WindowLength { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public int EncodingDim { get; init; } = 1024;
    public int TimeReductionInputDim { get; init; } = 128;
    public int TimeReductionStride { get; init; } = 4;
    public int NumHeads { get; init; } = 8;
    public int FeedForwardDim { get; init; } = 2048;
    public int NumLayers { get; init; } = 20;
    /// <summary>Segment length in input frames (divided by the time-reduction stride inside the Emformer).</summary>
    public int SegmentLength { get; init; } = 16;
    /// <summary>Right-context (look-ahead) length in input frames.</summary>
    public int RightContextLength { get; init; } = 4;
    /// <summary>Left-context length in reduced frames.</summary>
    public int LeftContextLength { get; init; } = 30;
    public int SymbolEmbeddingDim { get; init; } = 512;
    public int NumLstmLayers { get; init; } = 3;
    public double LstmLayerNormEpsilon { get; init; } = 1e-3;
    public int VocabSize { get; init; } = 4097;
    public int FrequencyMasks { get; init; } = 2;
    public int FrequencyMaskWidth { get; init; } = 27;
    public int TimeMasks { get; init; } = 10;
    public double TimeMaskRatio { get; init; } = 0.05;
}

/// <summary>
/// The Emformer RNN transducer (Shi et al. 2021, "Emformer: Efficient Memory Transformer Based Acoustic Model for Low
/// Latency Streaming Speech Recognition") as torchaudio's <c>emformer_rnnt_model</c> builds it: a bias-free linear layer on
/// the log-mel frames, time reduction by stacking frames, Emformer layers whose attention for each segment covers its
/// left context, the segment and a copy of its right context (look-ahead) processed as separate rows, an output linear
/// layer with layer norm, a prediction network of layer-normalized LSTMs between layer norms, and a joiner applying ReLU
/// to the summed encodings before its output layer. The memory bank is off (<c>max_memory_size = 0</c>), as in
/// <c>emformer_rnnt_base</c>.
/// </summary>
internal sealed class EmformerRnntNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly EmformerRnntConfiguration _o;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _width;

    internal sealed class Layer
    {
        public Layer(LayerNormalizationLayer<T> inputNorm, DenseLayer<T> query, DenseLayer<T> keyValue, DenseLayer<T> output,
            LayerNormalizationLayer<T> feedForwardNorm, DenseLayer<T> expand, DenseLayer<T> project, LayerNormalizationLayer<T> outputNorm)
        {
            InputNorm = inputNorm; Query = query; KeyValue = keyValue; Output = output;
            FeedForwardNorm = feedForwardNorm; Expand = expand; Project = project; OutputNorm = outputNorm;
        }

        public LayerNormalizationLayer<T> InputNorm { get; }
        public DenseLayer<T> Query { get; }
        public DenseLayer<T> KeyValue { get; }
        public DenseLayer<T> Output { get; }
        public LayerNormalizationLayer<T> FeedForwardNorm { get; }
        public DenseLayer<T> Expand { get; }
        public DenseLayer<T> Project { get; }
        public LayerNormalizationLayer<T> OutputNorm { get; }
    }

    internal sealed class Lstm
    {
        public Lstm(BiasFreeLinearLayer<T> inputGates, BiasFreeLinearLayer<T> recurrentGates, LayerNormalizationLayer<T> gateNorm, LayerNormalizationLayer<T> cellNorm)
        {
            InputGates = inputGates; RecurrentGates = recurrentGates; GateNorm = gateNorm; CellNorm = cellNorm;
        }

        public BiasFreeLinearLayer<T> InputGates { get; }
        public BiasFreeLinearLayer<T> RecurrentGates { get; }
        public LayerNormalizationLayer<T> GateNorm { get; }
        public LayerNormalizationLayer<T> CellNorm { get; }
    }

    public EmformerRnntNetwork(IEngine engine, List<LayerBase<T>> layers, EmformerRnntConfiguration options)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _o = options ?? throw new ArgumentNullException(nameof(options));
        if (_o.SegmentLength % _o.TimeReductionStride != 0 || _o.RightContextLength % _o.TimeReductionStride != 0)
            throw new ArgumentException("The segment and right-context lengths must be multiples of the time-reduction stride.", nameof(options));
        _width = _o.TimeReductionInputDim * _o.TimeReductionStride;
        if (_width % _o.NumHeads != 0) throw new ArgumentException("The Emformer width must divide into the heads.", nameof(options));
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.NumMels, nFft: _o.WindowLength, hopLength: _o.HopLength, logMel: true, logOffset: 1e-6);
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        var gelu = (IActivationFunction<T>)new GELUActivation<T>();
        InputLinear = Own(new BiasFreeLinearLayer<T>(_o.NumMels, _o.TimeReductionInputDim));
        for (int i = 0; i < _o.NumLayers; i++)
            Layers.Add(new Layer(Own(new LayerNormalizationLayer<T>(_width)), Own(new DenseLayer<T>(_width, identity)), Own(new DenseLayer<T>(2 * _width, identity)),
                Own(new DenseLayer<T>(_width, identity)), Own(new LayerNormalizationLayer<T>(_width)), Own(new DenseLayer<T>(_o.FeedForwardDim, gelu)),
                Own(new DenseLayer<T>(_width, identity)), Own(new LayerNormalizationLayer<T>(_width))));
        OutputLinear = Own(new DenseLayer<T>(_o.EncodingDim, identity));
        OutputNorm = Own(new LayerNormalizationLayer<T>(_o.EncodingDim));
        int s = _o.SymbolEmbeddingDim;
        Embedding = Own(new EmbeddingLayer<T>(_o.VocabSize, s));
        PredictorInputNorm = Own(new LayerNormalizationLayer<T>(s));
        for (int i = 0; i < _o.NumLstmLayers; i++)
            Lstms.Add(new Lstm(Own(new BiasFreeLinearLayer<T>(s, 4 * s)), Own(new BiasFreeLinearLayer<T>(s, 4 * s)),
                Own(new LayerNormalizationLayer<T>(4 * s, _o.LstmLayerNormEpsilon)), Own(new LayerNormalizationLayer<T>(s, _o.LstmLayerNormEpsilon))));
        PredictorLinear = Own(new DenseLayer<T>(_o.EncodingDim, identity));
        PredictorNorm = Own(new LayerNormalizationLayer<T>(_o.EncodingDim));
        Joiner = Own(new DenseLayer<T>(_o.VocabSize, identity));
    }

    public BiasFreeLinearLayer<T> InputLinear { get; }
    public List<Layer> Layers { get; } = new();
    public DenseLayer<T> OutputLinear { get; }
    public LayerNormalizationLayer<T> OutputNorm { get; }
    public EmbeddingLayer<T> Embedding { get; }
    public LayerNormalizationLayer<T> PredictorInputNorm { get; }
    public List<Lstm> Lstms { get; } = new();
    public DenseLayer<T> PredictorLinear { get; }
    public LayerNormalizationLayer<T> PredictorNorm { get; }
    public DenseLayer<T> Joiner { get; }

    public EmformerRnntConfiguration Configuration => _o;

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

    /// <summary>The encoder output <c>[T', EncodingDim]</c>: the utterance frames after time reduction, the last right-context
    /// frames serving only as look-ahead.</summary>
    public Tensor<T> Encode(double[,] features, bool training)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1), stride = _o.TimeReductionStride;
        var x = new Tensor<T>(new[] { frames, bins });
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) x[t, b] = NumOps.FromDouble(features[t, b]);
        var projected = InputLinear.Forward(x);
        int reduced = frames / stride, rc = _o.RightContextLength / stride, segment = _o.SegmentLength / stride;
        if (reduced <= rc) throw new ArgumentException("The audio is shorter than the right context.", nameof(features));
        var input = _engine.Reshape(_engine.TensorSlice(projected, new[] { 0, 0 }, new[] { reduced * stride, _o.TimeReductionInputDim }), new[] { reduced, _width });
        int length = reduced - rc, segments = (length + segment - 1) / segment;
        // _gen_right_context: the rc frames after each segment, and the input's last rc frames for the final segment.
        var blocks = new List<Tensor<T>>();
        for (int s = 0; s < segments - 1; s++) blocks.Add(_engine.TensorSlice(input, new[] { (s + 1) * segment, 0 }, new[] { rc, _width }));
        blocks.Add(_engine.TensorSlice(input, new[] { reduced - rc, 0 }, new[] { rc, _width }));
        var rightContext = rc > 0 ? _engine.TensorConcatenate(blocks.ToArray(), 0) : new Tensor<T>(new[] { 0, _width });
        var utterance = _engine.TensorSlice(input, new[] { 0, 0 }, new[] { length, _width });
        var mask = AttentionMask(length, segments, segment, rc);
        int rows = segments * rc + length;
        foreach (var layer in Layers)
        {
            var block = rc > 0 ? _engine.TensorConcatenate(new[] { rightContext, utterance }, 0) : utterance;
            var normed = layer.InputNorm.Forward(block);
            var attended = layer.Output.Forward(Attend(layer, normed, mask, rows));
            var result = _engine.TensorAdd(attended, block);
            result = _engine.TensorAdd(layer.Project.Forward(layer.Expand.Forward(layer.FeedForwardNorm.Forward(result))), result);
            result = layer.OutputNorm.Forward(result);
            if (rc > 0) rightContext = _engine.TensorSlice(result, new[] { 0, 0 }, new[] { segments * rc, _width });
            utterance = _engine.TensorSlice(result, new[] { segments * rc, 0 }, new[] { length, _width });
        }
        return OutputNorm.Forward(OutputLinear.Forward(utterance));
    }

    // Multi-head attention of the rows [right context; utterance] over themselves under the additive mask [rows, rows].
    private Tensor<T> Attend(Layer layer, Tensor<T> x, Tensor<T> mask, int rows)
    {
        int heads = _o.NumHeads, hd = _width / heads;
        var q = _engine.TensorMultiplyScalar(layer.Query.Forward(x), NumOps.FromDouble(Math.Pow(hd, -0.5)));
        var kv = layer.KeyValue.Forward(x);
        var k = _engine.TensorSlice(kv, new[] { 0, 0 }, new[] { rows, _width });
        var v = _engine.TensorSlice(kv, new[] { 0, _width }, new[] { rows, _width });
        Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, heads, hd }), new[] { 1, 0, 2 });
        var scores = _engine.BatchMatMul(Heads(q), _engine.TensorPermute(Heads(k), new[] { 0, 2, 1 }));
        scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(_engine.Reshape(mask, new[] { 1, rows, rows }), new[] { heads, rows, rows }));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(v));
        return _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { rows, _width });
    }

    // _gen_attention_mask without memory: right-context rows and query rows of segment s attend to s's right-context block
    // and to the utterance frames from LeftContextLength before the segment to its end.
    private Tensor<T> AttentionMask(int length, int segments, int segment, int rc)
    {
        int rows = segments * rc + length;
        var mask = new Tensor<T>(new[] { rows, rows });
        var blocked = NumOps.FromDouble(-1e9);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < rows; c++) mask[r, c] = blocked;
        for (int s = 0; s < segments; s++)
        {
            int start = Math.Max(s * segment - _o.LeftContextLength, 0), end = Math.Min((s + 1) * segment, length);
            void Open(int row)
            {
                for (int c = s * rc; c < (s + 1) * rc; c++) mask[row, c] = NumOps.Zero;
                for (int c = start; c < end; c++) mask[row, segments * rc + c] = NumOps.Zero;
            }
            for (int i = 0; i < rc; i++) Open(s * rc + i);
            for (int t = s * segment; t < end; t++) Open(segments * rc + t);
        }
        return mask;
    }

    /// <summary>The prediction network's output <c>[U + 1, EncodingDim]</c> for the label history: the blank, then each label.</summary>
    public Tensor<T> Predict(IReadOnlyList<int> labels)
    {
        int n = labels.Count + 1, s = _o.SymbolEmbeddingDim;
        var history = new Tensor<T>(new[] { 1, n });
        for (int u = 0; u < labels.Count; u++) history[0, u + 1] = NumOps.FromDouble(labels[u]);
        var x = PredictorInputNorm.Forward(_engine.Reshape(Embedding.Forward(history), new[] { n, s }));
        foreach (var lstm in Lstms)
        {
            var gatesFromInput = lstm.InputGates.Forward(x);
            var h = new Tensor<T>(new[] { 1, s });
            var c = new Tensor<T>(new[] { 1, s });
            var outputs = new Tensor<T>[n];
            for (int t = 0; t < n; t++)
            {
                var gates = lstm.GateNorm.Forward(_engine.TensorAdd(_engine.TensorSlice(gatesFromInput, new[] { t, 0 }, new[] { 1, 4 * s }), lstm.RecurrentGates.Forward(h)));
                Tensor<T> Gate(int g) => _engine.TensorSlice(gates, new[] { 0, g * s }, new[] { 1, s });
                c = lstm.CellNorm.Forward(_engine.TensorAdd(_engine.TensorMultiply(_engine.Sigmoid(Gate(1)), c), _engine.TensorMultiply(_engine.Sigmoid(Gate(0)), _engine.Tanh(Gate(2)))));
                h = _engine.TensorMultiply(_engine.Sigmoid(Gate(3)), _engine.Tanh(c));
                outputs[t] = h;
            }
            x = _engine.TensorConcatenate(outputs, 0);
        }
        return PredictorNorm.Forward(PredictorLinear.Forward(x));
    }

    /// <summary>The joiner's log-probabilities <c>[T, U + 1, VocabSize]</c>: log softmax(W relu(f_t + g_u)).</summary>
    public Tensor<T> Joint(Tensor<T> encoded, Tensor<T> predicted)
    {
        int frames = encoded.Shape[0], width = predicted.Shape[0], d = _o.EncodingDim;
        var sum = _engine.TensorAdd(_engine.TensorBroadcastTo(_engine.Reshape(encoded, new[] { frames, 1, d }), new[] { frames, width, d }),
            _engine.TensorBroadcastTo(_engine.Reshape(predicted, new[] { 1, width, d }), new[] { frames, width, d }));
        var logits = Joiner.Forward(_engine.Reshape(_engine.ReLU(sum), new[] { frames * width, d }));
        return _engine.TensorLogSoftmax(_engine.Reshape(logits, new[] { frames, width, _o.VocabSize }), axis: -1);
    }

    /// <summary>The transducer loss of a label sequence.</summary>
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
                var last = _engine.TensorSlice(g, new[] { labels.Count, 0 }, new[] { 1, _o.EncodingDim });
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
