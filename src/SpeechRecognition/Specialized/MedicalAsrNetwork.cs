using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>The shape of a <see cref="MedicalAsrNetwork{T}"/>.</summary>
internal sealed record MedicalAsrConfiguration
{
    public MedicalAsrDecoderType DecoderType { get; init; } = MedicalAsrDecoderType.ListenAttendSpell;
    public int SampleRate { get; init; } = 16000;
    /// <summary>Filterbank channels per 25 ms frame (section 4: 80).</summary>
    public int NumFilterbanks { get; init; } = 80;
    /// <summary>The highest frequency the filterbanks cover (section 4: frequencies above 3800 Hz are ignored).</summary>
    public double MaxFrequency { get; init; } = 3800;
    /// <summary>Frames stacked into one input step, which is also the stride (section 4: 3, so 240 values every 30 ms).</summary>
    public int StackedFrames { get; init; } = 3;
    /// <summary>Encoder LSTM layers.</summary>
    public int EncoderLayers { get; init; } = 6;
    /// <summary>LSTM units per direction (or in total when unidirectional) of each encoder layer.</summary>
    public int EncoderUnits { get; init; } = 768;
    /// <summary>Whether the encoder LSTMs are bidirectional.</summary>
    public bool Bidirectional { get; init; } = true;
    /// <summary>The unidirectional CTC model's output delay in steps (section 5.1: 10).</summary>
    public int OutputDelay { get; init; }
    /// <summary>The LAS decoder's LSTM layers (section 5.2: 3).</summary>
    public int DecoderLayers { get; init; } = 3;
    /// <summary>The LAS decoder's LSTM units (section 5.2: 768); also its grapheme embedding width.</summary>
    public int DecoderUnits { get; init; } = 768;
    /// <summary>Heads of the LAS multi-headed attention.</summary>
    public int AttentionHeads { get; init; } = 4;
    /// <summary>Output units: CD phones plus the blank (index 0) for CTC; graphemes with the boundary (index 0) for LAS.</summary>
    public int VocabSize { get; init; } = 128;
}

/// <summary>
/// Chiu et al. 2018's two recognizers for medical conversations (section 4): a CTC model, a stack of LSTM layers under a
/// softmax over context-dependent phones and the blank, and a LAS model, a bidirectional LSTM encoder with a
/// multi-headed attention and a unidirectional LSTM decoder predicting graphemes. Both read 80 filterbanks up to
/// 3800 Hz, stacked three frames at a time with stride three.
/// </summary>
internal sealed class MedicalAsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly MelSpectrogram<T> _mel;
    private readonly MedicalAsrConfiguration _o;

    public MedicalAsrNetwork(IEngine engine, List<LayerBase<T>> layers, MedicalAsrConfiguration configuration)
    {
        _engine = engine;
        _o = configuration;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        // 25 ms windows every 10 ms at 16 kHz.
        int window = _o.SampleRate / 40, hop = _o.SampleRate / 100;
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.NumFilterbanks, nFft: window, hopLength: hop,
            fMax: Math.Min(_o.MaxFrequency, _o.SampleRate / 2.0), logMel: true, logOffset: 1e-6);
        WindowSamples = window;

        for (int i = 0; i < _o.EncoderLayers; i++)
            EncoderLayers.Add((Own(new LSTMLayer<T>(_o.EncoderUnits)), _o.Bidirectional ? Own(new LSTMLayer<T>(_o.EncoderUnits)) : null));

        if (_o.DecoderType == MedicalAsrDecoderType.Ctc)
        {
            CtcHead = Own(new DenseLayer<T>(_o.VocabSize, identity));
        }
        else
        {
            int d = _o.DecoderUnits;
            Embedding = Own(new EmbeddingLayer<T>(_o.VocabSize, d));
            for (int i = 0; i < Math.Max(1, _o.DecoderLayers); i++) DecoderCells.Add(Own(new DenseLayer<T>(4 * d, identity)));
            Query = Own(new DenseLayer<T>(d, identity));
            Key = Own(new DenseLayer<T>(d, identity));
            Value = Own(new DenseLayer<T>(d, identity));
            AttentionOutput = Own(new DenseLayer<T>(d, identity));
            Output = Own(new DenseLayer<T>(_o.VocabSize, identity));
        }

        // Every layer sizes itself on its first input, and the decoder is reached only by the loss and decoding. One
        // forward of the training path materializes all of them, so the parameter vector has one size from construction
        // on (a later size change would rebind layers to the wrong widths on SetParameters and Clone).
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            Loss(Features(new Tensor<T>(new[] { WindowSamples })), new[] { 1 });
    }

    /// <summary>The samples of one analysis window; shorter audio is zero-padded to it.</summary>
    public int WindowSamples { get; }

    public List<(LSTMLayer<T> Forward, LSTMLayer<T>? Backward)> EncoderLayers { get; } = new();
    public DenseLayer<T>? CtcHead { get; }
    public EmbeddingLayer<T>? Embedding { get; }
    /// <summary>The decoder LSTMs' gates: per layer one linear layer over [input; state] to i, f, g, o (PyTorch's order).</summary>
    public List<DenseLayer<T>> DecoderCells { get; } = new();
    public DenseLayer<T>? Query { get; }
    public DenseLayer<T>? Key { get; }
    public DenseLayer<T>? Value { get; }
    public DenseLayer<T>? AttentionOutput { get; }
    public DenseLayer<T>? Output { get; }

    public MedicalAsrConfiguration Configuration => _o;

    /// <summary>The LAS boundary token that starts and ends a transcript.</summary>
    public const int Boundary = 0;

    /// <summary>
    /// The model input <c>[steps, StackedFrames * NumFilterbanks]</c>: per-channel normalized log filterbanks, each step
    /// the concatenation of <see cref="MedicalAsrConfiguration.StackedFrames"/> frames, advancing by as many.
    /// </summary>
    public Tensor<T> Features(Tensor<T> waveform)
    {
        if (waveform.Length < WindowSamples)
            waveform = _engine.TensorConcatenate(new[] { _engine.Reshape(waveform, new[] { waveform.Length }), new Tensor<T>(new[] { WindowSamples - waveform.Length }) }, 0);
        var mel = _mel.Forward(_engine.Reshape(waveform, new[] { waveform.Length }));
        int frames = mel.Shape[0], bins = mel.Shape[1], stack = Math.Max(1, _o.StackedFrames);
        var normalized = new double[frames, bins];
        for (int b = 0; b < bins; b++)
        {
            double mean = 0, square = 0;
            for (int t = 0; t < frames; t++) mean += NumOps.ToDouble(mel[t, b]);
            mean /= frames;
            for (int t = 0; t < frames; t++) { double v = NumOps.ToDouble(mel[t, b]) - mean; square += v * v; }
            double std = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++) normalized[t, b] = std > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - mean) / std : 0.0;
        }
        // A trailing partial stack repeats the last frame.
        int steps = (frames + stack - 1) / stack;
        var features = new Tensor<T>(new[] { steps, stack * bins });
        for (int s = 0; s < steps; s++)
            for (int k = 0; k < stack; k++)
            {
                int t = Math.Min(s * stack + k, frames - 1);
                for (int b = 0; b < bins; b++) features[s, k * bins + b] = NumOps.FromDouble(normalized[t, b]);
            }
        return features;
    }

    /// <summary>The encoder output <c>[steps, units]</c>. The unidirectional model with an output delay of D reads its
    /// input with the last step repeated D times and drops its first D outputs (section 4.1).</summary>
    public Tensor<T> Encode(Tensor<T> features)
    {
        int steps = features.Shape[0], delay = _o.Bidirectional ? 0 : Math.Max(0, _o.OutputDelay);
        var h = features;
        if (delay > 0)
        {
            var last = _engine.TensorSlice(features, new[] { steps - 1, 0 }, new[] { 1, features.Shape[1] });
            h = _engine.TensorConcatenate(new[] { features }.Concat(Enumerable.Repeat(last, delay)).ToArray(), 0);
        }
        foreach (var (forward, backward) in EncoderLayers)
        {
            var f = forward.Forward(h);
            h = backward is null ? f
                : _engine.TensorConcatenate(new[] { f, _engine.TensorFlip(backward.Forward(_engine.TensorFlip(h, new[] { 0 })), new[] { 0 }) }, 1);
        }
        return delay > 0 ? _engine.TensorSlice(h, new[] { delay, 0 }, new[] { steps, h.Shape[1] }) : h;
    }

    /// <summary>The CTC arm's logits <c>[steps, VocabSize]</c>.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded) =>
        (CtcHead ?? throw new InvalidOperationException("Only the CTC model has a CTC head.")).Forward(encoded);

    private (Tensor<T> H, Tensor<T> C) Step(DenseLayer<T> cell, Tensor<T> input, Tensor<T> h, Tensor<T> c)
    {
        int u = _o.DecoderUnits;
        var gates = cell.Forward(_engine.TensorConcatenate(new[] { input, h }, 1));
        var i = _engine.Sigmoid(_engine.TensorSlice(gates, new[] { 0, 0 }, new[] { 1, u }));
        var f = _engine.Sigmoid(_engine.TensorSlice(gates, new[] { 0, u }, new[] { 1, u }));
        var g = _engine.Tanh(_engine.TensorSlice(gates, new[] { 0, 2 * u }, new[] { 1, u }));
        var o = _engine.Sigmoid(_engine.TensorSlice(gates, new[] { 0, 3 * u }, new[] { 1, u }));
        c = _engine.TensorAdd(_engine.TensorMultiply(f, c), _engine.TensorMultiply(i, g));
        return (_engine.TensorMultiply(o, _engine.Tanh(c)), c);
    }

    /// <summary>
    /// Multi-headed attention (Vaswani et al. 2017, the paper's [14]) of the decoder state over the encoder output: each
    /// head scores its slice of the projected keys against its slice of the projected query, and the heads' contexts are
    /// concatenated and projected. Returns the context <c>[1, DecoderUnits]</c> and the weights <c>[heads, steps]</c>.
    /// </summary>
    public (Tensor<T> Context, Tensor<T> Weights) Attend(Tensor<T> keys, Tensor<T> values, Tensor<T> state)
    {
        var query = (Query ?? throw new InvalidOperationException("Only the LAS model attends.")).Forward(state);
        int heads = Math.Max(1, _o.AttentionHeads), d = _o.DecoderUnits, dh = d / heads, steps = keys.Shape[0];
        var q = _engine.Reshape(query, new[] { heads, 1, dh });
        var k = _engine.TensorPermute(_engine.Reshape(keys, new[] { steps, heads, dh }), new[] { 1, 2, 0 });
        var v = _engine.TensorPermute(_engine.Reshape(values, new[] { steps, heads, dh }), new[] { 1, 0, 2 });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, k), NumOps.FromDouble(1.0 / Math.Sqrt(dh)));
        var weights = _engine.Softmax(scores, -1);
        var context = _engine.Reshape(_engine.BatchMatMul(weights, v), new[] { 1, heads * dh });
        var output = AttentionOutput ?? throw new InvalidOperationException("Only the LAS model attends.");
        return (output.Forward(context), _engine.Reshape(weights, new[] { heads, steps }));
    }

    private Tensor<T> Embed(int id)
    {
        var ids = new Tensor<T>(new[] { 1, 1 });
        ids[0, 0] = NumOps.FromDouble(id);
        var embedding = Embedding ?? throw new InvalidOperationException("Only the LAS model embeds graphemes.");
        return _engine.Reshape(embedding.Forward(ids), new[] { 1, _o.DecoderUnits });
    }

    /// <summary>
    /// The LAS decoder's logits <c>[n + 1, VocabSize]</c> over inputs [boundary, y1..yn]. With a positive
    /// <paramref name="samplingProbability"/> each previous grapheme is, with that probability, the decoder's own
    /// prediction instead of the ground truth (scheduled sampling, section 4.2).
    /// </summary>
    public Tensor<T> DecoderLogits(Tensor<T> encoded, IReadOnlyList<int> labels, double samplingProbability = 0, Random? random = null)
    {
        var state = new DecoderState(this, encoded);
        var rows = new List<Tensor<T>>();
        int previous = Boundary;
        for (int i = 0; i <= labels.Count; i++)
        {
            if (i > 0)
            {
                previous = labels[i - 1];
                if (samplingProbability > 0 && random is not null && random.NextDouble() < samplingProbability)
                    previous = ArgMax(rows[i - 1]);
            }
            rows.Add(state.Next(previous));
        }
        return _engine.TensorConcatenate(rows.ToArray(), 0);
    }

    private int ArgMax(Tensor<T> row)
    {
        int best = 0;
        for (int k = 1; k < _o.VocabSize; k++) if (NumOps.GreaterThan(row[0, k], row[0, best])) best = k;
        return best;
    }

    /// <summary>The loss of one utterance, summed over its outputs: CTC for the CTC model, the cross-entropy of the
    /// graphemes and the closing boundary for LAS.</summary>
    public Tensor<T> Loss(Tensor<T> features, IReadOnlyList<int> labels, double samplingProbability = 0, Random? random = null)
    {
        var encoded = Encode(features);
        if (_o.DecoderType == MedicalAsrDecoderType.Ctc)
        {
            var logits = CtcLogits(encoded);
            int steps = logits.Shape[0];
            var loss = _engine.TensorCTCLoss(_engine.Reshape(_engine.TensorLogSoftmax(logits, axis: -1), new[] { steps, 1, _o.VocabSize }),
                new Tensor<int>(labels.ToArray(), new[] { labels.Count }), new[] { steps }, new[] { labels.Count }, 0);
            return _engine.Reshape(loss, new[] { 1 });
        }
        var tokens = DecoderLogits(encoded, labels, samplingProbability, random);
        var targets = labels.Concat(new[] { Boundary }).ToArray();
        int n = tokens.Shape[0];
        var weights = new Tensor<T>(new[] { n, _o.VocabSize });
        for (int i = 0; i < n; i++) weights[i, targets[i]] = NumOps.FromDouble(-1.0);
        return _engine.Reshape(_engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(tokens, axis: -1), weights), new[] { 0, 1 }, keepDims: false), new[] { 1 });
    }

    /// <summary>Greedy LAS decoding for <paramref name="steps"/> steps, each fed the previous step's best grapheme: the
    /// logits <c>[steps, VocabSize]</c> and the graphemes before the first boundary.</summary>
    public (Tensor<T> Logits, List<int> Tokens) Greedy(Tensor<T> encoded, int steps)
    {
        var state = new DecoderState(this, encoded);
        var rows = new List<Tensor<T>>();
        var tokens = new List<int>();
        int previous = Boundary;
        bool ended = false;
        for (int i = 0; i < Math.Max(1, steps); i++)
        {
            var row = state.Next(previous);
            rows.Add(row);
            previous = ArgMax(row);
            if (previous == Boundary) ended = true;
            if (!ended) tokens.Add(previous);
        }
        return (_engine.TensorConcatenate(rows.ToArray(), 0), tokens);
    }

    // The LAS decoder's recurrent state over one utterance: s_i = LSTM(s_{i-1}, [y_{i-1}; c_{i-1}]), c_i = Attend(s_i),
    // and the output distribution over [s_i; c_i] (Chan et al. 2016, the paper's [7]).
    private sealed class DecoderState
    {
        private readonly MedicalAsrNetwork<T> _n;
        private readonly Tensor<T> _keys, _values;
        private readonly Tensor<T>[] _h, _c;
        private Tensor<T> _context;

        public DecoderState(MedicalAsrNetwork<T> network, Tensor<T> encoded)
        {
            _n = network;
            _keys = (network.Key ?? throw new InvalidOperationException("Only the LAS model decodes.")).Forward(encoded);
            _values = (network.Value ?? throw new InvalidOperationException("Only the LAS model decodes.")).Forward(encoded);
            int u = network._o.DecoderUnits, layers = network.DecoderCells.Count;
            _h = new Tensor<T>[layers]; _c = new Tensor<T>[layers];
            for (int l = 0; l < layers; l++) { _h[l] = new Tensor<T>(new[] { 1, u }); _c[l] = new Tensor<T>(new[] { 1, u }); }
            _context = new Tensor<T>(new[] { 1, u });
        }

        public Tensor<T> Next(int previous)
        {
            var e = _n._engine;
            var input = e.TensorConcatenate(new[] { _n.Embed(previous), _context }, 1);
            for (int l = 0; l < _h.Length; l++)
            {
                (_h[l], _c[l]) = _n.Step(_n.DecoderCells[l], input, _h[l], _c[l]);
                input = _h[l];
            }
            (_context, _) = _n.Attend(_keys, _values, _h[_h.Length - 1]);
            var output = _n.Output ?? throw new InvalidOperationException("Only the LAS model decodes.");
            return output.Forward(e.TensorConcatenate(new[] { _h[_h.Length - 1], _context }, 1));
        }
    }
}
