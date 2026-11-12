using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>The shape of a <see cref="CodeSwitchingAsrNetwork{T}"/>.</summary>
internal sealed record CodeSwitchingAsrConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int NumMels { get; init; } = 80;
    /// <summary>LSTM units per direction of each encoder BLSTM layer (section 4.3: 320).</summary>
    public int EncoderUnits { get; init; } = 320;
    /// <summary>Encoder BLSTM layers (section 4.3: 6).</summary>
    public int EncoderLayers { get; init; } = 6;
    /// <summary>Width of the encoder output projection (ESPnet's <c>eprojs</c>).</summary>
    public int EncoderProjection { get; init; } = 320;
    /// <summary>LSTM units of the decoder (section 4.3: 320); also its token embedding width.</summary>
    public int DecoderUnits { get; init; } = 320;
    /// <summary>Stacked LSTM layers of the decoder (section 4.3: 1).</summary>
    public int DecoderLayers { get; init; } = 1;
    /// <summary>Dropout on encoder layer outputs and decoder embeddings in training (ESPnet's <c>dropout-rate</c>).</summary>
    public double DropoutRate { get; init; }
    /// <summary>Width of the location-aware attention (ESPnet's <c>adim</c>).</summary>
    public int AttentionDim { get; init; } = 320;
    /// <summary>Channels of the attention's location convolution (ESPnet's <c>aconv_chans</c>).</summary>
    public int AttentionConvChannels { get; init; } = 10;
    /// <summary>Half-width of the location convolution's kernel (ESPnet's <c>aconv_filts</c>).</summary>
    public int AttentionConvFilters { get; init; } = 100;
    /// <summary>Output vocabulary: the blank (0), Mandarin characters, English BPE units, then the boundary token.</summary>
    public int VocabSize { get; init; } = 6002;
    /// <summary>The first English token; ids from 1 below it are Mandarin.</summary>
    public int EnglishTokenOffset { get; init; } = 3001;
    /// <summary>The CTC weight 1 − λ1 of Equation 2 (section 4.3: 0.2).</summary>
    public double CtcWeight { get; init; } = 0.2;
    /// <summary>λ2 of Equation 2, the language-identification weight.</summary>
    public double LidWeight { get; init; } = 0.2;
    /// <summary>Whether the LID decoder reads the speech decoder's attention (LID_shared) or has its own (LID_indep).</summary>
    public bool SharedLidAttention { get; init; } = true;
}

/// <summary>
/// Zeng et al. 2019's code-switching recognizer (section 4.3): ESPnet's hybrid CTC/attention RNN model with a CNN front
/// end, a BLSTM encoder, a location-aware attention LSTM decoder, a CTC layer, and a language-identification decoder
/// that emits the language of each output token (Figure 1).
/// </summary>
internal sealed class CodeSwitchingAsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly MelSpectrogram<T> _mel;
    private readonly CodeSwitchingAsrConfiguration _o;

    /// <summary>The two languages, in LID label order.</summary>
    public static readonly string[] Languages = { "zh", "en" };

    public CodeSwitchingAsrNetwork(IEngine engine, List<LayerBase<T>> layers, CodeSwitchingAsrConfiguration configuration)
    {
        _engine = engine;
        _o = configuration;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.NumMels, nFft: 400, hopLength: 160, logMel: true, logOffset: 1e-6);

        // ESPnet's VGG2L front end: two blocks of two 3x3 convolutions with ReLU and 2x2 max pooling (64 then 128
        // channels), 4x in time and frequency.
        int inChannels = 1;
        foreach (int c in new[] { 64, 128 })
            for (int k = 0; k < 2; k++) { Convs.Add(Own(new Conv2DLayer<T>(inChannels, c, 3, 3, 1, 1, 1, 1))); inChannels = c; }
        _frontEndWidth = 128 * ((_o.NumMels + 3) / 4);

        for (int i = 0; i < _o.EncoderLayers; i++)
            EncoderLayers.Add((Own(new LSTMLayer<T>(_o.EncoderUnits)), Own(new LSTMLayer<T>(_o.EncoderUnits))));
        EncoderProjection = Own(new DenseLayer<T>(_o.EncoderProjection, (IActivationFunction<T>)new TanhActivation<T>()));
        CtcHead = Own(new DenseLayer<T>(_o.VocabSize, identity));

        Attention = new LocationAttention(engine, _o, Own);
        TokenEmbedding = Own(new EmbeddingLayer<T>(_o.VocabSize, _o.DecoderUnits));
        for (int i = 0; i < Math.Max(1, _o.DecoderLayers); i++) DecoderCells.Add(Own(new DenseLayer<T>(4 * _o.DecoderUnits, identity)));
        DecoderOutput = Own(new DenseLayer<T>(_o.VocabSize, identity));

        // The LID decoder: its own embedding of the previous language label (the languages, then the boundary) and LSTM,
        // over the speech decoder's attention contexts (LID_shared) or its own attention (LID_indep).
        LidAttention = _o.SharedLidAttention ? null : new LocationAttention(engine, _o, Own);
        LidEmbedding = Own(new EmbeddingLayer<T>(Languages.Length + 1, _o.DecoderUnits));
        LidCell = Own(new DenseLayer<T>(4 * _o.DecoderUnits, identity));
        LidOutput = Own(new DenseLayer<T>(Languages.Length + 1, identity));

        // Every layer sizes itself on its first input, and Predict reaches only the encoder and the CTC head. One
        // forward of the whole training path materializes the decoders too, so the parameter vector has one size from
        // construction on (a later size change would rebind layers to the wrong widths on SetParameters and Clone).
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            Loss(Features(new Tensor<T>(new[] { 400 })), new[] { 1 });
    }

    private readonly int _frontEndWidth;

    public List<Conv2DLayer<T>> Convs { get; } = new();
    public List<(LSTMLayer<T> Forward, LSTMLayer<T> Backward)> EncoderLayers { get; } = new();
    public DenseLayer<T> EncoderProjection { get; }
    public DenseLayer<T> CtcHead { get; }
    public LocationAttention Attention { get; }
    public EmbeddingLayer<T> TokenEmbedding { get; }
    /// <summary>The decoder LSTMs' gates: per layer one linear layer over [input; state] to i, f, g, o (PyTorch's order).</summary>
    public List<DenseLayer<T>> DecoderCells { get; } = new();
    public DenseLayer<T> DecoderOutput { get; }
    public LocationAttention? LidAttention { get; }
    public EmbeddingLayer<T> LidEmbedding { get; }
    public DenseLayer<T> LidCell { get; }
    public DenseLayer<T> LidOutput { get; }

    public CodeSwitchingAsrConfiguration Configuration => _o;

    /// <summary>The boundary token that starts and ends a transcript (the vocabulary's last).</summary>
    public int Boundary => _o.VocabSize - 1;

    /// <summary>The LID boundary label (after the languages).</summary>
    public int LidBoundary => Languages.Length;

    /// <summary>The LID label of a token: 0 for Mandarin, 1 for English.</summary>
    public int LanguageLabel(int token) => token < _o.EnglishTokenOffset ? 0 : 1;

    /// <summary>Log-mel features <c>[frames, bins]</c>, each bin normalized over the utterance.</summary>
    public Tensor<T> Features(Tensor<T> waveform)
    {
        // Audio shorter than one 25 ms window is zero-padded to it, so it still gives a frame.
        if (waveform.Length < 400) waveform = _engine.TensorConcatenate(new[] { _engine.Reshape(waveform, new[] { waveform.Length }), new Tensor<T>(new[] { 400 - waveform.Length }) }, 0);
        var mel = _mel.Forward(waveform);
        int frames = mel.Shape[0], bins = mel.Shape[1];
        var features = new Tensor<T>(new[] { frames, bins });
        for (int b = 0; b < bins; b++)
        {
            double mean = 0, square = 0;
            for (int t = 0; t < frames; t++) mean += NumOps.ToDouble(mel[t, b]);
            mean /= frames;
            for (int t = 0; t < frames; t++) { double v = NumOps.ToDouble(mel[t, b]) - mean; square += v * v; }
            double std = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++) features[t, b] = NumOps.FromDouble(std > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - mean) / std : 0.0);
        }
        return features;
    }

    /// <summary>The encoder output <c>[L, EncoderProjection]</c> of features <c>[frames, bins]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> features, bool training = false, Random? random = null)
    {
        int frames = features.Shape[0], bins = features.Shape[1];
        var x = _engine.Reshape(features, new[] { 1, 1, frames, bins });
        for (int i = 0; i < Convs.Count; i++)
        {
            x = _engine.ReLU(Convs[i].Forward(x));
            if (i % 2 == 1)
            {
                // ceil_mode pooling, as in VGG2L: an odd edge is zero-padded, which after ReLU leaves each maximum as is.
                int padTime = x.Shape[2] % 2, padFrequency = x.Shape[3] % 2;
                if (padTime + padFrequency > 0) x = _engine.TensorConstantPad(x, new[] { 0, padFrequency, 0, padTime }, NumOps.Zero);
                x = _engine.MaxPool2D(x, 2, 2);
            }
        }
        int length = x.Shape[2];
        if (length <= 0) throw new ArgumentException("The audio is too short for the convolution front end.");
        var h = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { length, _frontEndWidth });
        foreach (var (forward, backward) in EncoderLayers)
        {
            var f = forward.Forward(h);
            var b = _engine.TensorFlip(backward.Forward(_engine.TensorFlip(h, new[] { 0 })), new[] { 0 });
            h = Dropout(_engine.TensorConcatenate(new[] { f, b }, 1), training, random);
        }
        return EncoderProjection.Forward(h);
    }

    /// <summary>The CTC logits <c>[L, VocabSize]</c>.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded) => CtcHead.Forward(encoded);

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

    private Tensor<T> Dropout(Tensor<T> x, bool training, Random? random)
    {
        if (!training || random is null || _o.DropoutRate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - _o.DropoutRate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < _o.DropoutRate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    private Tensor<T> Embed(EmbeddingLayer<T> embedding, int id)
    {
        var ids = new Tensor<T>(new[] { 1, 1 });
        ids[0, 0] = NumOps.FromDouble(id);
        return _engine.Reshape(embedding.Forward(ids), new[] { 1, _o.DecoderUnits });
    }

    /// <summary>
    /// Teacher-forced decoding: the speech decoder's logits <c>[n + 1, VocabSize]</c> for inputs [boundary, y1..yn] and
    /// the LID decoder's logits <c>[n + 1, languages + 1]</c> for inputs [boundary, z1..zn].
    /// </summary>
    public (Tensor<T> Tokens, Tensor<T> Languages) DecoderLogits(Tensor<T> encoded, IReadOnlyList<int> labels, bool training = false, Random? random = null)
    {
        var tokenRows = new List<Tensor<T>>();
        var lidRows = new List<Tensor<T>>();
        var state = new DecoderState(this, encoded, training, random);
        for (int i = 0; i <= labels.Count; i++)
        {
            int previous = i == 0 ? Boundary : labels[i - 1];
            int previousLanguage = i == 0 ? LidBoundary : LanguageLabel(labels[i - 1]);
            var (token, language) = state.Next(previous, previousLanguage);
            tokenRows.Add(token);
            lidRows.Add(language);
        }
        return (_engine.TensorConcatenate(tokenRows.ToArray(), 0), _engine.TensorConcatenate(lidRows.ToArray(), 0));
    }

    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var weights = new Tensor<T>(new[] { n, v });
        for (int i = 0; i < n; i++) weights[i, targets[i]] = NumOps.FromDouble(-1.0);
        return _engine.Reshape(_engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, axis: -1), weights), new[] { 0, 1 }, keepDims: false), new[] { 1 });
    }

    /// <summary>The CTC loss of labels (summed over the utterance).</summary>
    public Tensor<T> CtcLoss(Tensor<T> encoded, IReadOnlyList<int> labels)
    {
        var logits = CtcLogits(encoded);
        int frames = logits.Shape[0];
        var loss = _engine.TensorCTCLoss(_engine.Reshape(_engine.TensorLogSoftmax(logits, axis: -1), new[] { frames, 1, _o.VocabSize }),
            new Tensor<int>(labels.ToArray(), new[] { labels.Count }), new[] { frames }, new[] { labels.Count }, 0);
        return _engine.Reshape(loss, new[] { 1 });
    }

    /// <summary>The three losses of Equation 2 for one utterance, each summed over its tokens, and their mix
    /// L = λ1 L_att + (1 − λ1) L_ctc + λ2 L_lid.</summary>
    public (Tensor<T> Total, Tensor<T> Attention, Tensor<T> Ctc, Tensor<T> Lid) Loss(Tensor<T> features, IReadOnlyList<int> labels, bool training = false,
        Random? random = null)
    {
        var encoded = Encode(features, training, random);
        var (tokens, languages) = DecoderLogits(encoded, labels, training, random);
        var tokenTargets = labels.Concat(new[] { Boundary }).ToArray();
        var lidTargets = labels.Select(LanguageLabel).Concat(new[] { LidBoundary }).ToArray();
        var attention = CrossEntropy(tokens, tokenTargets);
        var lid = CrossEntropy(languages, lidTargets);
        var ctc = CtcLoss(encoded, labels);
        var total = _engine.TensorAdd(_engine.TensorAdd(
            _engine.TensorMultiplyScalar(attention, NumOps.FromDouble(1 - _o.CtcWeight)),
            _engine.TensorMultiplyScalar(ctc, NumOps.FromDouble(_o.CtcWeight))),
            _engine.TensorMultiplyScalar(lid, NumOps.FromDouble(_o.LidWeight)));
        return (total, attention, ctc, lid);
    }

    /// <summary>Greedy attention decoding: the tokens until the boundary or <paramref name="maxLength"/>, and the LID
    /// decoder's probabilities over the languages and the boundary at every step taken, the last included.</summary>
    public (List<int> Tokens, List<double[]> Languages) Greedy(Tensor<T> encoded, int maxLength)
    {
        var tokens = new List<int>();
        var languages = new List<double[]>();
        var state = new DecoderState(this, encoded, training: false, random: null);
        int previous = Boundary, previousLanguage = LidBoundary;
        for (int step = 0; step < maxLength; step++)
        {
            var (token, language) = state.Next(previous, previousLanguage);
            int best = 0;
            for (int k = 1; k < _o.VocabSize; k++) if (NumOps.GreaterThan(token[0, k], token[0, best])) best = k;
            var probabilities = new double[Languages.Length + 1];
            double max = double.NegativeInfinity, sum = 0;
            for (int k = 0; k < probabilities.Length; k++) max = Math.Max(max, NumOps.ToDouble(language[0, k]));
            for (int k = 0; k < probabilities.Length; k++) { probabilities[k] = Math.Exp(NumOps.ToDouble(language[0, k]) - max); sum += probabilities[k]; }
            for (int k = 0; k < probabilities.Length; k++) probabilities[k] /= sum;
            languages.Add(probabilities);
            if (best == Boundary) break;
            tokens.Add(best);
            previous = best;
            previousLanguage = LanguageLabel(best);
        }
        return (tokens, languages);
    }

    // The two decoders' recurrent state over one utterance.
    private sealed class DecoderState
    {
        private readonly CodeSwitchingAsrNetwork<T> _n;
        private readonly Tensor<T> _encoded;
        private readonly Tensor<T> _projected;
        private readonly Tensor<T>? _lidProjected;
        private readonly bool _training;
        private readonly Random? _random;
        private readonly Tensor<T>[] _h, _c;
        private Tensor<T> _weights, _lidH, _lidC, _lidWeights;

        public DecoderState(CodeSwitchingAsrNetwork<T> network, Tensor<T> encoded, bool training, Random? random)
        {
            _n = network;
            _encoded = encoded;
            _training = training;
            _random = random;
            int length = encoded.Shape[0], u = network._o.DecoderUnits, layers = network.DecoderCells.Count;
            _projected = network.Attention.Project(encoded);
            _lidProjected = network.LidAttention?.Project(encoded);
            _h = new Tensor<T>[layers]; _c = new Tensor<T>[layers];
            for (int l = 0; l < layers; l++) { _h[l] = new Tensor<T>(new[] { 1, u }); _c[l] = new Tensor<T>(new[] { 1, u }); }
            _lidH = new Tensor<T>(new[] { 1, u }); _lidC = new Tensor<T>(new[] { 1, u });
            // ESPnet starts the attention weights uniform.
            _weights = new Tensor<T>(new[] { length });
            for (int t = 0; t < length; t++) _weights[t] = NumOps.FromDouble(1.0 / length);
            _lidWeights = _weights;
        }

        // One step: attention from the previous state, then the LSTM over [embedding; context] (ESPnet's Decoder).
        public (Tensor<T> Token, Tensor<T> Language) Next(int previous, int previousLanguage)
        {
            var e = _n._engine;
            Tensor<T> context;
            int top = _h.Length - 1;
            (context, _weights) = _n.Attention.Attend(_encoded, _projected, _h[top], _weights);
            var input = e.TensorConcatenate(new[] { _n.Dropout(_n.Embed(_n.TokenEmbedding, previous), _training, _random), context }, 1);
            for (int l = 0; l <= top; l++)
            {
                (_h[l], _c[l]) = _n.Step(_n.DecoderCells[l], input, _h[l], _c[l]);
                input = l < top ? _n.Dropout(_h[l], _training, _random) : _h[l];
            }
            var lidContext = context;
            if (_n.LidAttention is not null && _lidProjected is not null)
                (lidContext, _lidWeights) = _n.LidAttention.Attend(_encoded, _lidProjected, _lidH, _lidWeights);
            (_lidH, _lidC) = _n.Step(_n.LidCell, e.TensorConcatenate(new[] { _n.Embed(_n.LidEmbedding, previousLanguage), lidContext }, 1), _lidH, _lidC);
            return (_n.DecoderOutput.Forward(_h[top]), _n.LidOutput.Forward(_lidH));
        }
    }

    /// <summary>Location-aware attention (Chorowski et al. 2015; ESPnet's <c>AttLoc</c>): e = w^T tanh(W h + V s + U f),
    /// f the previous weights convolved, and weights softmax(2 e).</summary>
    internal sealed class LocationAttention
    {
        private readonly IEngine _engine;
        private readonly CodeSwitchingAsrConfiguration _o;

        public LocationAttention(IEngine engine, CodeSwitchingAsrConfiguration o, Func<LayerBase<T>, LayerBase<T>> own)
        {
            _o = o;
            _engine = engine;
            var identity = (IActivationFunction<T>)new IdentityActivation<T>();
            EncoderProjection = (DenseLayer<T>)own(new DenseLayer<T>(o.AttentionDim, identity));
            DecoderProjection = (BiasFreeLinearLayer<T>)own(new BiasFreeLinearLayer<T>(o.DecoderUnits, o.AttentionDim));
            LocationConv = (Conv1DLayer<T>)own(new Conv1DLayer<T>(inputChannels: 1, outputChannels: o.AttentionConvChannels,
                kernelSize: 2 * o.AttentionConvFilters + 1, dilation: 1, stride: 1, padding: o.AttentionConvFilters));
            LocationProjection = (BiasFreeLinearLayer<T>)own(new BiasFreeLinearLayer<T>(o.AttentionConvChannels, o.AttentionDim));
            Score = (DenseLayer<T>)own(new DenseLayer<T>(1, identity));
        }

        public DenseLayer<T> EncoderProjection { get; }
        public BiasFreeLinearLayer<T> DecoderProjection { get; }
        public Conv1DLayer<T> LocationConv { get; }
        public BiasFreeLinearLayer<T> LocationProjection { get; }
        public DenseLayer<T> Score { get; }

        public Tensor<T> Project(Tensor<T> encoded) => EncoderProjection.Forward(encoded);

        /// <summary>The context <c>[1, P]</c> and new weights <c>[L]</c> for decoder state <c>[1, U]</c>.</summary>
        public (Tensor<T> Context, Tensor<T> Weights) Attend(Tensor<T> encoded, Tensor<T> projected, Tensor<T> state, Tensor<T> previous)
        {
            int length = encoded.Shape[0], a = _o.AttentionDim;
            var location = LocationConv.Forward(_engine.Reshape(previous, new[] { 1, 1, length }));          // [1, C, L]
            var f = LocationProjection.Forward(_engine.Reshape(_engine.TensorPermute(location, new[] { 0, 2, 1 }), new[] { length, _o.AttentionConvChannels }));
            var s = _engine.TensorBroadcastTo(DecoderProjection.Forward(state), new[] { length, a });
            var e = Score.Forward(_engine.Tanh(_engine.TensorAdd(_engine.TensorAdd(projected, s), f)));     // [L, 1]
            var weights = _engine.Softmax(_engine.TensorMultiplyScalar(_engine.Reshape(e, new[] { 1, length }), NumOps.FromDouble(2.0)), axis: -1);
            return (_engine.TensorMatMul(weights, encoded), _engine.Reshape(weights, new[] { length }));
        }
    }
}
