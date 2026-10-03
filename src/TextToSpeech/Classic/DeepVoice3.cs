using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Deep Voice 3: fully convolutional attention-based text-to-speech.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning" (Ping et al.,
/// ICLR 2018), single-speaker configuration with the Griffin–Lim converter; unstated details follow the reference
/// implementation r9y9/deepvoice3_pytorch.</para>
/// <para>
/// The encoder (§3.4) projects character embeddings to the convolution width, applies non-causal gated convolution
/// blocks (<see cref="GatedConvolutionBlockLayer{T}"/>) and projects back to attention keys; values are
/// √0.5 (keys + embeddings). The decoder (§3.5) reads the previous group of r frames through a fully connected pre-net,
/// then alternates causal convolution blocks with attention blocks (§3.6: positional encodings with rates ω_query and
/// ω_key added to queries and keys, dot-product softmax attention, context normalization, √0.5 residuals), and predicts
/// the next r frames and a done flag. The converter (§3.7) maps the decoder's hidden states to the linear spectrogram.
/// Training minimizes L1 on mel, binary cross-entropy on done and L1 on the linear spectrogram, with gradients clipped
/// to norm 100 and value 5 (Table 4). Inference is autoregressive with attention restricted to a window around the
/// last attended position and stops on the done flag.
/// </para>
/// <para><b>For Beginners:</b> Deep Voice 3 replaces recurrent networks with stacks of convolutions, so it trains fast,
/// and uses attention to decide which letters it is speaking at each step.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning",
    "https://arxiv.org/abs/1710.07654",
    Year = 2018,
    Authors = "Ping et al."
)]
public partial class DeepVoice3<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly DeepVoice3Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Encoder (§3.4).
    private EmbeddingLayer<T>? _embedding;
    private WeightNormConv1DLayer<T>? _encoderIn;
    private readonly List<GatedConvolutionBlockLayer<T>> _encoderBlocks = new();
    private WeightNormConv1DLayer<T>? _encoderOut;

    // Decoder (§3.5) with attention blocks (§3.6).
    private readonly List<WeightNormConv1DLayer<T>> _prenet = new();
    private readonly List<GatedConvolutionBlockLayer<T>> _decoderBlocks = new();
    private readonly List<AttentionBlock> _attention = new();
    private WeightNormConv1DLayer<T>? _melProjection;
    private WeightNormConv1DLayer<T>? _doneProjection;
    private DropoutLayer<T>? _dropout;

    // Converter (§3.7), Griffin-Lim variant.
    private WeightNormConv1DLayer<T>? _converterIn;
    private readonly List<GatedConvolutionBlockLayer<T>> _converterBlocks = new();
    private WeightNormConv1DLayer<T>? _linearProjection;

    public override ModelOptions GetOptions() => _options;

    public DeepVoice3(NeuralNetworkArchitecture<T> architecture, string modelPath, DeepVoice3Options? options = null)
        : base(architecture)
    {
        _options = options ?? new DeepVoice3Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public DeepVoice3(
        NeuralNetworkArchitecture<T> architecture,
        DeepVoice3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DeepVoice3Options();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate }));
        MaxGradNorm = NumOps.FromDouble(_options.MaxGradientNorm);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a log-mel spectrogram from text.</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>The Griffin–Lim converter predicts the linear spectrogram, which a mel spectrogram does not carry (§3.7).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    protected override int TargetFftSize => _options.FftSize;

    /// <inheritdoc />
    protected override int TargetWindowSize => _options.WindowSize;

    /// <inheritdoc />
    /// <remarks>Table 4 states both a maximum gradient norm (100) and a gradient clipping value (5); the norm clip runs first.</remarks>
    protected override double GradientValueClip => _options.GradientClipValue;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        double p = o.DropoutRate;
        WeightNormConv1DLayer<T> Linear(int input, int output, double stdMul = 1.0)
            => new(input, output, 1, 1, 0, Math.Sqrt(stdMul * (1.0 - p) / input));

        _dropout = p > 0 ? new DropoutLayer<T>(p) : null;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EmbeddingDim);
        _encoderIn = Linear(o.EmbeddingDim, o.EncoderChannels);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new GatedConvolutionBlockLayer<T>(o.EncoderChannels, o.EncoderChannels, o.ConvKernelSize, 1,
                causal: false, p, residual: true, stdMultiplier: i == 0 ? 2.0 : 4.0));
        _encoderOut = Linear(o.EncoderChannels, o.EmbeddingDim, 4.0);

        int frameGroup = o.MelChannels * o.OutputsPerStep;
        int previous = frameGroup;
        foreach (int size in o.DecoderPrenetSizes)
        {
            _prenet.Add(Linear(previous, size));
            previous = size;
        }
        int channels = previous;
        for (int i = 0; i < o.NumDecoderLayers; i++)
        {
            _decoderBlocks.Add(new GatedConvolutionBlockLayer<T>(channels, channels, o.ConvKernelSize, 1,
                causal: true, p, residual: false, stdMultiplier: 4.0));
            _attention.Add(new AttentionBlock(this, channels, o.EmbeddingDim, o.AttentionDim, p));
        }
        _melProjection = Linear(channels, frameGroup, 4.0);
        _doneProjection = Linear(frameGroup, 1);

        _converterIn = Linear(channels, o.ConverterChannels);
        for (int i = 0; i < o.NumConverterLayers; i++)
            _converterBlocks.Add(new GatedConvolutionBlockLayer<T>(o.ConverterChannels, o.ConverterChannels, o.ConvKernelSize, 1,
                causal: false, p, residual: true, stdMultiplier: 4.0));
        _linearProjection = Linear(o.ConverterChannels, o.FftSize / 2 + 1, 4.0);

        var encoder = new List<ILayer<T>> { _embedding, _encoderIn };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_encoderOut);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_prenet);
        ComponentLayers.AddRange(_decoderBlocks);
        foreach (var block in _attention) ComponentLayers.AddRange(block.Layers);
        ComponentLayers.Add(_melProjection);
        ComponentLayers.Add(_doneProjection);
        ComponentLayers.Add(_converterIn);
        ComponentLayers.AddRange(_converterBlocks);
        ComponentLayers.Add(_linearProjection);
        if (_dropout is not null) ComponentLayers.Add(_dropout);
    }

    private bool HasPaperLayers => _embedding is not null;

    // A 1x1 weight-normalized convolution applied position-wise: [time, in] -> [time, out].
    private Tensor<T> Apply(WeightNormConv1DLayer<T> linear, Tensor<T> x)
    {
        int time = x.Shape[0];
        var channelsFirst = Engine.Reshape(Engine.TensorTranspose(x), new[] { 1, x.Shape[1], time });
        var y = linear.Forward(channelsFirst);
        return Engine.TensorTranspose(Engine.Reshape(y, new[] { y.Shape[1], time }));
    }

    private Tensor<T> Dropout(Tensor<T> x) => _dropout is null ? x : _dropout.Forward(x);

    /// <summary>The encoder: keys <c>h_k</c> and values <c>h_v = √0.5 (h_k + h_e)</c>, both <c>[characters, embedding]</c>.</summary>
    private (Tensor<T> Keys, Tensor<T> Values) Encode(Tensor<T> tokens)
    {
        var embedded = Dropout(_embedding!.Forward(tokens));
        // The projection to the convolution width is followed by a ReLU in the reference implementation.
        var x = Engine.ReLU(Apply(_encoderIn!, embedded));
        foreach (var block in _encoderBlocks) x = block.Forward(x);
        var keys = Apply(_encoderOut!, Dropout(x));
        var values = Engine.TensorMultiplyScalar(Engine.TensorAdd(keys, embedded), NumOps.FromDouble(Math.Sqrt(0.5)));
        return (keys, values);
    }

    /// <summary>
    /// Sinusoidal positional encoding at rate ω (§3.6): channel k of position i is sin(ω i / 10000^(k/d)) for even k and
    /// cos for odd k, scaled by the position weight. Positions start at 1.
    /// </summary>
    private Tensor<T> PositionalEncoding(int length, int dimension, double rate)
    {
        var encoding = new Tensor<T>(new[] { length, dimension });
        for (int i = 0; i < length; i++)
            for (int k = 0; k < dimension; k++)
            {
                double angle = rate * (i + 1) / Math.Pow(10000.0, (k - k % 2) / (double)dimension);
                encoding[i, k] = NumOps.FromDouble(_options.PositionWeight * (k % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle)));
            }
        return encoding;
    }

    /// <summary>
    /// The decoder over frame groups <c>[steps, r · mel]</c> (each row the previous group of r frames, zeros first):
    /// mel groups <c>[steps, r · mel]</c>, done logits <c>[steps]</c>, and the last hidden states <c>[steps, channels]</c>.
    /// With <paramref name="windows"/> the attention is restricted per layer to a window around the last attended position.
    /// </summary>
    private (Tensor<T> Mel, Tensor<T> Done, Tensor<T> Hidden) Decode(Tensor<T> inputs, Tensor<T> keys, Tensor<T> values,
        int[]? windows)
    {
        int steps = inputs.Shape[0];
        var keysWithPositions = Engine.TensorAdd(keys, PositionalEncoding(keys.Shape[0], keys.Shape[1], _options.KeyPositionRate));
        var x = inputs;
        for (int i = 0; i < _prenet.Count; i++)
        {
            if (i > 0) x = Dropout(x);
            x = Engine.ReLU(Apply(_prenet[i], x));
        }
        var queryPositions = PositionalEncoding(steps, x.Shape[1], _options.QueryPositionRate);
        for (int l = 0; l < _decoderBlocks.Count; l++)
        {
            var residual = x;
            x = _decoderBlocks[l].Forward(x);
            x = _attention[l].Forward(Engine.TensorAdd(x, queryPositions), keysWithPositions, values,
                windows is null ? null : windows[l]);
            x = Engine.TensorMultiplyScalar(Engine.TensorAdd(x, residual), NumOps.FromDouble(Math.Sqrt(0.5)));
        }
        var mel = Apply(_melProjection!, Dropout(x));
        var done = Engine.Reshape(Apply(_doneProjection!, mel), new[] { steps });
        return (mel, done, x);
    }

    /// <summary>The converter (§3.7): decoder states upsampled by repetition to frames, non-causal blocks, linear spectrogram.</summary>
    private Tensor<T> Convert(Tensor<T> hidden)
    {
        int steps = hidden.Shape[0], channels = hidden.Shape[1], r = _options.OutputsPerStep;
        var repeated = Engine.Reshape(Engine.TensorTile(Engine.Reshape(hidden, new[] { steps, 1, channels }), new[] { 1, r, 1 }),
            new[] { steps * r, channels });
        var x = Apply(_converterIn!, repeated);
        foreach (var block in _converterBlocks) x = block.Forward(x);
        return Apply(_linearProjection!, Dropout(x));
    }

    /// <inheritdoc />
    /// <remarks>Autoregressive inference: each step decodes the groups produced so far and appends the newest, with each
    /// attention layer restricted to a window from one position behind to three ahead of its last attended position
    /// (§3.6), until the done prediction exceeds 0.5 (after a minimum number of steps) or the step limit.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        if (input.Rank == 1)
            return SynthesizeOne(input).Mel;
        if (input.Rank != 2)
            throw new ArgumentException($"Expected characters [characters] or [batch, characters], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], characters = input.Shape[1];
        var outputs = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var row = new Tensor<T>(new[] { characters });
            for (int i = 0; i < characters; i++) row[i] = input[b, i];
            outputs.Add(SynthesizeOne(row).Mel);
        }
        int longest = outputs.Max(o => o.Shape[0]);
        var result = new Tensor<T>(new[] { batch, longest, _options.MelChannels });
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < outputs[b].Shape[0]; f++)
                for (int m = 0; m < _options.MelChannels; m++) result[b, f, m] = outputs[b][f, m];
        return result;
    }

    /// <summary>Predicts the log-magnitude linear spectrogram for text, the converter's output.</summary>
    public Tensor<T> PredictLinearSpectrogram(string text)
    {
        ThrowIfDisposed();
        SetTrainingMode(false);
        return SynthesizeOne(PreprocessText(text)).Linear;
    }

    private (Tensor<T> Mel, Tensor<T> Linear) SynthesizeOne(Tensor<T> tokens)
    {
        using var _ = new NoGradScope<T>();
        var (keys, values) = Encode(tokens);
        int group = _options.MelChannels * _options.OutputsPerStep;
        var fed = new List<double[]> { new double[group] };
        var windows = new int[_decoderBlocks.Count];
        Tensor<T>? lastMel = null, lastHidden = null;
        for (int step = 0; step < _options.MaxDecoderSteps; step++)
        {
            var inputs = new Tensor<T>(new[] { fed.Count, group });
            for (int s = 0; s < fed.Count; s++)
                for (int c = 0; c < group; c++) inputs[s, c] = NumOps.FromDouble(fed[s][c]);
            var (mel, done, hidden) = Decode(inputs, keys, values, windows);
            for (int l = 0; l < _attention.Count; l++) windows[l] = _attention[l].LastAttended;
            lastMel = mel;
            lastHidden = hidden;
            int last = fed.Count - 1;
            var frame = new double[group];
            for (int c = 0; c < group; c++) frame[c] = NumOps.ToDouble(mel[last, c]);
            fed.Add(frame);
            double doneProbability = 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(done[last])));
            if (doneProbability > 0.5 && step + 1 >= _options.MinDecoderSteps) break;
        }
        int steps = lastMel!.Shape[0];
        var melFrames = Engine.Reshape(lastMel, new[] { steps * _options.OutputsPerStep, _options.MelChannels });
        return (melFrames, Convert(lastHidden!));
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var (groups, objective) = BuildObjective(sample);
        return TrainWithCustomObjective(sample.Tokens, groups, objective, _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (groups, objective) = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, groups)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The multi-task objective (§3.5, §3.7): L1 on the decoder's mel frames, binary cross-entropy of the done prediction
    /// (1 on the final group), and L1 on the converter's linear spectrogram, with teacher forcing (each step reads the
    /// previous ground-truth group of r frames).
    /// </summary>
    private (Tensor<T> Groups, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var targets = DeriveAcousticTargets(sample);
        var linear = targets.LinearSpectrogram ?? throw new ArgumentException(
            $"{nameof(DeepVoice3<T>)} trains its converter on the linear spectrogram; set {nameof(sample.Audio)} or {nameof(sample.LinearSpectrogram)}.",
            nameof(sample));
        int frames = targets.MelFrames, r = _options.OutputsPerStep, mel = _options.MelChannels, bins = _options.FftSize / 2 + 1;
        if (linear.Shape[0] != frames || linear.Shape[1] != bins)
            throw new ArgumentException($"Expected a linear spectrogram [{frames}, {bins}], got [{string.Join(", ", linear.Shape)}].", nameof(sample));
        int steps = (frames + r - 1) / r;

        // Ground truth grouped into r-frame rows, zero-padded at the end.
        var groups = new Tensor<T>(new[] { steps, r * mel });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < mel; c++) groups[f / r, (f % r) * mel + c] = targets.Mel[f, c];
        var paddedLinear = new Tensor<T>(new[] { steps * r, bins });
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < bins; k++) paddedLinear[f, k] = linear[f, k];
        var doneTarget = new Tensor<T>(new[] { steps });
        doneTarget[steps - 1] = NumOps.One;

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            var (keys, values) = Encode(tokens);
            var inputs = steps == 1
                ? new Tensor<T>(new[] { 1, r * mel })
                : Engine.TensorConcatenate(new[]
                {
                    new Tensor<T>(new[] { 1, r * mel }),
                    Engine.TensorSlice(target, new[] { 0, 0 }, new[] { steps - 1, r * mel }),
                }, 0);
            var (predicted, done, hidden) = Decode(inputs, keys, values, null);
            var melLoss = MeanAbsolute(predicted, target);
            var linearLoss = MeanAbsolute(Convert(hidden), paddedLinear);
            return Engine.TensorAdd(Engine.TensorAdd(melLoss, BinaryCrossEntropy(done, doneTarget)), linearLoss);
        }

        return (groups, Objective);
    }

    private Tensor<T> MeanAbsolute(Tensor<T> prediction, Tensor<T> target)
        => Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(prediction, target)), new[] { 0, 1 }, keepDims: false);

    // mean(-[y log s(x) + (1 - y) log(1 - s(x))]) with the stable softplus form.
    private Tensor<T> BinaryCrossEntropy(Tensor<T> logits, Tensor<T> target)
    {
        Tensor<T> Softplus(Tensor<T> x) => Engine.TensorAdd(Engine.ReLU(x),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(x))), NumOps.One)));
        var positive = Engine.TensorMultiply(target, Softplus(Engine.TensorNegate(logits)));
        var negative = Engine.TensorMultiply(Engine.TensorAddScalar(Engine.TensorNegate(target), NumOps.One), Softplus(logits));
        return Engine.ReduceMean(Engine.TensorAdd(positive, negative), new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// Deep Voice 3's attention block (§3.6, Fig. 3; reference <c>AttentionLayer</c>): query, key and value projections to
    /// the attention width (query and key share their initial weights), dot-product scores, softmax (dropped out in
    /// training; optionally restricted to a window at inference), context normalized by √(input steps), an output
    /// projection, and a residual with the query scaled by √0.5.
    /// </summary>
    private sealed class AttentionBlock
    {
        private readonly DeepVoice3<T> _owner;
        private readonly WeightNormConv1DLayer<T> _query;
        private readonly WeightNormConv1DLayer<T> _key;
        private readonly WeightNormConv1DLayer<T> _value;
        private readonly WeightNormConv1DLayer<T> _out;

        public AttentionBlock(DeepVoice3<T> owner, int channels, int embedding, int attention, double dropout)
        {
            _owner = owner;
            double std(int input) => Math.Sqrt((1.0 - dropout) / input);
            _query = new WeightNormConv1DLayer<T>(channels, attention, 1, 1, 0, std(channels));
            _key = new WeightNormConv1DLayer<T>(embedding, attention, 1, 1, 0, std(embedding));
            if (channels == embedding) _key.SetParameters(_query.GetParameters());
            _value = new WeightNormConv1DLayer<T>(embedding, attention, 1, 1, 0, std(embedding));
            _out = new WeightNormConv1DLayer<T>(attention, channels, 1, 1, 0, std(attention));
        }

        public IEnumerable<LayerBase<T>> Layers => new LayerBase<T>[] { _query, _key, _value, _out };

        /// <summary>The input position with the highest weight at the last query step.</summary>
        public int LastAttended { get; private set; }

        public Tensor<T> Forward(Tensor<T> query, Tensor<T> keys, Tensor<T> values, int? window)
        {
            var engine = _owner.Engine;
            var numOps = _owner.NumOps;
            int queries = query.Shape[0], inputs = keys.Shape[0];
            var q = _owner.Apply(_query, query);
            var k = _owner.Apply(_key, keys);
            var v = _owner.Apply(_value, values);
            var scores = engine.TensorMatMul(q, engine.TensorTranspose(k));                       // [queries, inputs]
            if (window.HasValue)
            {
                // Only the newest query is constrained: positions outside [last - 1, last + 3) get a large negative logit.
                var mask = new Tensor<T>(new[] { queries, inputs });
                int from = Math.Max(0, window.Value - 1), to = Math.Min(inputs, window.Value + 3);
                for (int j = 0; j < inputs; j++)
                    if (j < from || j >= to) mask[queries - 1, j] = numOps.FromDouble(-1e9);
                scores = engine.TensorAdd(scores, mask);
            }
            var weights = engine.TensorSoftmax(scores, axis: 1);
            int best = 0;
            for (int j = 1; j < inputs; j++)
                if (numOps.ToDouble(weights[queries - 1, j]) > numOps.ToDouble(weights[queries - 1, best])) best = j;
            LastAttended = best;
            var context = engine.TensorMultiplyScalar(engine.TensorMatMul(_owner.Dropout(weights), v),
                numOps.FromDouble(Math.Sqrt(inputs)));
            var output = _owner.Apply(_out, context);
            return engine.TensorMultiplyScalar(engine.TensorAdd(output, query), numOps.FromDouble(Math.Sqrt(0.5)));
        }
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "DeepVoice3-Native" : "DeepVoice3-ONNX",
            Description = "Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning (Ping et al., 2018)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers + _options.NumConverterLayers,
        };
        m.AdditionalInfo["Architecture"] = "DeepVoice3";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(DeepVoice3<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
