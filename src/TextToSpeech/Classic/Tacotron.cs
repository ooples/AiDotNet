using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

using AiDotNet.ActivationFunctions;

using AiDotNet.NeuralNetworks.Layers;

using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Tacotron: an end-to-end attention-based sequence-to-sequence text-to-speech model with CBHG modules.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Tacotron: Towards End-to-End Speech Synthesis" (Wang et al., Interspeech 2017).</para>
/// <para>
/// The architecture is Table 1 of the paper: a character embedding, a pre-net and a CBHG encoder
/// (<see cref="CbhgLayer{T}"/>); a decoder pre-net feeding a 256-cell attention GRU with content-based tanh attention
/// over the encoder outputs; the attention GRU output and context projected into two residual 256-cell decoder GRUs,
/// which emit r = 2 mel frames per step from an all-zero &lt;GO&gt; frame; and a post-processing CBHG that predicts the
/// linear-frequency spectrogram, turned into a waveform by Griffin–Lim. The loss is the L1 error of both spectrograms.
/// Unstated details (padding, normalization order, the attention cell's input) follow the reference implementation
/// keithito/tacotron and are noted where they are applied.
/// </para>
/// <para>
/// Training data: the post-processing net trains on the linear spectrogram, which a mel spectrogram does not carry, so
/// a plain <c>Train(tokens, mel)</c> throws; pass a <see cref="TtsTrainingSample{T}"/> with the recording
/// (<see cref="TtsTrainingSample{T}.Audio"/>, from which both spectrograms are computed with the paper's analysis
/// settings) or with <see cref="TtsTrainingSample{T}.LinearSpectrogram"/>.
/// </para>
/// <para><b>For Beginners:</b> Tacotron reads text and, step by step, paints the spectrogram while learning on its own
/// which letters it is reading at each step; a second network sharpens the result into a full spectrogram that can be
/// turned into sound.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Tacotron: Towards End-to-End Speech Synthesis",
    "https://arxiv.org/abs/1703.10135",
    Year = 2017,
    Authors = "Wang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001,
                Source = "Wang et al. 2017, Sec. 4: Adam starting from a learning rate of 0.001, reduced to 0.0005, 0.0003 and 0.0001 after 500K, 1M and 2M global steps. No schedule is declared because those steps are not a constant decay factor (0.5, then 0.6, then 0.33), and a multi-step schedule applies one gamma to every milestone. Built by the model rather than by the factory because it constructs explicit options; the declaration verifies those values instead of replacing them.")]
public partial class Tacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly TacotronOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Encoder: character embedding -> pre-net -> CBHG (Table 1).
    private EmbeddingLayer<T>? _embedding;
    private readonly List<LayerBase<T>> _encoderPrenet = new();
    private CbhgLayer<T>? _encoderCbhg;

    // Decoder: pre-net -> attention GRU -> additive attention -> projection -> 2 residual GRUs -> r frames.
    private readonly List<LayerBase<T>> _decoderPrenet = new();
    private GRUCellLayer<T>? _attentionRnn;
    private BiasFreeLinearLayer<T>? _attentionQuery;
    private BiasFreeLinearLayer<T>? _attentionMemory;
    private BiasFreeLinearLayer<T>? _attentionScore;
    private DenseLayer<T>? _decoderInput;
    private GRUCellLayer<T>? _decoderRnn1;
    private GRUCellLayer<T>? _decoderRnn2;
    private DenseLayer<T>? _frameProjection;

    // Post-processing: CBHG on the mel frames -> linear spectrogram.
    private CbhgLayer<T>? _postCbhg;
    private DenseLayer<T>? _linearProjection;

    public override ModelOptions GetOptions() => _options;

    public Tacotron(NeuralNetworkArchitecture<T> architecture, string modelPath, TacotronOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TacotronOptions();
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

    public Tacotron(
        NeuralNetworkArchitecture<T> architecture,
        TacotronOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TacotronOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        // Adam from 0.001, reduced to 0.0005, 0.0003 and 0.0001 after 500K, 1M and 2M steps (§4).
        double baseRate = _options.LearningRate;
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = baseRate,
                    LearningRateScheduler = new LambdaLRScheduler(baseRate,
                        step => step < 500_000 ? 1.0 : step < 1_000_000 ? 0.5 : step < 2_000_000 ? 0.3 : 0.1),
                }));
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a log-mel spectrogram from text, the decoder's targets.</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>Tacotron learns its alignment, but its post-processing net predicts the linear spectrogram, which a mel
    /// spectrogram alone does not carry (§3.4).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    protected override int TargetFftSize => _options.FftSize;

    /// <inheritdoc />
    /// <remarks>50 ms frames (§4).</remarks>
    protected override int TargetWindowSize => _options.WindowSize;

    /// <inheritdoc />
    /// <remarks>"We also found pre-emphasis (0.97) to be helpful" (§4).</remarks>
    protected override double PreEmphasis => _options.PreEmphasis;

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
        IActivationFunction<T> relu = new ReLUActivation<T>();
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EmbeddingDim);
        AddPrenet(_encoderPrenet, o.PrenetSizes, relu);
        _encoderCbhg = new CbhgLayer<T>(o.PrenetSizes[^1], o.EncoderBankSize, o.CbhgChannels,
            new[] { o.CbhgChannels, o.PrenetSizes[^1] }, o.CbhgChannels, o.CbhgChannels);
        int encoderWidth = _encoderCbhg.OutputChannels;

        AddPrenet(_decoderPrenet, o.PrenetSizes, relu);
        _attentionRnn = new GRUCellLayer<T>(o.PrenetSizes[^1] + encoderWidth, o.AttentionDim);
        _attentionQuery = new BiasFreeLinearLayer<T>(o.AttentionDim, o.AttentionDim);
        _attentionMemory = new BiasFreeLinearLayer<T>(encoderWidth, o.AttentionDim);
        _attentionScore = new BiasFreeLinearLayer<T>(o.AttentionDim, 1);
        _decoderInput = new DenseLayer<T>(o.DecoderRnnDim, identity);
        _decoderRnn1 = new GRUCellLayer<T>(o.DecoderRnnDim, o.DecoderRnnDim);
        _decoderRnn2 = new GRUCellLayer<T>(o.DecoderRnnDim, o.DecoderRnnDim);
        _frameProjection = new DenseLayer<T>(o.MelChannels * o.OutputsPerStep, identity);

        _postCbhg = new CbhgLayer<T>(o.MelChannels, o.PostBankSize, o.CbhgChannels,
            new[] { o.PostProjectionChannels, o.MelChannels }, o.CbhgChannels, o.CbhgChannels);
        _linearProjection = new DenseLayer<T>(o.FftSize / 2 + 1, identity);

        var encoder = new List<ILayer<T>> { _embedding };
        encoder.AddRange(_encoderPrenet);
        encoder.Add(_encoderCbhg);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_decoderPrenet);
        ComponentLayers.AddRange(new LayerBase<T>[]
        {
            _attentionRnn, _attentionQuery, _attentionMemory, _attentionScore, _decoderInput, _decoderRnn1,
            _decoderRnn2, _frameProjection, _postCbhg, _linearProjection,
        });
    }

    // FC-256-ReLU -> Dropout(0.5) -> FC-128-ReLU -> Dropout(0.5) (Table 1).
    private void AddPrenet(List<LayerBase<T>> into, int[] sizes, IActivationFunction<T> relu)
    {
        foreach (int size in sizes)
        {
            into.Add(new DenseLayer<T>(size, relu));
            into.Add(new DropoutLayer<T>(_options.PrenetDropout));
        }
    }

    private bool HasPaperLayers => _embedding is not null;

    private static Tensor<T> RunAll(IEnumerable<LayerBase<T>> layers, Tensor<T> x)
    {
        foreach (var layer in layers) x = layer.Forward(x);
        return x;
    }

    /// <summary>The encoder: <c>[characters] → [characters, 2 · CBHG GRU units]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> tokens) => _encoderCbhg!.Forward(RunAll(_encoderPrenet, _embedding!.Forward(tokens)));

    /// <summary>
    /// Runs the attention decoder for <paramref name="steps"/> steps from the all-zero &lt;GO&gt; frame and returns the
    /// predicted mel frames, <c>[steps · r, melChannels]</c>. With <paramref name="teacher"/> each step reads the last
    /// ground-truth frame of the previous group (§3.3: "the last frame of the r predictions is fed"); without, the last
    /// predicted frame.
    /// </summary>
    private Tensor<T> Decode(Tensor<T> encoded, int steps, Tensor<T>? teacher)
    {
        var o = _options;
        int characters = encoded.Shape[0], r = o.OutputsPerStep;
        var memory = _attentionMemory!.Forward(encoded);                     // [chars, attention]
        var attentionState = new Tensor<T>(new[] { 1, o.AttentionDim });
        var decoderState1 = new Tensor<T>(new[] { 1, o.DecoderRnnDim });
        var decoderState2 = new Tensor<T>(new[] { 1, o.DecoderRnnDim });
        var context = new Tensor<T>(new[] { 1, encoded.Shape[1] });
        var previous = new Tensor<T>(new[] { 1, o.MelChannels });            // <GO>
        var groups = new List<Tensor<T>>(steps);

        for (int step = 0; step < steps; step++)
        {
            var prenet = RunAll(_decoderPrenet, previous);                  // [1, 128]
            attentionState = _attentionRnn!.Forward(Engine.TensorConcatenate(new[] { prenet, context }, 1), attentionState);

            // Content-based tanh attention: e_j = v^T tanh(W h_j + V d_t) (Bahdanau et al.; Vinyals et al.).
            var query = Engine.TensorTile(_attentionQuery!.Forward(attentionState), new[] { characters, 1 });
            var scores = _attentionScore!.Forward(Engine.Tanh(Engine.TensorAdd(memory, query)));   // [chars, 1]
            var weights = Engine.TensorSoftmax(Engine.Reshape(scores, new[] { 1, characters }), axis: 1);
            context = Engine.TensorMatMul(weights, encoded);                // [1, encoder]

            var input = _decoderInput!.Forward(Engine.TensorConcatenate(new[] { attentionState, context }, 1));
            decoderState1 = _decoderRnn1!.Forward(input, decoderState1);
            var output1 = Engine.TensorAdd(input, decoderState1);           // residual
            decoderState2 = _decoderRnn2!.Forward(output1, decoderState2);
            var output2 = Engine.TensorAdd(output1, decoderState2);

            var frames = Engine.Reshape(_frameProjection!.Forward(output2), new[] { r, o.MelChannels });
            groups.Add(frames);
            previous = teacher is not null
                ? Engine.TensorSlice(teacher, new[] { step * r + r - 1, 0 }, new[] { 1, o.MelChannels })
                : Engine.TensorSlice(frames, new[] { r - 1, 0 }, new[] { 1, o.MelChannels });
            if (teacher is null)
                previous = new Tensor<T>(previous._shape, previous.ToVector());
        }
        return groups.Count == 1 ? groups[0] : Engine.TensorConcatenate(groups.ToArray(), 0);
    }

    /// <summary>The post-processing net: mel frames → linear spectrogram, <c>[frames, fftSize / 2 + 1]</c>.</summary>
    private Tensor<T> PostProcess(Tensor<T> mel) => _linearProjection!.Forward(_postCbhg!.Forward(mel));

    /// <inheritdoc />
    /// <remarks>
    /// Returns the predicted log-mel spectrogram, <c>[MaxDecoderSteps · r, melChannels]</c>. The paper gives no stopping
    /// rule; the reference implementation decodes a fixed maximum number of steps and trims silence from the waveform,
    /// so the decoder runs <see cref="TacotronOptions.MaxDecoderSteps"/> steps.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
            return RunAll(Layers.Cast<LayerBase<T>>(), input);
        if (input.Rank == 1)
            return Decode(Encode(input), _options.MaxDecoderSteps, null);
        if (input.Rank != 2)
            throw new ArgumentException($"Expected characters [characters] or [batch, characters], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], characters = input.Shape[1];
        var outputs = new Tensor<T>[batch];
        for (int b = 0; b < batch; b++)
        {
            var row = new Tensor<T>(new[] { characters });
            for (int i = 0; i < characters; i++) row[i] = input[b, i];
            outputs[b] = Engine.Reshape(Decode(Encode(row), _options.MaxDecoderSteps, null),
                new[] { 1, _options.MaxDecoderSteps * _options.OutputsPerStep, _options.MelChannels });
        }
        return batch == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
    }

    /// <summary>Predicts the log-magnitude linear spectrogram for text, the post-processing net's output.</summary>
    public Tensor<T> PredictLinearSpectrogram(string text)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers)
            throw new NotSupportedException("The linear spectrogram needs the paper's layers; this model was built from caller-supplied layers.");
        SetTrainingMode(false);
        var tokens = PreprocessText(text);
        return PostProcess(Decode(Encode(tokens), _options.MaxDecoderSteps, null));
    }

    /// <summary>
    /// Synthesizes a waveform the paper's way (§3.4): Griffin–Lim (50 iterations) on the predicted magnitudes raised to
    /// the power 1.2, then de-emphasis.
    /// </summary>
    public Tensor<T> SynthesizeWaveform(string text)
    {
        var logMagnitude = PredictLinearSpectrogram(text);
        int frames = logMagnitude.Shape[0], bins = logMagnitude.Shape[1];
        var magnitude = new Tensor<T>(new[] { frames, bins });
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < bins; k++)
                magnitude[f, k] = NumOps.FromDouble(Math.Pow(Math.Exp(NumOps.ToDouble(logMagnitude[f, k])), _options.GriffinLimPower));
        var griffinLim = new AiDotNet.Diffusion.Audio.GriffinLim<T>(_options.FftSize, _options.HopSize,
            new CenteredHannWindow(_options.WindowSize), _options.GriffinLimIterations, momentum: 0.0, seed: 0);
        var waveform = griffinLim.Reconstruct(magnitude);
        // Undo the 0.97 pre-emphasis: x[n] = y[n] + k x[n - 1].
        double k0 = _options.PreEmphasis, previous = 0;
        for (int n = 0; n < waveform.Length; n++)
        {
            double value = NumOps.ToDouble(waveform[n]) + k0 * previous;
            waveform[n] = NumOps.FromDouble(value);
            previous = value;
        }
        return waveform;
    }

    /// <summary>A periodic Hann window of the analysis length, centred and zero-padded to the FFT size, as
    /// <see cref="TacotronSpectrogram"/> analyses with.</summary>
    private sealed class CenteredHannWindow : IWindowFunction<T>
    {
        private readonly int _length;
        public CenteredHannWindow(int length) => _length = length;

        public Vector<T> Create(int windowSize)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var window = new Vector<T>(windowSize);
            int length = Math.Min(_length, windowSize), offset = (windowSize - length) / 2;
            for (int i = 0; i < length; i++)
                window[offset + i] = ops.FromDouble(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / length));
            return window;
        }

        public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hanning;
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
        var (mel, objective) = BuildObjective(sample);
        return TrainWithCustomObjective(sample.Tokens, mel, objective, _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (mel, objective) = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The paper objective (§3.4): L1 on the decoder's mel frames plus L1 on the post-processing net's linear spectrogram,
    /// equally weighted, both over the targets zero-padded to a whole number of r-frame groups ("we use a simple L1 loss
    /// ... without masking ... it allows the model to learn when to stop").
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var targets = DeriveAcousticTargets(sample);
        var linear = targets.LinearSpectrogram ?? throw new ArgumentException(
            $"{nameof(Tacotron<T>)} trains its post-processing net on the linear spectrogram; set {nameof(sample.Audio)} or {nameof(sample.LinearSpectrogram)}.",
            nameof(sample));
        int frames = targets.MelFrames, r = _options.OutputsPerStep;
        if (linear.Shape[0] != frames || linear.Shape[1] != _options.FftSize / 2 + 1)
            throw new ArgumentException(
                $"Expected a linear spectrogram [{frames}, {_options.FftSize / 2 + 1}], got [{string.Join(", ", linear.Shape)}].", nameof(sample));
        int steps = (frames + r - 1) / r;
        var mel = PadFrames(targets.Mel, steps * r);
        var paddedLinear = PadFrames(linear, steps * r);

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            var predictedMel = Decode(Encode(tokens), steps, target);
            var predictedLinear = PostProcess(predictedMel);
            return Engine.TensorAdd(MeanAbsolute(predictedMel, target), MeanAbsolute(predictedLinear, paddedLinear));
        }

        return (mel, Objective);
    }

    private Tensor<T> PadFrames(Tensor<T> x, int frames)
    {
        if (x.Shape[0] == frames) return x;
        var padded = new Tensor<T>(new[] { frames, x.Shape[1] });
        for (int f = 0; f < x.Shape[0]; f++)
            for (int c = 0; c < x.Shape[1]; c++) padded[f, c] = x[f, c];
        return padded;
    }

    private Tensor<T> MeanAbsolute(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorAbs(Engine.TensorSubtract(prediction, target));
        return Engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), keepDims: false);
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
            Name = _useNativeMode ? "Tacotron-Native" : "Tacotron-ONNX",
            Description = "Tacotron: Towards End-to-End Speech Synthesis (Wang et al., 2017)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.EncoderBankSize + _options.PostBankSize,
        };
        m.AdditionalInfo["Architecture"] = "Tacotron";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Tacotron<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
