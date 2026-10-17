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
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Transformer TTS: Tacotron 2's text-to-mel pipeline with its recurrent encoder and decoder replaced by a Transformer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Neural Speech Synthesis with Transformer Network" (Li et al., AAAI 2019).</para>
/// <para>
/// The structure is the paper's Fig. 3: phoneme embeddings through Tacotron 2's 3-layer convolutional encoder pre-net and
/// a linear projection (§3.3), plus a scaled positional encoding with a trainable weight α (§3.2,
/// <see cref="ScaledPositionalEncodingLayer{T}"/>); a Transformer encoder (§3.5); the mel spectrogram through a
/// 2-layer fully connected decoder pre-net and a linear projection plus its own scaled positional encoding (§3.4); a
/// Transformer decoder with masked self-attention and encoder–decoder attention (§3.6,
/// <see cref="PostNormTransformerDecoderBlock{T}"/>); mel and stop linear projections and a 5-layer convolutional
/// post-net producing a residual (§3.7). Training is teacher-forced and minimizes the mel mean squared error before and
/// after the post-net plus the stop-token binary cross-entropy with a positive weight on the final frame.
/// Inference is autoregressive until the stop token fires.
/// </para>
/// <para>The model learns its own alignment through attention, so a plain <c>Train(tokens, mel)</c> is its full
/// training signal.</para>
/// <para><b>For Beginners:</b> Like Tacotron 2, but every recurrent network is replaced by attention, so training
/// runs over all frames in parallel.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Neural Speech Synthesis with Transformer Network",
    "https://arxiv.org/abs/1809.08895",
    Year = 2019,
    Authors = "Li et al."
)]
public partial class TransformerTTS<T> : TtsModelBase<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly TransformerTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Encoder: phoneme embedding -> 3-conv pre-net -> linear -> +alpha PE -> Transformer encoder.
    private EmbeddingLayer<T>? _embedding;
    private ConvBatchNormStackLayer<T>? _encoderPrenet;
    private DenseLayer<T>? _encoderPrenetProjection;
    private ScaledPositionalEncodingLayer<T>? _encoderPositions;
    private readonly List<FeedForwardTransformerBlock<T>> _encoderBlocks = new();

    // Decoder: 2-FC pre-net -> linear -> +alpha PE -> Transformer decoder -> mel linear, stop linear -> post-net.
    private readonly List<LayerBase<T>> _decoderPrenet = new();
    private DenseLayer<T>? _decoderPrenetProjection;
    private ScaledPositionalEncodingLayer<T>? _decoderPositions;
    private readonly List<PostNormTransformerDecoderBlock<T>> _decoderBlocks = new();
    private DenseLayer<T>? _melLinear;
    private DenseLayer<T>? _stopLinear;
    private ConvBatchNormStackLayer<T>? _postnet;

    public override ModelOptions GetOptions() => _options;

    public TransformerTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, TransformerTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TransformerTTSOptions();
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

    public TransformerTTS(
        NeuralNetworkArchitecture<T> architecture,
        TransformerTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TransformerTTSOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with a WaveNet vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>The trained scale α of the encoder's positional encoding (§3.2, §4.6).</summary>
    public T EncoderPositionScale => _encoderPositions is null ? NumOps.Zero : _encoderPositions.Alpha;

    /// <summary>The trained scale α of the decoder's positional encoding.</summary>
    public T DecoderPositionScale => _decoderPositions is null ? NumOps.Zero : _decoderPositions.Alpha;

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
        int d = o.HiddenDim;
        IActivationFunction<T> identity = new IdentityActivation<T>();
        IActivationFunction<T> relu = new ReLUActivation<T>();

        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EncoderDim);
        _encoderPrenet = new ConvBatchNormStackLayer<T>(o.EncoderDim, Enumerable.Repeat(o.EncoderPrenetChannels, o.EncoderPrenetLayers).ToArray(),
            o.PrenetKernelSize, useTanh: false, linearLast: false, o.PrenetDropout);
        _encoderPrenetProjection = new DenseLayer<T>(d, identity);
        _encoderPositions = new ScaledPositionalEncodingLayer<T>(d);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new FeedForwardTransformerBlock<T>(d, o.NumHeads, o.FeedForwardDim, 1, 1, o.DropoutRate));

        foreach (int size in o.DecoderPrenetSizes)
        {
            _decoderPrenet.Add(new DenseLayer<T>(size, relu));
            _decoderPrenet.Add(new DropoutLayer<T>(o.PrenetDropout));
        }
        _decoderPrenetProjection = new DenseLayer<T>(d, identity);
        _decoderPositions = new ScaledPositionalEncodingLayer<T>(d);
        for (int i = 0; i < o.NumDecoderLayers; i++)
            _decoderBlocks.Add(new PostNormTransformerDecoderBlock<T>(d, o.NumHeads, o.FeedForwardDim, o.DropoutRate));
        _melLinear = new DenseLayer<T>(o.MelChannels, identity);
        _stopLinear = new DenseLayer<T>(1, identity);
        var postnetChannels = Enumerable.Repeat(o.PostnetDim, o.PostnetLayers - 1).Append(o.MelChannels).ToArray();
        _postnet = new ConvBatchNormStackLayer<T>(o.MelChannels, postnetChannels, o.PostnetKernelSize, useTanh: true,
            linearLast: true, o.PostnetDropout);

        var encoder = new List<ILayer<T>> { _embedding, _encoderPrenet, _encoderPrenetProjection, _encoderPositions };
        encoder.AddRange(_encoderBlocks);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_decoderPrenet);
        ComponentLayers.Add(_decoderPrenetProjection);
        ComponentLayers.Add(_decoderPositions);
        ComponentLayers.AddRange(_decoderBlocks);
        ComponentLayers.Add(_melLinear);
        ComponentLayers.Add(_stopLinear);
        ComponentLayers.Add(_postnet);
    }

    private bool HasPaperLayers => _embedding is not null;

    /// <summary>The encoder (§3.3, §3.5): <c>[phonemes] → [phonemes, d_model]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> tokens) => RunEncoder(tokens);

    /// <summary>
    /// The decoder on the frames fed so far (§3.4, §3.6–3.7): <c>[frames, mel] → (mel [frames, mel], stop logits [frames])</c>.
    /// Causal self-attention makes each output depend only on the frames at and before it.
    /// </summary>
    private (Tensor<T> Mel, Tensor<T> StopLogits) DecodeFrames(Tensor<T> previousFrames, Tensor<T> memory)
    {
        var x = previousFrames;
        foreach (var layer in _decoderPrenet) x = layer.Forward(x);
        x = _decoderPositions!.Forward(_decoderPrenetProjection!.Forward(x));
        foreach (var block in _decoderBlocks) x = block.Forward(x, memory);
        var stop = Engine.Reshape(_stopLinear!.Forward(x), new[] { previousFrames.Shape[0] });
        return (_melLinear!.Forward(x), stop);
    }

    /// <summary>The post-net's residual refinement (§3.7).</summary>
    private Tensor<T> Refine(Tensor<T> mel) => Engine.TensorAdd(mel, _postnet!.Forward(mel));

    /// <inheritdoc />
    /// <remarks>Autoregressive inference: from the all-zero first frame, each step decodes the frames so far and appends
    /// the last prediction, until the stop token's probability exceeds 0.5 or <see cref="TransformerTTSOptions.MaxDecoderSteps"/>;
    /// the post-net then refines the whole spectrogram.</remarks>
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
            return SynthesizeOne(input);
        if (input.Rank != 2)
            throw new ArgumentException($"Expected phonemes [phonemes] or [batch, phonemes], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], tokens = input.Shape[1];
        var outputs = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var row = new Tensor<T>(new[] { tokens });
            for (int i = 0; i < tokens; i++) row[i] = input[b, i];
            outputs.Add(SynthesizeOne(row));
        }
        int longest = outputs.Max(o => o.Shape[0]);
        var result = new Tensor<T>(new[] { batch, longest, _options.MelChannels });
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < outputs[b].Shape[0]; f++)
                for (int m = 0; m < _options.MelChannels; m++) result[b, f, m] = outputs[b][f, m];
        return result;
    }

    private Tensor<T> SynthesizeOne(Tensor<T> tokens)
    {
        var memory = Encode(tokens);
        int channels = _options.MelChannels;
        var fed = new List<double[]> { new double[channels] };   // all-zero first frame
        var produced = new List<double[]>();
        for (int step = 0; step < _options.MaxDecoderSteps; step++)
        {
            var frames = new Tensor<T>(new[] { fed.Count, channels });
            for (int f = 0; f < fed.Count; f++)
                for (int c = 0; c < channels; c++) frames[f, c] = NumOps.FromDouble(fed[f][c]);
            var (mel, stop) = DecodeFrames(frames, memory);
            int last = fed.Count - 1;
            var frame = new double[channels];
            for (int c = 0; c < channels; c++) frame[c] = NumOps.ToDouble(mel[last, c]);
            produced.Add(frame);
            fed.Add(frame);
            double stopProbability = 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(stop[last])));
            if (stopProbability > 0.5) break;
        }
        var output = new Tensor<T>(new[] { produced.Count, channels });
        for (int f = 0; f < produced.Count; f++)
            for (int c = 0; c < channels; c++) output[f, c] = NumOps.FromDouble(produced[f][c]);
        return Refine(output);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }
        var mel = expectedOutput.Rank == 3 ? Engine.Reshape(expectedOutput, new[] { expectedOutput.Shape[1], expectedOutput.Shape[2] }) : expectedOutput;
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        TrainWithCustomObjective(tokens, mel, Objective, _optimizer);
    }

    /// <inheritdoc />
    /// <remarks>Self-aligning through encoder–decoder attention, so a token/mel pair is the whole supervision.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        var targets = DeriveAcousticTargets(sample);
        Train(sample.Tokens, targets.Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var mel = DeriveAcousticTargets(sample).Mel;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(sample.Tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <remarks>Synthesis stops at the predicted stop token, so a prediction need not match a target frame for frame; the
    /// teacher-forced objective is what training lowers.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var mel = target.Rank == 3 ? Engine.Reshape(target, new[] { target.Shape[1], target.Shape[2] }) : target;
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The teacher-forced objective (§3.7, following Tacotron 2): mel MSE before and after the post-net, plus the
    /// stop-token binary cross-entropy with a positive weight on the single final "stop" frame.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var memory = Encode(tokens);
        // Decoder input: the all-zero first frame, then the ground truth shifted by one.
        var shifted = frames == 1
            ? new Tensor<T>(new[] { 1, channels })
            : Engine.TensorConcatenate(new[]
            {
                new Tensor<T>(new[] { 1, channels }),
                Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { frames - 1, channels }),
            }, 0);
        var (before, stopLogits) = DecodeFrames(shifted, memory);
        var after = Refine(before);

        var stopTarget = new Tensor<T>(new[] { frames });
        stopTarget[frames - 1] = NumOps.One;
        var loss = Engine.TensorAdd(MeanSquared(before, mel), MeanSquared(after, mel));
        return Engine.TensorAdd(loss, WeightedBinaryCrossEntropy(stopLogits, stopTarget, _options.StopTokenPositiveWeight));
    }

    private Tensor<T> MeanSquared(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(prediction, target);
        return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// <c>mean(−[w y log σ(x) + (1 − y) log(1 − σ(x))])</c>, written with the stable softplus
    /// <c>log(1 + e^x) = max(x, 0) + log(1 + e^−|x|)</c>: <c>−log σ(x) = softplus(−x)</c>, <c>−log(1 − σ(x)) = softplus(x)</c>.
    /// </summary>
    private Tensor<T> WeightedBinaryCrossEntropy(Tensor<T> logits, Tensor<T> target, double positiveWeight)
    {
        Tensor<T> Softplus(Tensor<T> x) => Engine.TensorAdd(Engine.ReLU(x),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(x))), NumOps.One)));
        var positive = Engine.TensorMultiplyScalar(Engine.TensorMultiply(target, Softplus(Engine.TensorNegate(logits))),
            NumOps.FromDouble(positiveWeight));
        var negativeMask = Engine.TensorAddScalar(Engine.TensorNegate(target), NumOps.One);
        var negative = Engine.TensorMultiply(negativeMask, Softplus(logits));
        return Engine.ReduceMean(Engine.TensorAdd(positive, negative), new[] { 0 }, keepDims: false);
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
            Name = _useNativeMode ? "TransformerTTS-Native" : "TransformerTTS-ONNX",
            Description = "Neural Speech Synthesis with Transformer Network (Li et al., 2019)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "TransformerTTS";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(TransformerTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
