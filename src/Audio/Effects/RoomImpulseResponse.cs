using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// FiNS — Filtered Noise Shaping network that estimates a time-domain room impulse response
/// directly from reverberant speech.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A strided 1-D convolutional encoder compresses the reverberant waveform to a single latent
/// embedding z, and a filtered noise shaping decoder expands z into the impulse response as two
/// parts. The early component (first E samples) is predicted directly on an extra decoder output
/// channel. The late field is synthesised by passing a noise signal through a trainable bank of M
/// FIR band-pass filters and shaping each band with a predicted time-domain mask,
/// <c>h_l,m(n) = sigmoid(y_m(n)) * s_m(n)</c>. A 1x1 convolution mixes the M late bands and the
/// early component into the final monophonic RIR.
/// </para>
/// <para>
/// <b>For Beginners:</b> Clap in a room and you hear a sharp crack followed by a smeared tail of
/// echoes. Those two halves are different problems. The crack has real structure, so the model
/// predicts it sample by sample. The tail is essentially noise fading out — and fading at a
/// different rate in each frequency band, because rooms absorb treble faster than bass. So instead
/// of predicting 48,000 tail samples, the model takes noise, splits it into 10 frequency bands, and
/// only learns how loud each band should be over time. That is the "filtered noise shaping" idea,
/// and it is why this network can produce a full one-second response from 128 numbers.
/// </para>
/// <para>
/// The forward pass is written out explicitly rather than run as a flat layer walk: the encoder has
/// a parallel residual branch per block, the decoder is FiLM-conditioned on a latent obtained by
/// pooling, and the filterbank is driven by noise rather than by the previous layer's output. None
/// of that is expressible as a sequential chain.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 48000, outputSize: 48000);
/// var model = new RoomImpulseResponse&lt;float&gt;(arch);
/// var rir   = model.Predict(reverbSpeech);   // the estimated impulse response
/// var clean = model.Enhance(reverbSpeech);   // dereverberated audio
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.Denoising)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Filtered Noise Shaping for Time Domain Room Impulse Response Estimation from Reverberant Speech",
    "https://arxiv.org/abs/2107.07503",
    Year = 2021,
    Authors = "Christian J. Steinmetz, Vamsi Krishna Ithapu, Paul Calamia")]
public class RoomImpulseResponse<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly RoomImpulseResponseOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;
    private double _noiseFloor;

    // Ordered VIEWS into Layers, bound by index against the emission order documented on
    // LayerHelper<T>.CreateDefaultRoomImpulseResponseLayers. Rebound by BindLayerViewsFromLayers
    // after deserialization, which replaces every entry in Layers.
    private readonly List<ILayer<T>> _encoderConv = [];
    private readonly List<ILayer<T>> _encoderNorm = [];
    private readonly List<ILayer<T>> _encoderAct = [];
    private readonly List<ILayer<T>> _encoderResConv = [];
    private readonly List<ILayer<T>> _encoderResNorm = [];
    private readonly List<ILayer<T>> _decoderFilmA = [];
    private readonly List<ILayer<T>> _decoderUpsample = [];
    private readonly List<ILayer<T>> _decoderFilmB = [];
    private readonly List<ILayer<T>> _decoderRefine = [];
    private ILayer<T>? _mlp1, _mlp2, _mlp3;
    private ILayer<T>? _maskHead, _noiseFilterbank, _mixConv;

    /// <summary>True once the views above address a full FiNS stack.</summary>
    private bool _bound;

    /// <summary>
    /// The fixed noise realisation driving the filterbank, generated once and persisted.
    /// </summary>
    /// <remarks>
    /// Deliberately FIXED rather than redrawn per call. FiNS shapes a noise signal into the late
    /// field; if that signal were resampled on every forward, two Predict calls on identical input
    /// would disagree, and every determinism, clone-fidelity and finite-difference gradient check
    /// would fail for a reason that has nothing to do with the learned weights. Serialized with the
    /// model so a reload reproduces the same response.
    /// </remarks>
    [Buffer]
    private Tensor<T>? _noiseSignal;

    #endregion

    #region Constructors

    /// <summary>Creates an RIR estimation model in ONNX inference mode.</summary>
    public RoomImpulseResponse(NeuralNetworkArchitecture<T> architecture, string modelPath, RoomImpulseResponseOptions? options = null)
        : base(architecture, new MultiResolutionStftLoss<T>((options ?? new RoomImpulseResponseOptions()).StftFrameSizes))
    {
        _options = options ?? new RoomImpulseResponseOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an RIR estimation model in native training mode.</summary>
    public RoomImpulseResponse(NeuralNetworkArchitecture<T> architecture, RoomImpulseResponseOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        // FiNS trains on the multi-resolution STFT loss, which the paper reports worked best ALONE
        // (no time-domain term). Passing it here replaces the audio base's MeanSquaredErrorLoss
        // default: sample-wise MSE is a poor objective for an impulse response, where a small time
        // shift changes every sample while sounding identical, and it cannot express the per-band
        // decay behaviour the filtered-noise-shaping decoder exists to model.
        : base(architecture, new MultiResolutionStftLoss<T>((options ?? new RoomImpulseResponseOptions()).StftFrameSizes))
    {
        _options = options ?? new RoomImpulseResponseOptions();
        _useNativeMode = true;
        // Build the default optimizer from the model's OWN configured rate. A bare
        // AdamWOptimizer(this) silently takes AdamWOptimizerOptions' global 1e-3 default and drops
        // RoomImpulseResponseOptions.LearningRate (1e-4) — a 10x over-rate on a model whose output
        // head is RIRLength-wide.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<RoomImpulseResponse<T>> CreateAsync(RoomImpulseResponseOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new RoomImpulseResponseOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("rir_estimator", "rir_estimator.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.RIRLength, outputSize: options.RIRLength);
        return new RoomImpulseResponse<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.RIRLength;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        EnhancementStrength = _options.DereverberationStrength;
        // Estimate RIR from audio
        var features = PreprocessAudio(audio);
        Tensor<T> estimatedRIR;
        if (IsOnnxMode && OnnxEncoder is not null) estimatedRIR = OnnxEncoder.Run(features);
        else estimatedRIR = Predict(features);
        // Apply dereverberation using estimated RIR
        return ApplyDereverberation(audio, estimatedRIR);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        EstimateNoiseProfile(reference);
        return Enhance(audio);
    }

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++) _streamingBuffer.Add(audioChunk[i]);

        int frameSize = _options.RIRLength;
        if (_streamingBuffer.Count < frameSize)
            return new Tensor<T>([0]);

        var frame = new Tensor<T>([frameSize]);
        for (int i = 0; i < frameSize; i++) frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, frameSize / 2);
        return Enhance(frame);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        base.ResetState();
        _streamingBuffer = null;
    }

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // Compute mean energy as a simple noise floor estimate
        T sum = NumOps.Zero;
        for (int i = 0; i < noiseOnlyAudio.Length; i++)
        {
            T val = noiseOnlyAudio[i];
            sum = NumOps.Add(sum, NumOps.Multiply(val, val));
        }
        _noiseFloor = NumOps.ToDouble(sum) / Math.Max(1, noiseOnlyAudio.Length);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultRoomImpulseResponseLayers(
                numEncoderBlocks: _options.NumEncoderBlocks,
                encoderKernelSize: _options.EncoderKernelSize,
                encoderStride: _options.EncoderStride,
                encoderMaxChannels: _options.EncoderMaxChannels,
                latentDim: _options.LatentDim,
                numDecoderBlocks: _options.NumDecoderBlocks,
                numNoiseBands: _options.NumNoiseBands,
                noiseFilterOrder: _options.NoiseFilterOrder));
        }

        BindLayerViewsFromLayers();
    }

    /// <summary>
    /// (Re)binds the per-stage views into <c>Layers</c> that <see cref="FiNSForward"/> walks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>Layers</c> is the single ownership graph; the fields below are ordered views into it,
    /// bound by index against the emission order documented on
    /// <c>LayerHelper&lt;T&gt;.CreateDefaultRoomImpulseResponseLayers</c>. Deserialization REPLACES
    /// every entry in <c>Layers</c> with a restored instance, so this must run again afterwards —
    /// a view left pointing at the constructor's layer would make the forward silently ignore every
    /// restored weight while the flat parameter APIs reported the correct values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model keeps shortcuts to particular layers so the forward pass can
    /// wire them up in a custom order. When a saved model is loaded, the layers are rebuilt as new
    /// objects, so those shortcuts have to be re-pointed at the new ones or the model would quietly
    /// keep using its untrained originals.
    /// </para>
    /// </remarks>
    private void BindLayerViewsFromLayers()
    {
        _encoderConv.Clear(); _encoderNorm.Clear(); _encoderAct.Clear();
        _encoderResConv.Clear(); _encoderResNorm.Clear();
        _decoderFilmA.Clear(); _decoderUpsample.Clear();
        _decoderFilmB.Clear(); _decoderRefine.Clear();
        _bound = false;

        // A custom Architecture.Layers chain is NOT the FiNS topology, so leave the views unbound
        // and let the base class walk it sequentially — the documented custom-layers contract.
        int expected = _options.NumEncoderBlocks * LayerHelper<T>.RoomImpulseResponseLayersPerEncoderBlock
                     + _options.NumDecoderBlocks * LayerHelper<T>.RoomImpulseResponseLayersPerDecoderBlock
                     + LayerHelper<T>.RoomImpulseResponseFixedLayers;
        if (Layers.Count != expected) return;

        int idx = 0;
        for (int i = 0; i < _options.NumEncoderBlocks; i++)
        {
            _encoderConv.Add(Layers[idx++]);
            _encoderNorm.Add(Layers[idx++]);
            _encoderAct.Add(Layers[idx++]);
            _encoderResConv.Add(Layers[idx++]);
            _encoderResNorm.Add(Layers[idx++]);
        }

        _mlp1 = Layers[idx++];
        _mlp2 = Layers[idx++];
        _mlp3 = Layers[idx++];

        for (int j = 0; j < _options.NumDecoderBlocks; j++)
        {
            _decoderFilmA.Add(Layers[idx++]);
            _decoderUpsample.Add(Layers[idx++]);
            _decoderFilmB.Add(Layers[idx++]);
            _decoderRefine.Add(Layers[idx++]);
        }

        _maskHead = Layers[idx++];
        _noiseFilterbank = Layers[idx++];
        _mixConv = Layers[idx++];

        // Every layer must have been claimed by exactly one field above. The cursor walks Layers in
        // the order InitializeLayers built them, so a mismatch means the two have drifted apart —
        // and the symptom of that is silent: a field binds to its NEIGHBOUR's layer and the model
        // computes a different function than it was built to, with correct shapes throughout.
        // Checking it here also gives the final idx++ a reader, which is what CodeQL flagged: the
        // increment was dead, so appending a layer would have gone unnoticed rather than tripping.
        if (idx != Layers.Count)
        {
            throw new InvalidOperationException(
                $"{nameof(RoomImpulseResponse<T>)} bound {idx} layers but Layers holds {Layers.Count}. " +
                "InitializeLayers and BindLayerReferences have drifted apart.");
        }

        _bound = true;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        // Unbound means a caller supplied a custom layer chain; honour the sequential contract.
        if (!_bound)
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

        return FiNSForward(input);
    }

    /// <summary>
    /// Routes TRAINING through the same explicit FiNS graph the inference path uses.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without this the tape walks <c>Layers</c> sequentially, which hands the noise filterbank the
    /// decoder's multi-channel activation instead of the single-channel noise signal — observed as
    /// "Input channels (64) must match kernel in_channels (1)" in every training invariant while the
    /// inference tests passed. Any model whose forward is explicit must override BOTH entry points
    /// or it trains a different function from the one it evaluates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A network has to compute the same thing when it is learning as when it
    /// is answering. This makes the training path reuse the exact custom wiring described above.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        ThrowIfDisposed();
        return _bound ? FiNSForward(input) : base.ForwardForTraining(input);
    }

    /// <summary>
    /// Exposes the FiNS stage activations, which a flat layer walk cannot produce for this graph.
    /// </summary>
    public override System.Collections.Generic.Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_bound) return base.GetNamedLayerActivations(input);

        var activations = new System.Collections.Generic.Dictionary<string, Tensor<T>>();
        var eng = Engine;
        int totalSamples = 1;
        for (int d = 0; d < input.Rank; d++) totalSamples *= input.Shape[d];
        var x = eng.Reshape(input, [1, 1, totalSamples]);

        for (int i = 0; i < _encoderConv.Count; i++)
        {
            var main = _encoderAct[i].Forward(ApplyChannelNorm(eng, _encoderNorm[i], _encoderConv[i].Forward(x)));
            var residual = ApplyChannelNorm(eng, _encoderResNorm[i], _encoderResConv[i].Forward(x));
            x = AddOverCommonLength(eng, main, residual);
            activations[$"Encoder_{i}"] = x.Clone();
        }

        // These views are non-null exactly when _bound is true, which the caller above has already
        // checked - but the compiler cannot see that invariant. Assert it once here rather than
        // suppressing the warning at each dereference, so a future change that breaks the pairing
        // fails with a message naming the invariant instead of a NullReferenceException.
        if (_mlp1 is null || _mlp2 is null || _mlp3 is null)
        {
            throw new InvalidOperationException(
                "Layer views are bound (_bound) but the latent MLP stack is null; BindLayerViewsFromLayers did not complete.");
        }

        var z = _mlp3.Forward(_mlp2.Forward(_mlp1.Forward(MeanOverTime(eng, x))));
        activations["Latent"] = z.Clone();
        activations["RIR"] = FiNSForwardSingle(input).Clone();
        return activations;
    }

    /// <summary>
    /// The explicit FiNS graph: time-domain encoder, latent, filtered noise shaping decoder.
    /// </summary>
    private Tensor<T> FiNSForward(Tensor<T> input)
    {
        // A batched call is B independent recordings, each producing its own impulse response.
        // Flattening them into one signal would estimate a single response for the concatenation,
        // which is why BatchConsistency compares a batched call against per-item calls.
        int batch = input.Rank >= 2 ? input.Shape[0] : 1;
        if (batch <= 1) return FiNSForwardSingle(input);

        int perItem = 1;
        for (int d = 1; d < input.Rank; d++) perItem *= input.Shape[d];

        var stacked = new Tensor<T>([batch, _options.RIRLength]);
        var item = new Tensor<T>([perItem]);
        for (int b = 0; b < batch; b++)
        {
            int itemOffset = b * perItem;
            for (int n = 0; n < perItem; n++) item[n] = input[itemOffset + n];
            var response = FiNSForwardSingle(item);
            for (int n = 0; n < _options.RIRLength; n++) stacked[b, n] = response[n];
        }
        return stacked;
    }

    /// <summary>Runs the FiNS graph for a single recording, returning a response of RIRLength.</summary>
    private Tensor<T> FiNSForwardSingle(Tensor<T> input)
    {
        var eng = Engine;
        int totalSamples = 1;
        for (int d = 0; d < input.Rank; d++) totalSamples *= input.Shape[d];
        var x = eng.Reshape(input, [1, 1, totalSamples]);

        // --- Encoder: main branch + parallel 1x1 residual at the same stride (paper Fig. 1) ---
        //
        // NOTE on BatchNorm layout, recorded because it is easy to re-derive and misread.
        // BatchNormalizationLayer resolves a rank-3 input as [C, H, W] with channels in AXIS 0
        // (OnFirstForward), and the features-last flatten below it only triggers when the LAST axis
        // is the feature axis. These activations are [B, C, L] — channels in the middle — so neither
        // path applies and the layer does not compute per-channel statistics here.
        //
        // That is a latent layout gap, NOT the cause of any failing test: this model's invariants
        // are green. An earlier investigation blamed it for Training_ShouldReduceLoss and was WRONG;
        // the real cause was the generated fixture still pinning the spectral model's [1,64,32]
        // input and 4-element target, so a 2048-sample input was scored against a target of the
        // wrong length. Three attempts to "fix" BatchNorm for this case (rank-4 NCHW [B,C,1,L],
        // rank-2 [L,C], and a guarded in-layer permute) each made things strictly worse, which in
        // hindsight is what a remedy for a non-existent problem looks like. Do not repeat them
        // without first reproducing a real defect under the corrected fixture.

        for (int i = 0; i < _encoderConv.Count; i++)
        {
            var main = _encoderAct[i].Forward(ApplyChannelNorm(eng, _encoderNorm[i], _encoderConv[i].Forward(x)));
            var residual = ApplyChannelNorm(eng, _encoderResNorm[i], _encoderResConv[i].Forward(x));
            // Strided conv and strided 1x1 can differ by one frame depending on padding; add over
            // the common span so the residual never dictates the main branch's length.
            x = AddOverCommonLength(eng, main, residual);
        }

        // --- Adaptive average pooling over time, then the 3-layer MLP producing z ---
        // Same invariant as GetNamedLayerActivations: reached only when _bound is true, so these
        // views are non-null. Asserted once so the dereferences below need no suppression.
        if (_mlp1 is null || _mlp2 is null || _mlp3 is null ||
            _maskHead is null || _noiseFilterbank is null || _mixConv is null)
        {
            throw new InvalidOperationException(
                "Layer views are bound (_bound) but the FiNS stack is null; BindLayerViewsFromLayers did not complete.");
        }

        var pooled = MeanOverTime(eng, x);
        var z = _mlp3.Forward(_mlp2.Forward(_mlp1.Forward(pooled)));

        // --- Decoder: upsample from a seed, FiLM-conditioned on z at both stages ---
        int blocks = _decoderUpsample.Count;
        int seedLength = Math.Max(1, _options.RIRLength >> blocks);
        int decoderChannels = Math.Max(_options.NumNoiseBands + 1, _options.EncoderMaxChannels / 8);
        var h = SeedFromLatent(eng, z, decoderChannels, seedLength);

        for (int j = 0; j < blocks; j++)
        {
            h = ApplyFilm(eng, h, _decoderFilmA[j].Forward(z));
            h = _decoderUpsample[j].Forward(h);
            h = ApplyFilm(eng, h, _decoderFilmB[j].Forward(z));
            h = _decoderRefine[j].Forward(h);
        }
        h = ResizeTime(eng, h, _options.RIRLength);

        // --- Heads: M masks plus the early component on the extra channel ---
        var heads = _maskHead.Forward(h);                        // [1, M+1, L]
        int m = _options.NumNoiseBands;
        var masks = eng.TensorNarrow(heads, 1, 0, m);             // [1, M, L]
        var early = eng.TensorNarrow(heads, 1, m, 1);             // [1, 1, L]
        early = ZeroBeyond(eng, early, _options.EarlyResponseLength);

        // --- Filtered noise shaping: sigmoid mask times band-filtered noise ---
        var subbands = _noiseFilterbank.Forward(GetNoiseSignal(_options.RIRLength));  // [1, M, L]
        subbands = ResizeTime(eng, subbands, _options.RIRLength);
        var late = eng.TensorMultiply(eng.Sigmoid(masks), subbands);                   // [1, M, L]

        // --- Mix the M late bands and the early component with a 1x1 convolution ---
        var mixed = _mixConv.Forward(eng.TensorConcatenate([late, early], 1));        // [1, 1, L]
        return eng.Reshape(mixed, [_options.RIRLength]);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Pass the model's configured optimizer. The 2-arg TrainWithTape resolves
            // optimizer: null and falls back to NeuralNetworkBase's lazily-created DEFAULT Adam at
            // its global 1e-3 rate, so _optimizer — including any instance the CALLER supplied to
            // the constructor — was never used for a single step, and _options.LearningRate (1e-4)
            // had no effect on training at all. That 10x over-rate is what produced the measured
            // first-step loss HUMP the generated audio probes had been working around with wider
            // iteration windows (memorization 0.645785 -> 0.926694 at step 2; MoreData still rising
            // between 2 and 5 steps).
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Returns the waveform unchanged: FiNS is a TIME-DOMAIN model whose encoder consumes raw
    /// samples. The previous implementation ran a mel spectrogram here, which is the input
    /// representation of the spectral model this replaced.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    #endregion

    #region FiNS forward helpers

    /// <summary>
    /// Applies a <see cref="BatchNormalizationLayer{T}"/> to a <c>[B, C, L]</c> activation with
    /// per-channel statistics pooled over time — BatchNorm1d semantics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BatchNormalizationLayer resolves RANK-3 input as <c>[C, H, W]</c> with channels in AXIS 0.
    /// These activations are <c>[B, C, L]</c>, so passing them straight in does not merely mis-pool
    /// statistics — it corrupts the shape: a <c>[1, 16, 32]</c> conv output came back as a 16-BATCH,
    /// and the encoder emitted <c>[16, 16, 16]</c> instead of <c>[1, 16, 16]</c>. That went unnoticed
    /// while the surrounding helpers read elements with <c>x[0, c, n]</c> loops, which silently
    /// consume only the first slice.
    /// </para>
    /// <para>
    /// Presented instead as rank-2 <c>[L, C]</c> — time on the batch axis, channels as features in
    /// axis 1 — the layout the layer documents for rank 2, giving exactly BatchNorm1d's per-channel
    /// statistics pooled over time with no degenerate axis.
    /// </para>
    /// </remarks>
    private static Tensor<T> ApplyChannelNorm(IEngine eng, ILayer<T> norm, Tensor<T> x)
    {
        int batch = x.Shape[0], channels = x.Shape[1], length = x.Shape[2];
        var plane = eng.Reshape(x, [channels, length]);
        var timeMajor = eng.TensorPermute(plane, [1, 0]);        // [L, C]
        var normalized = norm.Forward(timeMajor);
        var channelMajor = eng.TensorPermute(normalized, [1, 0]); // [C, L]
        return eng.Reshape(channelMajor, [batch, channels, length]);
    }

    /// <summary>Adds two rank-3 tensors over the time span they share.</summary>
    /// <remarks>
    /// A strided kernel-15 convolution and a strided 1x1 convolution do not always agree on the
    /// final frame, so the residual add is taken over the common length rather than asserting an
    /// exact match — the main branch keeps ownership of the block's output length.
    /// </remarks>
    private static Tensor<T> AddOverCommonLength(IEngine eng, Tensor<T> main, Tensor<T> residual)
    {
        int mainLength = main.Shape[2];
        int residualLength = residual.Shape[2];
        if (mainLength == residualLength) return eng.TensorAdd(main, residual);

        int common = Math.Min(mainLength, residualLength);
        var trimmedResidual = eng.TensorNarrow(residual, 2, 0, common);
        if (mainLength == common) return eng.TensorAdd(main, trimmedResidual);

        // Main is longer: add into its leading span and keep its tail untouched.
        var head = eng.TensorAdd(eng.TensorNarrow(main, 2, 0, common), trimmedResidual);
        var tail = eng.TensorNarrow(main, 2, common, mainLength - common);
        return eng.TensorConcatenate([head, tail], 2);
    }

    /// <summary>Adaptive average pooling over the time axis: [1, C, L] -> [1, C].</summary>
    /// <remarks>
    /// An engine reduction, not an element loop. Everything in this forward must be expressible as
    /// engine ops or the fused compiled path cannot trace it: a tracing compiler records the graph
    /// once and replays it, so any value read by a raw C# loop is baked in as a constant and the
    /// replayed plan silently stops depending on the parameters behind it.
    /// </remarks>
    private static Tensor<T> MeanOverTime(IEngine eng, Tensor<T> x)
        => eng.ReduceMean(x, [2], keepDims: false);

    /// <summary>Broadcasts the latent into the decoder's seed sequence [1, channels, length].</summary>
    /// <remarks>
    /// Pure engine ops so the graph stays traceable. When the decoder is narrower than the latent
    /// the leading channels are taken; when it is wider the latent is repeated to cover them, which
    /// is the tiling the previous element loop expressed with a modulo index.
    /// </remarks>
    private Tensor<T> SeedFromLatent(IEngine eng, Tensor<T> z, int channels, int length)
    {
        int latent = z.Shape[^1];

        var widened = z;
        if (channels > latent)
        {
            int copies = (channels + latent - 1) / latent;
            var tiled = new Tensor<T>[copies];
            for (int c = 0; c < copies; c++) tiled[c] = z;
            widened = eng.TensorConcatenate(tiled, 1);
        }
        if (widened.Shape[^1] != channels) widened = eng.TensorNarrow(widened, 1, 0, channels);

        // [1, channels] -> [1, channels, 1] -> [1, channels, length]
        return eng.TensorBroadcastTo(eng.TensorExpandDims(widened, 2), [1, channels, length]);
    }

    /// <summary>
    /// FiLM conditioning (Perez et al. 2018): a per-channel scale and shift predicted from z.
    /// </summary>
    /// <remarks>
    /// The prediction is split as [scale | shift] — the standard FiLM layout — and applied with
    /// broadcast engine ops. The previous version read individual elements in a C# loop with an
    /// interleaved index, which a tracing compiler bakes in as constants, freezing the decoder's
    /// dependence on z in the compiled plan.
    /// </remarks>
    private static Tensor<T> ApplyFilm(IEngine eng, Tensor<T> h, Tensor<T> filmParams)
    {
        int channels = h.Shape[1];
        int available = filmParams.Shape[^1];
        int half = Math.Min(channels, available / 2);
        if (half <= 0) return h;

        var scale = eng.TensorExpandDims(eng.TensorNarrow(filmParams, 1, 0, half), 2);        // [1, half, 1]
        var shift = eng.TensorExpandDims(eng.TensorNarrow(filmParams, 1, half, half), 2);     // [1, half, 1]

        if (half < channels)
        {
            // Condition the leading channels and pass the remainder through unchanged, all with
            // engine ops so the graph stays static.
            var head = eng.TensorNarrow(h, 1, 0, half);
            var tail = eng.TensorNarrow(h, 1, half, channels - half);
            var conditioned = eng.TensorBroadcastAdd(eng.TensorBroadcastMultiply(head, scale), shift);
            return eng.TensorConcatenate([conditioned, tail], 1);
        }

        return eng.TensorBroadcastAdd(eng.TensorBroadcastMultiply(h, scale), shift);
    }

    /// <summary>Trims or zero-pads the time axis to exactly <paramref name="length"/>.</summary>
    /// <remarks>
    /// The decoder's upsampling factors need not land exactly on RIRLength for every configuration,
    /// so the response is squared up here rather than constraining the option to a power of two.
    /// Uses Engine.Pad rather than an element copy so the graph stays traceable.
    /// </remarks>
    private static Tensor<T> ResizeTime(IEngine eng, Tensor<T> x, int length)
    {
        int current = x.Shape[2];
        if (current == length) return x;
        if (current > length) return eng.TensorNarrow(x, 2, 0, length);
        return eng.Pad(x, 0, 0, 0, length - current, MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>Zeroes every sample past <paramref name="keep"/> (paper: h_d(n) = 0 for n &gt; E).</summary>
    private static Tensor<T> ZeroBeyond(IEngine eng, Tensor<T> x, int keep)
    {
        int length = x.Shape[2];
        if (keep >= length) return x;
        var head = eng.TensorNarrow(x, 2, 0, keep);
        var zeros = new Tensor<T>([x.Shape[0], x.Shape[1], length - keep]);
        return eng.TensorConcatenate([head, zeros], 2);
    }

    /// <summary>
    /// The fixed white-noise realisation shaped into the late field, created on first use.
    /// </summary>
    private Tensor<T> GetNoiseSignal(int length)
    {
        if (_noiseSignal is not null && _noiseSignal.Shape[^1] == length) return _noiseSignal;

        // Fixed seed: the noise is part of the model, not a per-call random draw. See _noiseSignal.
        var rng = RandomHelper.CreateSeededRandom(20210715);
        var noise = new Tensor<T>([1, 1, length]);
        for (int n = 0; n < length; n++)
            noise[0, 0, n] = NumOps.FromDouble(rng.NextDouble() * 2.0 - 1.0);
        _noiseSignal = noise;
        return noise;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RoomImpulseResponse-Native" : "RoomImpulseResponse-ONNX",
            Description = "Neural Room Impulse Response estimation for dereverberation (2023-2024)",
            Complexity = _options.NumEncoderBlocks
        };
        m.AdditionalInfo["RIRLength"] = _options.RIRLength.ToString();
        m.AdditionalInfo["EncoderMaxChannels"] = _options.EncoderMaxChannels.ToString();
        m.AdditionalInfo["LatentDim"] = _options.LatentDim.ToString();
        m.AdditionalInfo["NumNoiseBands"] = _options.NumNoiseBands.ToString();
        m.AdditionalInfo["EarlyResponseLength"] = _options.EarlyResponseLength.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.EncoderMaxChannels);
        w.Write(_options.NumEncoderBlocks); w.Write(_options.RIRLength);
        w.Write(_options.LatentDim); w.Write(_options.NumNoiseBands);
        w.Write(_options.EarlyResponseLength); w.Write(_options.NumDecoderBlocks);
        w.Write(_options.DereverberationStrength); w.Write(_options.RT60WindowSeconds);
        w.Write(_options.NoiseFilterOrder);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.EncoderMaxChannels = r.ReadInt32();
        _options.NumEncoderBlocks = r.ReadInt32(); _options.RIRLength = r.ReadInt32();
        _options.LatentDim = r.ReadInt32(); _options.NumNoiseBands = r.ReadInt32();
        _options.EarlyResponseLength = r.ReadInt32(); _options.NumDecoderBlocks = r.ReadInt32();
        _options.DereverberationStrength = r.ReadDouble(); _options.RT60WindowSeconds = r.ReadDouble();
        _options.NoiseFilterOrder = r.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);

        // The base deserializer has just replaced Layers with the restored instances. Rebind the
        // per-stage views so FiNSForward consumes those weights and not the constructor's layers.
        BindLayerViewsFromLayers();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new RoomImpulseResponse<T>(Architecture, mp, _options);
        return new RoomImpulseResponse<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> ApplyDereverberation(Tensor<T> audio, Tensor<T> estimatedRIR)
    {
        // Simplified spectral dereverberation using estimated RIR
        var output = new Tensor<T>([audio.Length]);
        double strength = EnhancementStrength;
        for (int i = 0; i < audio.Length; i++)
        {
            double clean = NumOps.ToDouble(audio[i]);
            // Subtract estimated reverb contribution
            double reverbContrib = 0;
            for (int j = 1; j < Math.Min(estimatedRIR.Length, i); j++)
            {
                double rirVal = NumOps.ToDouble(estimatedRIR[j]);
                double audioVal = NumOps.ToDouble(audio[i - j]);
                reverbContrib += rirVal * audioVal;
            }
            double dereverbed = clean - strength * reverbContrib;
            output[i] = NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, dereverbed)));
        }
        return output;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RoomImpulseResponse<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
