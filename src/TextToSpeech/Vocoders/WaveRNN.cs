using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveRNN: efficient neural audio synthesis with a single recurrent layer whose state is split to predict the coarse
/// and the fine byte of each 16-bit sample through a dual softmax.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Efficient Neural Audio Synthesis" (Kalchbrenner et al., ICML 2018).</para>
/// <para>
/// Training maximizes <c>log P(c_t) + log P(f_t | c_t)</c> of every sample of a 960-sample sequence with full
/// back-propagation through time (§2, §5), the inputs being the true previous sample and current coarse byte. Synthesis
/// runs the cell once per sample: the coarse half of the state does not see the current coarse byte, so the coarse byte
/// is drawn first and the fine byte second from the state that has seen it. Magnitude pruning of the recurrent weights
/// (§3, Sparse WaveRNN) runs when a sparsity target is set.
/// </para>
/// <para><b>For Beginners:</b> WaveRNN writes audio one sample at a time with a small recurrent network: it first
/// chooses the sample's rough level (the top 8 bits), then refines it (the bottom 8 bits).</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a WaveRNN vocoder for efficient autoregressive synthesis
/// // with dual softmax output and subscale sample generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new WaveRNN&lt;double&gt;(architecture, "wavernn.onnx");
///
/// // Training mode with native layers
/// var trainModel = new WaveRNN&lt;double&gt;(architecture, new WaveRNNOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Efficient Neural Audio Synthesis",
    "https://arxiv.org/abs/1802.08435",
    Year = 2018,
    Authors = "Kalchbrenner et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, ReferenceBatchSize = 32,
                Source = "The paper states no optimizer (500k steps, Sec. 3.1); fatchord/WaveRNN hparams.py: Adam at 1e-4, batch 32, gradient norm clipped to 4.")]
public partial class WaveRNN<T> : SegmentVocoderBase<T>
{
    private WaveRnnCore<T>? _core;
    private TacotronSpectrogram? _features;
    private Tensor<T>? _mask;

    /// <summary>Creates a WaveRNN that runs an exported ONNX graph.</summary>
    public WaveRNN(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveRNNOptions? options = null)
        : base(architecture, modelPath, options ?? new WaveRNNOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable WaveRNN.</summary>
    public WaveRNN(NeuralNetworkArchitecture<T> architecture, WaveRNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new WaveRNNOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private WaveRNNOptions PaperOptions => (WaveRNNOptions)VocoderSettings;

    private WaveRnnCore<T> Core => _core ?? throw new InvalidOperationException("The WaveRNN network has not been created.");

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleScales.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SequenceSamples;

    /// <inheritdoc />
    /// <remarks>The teacher-forced likelihood has no random draws.</remarks>
    protected override int EvaluationDraws => 1;

    /// <inheritdoc />
    protected override double GradientClipNorm => PaperOptions.MaxGradientNorm;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateNetwork()
    {
        var o = PaperOptions;
        if (o.DropoutRate != 0.0)
            throw new ArgumentException("WaveRNN's published recurrent network has no dropout; DropoutRate must be zero.", nameof(o.DropoutRate));
        if (o.RnnDim <= 0 || o.RnnDim % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(o.RnnDim), "WaveRNN needs two equal nonempty state halves.");
        if (o.UpsampleScales.Length == 0 || o.UpsampleScales.Any(s => s <= 0))
            throw new ArgumentException("WaveRNN needs positive conditioning upsampling factors.", nameof(o.UpsampleScales));
        if (o.SequenceSamples <= 0) throw new ArgumentOutOfRangeException(nameof(o.SequenceSamples));
        if (double.IsNaN(o.SparsityTarget) || o.SparsityTarget < 0 || o.SparsityTarget > 1)
            throw new ArgumentOutOfRangeException(nameof(o.SparsityTarget));
        if (o.PruningStartStep < 0 || o.PruningSteps <= 0 || o.PruningInterval <= 0)
            throw new ArgumentException("The pruning schedule needs a nonnegative start and positive duration and interval.");
        if (o.PruneBlockRows <= 0 || o.PruneBlockColumns <= 0 ||
            o.RnnDim % o.PruneBlockRows != 0 || o.RnnDim % o.PruneBlockColumns != 0)
            throw new ArgumentException("Pruning blocks must tile each recurrent gate matrix.");
        _core = new WaveRnnCore<T>(Engine, o.RnnDim, o.MelChannels, o.UpsampleScales, 256);
        _features = new TacotronSpectrogram(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency);
        return _core.Layers;
    }

    /// <inheritdoc />
    protected override (Tensor<T> Condition, Tensor<T> Audio) TrainingPair(Tensor<T> condition, Tensor<T> audio, Random random)
    {
        audio = Flat(audio);
        // Section 5.1 uses 960 samples, which is not a multiple of the 300-sample feature hop.
        // Keep every target sample; the last conditioning frame can cover only part of the sequence.
        int hop = UpsampleFactor;
        int samples = Math.Min(PaperOptions.SequenceSamples, Math.Min(audio.Length, checked(condition.Shape[2] * hop)));
        if (samples <= 0) throw new ArgumentException("WaveRNN training needs nonempty aligned audio and conditioning.");
        int frames = 1 + (samples - 1) / hop;
        int lastStart = Math.Min(condition.Shape[2] - frames, (audio.Length - samples) / hop);
        int start = lastStart > 0 ? random.Next(lastStart + 1) : 0;
        return (Engine.TensorSlice(condition, new[] { 0, 0, start }, new[] { 1, condition.Shape[1], frames }),
            Engine.TensorSlice(audio, new[] { start * hop }, new[] { samples }));
    }

    /// <inheritdoc />
    protected override Tensor<T> EvaluationAudio(Tensor<T> condition, Tensor<T> target)
    {
        if (target.Length > checked(condition.Shape[2] * UpsampleFactor))
            throw new ArgumentException("The conditioning must cover every evaluated waveform sample.", nameof(condition));
        return target;
    }

    /// <inheritdoc />
    /// <remarks>Tacotron 2's log-mel spectrogram.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var features = _features ?? throw new InvalidOperationException("The WaveRNN network has not been created.");
        // Computed in double and narrowed once, whatever T is.
        var rows = features.LogMel(TensorCopyHelper.CastLogical<T, double>(Engine.Reshape(audio, new[] { audio.Length })));
        int frames = rows.Shape[0], bands = rows.Shape[1];                                     // [frames, bands]
        return TensorCopyHelper.CastLogical<double, T>(
            Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, bands, frames }));
    }

    /// <summary>The 16-bit value (0..65535) of a sample in [−1, 1].</summary>
    internal static int ToSixteenBit(double x) => (int)Math.Max(0, Math.Min(65535, Math.Round((Math.Max(-1, Math.Min(1, x)) + 1) * 32767.5)));

    /// <summary>The sample in [−1, 1] of a 16-bit value.</summary>
    internal static double FromSixteenBit(int s) => s / 32767.5 - 1;

    /// <summary>The coarse (high) and fine (low) bytes of a waveform <c>[samples]</c>.</summary>
    public (int[] Coarse, int[] Fine) Split(Tensor<T> audio)
    {
        // ToSixteenBit per sample, in double whatever T is (65535 is not representable in every T): round half to
        // even of (clamp(x, -1, 1) + 1) * 32767.5, then the high and low bytes.
        var x = Engine.Reshape(TensorCopyHelper.CastLogical<T, double>(audio), new[] { audio.Length });
        var sixteen = Engine.TensorClamp(
            Engine.TensorRound(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(Engine.TensorClamp(x, -1.0, 1.0), 1.0), 32767.5)),
            0.0, 65535.0);
        var high = Engine.TensorFloor(Engine.TensorDivideScalar(sixteen, 256.0));
        var low = Engine.TensorSubtract(sixteen, Engine.TensorMultiplyScalar(high, 256.0));
        return (TensorCopyHelper.CastLogical<double, int>(high).ToArray(), TensorCopyHelper.CastLogical<double, int>(low).ToArray());
    }

    // Bytes b as b / 127.5 - 1 in T, shaped [1, channels, count] (§2: "encoded as scalars in [0, 255] and scaled to
    // the interval [−1, 1]").
    private static Tensor<T> ScaledBytes(int[] bytes, int channels)
    {
        var engine = AiDotNetEngine.Current;
        var values = engine.TensorAddScalar(
            engine.TensorDivideScalar(new Tensor<int>(bytes, new[] { 1, channels, bytes.Length / channels }).Cast<double>(), 127.5), -1.0);
        return TensorCopyHelper.CastLogical<double, T>(values);
    }

    private Tensor<T> Condition(Tensor<T> mel, int samples)
        => Engine.TensorSlice(Core.Upsample(mel), new[] { 0, 0, 0 }, new[] { 1, mel.Shape[1], samples });

    /// <summary>The coarse and fine logits <c>[1, 256, samples]</c> of every sample of <paramref name="audio"/>
    /// given its previous samples and coarse byte (teacher forcing; the sample before the first is silence).</summary>
    public (Tensor<T> Coarse, Tensor<T> Fine) Logits(Tensor<T> mel, Tensor<T> audio)
        => LogitsFromBytes(mel, QuantizedBytes(Engine, audio));

    // A same-T discrete data node keeps the original double quantizer off the graph while declaring
    // its current waveform dependency. No derivative flows through byte rounding or teacher forcing.
    internal static Tensor<T> QuantizedBytes(AiDotNet.Tensors.Engines.IEngine engine, Tensor<T> audio)
    {
        void Compute(AiDotNet.Tensors.Engines.IEngine replayEngine, Tensor<T> output)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            using var noDoubleGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
            var x = replayEngine.Reshape(TensorCopyHelper.CastLogical<T, double>(audio), new[] { audio.Length });
            var sixteen = replayEngine.TensorClamp(replayEngine.TensorRound(replayEngine.TensorMultiplyScalar(
                replayEngine.TensorAddScalar(replayEngine.TensorClamp(x, -1.0, 1.0), 1.0), 32767.5)), 0.0, 65535.0);
            var high = replayEngine.TensorFloor(replayEngine.TensorDivideScalar(sixteen, 256.0));
            var low = replayEngine.TensorSubtract(sixteen, replayEngine.TensorMultiplyScalar(high, 256.0));
            var value = TensorCopyHelper.CastLogical<double, T>(replayEngine.TensorStack(new[] { high, low }, 0));
            AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine, value, output);
        }
        var shape = new[] { 2, audio.Length };
        var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.Current;
        if (scope is null)
        {
            var value = new Tensor<T>(shape);
            Compute(engine, value);
            return value;
        }
        return scope.RecordUnary(AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom,
            "WaveRnnQuantizedBytes", audio, shape, Compute, static (_, _, _, _, _, _) => { });
    }

    private static Tensor<T> ScaleByteTensor(AiDotNet.Tensors.Engines.IEngine engine, Tensor<T> bytes)
    {
        void Compute(AiDotNet.Tensors.Engines.IEngine replayEngine, Tensor<T> output)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            using var noDoubleGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
            var values = replayEngine.TensorAddScalar(
                replayEngine.TensorDivideScalar(TensorCopyHelper.CastLogical<T, double>(bytes), 127.5), -1.0);
            AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine,
                TensorCopyHelper.CastLogical<double, T>(values), output);
        }
        var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.Current;
        if (scope is null)
        {
            var value = new Tensor<T>(bytes._shape);
            Compute(engine, value);
            return value;
        }
        return scope.RecordUnary(AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom,
            "WaveRnnScaledBytes", bytes, bytes._shape, Compute, static (_, _, _, _, _, _) => { });
    }

    private (Tensor<T> Coarse, Tensor<T> Fine) LogitsFromBytes(Tensor<T> mel, Tensor<T> bytes)
    {
        mel = MelInput(mel);
        int n = bytes.Shape[1];
        Tensor<T> Row(int index) => Engine.Reshape(Engine.TensorSlice(bytes, new[] { index, 0 }, new[] { 1, n }),
            new[] { 1, 1, n });
        var coarse = Row(0);
        var fine = Row(1);
        int silence = ToSixteenBit(0);
        Tensor<T> Previous(Tensor<T> row, int initial)
        {
            if (n == 0) return row;
            var first = Tensor<T>.CreateDefault(new[] { 1, 1, 1 }, NumOps.FromDouble(initial));
            return n == 1 ? first : Engine.TensorConcatenate(new[]
                { first, Engine.TensorSlice(row, new[] { 0, 0, 0 }, new[] { 1, 1, n - 1 }) }, 2);
        }
        var scaled = ScaleByteTensor(Engine, Engine.TensorConcatenate(new[]
            { Previous(coarse, silence >> 8), Previous(fine, silence & 255), coarse }, 1));
        var previous = Engine.TensorNarrow(scaled, 1, 0, 2);
        var current = Engine.TensorNarrow(scaled, 1, 2, 1);
        var inputs = Core.Inputs(previous, current, Condition(mel, n));
        var h = new Tensor<T>(new[] { 1, Core.Hidden, 1 });
        var states = new Tensor<T>[n];
        for (int t = 0; t < n; t++)
        {
            h = Core.Step(h, Engine.TensorSlice(inputs, new[] { 0, 0, t }, new[] { 1, 3 * Core.Hidden, 1 }));
            states[t] = h;
        }
        var all = Engine.TensorConcatenate(states, 2);
        return (Core.CoarseLogits(all), Core.FineLogits(all));
    }

    /// <inheritdoc />
    /// <remarks>The negative log-likelihood per sample, −log P(c_t) − log P(f_t | c_t).</remarks>
    protected override Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        var bytes = QuantizedBytes(Engine, audio);
        var (coarseLogits, fineLogits) = LogitsFromBytes(mel, bytes);
        int n = audio.Length;
        Tensor<T> Targets(int row) => Engine.Reshape(Engine.TensorSlice(bytes, new[] { row, 0 }, new[] { 1, n }),
            new[] { 1, 1, n });
        Tensor<T> CrossEntropy(Tensor<T> logits, Tensor<T> targets)
        {
            var shape = logits.Shape.ToArray();
            var classes = Engine.Reshape(Engine.TensorArange<T>(0, 256), new[] { 1, 256, 1 });
            var oneHot = Engine.TensorEquals(Engine.TensorBroadcastTo(classes, shape),
                Engine.TensorBroadcastTo(targets, shape));
            return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(oneHot,
                LogSoftmaxOverClasses(logits)), new[] { 0, 1, 2 }, keepDims: false), NumOps.FromDouble(-1.0 / n));
        }
        return Engine.TensorAdd(CrossEntropy(coarseLogits, Targets(0)), CrossEntropy(fineLogits, Targets(1)));
    }

    /// <inheritdoc />
    /// <remarks>Per sample: the coarse byte from the coarse half (blind to it), then the fine byte from the state that
    /// has seen it.</remarks>
    protected override Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        int samples = mel.Shape[2] * UpsampleFactor;
        var condition = Condition(mel, samples);
        var core = Core;
        var h = new Tensor<T>(new[] { 1, core.Hidden, 1 });
        var wave = new double[samples];
        int silence = ToSixteenBit(0), previousCoarse = silence >> 8, previousFine = silence & 255;
        var blindCoarse = new Tensor<T>(new[] { 1, 1, 1 });
        for (int t = 0; t < samples; t++)
        {
            var previous = ScaledBytes(new[] { previousCoarse, previousFine }, 2);
            var column = Engine.TensorSlice(condition, new[] { 0, 0, t }, new[] { 1, mel.Shape[1], 1 });
            // The current coarse byte reaches only the fine half, so any value gives the coarse half's state.
            var blind = core.Step(h, core.Inputs(previous, blindCoarse, column));
            int c = SampleClass(core.CoarseLogits(blind), random);
            h = core.Step(h, core.Inputs(previous, ScaledBytes(new[] { c }, 1), column));
            int f = SampleClass(core.FineLogits(h), random);
            wave[t] = FromSixteenBit((c << 8) | f);
            previousCoarse = c;
            previousFine = f;
        }
        return new Tensor<double>(wave, new[] { 1, 1, samples }).Cast<T>();
    }

    /// <summary>The fraction of the recurrent weights that are zero.</summary>
    public double RecurrentSparsity
    {
        get
        {
            var kernel = Core.Recurrent.Kernel();
            return (kernel.Length - Engine.TensorCountNonzero(kernel)) / (double)kernel.Length;
        }
    }

    /// <inheritdoc />
    /// <remarks>Gradual magnitude pruning of each gate's recurrent matrix (§3.1; Zhu and Gupta 2017): the mask is
    /// recomputed every <see cref="WaveRNNOptions.PruningInterval"/> steps to prune the fraction
    /// <c>z = Z(1 − (1 − (t − t₀)/S)³)</c> of blocks with the smallest summed magnitude, and reapplied after every
    /// step.</remarks>
    protected override void AfterTrainingStep(int step)
    {
        var o = PaperOptions;
        if (o.SparsityTarget <= 0 || step < o.PruningStartStep)
            return;
        int since = step - o.PruningStartStep;
        if (_mask is null || since % o.PruningInterval == 0 || since == o.PruningSteps)
        {
            double progress = Math.Min(1.0, since / (double)o.PruningSteps);
            double z = o.SparsityTarget * (1 - Math.Pow(1 - progress, 3));
            // The mask is reapplied on every step until the next interval, so it is built outside any caller arena
            // (the facade trains inside one): an arena reset would reissue its storage and prune unintended weights.
            using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
            {
                _mask = PruningMask(z);
            }
        }
        Core.Recurrent.MaskKernel(_mask);
    }

    // The mask keeping all but the fraction z of each gate matrix's blocks with the smallest summed |w|.
    private Tensor<T> PruningMask(double z)
    {
        var o = PaperOptions;
        var kernel = Core.Recurrent.Kernel();
        int hidden = Core.Hidden, br = o.PruneBlockRows, bc = o.PruneBlockColumns;
        if (hidden % br != 0 || hidden % bc != 0)
            throw new InvalidOperationException($"The pruning block {br}x{bc} must tile the {hidden}x{hidden} gate matrices.");
        int rowsOfBlocks = hidden / br, colsOfBlocks = hidden / bc, blocks = rowsOfBlocks * colsOfBlocks;
        int pruned = (int)Math.Floor(z * blocks);

        // Summed |w| of each block: the kernel viewed as [gate, block row, row, block column, column]
        int[] blockShape = { 3, rowsOfBlocks, br, colsOfBlocks, bc };
        var magnitudes = Engine.ReduceSum(
            Engine.Reshape(Engine.TensorAbs(TensorCopyHelper.CastLogical<T, double>(kernel)), blockShape), new[] { 2, 4 }, false).ToArray();
        var keep = new int[3 * blocks];
        for (int gate = 0; gate < 3; gate++)
        {
            int offset = gate * blocks;
            var order = Enumerable.Range(0, blocks).OrderBy(b => magnitudes[offset + b]).ThenBy(b => b);
            var prunedBlocks = new HashSet<int>(order.Take(pruned));
            for (int b = 0; b < blocks; b++) keep[offset + b] = prunedBlocks.Contains(b) ? 0 : 1;
        }

        var blockMask = new Tensor<int>(keep, new[] { 3, rowsOfBlocks, 1, colsOfBlocks, 1 }).Cast<T>();
        return Engine.Reshape(Engine.TensorBroadcastTo(blockMask, blockShape), kernel._shape);
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
    {
        var optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this,
            learningRateOverride: PaperOptions.LearningRate, weightDecayOverride: PaperOptions.WeightDecay)
            ?? throw new InvalidOperationException("WaveRNN's native Adam recipe could not be created.");
        if (optimizer.GetOptions() is GradientBasedOptimizerOptions<T, Tensor<T>, Tensor<T>> settings)
            settings.EnableGradientClipping = false; // The model applies its single global norm clip.
        return optimizer;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "WaveRNN-ONNX" : "WaveRNN-Native",
            Description = "Efficient Neural Audio Synthesis (Kalchbrenner et al., 2018)",
            FeatureCount = o.MelChannels,
            Complexity = o.RnnDim,
        };
        m.AdditionalInfo["Architecture"] = o.SparsityTarget > 0 ? "Sparse WaveRNN" : "WaveRNN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
