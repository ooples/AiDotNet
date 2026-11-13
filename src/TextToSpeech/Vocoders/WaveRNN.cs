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
        var rows = _features!.LogMel(Engine.Reshape(audio, new[] { audio.Length }));   // [frames, bands]
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });
    }

    /// <summary>The 16-bit value (0..65535) of a sample in [−1, 1].</summary>
    internal static int ToSixteenBit(double x) => (int)Math.Max(0, Math.Min(65535, Math.Round((Math.Max(-1, Math.Min(1, x)) + 1) * 32767.5)));

    /// <summary>The sample in [−1, 1] of a 16-bit value.</summary>
    internal static double FromSixteenBit(int s) => s / 32767.5 - 1;

    /// <summary>The coarse (high) and fine (low) bytes of a waveform <c>[samples]</c>.</summary>
    /// <remarks>The 16-bit value round((x + 1) · 32767.5) is assembled in integers from y = x · 32767.5, which stays
    /// within ±32767.5 (representable in every T, half precision included): with n = ⌊y⌋, y + 32767.5 lies in
    /// (n + 32767, n + 32768] and rounds (half to even) to n + 32768, except at a tie (y = n) with n odd.</remarks>
    public (int[] Coarse, int[] Fine) Split(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var y = Engine.TensorMultiplyScalar(Engine.TensorClamp(Engine.Reshape(audio, new[] { audio.Length }),
            NumOps.FromDouble(-1), NumOps.One), NumOps.FromDouble(32767.5));
        var floor = Engine.TensorFloor(y);
        var whole = floor.Cast<int>().ToArray();
        var exact = Engine.TensorEquals(y, floor).Cast<int>().ToArray();
        var coarse = new int[whole.Length];
        var fine = new int[whole.Length];
        for (int i = 0; i < whole.Length; i++)
        {
            int n = whole[i];
            int s = Math.Max(0, Math.Min(65535, n + 32768 - (exact[i] != 0 && (n & 1) != 0 ? 1 : 0)));
            coarse[i] = s >> 8;
            fine[i] = s & 255;
        }
        return (coarse, fine);
    }

    // Byte b scaled to [−1, 1] (§2: "encoded as scalars in [0, 255] and scaled to the interval [−1, 1]"),
    // gathered from the 256-entry table of b / 127.5 − 1.
    private Tensor<T> ScaledBytes(int[] bytes, int[] shape)
    {
        var table = Engine.TensorAddScalar(Engine.TensorDivideScalar(Engine.TensorArange<T>(0, 256, 1), NumOps.FromDouble(127.5)),
            NumOps.FromDouble(-1));
        return Engine.Reshape(Engine.TensorIndexSelect(table, new Tensor<int>(bytes, new[] { bytes.Length }), 0), shape);
    }

    private Tensor<T> Condition(Tensor<T> mel, int samples)
        => Engine.TensorSlice(_core!.Upsample(mel), new[] { 0, 0, 0 }, new[] { 1, mel.Shape[1], samples });

    /// <summary>The coarse and fine logits <c>[1, 256, samples]</c> of every sample of <paramref name="audio"/>
    /// given its previous samples and coarse byte (teacher forcing; the sample before the first is silence).</summary>
    public (Tensor<T> Coarse, Tensor<T> Fine) Logits(Tensor<T> mel, Tensor<T> audio)
    {
        mel = MelInput(mel);
        var (coarse, fine) = Split(Flat(audio));
        int n = coarse.Length;
        int silence = ToSixteenBit(0);
        // The previous sample's bytes: silence, then the waveform shifted by one.
        var previousBytes = new int[2 * n];
        previousBytes[0] = silence >> 8;
        previousBytes[n] = silence & 255;
        Array.Copy(coarse, 0, previousBytes, 1, n - 1);
        Array.Copy(fine, 0, previousBytes, n + 1, n - 1);
        var previous = ScaledBytes(previousBytes, new[] { 1, 2, n });
        var current = ScaledBytes(coarse, new[] { 1, 1, n });
        var inputs = _core!.Inputs(previous, current, Condition(mel, n));
        var h = new Tensor<T>(new[] { 1, _core.Hidden, 1 });
        var states = new Tensor<T>[n];
        for (int t = 0; t < n; t++)
        {
            h = _core.Step(h, Engine.TensorSlice(inputs, new[] { 0, 0, t }, new[] { 1, 3 * _core.Hidden, 1 }));
            states[t] = h;
        }
        var all = Engine.TensorConcatenate(states, 2);
        return (_core.CoarseLogits(all), _core.FineLogits(all));
    }

    /// <inheritdoc />
    /// <remarks>The negative log-likelihood per sample, −log P(c_t) − log P(f_t | c_t).</remarks>
    protected override Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        var (coarse, fine) = Split(audio);
        var (coarseLogits, fineLogits) = Logits(mel, audio);
        return Engine.TensorAdd(ClassCrossEntropy(coarseLogits, coarse), ClassCrossEntropy(fineLogits, fine));
    }

    /// <inheritdoc />
    /// <remarks>Per sample: the coarse byte from the coarse half (blind to it), then the fine byte from the state that
    /// has seen it.</remarks>
    protected override Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        int samples = mel.Shape[2] * UpsampleFactor;
        var condition = Condition(mel, samples);
        var core = _core!;
        var h = new Tensor<T>(new[] { 1, core.Hidden, 1 });
        var coarse = new int[samples];
        var fine = new int[samples];
        int silence = ToSixteenBit(0), previousCoarse = silence >> 8, previousFine = silence & 255;
        for (int t = 0; t < samples; t++)
        {
            var previous = ScaledBytes(new[] { previousCoarse, previousFine }, new[] { 1, 2, 1 });
            var column = Engine.TensorSlice(condition, new[] { 0, 0, t }, new[] { 1, mel.Shape[1], 1 });
            // The current coarse byte reaches only the fine half, so any value gives the coarse half's state.
            var blind = core.Step(h, core.Inputs(previous, new Tensor<T>(new[] { 1, 1, 1 }), column));
            int c = SampleClass(core.CoarseLogits(blind), random);
            h = core.Step(h, core.Inputs(previous, ScaledBytes(new[] { c }, new[] { 1, 1, 1 }), column));
            int f = SampleClass(core.FineLogits(h), random);
            coarse[t] = c;
            fine[t] = f;
            previousCoarse = c;
            previousFine = f;
        }
        // (256 c + f) / 32767.5 - 1, as (128 c + f / 2) / 16383.75 - 1: halving both sides is exact, and the halved
        // value stays below half precision's 65504.
        var halved = Engine.TensorAdd(Engine.TensorMultiplyScalar(new Tensor<int>(coarse, new[] { samples }).Cast<T>(), NumOps.FromDouble(128)),
            Engine.TensorMultiplyScalar(new Tensor<int>(fine, new[] { samples }).Cast<T>(), NumOps.FromDouble(0.5)));
        var wave = Engine.TensorAddScalar(Engine.TensorDivideScalar(halved, NumOps.FromDouble(16383.75)), NumOps.FromDouble(-1));
        return Engine.Reshape(wave, new[] { 1, 1, samples });
    }

    /// <summary>The fraction of the recurrent weights that are zero.</summary>
    public double RecurrentSparsity
    {
        get
        {
            var kernel = _core!.Recurrent.Kernel();
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
            _mask = PruningMask(z);
        }
        _core!.Recurrent.MaskKernel(_mask);
    }

    // The mask keeping all but the fraction z of each gate matrix's blocks with the smallest summed |w|.
    private Tensor<T> PruningMask(double z)
    {
        var o = PaperOptions;
        var kernel = _core!.Recurrent.Kernel();
        int hidden = _core.Hidden, br = o.PruneBlockRows, bc = o.PruneBlockColumns;
        if (hidden % br != 0 || hidden % bc != 0)
            throw new InvalidOperationException($"The pruning block {br}x{bc} must tile the {hidden}x{hidden} gate matrices.");
        int rowsOfBlocks = hidden / br, colsOfBlocks = hidden / bc, blocks = rowsOfBlocks * colsOfBlocks;
        int pruned = (int)Math.Floor(z * blocks);
        // Summed |w| of each block, [gate, block row, block column] in block order.
        var blocked = Engine.Reshape(Engine.TensorAbs(kernel), new[] { 3, rowsOfBlocks, br, colsOfBlocks, bc });
        var sums = Engine.ReduceSum(blocked, new[] { 2, 4 }, false).ToArray();
        var prune = new int[3 * blocks];
        for (int gate = 0; gate < 3; gate++)
        {
            int offset = gate * blocks;
            foreach (int b in Enumerable.Range(0, blocks).OrderBy(b => sums[offset + b]).ThenBy(b => b).Take(pruned))
                prune[offset + b] = 1;
        }
        var keep = Engine.TensorAddScalar(Engine.TensorNegate(
            new Tensor<int>(prune, new[] { 3, rowsOfBlocks, 1, colsOfBlocks, 1 }).Cast<T>()), NumOps.One);
        return Engine.Reshape(Engine.TensorBroadcastTo(keep, new[] { 3, rowsOfBlocks, br, colsOfBlocks, bc }), kernel._shape);
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
