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
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 32,
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
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The upsampling scales ({string.Join("x", o.UpsampleScales)}) must multiply to the hop ({o.HopSize}).");
        _core = new WaveRnnCore<T>(Engine, o.RnnDim, o.MelChannels, o.UpsampleScales, 256);
        _features = new TacotronSpectrogram(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency);
        return _core.Layers;
    }

    /// <inheritdoc />
    /// <remarks>Tacotron 2's log-mel spectrogram.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(audio[i]);
        var rows = _features!.LogMel(samples);
        int frames = rows.GetLength(0), bands = rows.GetLength(1);
        var mel = new Tensor<T>(new[] { 1, bands, frames });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < bands; m++) mel[0, m, f] = NumOps.FromDouble(rows[f, m]);
        return mel;
    }

    /// <summary>The 16-bit value (0..65535) of a sample in [−1, 1].</summary>
    internal static int ToSixteenBit(double x) => (int)Math.Max(0, Math.Min(65535, Math.Round((Math.Max(-1, Math.Min(1, x)) + 1) * 32767.5)));

    /// <summary>The sample in [−1, 1] of a 16-bit value.</summary>
    internal static double FromSixteenBit(int s) => s / 32767.5 - 1;

    // A byte scaled to [−1, 1] (§2: "encoded as scalars in [0, 255] and scaled to the interval [−1, 1]").
    private static double Scaled(int b) => b / 127.5 - 1;

    /// <summary>The coarse (high) and fine (low) bytes of a waveform <c>[samples]</c>.</summary>
    public (int[] Coarse, int[] Fine) Split(Tensor<T> audio)
    {
        var coarse = new int[audio.Length];
        var fine = new int[audio.Length];
        for (int i = 0; i < audio.Length; i++)
        {
            int s = ToSixteenBit(NumOps.ToDouble(audio[i]));
            coarse[i] = s >> 8;
            fine[i] = s & 255;
        }
        return (coarse, fine);
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
        var previous = new Tensor<T>(new[] { 1, 2, n });
        var current = new Tensor<T>(new[] { 1, 1, n });
        int silence = ToSixteenBit(0);
        for (int t = 0; t < n; t++)
        {
            previous[0, 0, t] = NumOps.FromDouble(Scaled(t > 0 ? coarse[t - 1] : silence >> 8));
            previous[0, 1, t] = NumOps.FromDouble(Scaled(t > 0 ? fine[t - 1] : silence & 255));
            current[0, 0, t] = NumOps.FromDouble(Scaled(coarse[t]));
        }
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
        var wave = new Tensor<T>(new[] { 1, 1, samples });
        int silence = ToSixteenBit(0), previousCoarse = silence >> 8, previousFine = silence & 255;
        for (int t = 0; t < samples; t++)
        {
            var previous = new Tensor<T>(new[] { 1, 2, 1 });
            previous[0, 0, 0] = NumOps.FromDouble(Scaled(previousCoarse));
            previous[0, 1, 0] = NumOps.FromDouble(Scaled(previousFine));
            var column = Engine.TensorSlice(condition, new[] { 0, 0, t }, new[] { 1, mel.Shape[1], 1 });
            // The current coarse byte reaches only the fine half, so any value gives the coarse half's state.
            var blind = core.Step(h, core.Inputs(previous, new Tensor<T>(new[] { 1, 1, 1 }), column));
            int c = SampleClass(core.CoarseLogits(blind), random);
            var current = new Tensor<T>(new[] { 1, 1, 1 });
            current[0, 0, 0] = NumOps.FromDouble(Scaled(c));
            h = core.Step(h, core.Inputs(previous, current, column));
            int f = SampleClass(core.FineLogits(h), random);
            wave[0, 0, t] = NumOps.FromDouble(FromSixteenBit((c << 8) | f));
            previousCoarse = c;
            previousFine = f;
        }
        return wave;
    }

    /// <summary>The fraction of the recurrent weights that are zero.</summary>
    public double RecurrentSparsity
    {
        get
        {
            var kernel = _core!.Recurrent.Kernel();
            int zeros = 0;
            for (int i = 0; i < kernel.Length; i++) if (NumOps.ToDouble(kernel[i]) == 0) zeros++;
            return zeros / (double)kernel.Length;
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
        if (_mask is null || since % Math.Max(1, o.PruningInterval) == 0 || since == o.PruningSteps)
        {
            double progress = Math.Min(1.0, since / (double)Math.Max(1, o.PruningSteps));
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
        int hidden = _core.Hidden, br = Math.Max(1, o.PruneBlockRows), bc = Math.Max(1, o.PruneBlockColumns);
        if (hidden % br != 0 || hidden % bc != 0)
            throw new InvalidOperationException($"The pruning block {br}x{bc} must tile the {hidden}x{hidden} gate matrices.");
        var mask = new Tensor<T>(kernel._shape);
        for (int i = 0; i < mask.Length; i++) mask[i] = NumOps.One;
        int rowsOfBlocks = hidden / br, colsOfBlocks = hidden / bc, blocks = rowsOfBlocks * colsOfBlocks;
        int pruned = (int)Math.Floor(z * blocks);
        for (int gate = 0; gate < 3; gate++)
        {
            var magnitude = new (double Value, int Index)[blocks];
            for (int b = 0; b < blocks; b++)
            {
                int r0 = gate * hidden + b / colsOfBlocks * br, c0 = b % colsOfBlocks * bc;
                double sum = 0;
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < bc; c++) sum += Math.Abs(NumOps.ToDouble(kernel[(r0 + r) * hidden + c0 + c]));
                magnitude[b] = (sum, b);
            }
            foreach (var (_, b) in magnitude.OrderBy(m => m.Value).ThenBy(m => m.Index).Take(pruned))
            {
                int r0 = gate * hidden + b / colsOfBlocks * br, c0 = b % colsOfBlocks * bc;
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < bc; c++) mask[(r0 + r) * hidden + c0 + c] = NumOps.Zero;
            }
        }
        return mask;
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                UseAdaptiveBetas = false,
            }));

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
