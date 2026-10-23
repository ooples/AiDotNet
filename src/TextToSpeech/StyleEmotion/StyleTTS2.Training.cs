using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>The modules a StyleTTS 2 training step updates (Li et al. 2023, §3.3 and Appendix C).</summary>
public enum StyleTts2TrainingStage
{
    /// <summary>Stage 1 (the reference's <c>train_first.py</c>): waveform reconstruction from the aligned phonemes, the
    /// extracted pitch and energy and the acoustic style — the text encoder, acoustic style encoder and decoder, and once
    /// the aligner's objectives start, the text aligner and pitch extractor, adversarially against the wave
    /// discriminators and with WavLM feature matching.</summary>
    Pretraining,

    /// <summary>Stage 2 (<c>train_second.py</c>): PL-BERT, the prosody predictor and prosodic style encoder; then the
    /// style diffusion and the wave discriminators; then joint fine-tuning of the decoder and acoustic style encoder with
    /// SLM adversarial training and differentiable duration modeling.</summary>
    Joint,
}

public partial class StyleTTS2<T>
{
    private StyleTts2Losses<T>? _losses;
    private StyleTts2ModuleOptimizer<T>? _generatorOptimizer;
    private StyleTts2ModuleOptimizer<T>? _discriminatorOptimizer;
    private StyleTts2TrainingStage _optimizerStage;
    private Dictionary<Tensor<T>, string>? _moduleOfParameter;
    private HashSet<string> _steppedModules = new();
    private StyleTts2TrainingStage _stage = StyleTts2TrainingStage.Pretraining;

    /// <summary>The modules a training step updates.</summary>
    public StyleTts2TrainingStage CurrentStage
    {
        get => _stage;
        set => _stage = value;
    }

    /// <summary>Stage-1 updates taken (they decide when the aligner's objectives start).</summary>
    public int PretrainingUpdates { get; private set; }

    /// <summary>Stage-2 updates taken (they decide when the diffusion, discriminators and joint training start).</summary>
    public int JointUpdates { get; private set; }

    /// <summary>Training samples skipped because their segment was shorter than
    /// <see cref="StyleTtsBaseOptions.MinTrainingFrames"/>, as the reference skips them.</summary>
    public int SkippedSamples { get; private set; }

    /// <inheritdoc />
    /// <remarks>StyleTTS 2 reconstructs the waveform: it trains on the recording.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    private StyleTts2Losses<T> Losses => _losses ??= new StyleTts2Losses<T>(Engine, _options.SampleRate, _options.SlmSampleRate);

    private bool TmaActive => PretrainingUpdates >= _options.TmaStartStep;
    private bool DiffusionActive => JointUpdates >= _options.DiffusionStartStep;
    private bool JointActive => JointUpdates >= _options.JointStartStep;

    // ---------------------------------------------------------------- entry points

    /// <inheritdoc />
    /// <remarks>A token/waveform pair is a full StyleTTS 2 training sample.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        Validation.Guard.NotNull(sample);
        var ids = ToIds(sample.Tokens, nameof(sample));
        var wave = Padded(sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample)));
        var mel = MelOf(wave);
        var step = DrawStep(_trainingRandom);
        ApplyStageModes();
        if (_stage == StyleTts2TrainingStage.Pretraining)
        {
            var prepared = PreparePretraining(ids, wave, mel, step, training: true);
            if (prepared is null)
            {
                SkippedSamples++;
                return NumOps.Zero;
            }
            var loss = TrainPretraining(sample.Tokens, prepared, step);
            PretrainingUpdates++;
            return loss;
        }
        var joint = PrepareJoint(ids, wave, mel, ReferenceMelOf(sample, mel), step, evaluation: false);
        if (joint is null)
        {
            SkippedSamples++;
            return NumOps.Zero;
        }
        var result = TrainJoint(sample.Tokens, joint, step);
        JointUpdates++;
        return result;
    }

    /// <inheritdoc />
    /// <remarks>The current stage's generator objective without its adversarial terms (which measure the discriminators
    /// as much as the generator), with dropout off and fixed draws, without updating any weight.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        var ids = ToIds(sample.Tokens, nameof(sample));
        var wave = Padded(sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample)));
        var mel = MelOf(wave);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var step = DrawStep(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
            step.Hard = true;
            string tooShort = $"{GetType().Name} evaluates segments of at least {_options.MinTrainingFrames} frames; this one has {mel.Shape[1]}.";
            if (_stage == StyleTts2TrainingStage.Pretraining)
            {
                var prepared = PreparePretraining(ids, wave, mel, step, training: false) ?? throw new ArgumentException(tooShort, nameof(sample));
                return PretrainingLoss(prepared, step, training: false, adversarialTerms: false)[0];
            }
            var joint = PrepareJoint(ids, wave, mel, ReferenceMelOf(sample, mel), step, evaluation: true) ?? throw new ArgumentException(tooShort, nameof(sample));
            return JointLoss(joint, step, training: false, adversarialTerms: false)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- data

    /// <summary>The recording with the reference data loader's zero padding on each side.</summary>
    private Tensor<T> Padded(Tensor<T> audio)
    {
        int pad = _options.WavePadding, length = audio.Length;
        var wave = new Tensor<T>(new[] { length + 2 * pad });
        for (int i = 0; i < length; i++) wave[pad + i] = audio[i];
        return wave;
    }

    /// <summary>The mel spectrogram <c>[mels, frames]</c> of a waveform, trimmed to an even number of frames as the
    /// reference's data loader trims it.</summary>
    private Tensor<T> MelOf(Tensor<T> wave)
    {
        var mel = ComputeMel(wave);
        int frames = mel.Shape[1] / 2 * 2;
        return frames == mel.Shape[1] ? mel : Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { _options.MelChannels, frames });
    }

    /// <summary>The mel spectrogram of the sample's reference recording of the same speaker (the reference's
    /// <c>ref_mels</c>), or of the sample itself when it has none.</summary>
    private Tensor<T> ReferenceMelOf(TtsTrainingSample<T> sample, Tensor<T> own) =>
        sample.SpeakerReference is { } reference ? MelOf(Padded(reference)) : own;

    /// <summary>A step's random draws: whether the decoder reads the hard alignment, and the seed of every other draw, so a
    /// discriminator's fake and the generator's pass draw alike.</summary>
    private sealed class StepDraws
    {
        public bool Hard;
        public int Seed;
        public Random Fork(int stream) => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(Seed * 31 + stream));
    }

    private StepDraws DrawStep(Random random) =>
        new() { Seed = random.Next(), Hard = random.NextDouble() < _options.HardAlignmentProbability };

    private Tensor<T> FromMatrix(double[,] m)
    {
        var t = new Tensor<T>(new[] { m.GetLength(0), m.GetLength(1) });
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) t[i, j] = NumOps.FromDouble(m[i, j]);
        return t;
    }

    private double[,] ToMatrix(Tensor<T> tensor)
    {
        var m = new double[tensor.Shape[0], tensor.Shape[1]];
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) m[i, j] = NumOps.ToDouble(tensor[i, j]);
        return m;
    }

    /// <summary>The aligner's attention <c>[L, F/2]</c> (the softmax over frames, the start token's step dropped) and its
    /// token logits.</summary>
    private (Tensor<T> Soft, Tensor<T> Logits) Attention(int[] ids, Tensor<T> mel, bool training, Random random)
    {
        var (logits, energies) = AlignerNetwork.Decode(mel, ids, training, random);
        int frames = energies.Shape[1];
        var feature = Engine.TensorSlice(energies, new[] { 1, 0 }, new[] { ids.Length, frames });
        return (Engine.Softmax(feature, axis: 1), logits);
    }

    private Tensor<T> Aligned(Tensor<T> features, Tensor<T> alignment) =>
        Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(features, new[] { features.Shape[1], features.Shape[2] }), alignment),
            new[] { 1, features.Shape[1], alignment.Shape[1] });

    private Tensor<T> Crop(Tensor<T> x, int start, int frames) =>
        Engine.TensorSlice(x, new[] { 0, 0, start }, new[] { 1, x.Shape[1], frames });

    private Tensor<T> MelCrop(Tensor<T> mel, int start, int frames) =>
        Engine.TensorSlice(mel, new[] { 0, start }, new[] { _options.MelChannels, frames });

    private Tensor<T> WaveCrop(Tensor<T> wave, int start, int frames) =>
        Engine.TensorSlice(wave, new[] { start * _options.HopSize }, new[] { frames * _options.HopSize });

    // ---------------------------------------------------------------- stage 1

    /// <summary>What a stage-1 step reads that its trained modules do not produce.</summary>
    private sealed class PretrainingStep
    {
        public int[] Ids = Array.Empty<int>();
        public Tensor<T> Mel = default!;            // [mels, F]
        public int Start;                           // aligner frames
        public int Length;                          // aligner frames of the segment
        public Tensor<T> Target = default!;         // [mels, 2·Length]
        public Tensor<T> StyleReference = default!; // [mels, frames]
        public Tensor<T> Wave = default!;           // [2·Length·hop]
        public Tensor<T> Energy = default!;         // [1, 2·Length]
        public Tensor<T> Pitch = default!;          // [1, 2·Length]
    }

    private PretrainingStep? PreparePretraining(int[] ids, Tensor<T> wave, Tensor<T> mel, StepDraws step, bool training)
    {
        var random = step.Fork(0);
        int half = mel.Shape[1] / 2;
        int length = Math.Min(half - 1, _options.MaxTrainingFrames / 2), styleLength = half - 1;
        if (length <= 0 || 2 * length < _options.MinTrainingFrames) return null;
        int start = random.Next(0, half - length), styleStart = random.Next(0, half - styleLength);
        var target = MelCrop(mel, 2 * start, 2 * length);
        var prepared = new PretrainingStep
        {
            Ids = ids, Mel = mel, Start = start, Length = length, Target = target,
            StyleReference = _options.Multispeaker ? MelCrop(mel, 2 * styleStart, 2 * styleLength) : target,
            Wave = WaveCrop(wave, 2 * start, 2 * length),
            Energy = FrontEnd.LogNorm(target),
        };
        // The reference extracts the pitch once per step, outside the tape, in the extractor's training mode.
        using (new NoGradScope<T>()) prepared.Pitch = PitchExtractorNetwork.Forward(target, training, step.Fork(2));
        return prepared;
    }

    private sealed class PretrainingPass
    {
        public Tensor<T> Wave = default!;
        public Tensor<T> Soft = default!;
        public Tensor<T> Logits = default!;
        public double[,] Hard = default!;
    }

    /// <summary>Stage 1's generator forward: the aligner's attention and its monotonic path, the text features aligned by
    /// one of them (§3.3's 50%-50% strategy), and the decoder's waveform from them, the extracted pitch and energy and the
    /// acoustic style.</summary>
    private PretrainingPass Pretrain(PretrainingStep p, StepDraws step, bool training, bool refineStyleNorms = true)
    {
        var random = step.Fork(1);
        var (soft, logits) = Attention(p.Ids, p.Mel, training, random);
        var hard = StyleTTS<T>.HardPath(ToMatrix(soft));
        var text = TextEncoderNetwork.Forward(ToTensor(p.Ids), training, random);
        var en = Crop(Aligned(text, step.Hard ? FromMatrix(hard) : soft), p.Start, p.Length);
        Tensor<T> style;
        if (refineStyleNorms) style = StyleEncoderNetwork.Forward(p.StyleReference);
        else
        {
            foreach (var layer in _styleLayers) layer.SetTrainingMode(false);
            try { style = StyleEncoderNetwork.Forward(p.StyleReference); }
            finally { foreach (var layer in _styleLayers) layer.SetTrainingMode(training); }
        }
        var wave = DecoderNetwork.Forward(en, p.Pitch, p.Energy, style, training, random);
        return new PretrainingPass { Wave = wave, Soft = soft, Logits = logits, Hard = hard };
    }

    private Tensor<T> Scaled(Tensor<T> x, double factor) => Engine.TensorMultiplyScalar(x, NumOps.FromDouble(factor));

    private Tensor<T> Total(params (Tensor<T>? Term, double Weight)[] terms)
    {
        Tensor<T>? total = null;
        foreach (var (term, weight) in terms)
        {
            if (term is null || weight == 0) continue;
            var scaled = Engine.Reshape(Scaled(term, weight), new[] { 1 });
            total = total is null ? scaled : Engine.TensorAdd(total, scaled);
        }
        return total ?? new Tensor<T>(new[] { 1 });
    }

    /// <summary>Stage 1's generator objective (<c>train_first.py</c>): λ_mel times the multi-resolution mel loss and, once
    /// the aligner's objectives start, its token cross-entropy, the monotonic loss (×10), the wave discriminators'
    /// adversarial loss and WavLM feature matching.</summary>
    private Tensor<T> PretrainingLoss(PretrainingStep p, StepDraws step, bool training, bool adversarialTerms = true)
    {
        var o = _options;
        var pass = Pretrain(p, step, training);
        var mel = Losses.MultiResolutionMel(pass.Wave, p.Wave);
        if (!TmaActive) return Total((mel, o.MelLossWeight));
        int length = p.Ids.Length, tokens = o.VocabSize;
        var logits = Engine.TensorSlice(pass.Logits, new[] { 0, 0 }, new[] { length, tokens });
        var selection = new Tensor<T>(new[] { length, tokens });
        for (int i = 0; i < length; i++) selection[i, p.Ids[i]] = NumOps.FromDouble(-1.0 / length);
        var s2s = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits, axis: -1), selection), new[] { 0, 1 }, keepDims: false);
        var mono = Scaled(Losses.L1(pass.Soft, FromMatrix(pass.Hard)), 10.0);
        var slm = Losses.SlmFeatureMatching(SlmNetwork, p.Wave, pass.Wave);
        Tensor<T>? generator = null;
        if (adversarialTerms)
        {
            List<(Tensor<T>, List<Tensor<T>>)> real;
            using (new NoGradScope<T>()) real = WaveDiscriminators.Forward(p.Wave);
            generator = Losses.Generator(real, WaveDiscriminators.Forward(pass.Wave));
        }
        return Total((mel, o.MelLossWeight), (mono, o.MonotonicLossWeight), (s2s, o.S2sLossWeight), (generator, o.GeneratorLossWeight),
            (slm, o.SlmLossWeight));
    }

    private T TrainPretraining(Tensor<T> tokens, PretrainingStep prepared, StepDraws step)
    {
        if (TmaActive)
        {
            // The discriminators see the generator's waveform from this step's draws, before its update; the acoustic
            // style encoder runs without refining its spectral norms here, so only the generator's pass refines them.
            Tensor<T> fake;
            using (new NoGradScope<T>()) fake = Pretrain(prepared, step, training: true, refineStyleNorms: false).Wave;
            var real = prepared.Wave;
            RunStep(tokens, new[] { "mpd", "msd" }, discriminator: true,
                () => Losses.Discriminator(WaveDiscriminators.Forward(real), WaveDiscriminators.Forward(fake)));
        }
        var modules = new List<string> { "text_encoder", "style_encoder", "decoder" };
        if (TmaActive) modules.AddRange(new[] { "text_aligner", "pitch_extractor" });
        return RunStep(tokens, modules, discriminator: false, () => PretrainingLoss(prepared, step, training: true));
    }

    // ---------------------------------------------------------------- optimizers and module modes

    /// <summary>Every trained module's layers, by the reference's module key.</summary>
    private IEnumerable<(string Key, IReadOnlyList<LayerBase<T>> Layers)> ModuleLayers()
    {
        yield return ("text_encoder", _textEncoderLayers);
        yield return ("style_encoder", _styleLayers);
        yield return ("predictor_encoder", _predictorEncoderLayers);
        yield return ("predictor", _predictorLayers);
        yield return ("decoder", _decoderLayers);
        yield return ("bert", _bertLayers);
        yield return ("bert_encoder", _bertEncoderLayers);
        yield return ("diffusion", _diffusionLayers);
        yield return ("text_aligner", _alignerLayers);
        yield return ("pitch_extractor", _pitchLayers);
        var wave = WaveDiscriminators.Layers;
        int periodLayers = WaveDiscriminators.Periods.Layers.Count;
        yield return ("mpd", wave.Take(periodLayers).ToList());
        yield return ("msd", wave.Skip(periodLayers).ToList());
        yield return ("wd", _slmDiscriminatorLayers);
    }

    private Dictionary<Tensor<T>, string> ModuleOfParameter()
    {
        if (_moduleOfParameter is not null) return _moduleOfParameter;
        var map = new Dictionary<Tensor<T>, string>(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var (key, layers) in ModuleLayers())
            foreach (var tensor in Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1))
                map[tensor] = key;
        return _moduleOfParameter = map;
    }

    private AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> ModuleOptions(string key)
    {
        var o = _options;
        bool joint = _stage == StyleTts2TrainingStage.Joint;
        double rate = o.LearningRate, beta1 = o.Beta1, decay = o.WeightDecay;
        if (joint && key == "bert")
        {
            rate = o.BertLearningRate;
            beta1 = o.BertBeta1;
            decay = o.BertWeightDecay;
        }
        else if (joint && key is "decoder" or "style_encoder")
        {
            rate = o.FineTuneLearningRate;
        }
        // The reference clips no gradient.
        return new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate, Beta1 = beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = decay,
            EnableGradientClipping = false,
        };
    }

    private StyleTts2ModuleOptimizer<T> NewOptimizer(string component)
    {
        var map = ModuleOfParameter();
        var optimizer = new StyleTts2ModuleOptimizer<T>(this, ModuleOptions(component == "discriminator" ? "mpd" : "text_encoder"),
            p => map.TryGetValue(p, out var key) ? key : null, ModuleOptions);
        PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, component);
        return optimizer;
    }

    /// <summary>The optimizers of the current stage (the reference builds fresh ones for each stage).</summary>
    private (StyleTts2ModuleOptimizer<T> Generator, StyleTts2ModuleOptimizer<T> Discriminator) Optimizers()
    {
        if (_generatorOptimizer is null || _discriminatorOptimizer is null || _optimizerStage != _stage)
        {
            _optimizerStage = _stage;
            _generatorOptimizer = NewOptimizer("generator");
            _discriminatorOptimizer = NewOptimizer("discriminator");
        }
        return (_generatorOptimizer, _discriminatorOptimizer);
    }

    /// <summary>
    /// The stage's training modes (the reference's <c>.train()</c> / <c>.eval()</c>): every module in stage 1; in stage 2
    /// only the prosody predictor, PL-BERT, its projection and the wave discriminators.
    /// </summary>
    private void ApplyStageModes()
    {
        bool joint = _stage == StyleTts2TrainingStage.Joint;
        foreach (var (key, layers) in ModuleLayers())
        {
            bool train = !joint || key is "predictor" or "bert" or "bert_encoder" or "mpd" or "msd";
            foreach (var layer in layers) layer.SetTrainingMode(train);
        }
    }

    /// <summary>One update of <paramref name="modules"/> on <paramref name="objective"/>.</summary>
    private T RunStep(Tensor<T> tokens, IEnumerable<string> modules, bool discriminator, Func<Tensor<T>> objective,
        Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? gradientTransform = null)
    {
        var (generatorOptimizer, discriminatorOptimizer) = Optimizers();
        var optimizer = discriminator ? discriminatorOptimizer : generatorOptimizer;
        _steppedModules = new HashSet<string>(modules);
        optimizer.GradientTransform = gradientTransform;
        try
        {
            return TrainWithCustomObjective(tokens, tokens, (_, _) =>
            {
                ApplyStageModes();
                return objective();
            }, optimizer);
        }
        finally
        {
            optimizer.GradientTransform = null;
            _steppedModules = new HashSet<string>();
            ApplyStageModes();
        }
    }

    /// <inheritdoc />
    /// <remarks>The modules the current step updates (see <see cref="StyleTts2TrainingStage"/>); WavLM is never trained.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var map = ModuleOfParameter();
        return parameters.Where(p => map.TryGetValue(p, out var key) && _steppedModules.Contains(key)).ToList();
    }
}
