using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.StyleEmotion;

public partial class StyleTTS<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _discriminatorStep;

    /// <summary>Training samples skipped because their segment was shorter than
    /// <see cref="StyleTTSOptions.MinTrainingFrames"/>, as the reference skips them.</summary>
    public int SkippedSamples { get; private set; }

    /// <summary>The mel spectrogram <c>[mels, frames]</c> of a sample (its own, or computed from its recording), trimmed to
    /// an even number of frames as the reference's batches are.</summary>
    private Tensor<T> MelOf(TtsTrainingSample<T> sample)
    {
        Tensor<T> mel;
        if (sample.Mel is not null) mel = MelsFirst(sample.Mel);
        else if (sample.Audio is not null) mel = ComputeMel(sample.Audio);
        else throw new ArgumentException($"{GetType().Name} trains on a mel spectrogram or the recording.", nameof(sample));
        int frames = mel.Shape[1] / 2 * 2;
        return frames == mel.Shape[1] ? mel : Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { _options.MelChannels, frames });
    }

    /// <inheritdoc />
    /// <remarks>StyleTTS trains on phonemes and the recording's mel spectrogram alone, so a token/mel pair
    /// <c>[frames, mels]</c> is a full training sample.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Mel = expectedOutput });
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
        var mel = MelOf(sample);
        var step = DrawStep(_trainingRandom, training: true);
        var prepared = Prepare(ids, mel, step, training: true);
        if (prepared is null)
        {
            SkippedSamples++;
            return NumOps.Zero;
        }

        bool adversarial = CurrentStage == StyleTtsTrainingStage.Reconstruction || _options.Objective == StyleTtsObjective.Reference;
        if (adversarial)
        {
            // The fake spectrogram the discriminator sees: the generator's forward with this step's draws, before any
            // update. The style encoder runs in inference mode here, so only the generator's pass refines its spectral
            // norms, once per step as in the reference.
            Tensor<T> fake;
            using (new NoGradScope<T>()) fake = Generate(prepared, step, training: true, refineStyleNorms: false).Mel;
            var real = prepared.RealForDiscriminator;
            _discriminatorStep = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => DiscriminatorLoss(real, fake), DiscriminatorOptimizer());
            }
            finally
            {
                _discriminatorStep = false;
            }
        }
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => GeneratorLoss(prepared, step, training: true), GeneratorOptimizer());
        if (CurrentStage == StyleTtsTrainingStage.Reconstruction) ReconstructionUpdates++;
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>The current stage's generator objective without its adversarial terms (which measure the discriminator as
    /// much as the generator), with dropout off and fixed draws (the hard alignment, no stretch), without updating any
    /// weight.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        var ids = ToIds(sample.Tokens, nameof(sample));
        var mel = MelOf(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var step = DrawStep(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), training: false);
            var prepared = Prepare(ids, mel, step, training: false)
                ?? throw new ArgumentException($"{GetType().Name} evaluates segments of at least {_options.MinTrainingFrames} frames; " +
                    $"this one has {mel.Shape[1]}.", nameof(sample));
            return GeneratorLoss(prepared, step, training: false, adversarialTerms: false)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- one step's draws and fixed inputs

    /// <summary>A step's random draws: whether the decoder reads the hard alignment, the time stretch, and the seed of
    /// every dropout and token-masking draw (so the discriminator's fake and the generator's pass draw alike).</summary>
    private sealed class StepDraws
    {
        public bool Hard;
        public double Stretch = 1.0;
        public int Seed;
        /// <summary>A generator of the step's draws; each <paramref name="stream"/> is independent of the others.</summary>
        public Random Fork(int stream = 1) => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(Seed * 31 + stream));
    }

    private StepDraws DrawStep(Random random, bool training)
    {
        var step = new StepDraws { Seed = random.Next() };
        step.Hard = !training || random.NextDouble() < _options.HardAlignmentProbability;
        if (training && CurrentStage == StyleTtsTrainingStage.Prosody)
        {
            double magnitude = random.NextDouble();
            step.Stretch = 1 + (random.NextDouble() - 0.5) * 2 * _options.MaxStretch * magnitude;
        }
        return step;
    }

    /// <summary>What a step reads that no trained module produces.</summary>
    private sealed class PreparedStep
    {
        public int[] Ids = Array.Empty<int>();
        public Tensor<T> Mel = default!;                 // [mels, frames], the (stretched) recording
        public int Segment;                              // aligner frames of the training segment
        public double[,] Hard = default!;                // [L, F] monotonic alignment of Mel
        public Tensor<T> Target = default!;              // [1, mels, 2·Segment]
        public Tensor<T> Energy = default!;              // [1, 2·Segment]
        // Stage 2: the original recording's style and alignment, and the frozen modules' outputs.
        public Tensor<T>? UtteranceStyle;
        public double[,]? OriginalHard;
        public Tensor<T>? Text;                          // [1, C, L]
        public Tensor<T>? SegmentStyle;
        public Tensor<T>? Reconstruction;                // decoder(en, real F0, real energy, style)
        public Tensor<T>? RealF0;
        public Tensor<T> RealForDiscriminator = default!;
    }

    private static double[,] ToMatrix(Tensor<T> tensor, INumericOperations<T> ops)
    {
        int rows = tensor.Shape[0], columns = tensor.Shape[1];
        var m = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) m[i, j] = ops.ToDouble(tensor[i, j]);
        return m;
    }

    /// <summary>Monotonic alignment search (the reference's <c>maximum_path</c>, resemble-ai/monotonic_align) over a
    /// <c>[phonemes, frames]</c> score matrix, as a 0/1 path.</summary>
    internal static double[,] HardPath(double[,] scores)
    {
        var durations = MonotonicAlignment.MaximumPath(scores);
        var path = new double[scores.GetLength(0), scores.GetLength(1)];
        int frame = 0;
        for (int i = 0; i < durations.Length; i++)
            for (int f = 0; f < durations[i]; f++) path[i, frame++] = 1.0;
        return path;
    }

    private Tensor<T> FromMatrix(double[,] m)
    {
        var t = new Tensor<T>(new[] { m.GetLength(0), m.GetLength(1) });
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) t[i, j] = NumOps.FromDouble(m[i, j]);
        return t;
    }

    /// <summary>The aligner's soft attention <c>[L, F]</c> (softmax over the frames of each phoneme's energies, the start
    /// token's step dropped) and its token logits <c>[L + 1, tokens]</c>.</summary>
    private (Tensor<T> Soft, Tensor<T> Logits) Attention(int[] ids, Tensor<T> mel, bool training, Random random)
    {
        var (logits, energies) = AlignerModule.Decode(mel, ids, training, random);
        int frames = energies.Shape[1];
        var feature = Engine.TensorSlice(energies, new[] { 1, 0 }, new[] { ids.Length, frames });
        return (Engine.Softmax(feature, axis: 1), logits);
    }

    /// <summary>The reference's <c>TimeStrech</c>: bilinear interpolation along time to <c>⌊frames · scale⌋</c> frames
    /// (align_corners off), trimmed to an even count.</summary>
    private Tensor<T> Stretch(Tensor<T> mel, double scale)
    {
        int mels = mel.Shape[0], frames = mel.Shape[1], output = (int)Math.Floor(frames * scale);
        output = output / 2 * 2;
        var result = new Tensor<T>(new[] { mels, output });
        int full = (int)Math.Floor(frames * scale);
        double ratio = (double)frames / full;
        for (int j = 0; j < output; j++)
        {
            double source = Math.Max(0.0, (j + 0.5) * ratio - 0.5);
            int left = Math.Min((int)Math.Floor(source), frames - 1), right = Math.Min(left + 1, frames - 1);
            double w = source - left;
            for (int m = 0; m < mels; m++)
                result[m, j] = NumOps.FromDouble((1 - w) * NumOps.ToDouble(mel[m, left]) + w * NumOps.ToDouble(mel[m, right]));
        }
        return result;
    }

    /// <summary>Everything a step reads that its trained modules do not produce; null when the segment is too short.</summary>
    private PreparedStep? Prepare(int[] ids, Tensor<T> mel, StepDraws step, bool training)
    {
        var random = step.Fork(0);
        var prepared = new PreparedStep { Ids = ids };
        if (CurrentStage == StyleTtsTrainingStage.Prosody)
        {
            using var _ = new NoGradScope<T>();
            var (soft, _) = Attention(ids, mel, training, random);
            prepared.OriginalHard = HardPath(ToMatrix(soft, NumOps));
            prepared.Text = TextEncoderModule.Forward(ToTensor(ids), training, random);
            prepared.UtteranceStyle = StyleEncoderModule.Forward(mel);
            mel = Stretch(mel, step.Stretch);
        }
        prepared.Mel = mel;
        int frames = mel.Shape[1];
        prepared.Segment = (int)(frames / 2.0 - 1);
        if (2 * prepared.Segment < _options.MinTrainingFrames) return null;
        if (CurrentStage == StyleTtsTrainingStage.Prosody)
        {
            using var _ = new NoGradScope<T>();
            var (soft, _) = Attention(ids, mel, training, random);
            prepared.Hard = HardPath(ToMatrix(soft, NumOps));
        }
        var target = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { _options.MelChannels, 2 * prepared.Segment });
        prepared.Target = Engine.Reshape(target, new[] { 1, _options.MelChannels, 2 * prepared.Segment });
        prepared.Energy = LogNorm(target);
        if (CurrentStage == StyleTtsTrainingStage.Prosody)
        {
            using var _ = new NoGradScope<T>();
            prepared.SegmentStyle = StyleEncoderModule.Forward(target);
            var f0 = PitchModule.Forward(target, training, random);
            prepared.RealF0 = f0;
            var asr = Crop(Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(prepared.Text!, new[] { _options.HiddenDim, ids.Length }),
                FromMatrix(prepared.Hard)), new[] { 1, _options.HiddenDim, prepared.Hard.GetLength(1) }), prepared.Segment);
            prepared.Reconstruction = DecoderModule.Forward(asr, f0, prepared.Energy, prepared.SegmentStyle, training, random);
            prepared.RealForDiscriminator = prepared.Reconstruction;
        }
        else
        {
            prepared.RealForDiscriminator = prepared.Target;
        }
        return prepared;
    }

    private Tensor<T> Crop(Tensor<T> x, int frames) =>
        Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, x.Shape[1], frames });

    // ---------------------------------------------------------------- the generator's pass and losses

    private sealed class GeneratorPass
    {
        public Tensor<T> Mel = default!;
        public Tensor<T>? Soft;
        public Tensor<T>? Logits;
        public Tensor<T>? Durations;
        public Tensor<T>? F0;
        public Tensor<T>? Energy;
        public double[,]? Hard;
    }

    private bool TmaActive => ReconstructionUpdates >= _options.TmaStartStep;

    /// <summary>The generator's forward for a step: in stage 1 the reconstruction from the aligned phonemes, extracted
    /// pitch, energy and style; in stage 2 the predicted durations, pitch and energy and the decoder's output from them.</summary>
    private GeneratorPass Generate(PreparedStep p, StepDraws step, bool training, bool refineStyleNorms = true)
    {
        var random = step.Fork(1);
        var pass = new GeneratorPass();
        int c = _options.HiddenDim;
        if (CurrentStage == StyleTtsTrainingStage.Reconstruction)
        {
            var (soft, logits) = Attention(p.Ids, p.Mel, training, random);
            pass.Soft = soft;
            pass.Logits = logits;
            // The monotonic hard alignment of this pass's attention; the decoder reads it or the soft attention (§II-B's
            // 50%-50% strategy).
            pass.Hard = HardPath(ToMatrix(soft, NumOps));
            var alignment = step.Hard ? FromMatrix(pass.Hard) : soft;
            var text = TextEncoderModule.Forward(ToTensor(p.Ids), training, random);
            var asr = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(text, new[] { c, p.Ids.Length }), alignment), new[] { 1, c, alignment.Shape[1] });
            var en = Crop(asr, p.Segment);
            var target = Engine.Reshape(p.Target, new[] { _options.MelChannels, 2 * p.Segment });
            var f0 = PitchModule.Forward(target, training, random);
            Tensor<T> style;
            if (refineStyleNorms) style = StyleEncoderModule.Forward(target);
            else
            {
                foreach (var layer in _styleLayers) layer.SetTrainingMode(false);
                try { style = StyleEncoderModule.Forward(target); }
                finally { foreach (var layer in _styleLayers) layer.SetTrainingMode(training); }
            }
            pass.Mel = DecoderModule.Forward(en, f0, p.Energy, style, training, random);
            return pass;
        }

        var predictor = PredictorModule;
        var encodedForDuration = predictor.EncodeText(p.Text!, p.UtteranceStyle!, training, random);
        pass.Durations = predictor.Durations(encodedForDuration, training, random);
        var encoded = predictor.EncodeText(p.Text!, p.UtteranceStyle!, training, random);
        int width = c + _options.StyleDim, frames = p.Hard.GetLength(1);
        var aligned = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), FromMatrix(p.Hard)), new[] { 1, width, frames });
        var (f0Fake, energyFake) = predictor.Prosody(Crop(aligned, p.Segment), p.SegmentStyle!, training, random);
        pass.F0 = f0Fake;
        pass.Energy = energyFake;
        var asrStretched = Crop(Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(p.Text!, new[] { c, p.Ids.Length }), FromMatrix(p.Hard)),
            new[] { 1, c, frames }), p.Segment);
        pass.Mel = DecoderModule.Forward(asrStretched, f0Fake, energyFake, p.SegmentStyle!, training, random);
        return pass;
    }

    private Tensor<T> MeanAbsolute(Tensor<T> a, Tensor<T> b)
    {
        var diff = Engine.TensorAbs(Engine.TensorSubtract(a, Engine.Reshape(b, a._shape)));
        return Engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), keepDims: false);
    }

    private Tensor<T> SmoothL1(Tensor<T> a, Tensor<T> b)
    {
        var abs = Engine.TensorAbs(Engine.TensorSubtract(a, Engine.Reshape(b, a._shape)));
        var q = Engine.TensorMin(abs, NumOps.One);
        var loss = Engine.TensorMultiply(q, Engine.TensorSubtract(abs, Engine.TensorMultiplyScalar(q, NumOps.FromDouble(0.5))));
        return Engine.ReduceMean(loss, Enumerable.Range(0, loss.Rank).ToArray(), keepDims: false);
    }

    /// <summary>The reference's <c>adv_loss</c>: binary cross-entropy of the logit clamped to ±10.</summary>
    private Tensor<T> Adversarial(Tensor<T> logit, bool real)
    {
        var clamped = Engine.TensorClamp(logit, NumOps.FromDouble(-10), NumOps.FromDouble(10));
        var loss = Engine.Softplus(real ? Engine.TensorMultiplyScalar(clamped, NumOps.FromDouble(-1)) : clamped);
        return Engine.ReduceMean(loss, Enumerable.Range(0, loss.Rank).ToArray(), keepDims: false);
    }

    private Tensor<T> Scaled(Tensor<T> x, double factor) => Engine.TensorMultiplyScalar(x, NumOps.FromDouble(factor));

    /// <summary>The current stage's generator objective (§II-B, Eq. 7 and 12, or the released code's weighting).</summary>
    private Tensor<T> GeneratorLoss(PreparedStep p, StepDraws step, bool training, bool adversarialTerms = true)
    {
        bool paper = _options.Objective == StyleTtsObjective.Paper;
        var pass = Generate(p, step, training);
        Tensor<T> loss;
        bool adversarial;
        if (CurrentStage == StyleTtsTrainingStage.Reconstruction)
        {
            loss = Scaled(MeanAbsolute(pass.Mel, p.Target), paper ? 1.0 : 5.0);
            if (TmaActive)
            {
                // L_s2s: the decoder steps after the start token predict the phonemes (cross-entropy, mean over them).
                int tokens = _options.VocabSize, length = p.Ids.Length;
                var logits = Engine.TensorSlice(pass.Logits!, new[] { 0, 0 }, new[] { length, tokens });
                var selection = new Tensor<T>(new[] { length, tokens });
                for (int i = 0; i < length; i++) selection[i, p.Ids[i]] = NumOps.FromDouble(-1.0 / length);
                var s2s = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits, axis: -1), selection), new[] { 0, 1 }, keepDims: false);
                var mono = MeanAbsolute(pass.Soft!, FromMatrix(pass.Hard!));
                loss = Engine.TensorAdd(loss, Scaled(s2s, paper ? 0.2 : 1.0));
                loss = Engine.TensorAdd(loss, Scaled(mono, paper ? 5.0 : 10.0));
            }
            adversarial = true;
        }
        else
        {
            var reconstruction = p.Reconstruction!;
            loss = Scaled(MeanAbsolute(pass.Mel, reconstruction), paper ? 1.0 : 5.0);
            // L_dur over the phonemes between the pad symbols, against the original recording's alignment.
            int length = p.Ids.Length;
            if (length > 2)
            {
                var target = new Tensor<T>(new[] { length - 2 });
                for (int i = 1; i < length - 1; i++)
                {
                    double sum = 0;
                    for (int f = 0; f < p.OriginalHard!.GetLength(1); f++) sum += p.OriginalHard[i, f];
                    target[i - 1] = NumOps.FromDouble(sum);
                }
                loss = Engine.TensorAdd(loss, MeanAbsolute(Engine.TensorSlice(pass.Durations!, new[] { 1 }, new[] { length - 2 }), target));
            }
            var f0Real = p.RealF0!;
            var pitchLoss = paper ? Scaled(MeanAbsolute(pass.F0!, f0Real), 0.1) : Scaled(SmoothL1(f0Real, pass.F0!), 0.1);
            var energyLoss = paper ? MeanAbsolute(pass.Energy!, p.Energy) : SmoothL1(p.Energy, pass.Energy!);
            loss = Engine.TensorAdd(Engine.TensorAdd(loss, pitchLoss), energyLoss);
            adversarial = !paper;
        }
        if (adversarial && adversarialTerms)
        {
            var discriminator = DiscriminatorModule;
            List<Tensor<T>> realFeatures;
            using (new NoGradScope<T>()) realFeatures = discriminator.Forward(Image(p.RealForDiscriminator)).Features;
            var (logit, fakeFeatures) = discriminator.Forward(Image(pass.Mel));
            var featureMatching = realFeatures.Zip(fakeFeatures, (r, f) => MeanAbsolute(f, r)).Aggregate(Engine.TensorAdd);
            loss = Engine.TensorAdd(loss, Adversarial(logit, real: true));
            loss = Engine.TensorAdd(loss, Scaled(featureMatching, paper ? 0.2 : 0.1));
        }
        return loss;
    }

    private Tensor<T> Image(Tensor<T> mel) => Engine.Reshape(mel, new[] { 1, 1, _options.MelChannels, mel.Length / _options.MelChannels });

    /// <summary>The discriminator's objective: real and fake cross-entropies and the R1 penalty on the real input.</summary>
    private Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> fake)
    {
        var discriminator = DiscriminatorModule;
        var loss = Engine.TensorAdd(Adversarial(discriminator.Forward(Image(real)).Logit, real: true),
            Adversarial(discriminator.Forward(Image(fake)).Logit, real: false));
        if (_options.R1Weight > 0) loss = Engine.TensorAdd(loss, Scaled(R1Penalty(real), _options.R1Weight));
        return loss;
    }

    /// <summary>The reference's <c>r1_reg</c>: half the squared norm of the discriminator's gradient at the real input,
    /// differentiable into the discriminator (as StyleGAN's R1 penalty is built here).</summary>
    private Tensor<T> R1Penalty(Tensor<T> real)
    {
        if (!AiDotNet.Autodiff.GradientRecording.IsRecording<T>()) return new Tensor<T>(new[] { 1 });
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        var input = new Tensor<T>(Image(real).Shape.ToArray());
        real.AsSpan().CopyTo(input.AsWritableSpan());
        Tensor<T> gradient;
        using (var inner = new GradientTape<T>())
        {
            var logit = DiscriminatorModule.Forward(input).Logit;
            var summed = Engine.ReduceSum(logit, new[] { 0 }, keepDims: false);
            var gradients = inner.ComputeGradients(summed, new[] { input }, createGraph: true);
            gradient = gradients.TryGetValue(input, out var g) ? g : new Tensor<T>(input.Shape.ToArray());
        }
        var flat = Engine.Reshape(gradient, new[] { gradient.Length });
        return Scaled(Engine.ReduceSum(Engine.TensorMultiply(flat, flat), new[] { 0 }, keepDims: false), 0.5);
    }

    // ---------------------------------------------------------------- optimizers and trainable parameters

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> NewOptimizer(string component)
    {
        var o = _options;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate,
            Beta1 = o.Beta1,
            Beta2 = o.Beta2,
            Epsilon = o.Epsilon,
            WeightDecay = o.WeightDecay,
        };
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options), component);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GeneratorOptimizer() => _generatorOptimizer ??= NewOptimizer("generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiscriminatorOptimizer() => _discriminatorOptimizer ??= NewOptimizer("discriminator");

    /// <inheritdoc />
    /// <remarks>The discriminator's step trains the discriminator; stage 1's generator step the text encoder, style
    /// encoder and decoder (and, once the aligner's objectives start, the aligner and pitch extractor); stage 2's the
    /// duration and prosody predictors.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        IEnumerable<LayerBase<T>> layers;
        if (_discriminatorStep) layers = _discriminatorLayers;
        else if (CurrentStage == StyleTtsTrainingStage.Prosody) layers = _predictorLayers;
        else
        {
            layers = _textEncoderLayers.Concat(_styleLayers).Concat(_decoderLayers);
            if (TmaActive) layers = layers.Concat(_alignerLayers).Concat(_pitchLayers);
        }
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
