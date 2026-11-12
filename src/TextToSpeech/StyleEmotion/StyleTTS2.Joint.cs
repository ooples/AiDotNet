using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.StyleEmotion;

public partial class StyleTTS2<T>
{
    /// <summary>What a stage-2 step reads that its trained modules do not produce (computed outside the tape, as the
    /// reference computes it under <c>torch.no_grad</c> or with modules it does not step).</summary>
    private sealed class JointStep
    {
        public int[] Ids = Array.Empty<int>();
        public Tensor<T> Mel = default!;              // [mels, F], the whole utterance
        public Tensor<T> PaddedWave = default!;       // the whole recording, padded
        public Tensor<T> Mono = default!;             // [L, F/2] monotonic alignment
        public int[] Durations = Array.Empty<int>();  // frames per phoneme of Mono
        public Tensor<T> Asr = default!;              // [1, C, Length], the aligned text features of the segment
        public Tensor<T>? Reference;                  // [1, 2·style], the reference style (multispeaker, diffusion on)
        public Tensor<T> Target = default!;           // [1, 2·style], the utterance's styles (the denoiser's target)
        public int Start, Length;
        public Tensor<T> Segment = default!;          // [mels, 2·Length]
        public Tensor<T> StyleReference = default!;   // [mels, frames]
        public Tensor<T> Real = default!;             // the recording's segment
        public Tensor<T> Reconstruction = default!;   // the decoder's output from the real pitch and energy
        public Tensor<T> WaveTarget = default!;       // what the mel and adversarial losses compare with
        public Tensor<T> Pitch = default!;
        public Tensor<T> Energy = default!;
    }

    private JointStep? PrepareJoint(int[] ids, Tensor<T> wave, Tensor<T> mel, Tensor<T> referenceMel, StepDraws step, bool evaluation)
    {
        var o = _options;
        var random = step.Fork(0);
        int half = mel.Shape[1] / 2;
        int length = Math.Min(half - 1, o.MaxTrainingFrames / 2), styleLength = half - 1;
        if (length <= 0 || 2 * length < o.MinTrainingFrames) return null;
        using var _ = new NoGradScope<T>();
        if (!evaluation) ApplyStageModes();
        var p = new JointStep { Ids = ids, Mel = mel, PaddedWave = wave, Length = length };
        var (soft, _) = Attention(ids, mel, false, random);
        var mono = StyleTTS<T>.HardPath(ToMatrix(soft));
        p.Mono = FromMatrix(mono);
        p.Durations = Enumerable.Range(0, ids.Length).Select(i => Enumerable.Range(0, mono.GetLength(1)).Count(f => mono[i, f] > 0)).ToArray();
        var text = TextEncoderNetwork.Forward(ToTensor(ids), false, random);
        if (o.Multispeaker && DiffusionActive)
            p.Reference = Engine.TensorConcatenate(new[] { StyleEncoderNetwork.Forward(referenceMel), PredictorEncoderNetwork.Forward(referenceMel) }, 1);
        p.Target = Engine.TensorConcatenate(new[] { StyleEncoderNetwork.Forward(mel), PredictorEncoderNetwork.Forward(mel) }, 1);
        p.Start = random.Next(0, half - length);
        int styleStart = random.Next(0, half - styleLength);
        p.Asr = Crop(Aligned(text, p.Mono), p.Start, length);
        p.Segment = MelCrop(mel, 2 * p.Start, 2 * length);
        p.StyleReference = o.Multispeaker ? MelCrop(mel, 2 * styleStart, 2 * styleLength) : p.Segment;
        p.Real = WaveCrop(wave, 2 * p.Start, 2 * length);
        p.Pitch = PitchExtractorNetwork.Forward(p.Segment, false, random);
        p.Energy = FrontEnd.LogNorm(p.Segment);
        p.Reconstruction = DecoderNetwork.Forward(p.Asr, p.Pitch, p.Energy, StyleEncoderNetwork.Forward(p.StyleReference), false, random);
        // Before joint training the decoder is fixed, so the predictors learn to reproduce its reconstruction.
        p.WaveTarget = JointActive ? p.Real : p.Reconstruction;
        if (o.EstimateSigmaData && DiffusionActive && !evaluation) DiffusionNetwork.SigmaData = StandardDeviation(p.Target);
        return p;
    }

    // torch.std: the unbiased standard deviation.
    private double StandardDeviation(Tensor<T> x)
    {
        int n = x.Length;
        double mean = 0, sum = 0;
        for (int i = 0; i < n; i++) mean += NumOps.ToDouble(x[i]) / n;
        for (int i = 0; i < n; i++) sum += Math.Pow(NumOps.ToDouble(x[i]) - mean, 2);
        return Math.Sqrt(sum / (n - 1));
    }

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = NumOps.FromDouble(StyleTts2StyleDiffusion<T>.Gaussian(random));
        return t;
    }

    private Tensor<T> SmoothL1(Tensor<T> a, Tensor<T> b)
    {
        var abs = Engine.TensorAbs(Engine.TensorSubtract(a, Engine.Reshape(b, a._shape)));
        var q = Engine.TensorMin(abs, NumOps.One);
        var loss = Engine.TensorMultiply(q, Engine.TensorSubtract(abs, Engine.TensorMultiplyScalar(q, NumOps.FromDouble(0.5))));
        return Engine.ReduceMean(loss, Enumerable.Range(0, loss.Rank).ToArray(), keepDims: false);
    }

    private Tensor<T> PhonemesFirst(Tensor<T> rows) => Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    private sealed class JointPass
    {
        public Tensor<T> Wave = default!;
        public Tensor<T> DurationLogits = default!;
        public Tensor<T> Pitch = default!;
        public Tensor<T> Energy = default!;
        public Tensor<T>? Sampled;
        public Tensor<T>? Diffusion;
    }

    /// <summary>Stage 2's generator forward (<c>train_second.py</c>): PL-BERT's features, the denoiser's loss and a style
    /// sampled from it (once the diffusion starts), the duration logits and prosody from the utterance's prosodic style,
    /// and the decoder's waveform from the predicted pitch and energy.</summary>
    private JointPass Joint(JointStep p, StepDraws step, bool training, bool waveOnly = false)
    {
        var o = _options;
        var random = step.Fork(1);
        var pass = new JointPass();
        var bert = BertNetwork.Encode(p.Ids, training, random);
        var dEn = PhonemesFirst(BertEncoderLayer.Forward(bert));
        var globalProsody = PredictorEncoderNetwork.Forward(p.Mel);
        if (DiffusionActive && !waveOnly)
        {
            var draws = step.Fork(3);
            int steps = draws.Next(3, 5);
            var noise = Gaussian(new[] { 1, 2 * o.StyleDim }, draws);
            pass.Sampled = DiffusionNetwork.Sample(noise, bert, p.Reference, steps, 1.0, draws, maskProbability: 0.1);
            // The multispeaker model trains the denoiser through AudioDiffusionConditional (embedding_mask_proba 0.1); the
            // single-speaker model calls KDiffusion directly, without masking.
            pass.Diffusion = DiffusionNetwork.Loss(p.Target, bert, p.Reference, o.Multispeaker ? o.EmbeddingMaskProbability : 0.0, draws);
        }
        var encoded = PredictorNetwork.EncodeText(dEn, globalProsody, training, random);
        pass.DurationLogits = PredictorNetwork.Durations(encoded, training, random);
        var prosodyFeatures = Crop(Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), p.Mono),
            new[] { 1, encoded.Shape[1], p.Mono.Shape[1] }), p.Start, p.Length);
        var segmentProsody = PredictorEncoderNetwork.Forward(p.StyleReference);
        var acoustic = StyleEncoderNetwork.Forward(p.StyleReference);
        (pass.Pitch, pass.Energy) = PredictorNetwork.Prosody(prosodyFeatures, segmentProsody, training, random);
        pass.Wave = DecoderNetwork.Forward(p.Asr, pass.Pitch, pass.Energy, acoustic, false, random);
        return pass;
    }

    /// <summary>Stage 2's generator objective: λ_mel mel loss, λ_F0 smooth-L1 pitch (÷10), λ_norm smooth-L1 energy,
    /// λ_ce duration-bin cross-entropy, λ_dur duration L1, λ_slm WavLM feature matching, and once the diffusion starts
    /// λ_gen adversarial, λ_sty style reconstruction and λ_diff score matching.</summary>
    private Tensor<T> JointLoss(JointStep p, StepDraws step, bool training, bool adversarialTerms = true)
    {
        var o = _options;
        var pass = Joint(p, step, training);
        var pitch = Scaled(SmoothL1(p.Pitch, pass.Pitch), 0.1);
        var energy = SmoothL1(p.Energy, pass.Energy);
        var mel = Losses.MultiResolutionMel(pass.Wave, p.WaveTarget);
        var slm = Losses.SlmFeatureMatching(SlmNetwork, p.WaveTarget, pass.Wave);
        Tensor<T>? generator = null;
        if (DiffusionActive && adversarialTerms)
        {
            List<(Tensor<T>, List<Tensor<T>>)> real;
            using (new NoGradScope<T>()) real = WaveDiscriminators.Forward(p.WaveTarget);
            generator = Losses.Generator(real, WaveDiscriminators.Forward(pass.Wave));
        }
        // Duration bins: bin k of phoneme i is 1 for k < its frames; BCE with logits over all bins, L1 between the sum of
        // the sigmoids and the frames over the phonemes between the pad symbols.
        int length = p.Ids.Length, bins = o.DurationBins;
        var target = new Tensor<T>(new[] { length, bins });
        for (int i = 0; i < length; i++)
            for (int k = 0; k < Math.Min(bins, p.Durations[i]); k++) target[i, k] = NumOps.One;
        var logits = pass.DurationLogits;
        var bce = Engine.ReduceMean(Engine.TensorSubtract(Engine.Softplus(logits), Engine.TensorMultiply(logits, target)), new[] { 0, 1 }, keepDims: false);
        Tensor<T>? duration = null;
        if (length > 2)
        {
            var predicted = Engine.ReduceSum(Engine.Sigmoid(Engine.TensorSlice(logits, new[] { 1, 0 }, new[] { length - 2, bins })), new[] { 1 }, keepDims: false);
            var frames = new Tensor<T>(new[] { length - 2 });
            for (int i = 1; i < length - 1; i++) frames[i - 1] = NumOps.FromDouble(p.Durations[i]);
            duration = Losses.L1(predicted, frames);
        }
        Tensor<T>? style = pass.Sampled is null ? null : Losses.L1(pass.Sampled, p.Target);
        return Total((mel, o.MelLossWeight), (pitch, o.F0LossWeight), (bce, o.DurationCrossEntropyWeight), (energy, o.NormLossWeight),
            (duration, o.DurationLossWeight), (generator, o.GeneratorLossWeight), (slm, o.SlmLossWeight), (style, o.StyleLossWeight),
            (pass.Diffusion, o.DiffusionLossWeight));
    }

    private T TrainJoint(Tensor<T> tokens, JointStep prepared, StepDraws step)
    {
        if (DiffusionActive)
        {
            Tensor<T> fake;
            using (new NoGradScope<T>())
            {
                ApplyStageModes();
                fake = Joint(prepared, step, training: true, waveOnly: true).Wave;
            }
            var real = prepared.WaveTarget;
            RunStep(tokens, new[] { "mpd", "msd" }, discriminator: true,
                () => Losses.Discriminator(WaveDiscriminators.Forward(real), WaveDiscriminators.Forward(fake)));
        }
        var modules = new List<string> { "bert_encoder", "bert", "predictor", "predictor_encoder" };
        if (DiffusionActive) modules.Add("diffusion");
        if (JointActive) modules.AddRange(new[] { "style_encoder", "decoder" });
        var loss = RunStep(tokens, modules, discriminator: false, () => JointLoss(prepared, step, training: true));
        if (JointActive) SlmAdversarialStep(tokens, prepared, step);
        return loss;
    }

    // ---------------------------------------------------------------- SLM adversarial training (§3.2.2, Modules/slmadv.py)

    private sealed class SlmStep
    {
        public int[] Ids = Array.Empty<int>();
        public bool UseTarget;          // the utterance's own styles instead of sampled ones
        public int Frames;              // l, the predicted length
        public int Length, PredictedStart, RealStart;
        public Tensor<T> Real = default!;
        public Tensor<T> Generated = default!;
    }

    private int PredictedFrames(Tensor<T> logits)
    {
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(logits[i])));
        return (int)Math.Round(sum, MidpointRounding.ToEven);
    }

    /// <summary>The SLM step's generator forward: styles (the utterance's or sampled), durations, the differentiable
    /// alignment, prosody and the decoder's waveform of a crop of the predicted frames. Null when the predicted or the
    /// real utterance is not longer than the segment (the reference skips it).</summary>
    private (Tensor<T> Wave, SlmStep Step)? SlmForward(JointStep p, SlmStep s, StepDraws step, int? knownFrames)
    {
        var o = _options;
        var random = step.Fork(5);
        var bert = BertNetwork.Encode(s.Ids, true, random);
        var dEn = PhonemesFirst(BertEncoderLayer.Forward(bert));
        Tensor<T> styles;
        if (s.UseTarget) styles = p.Target;
        else
        {
            var draws = step.Fork(6);
            int steps = draws.Next(3, 5);
            styles = DiffusionNetwork.Sample(Gaussian(new[] { 1, 2 * o.StyleDim }, draws), bert, p.Reference, steps, 1.0, draws, maskProbability: 0.1);
        }
        var acoustic = Engine.TensorSlice(styles, new[] { 0, 0 }, new[] { 1, o.StyleDim });
        var prosodic = Engine.TensorSlice(styles, new[] { 0, o.StyleDim }, new[] { 1, o.StyleDim });
        var logits = PredictorNetwork.Durations(PredictorNetwork.EncodeText(dEn, prosodic, true, random), true, random);
        int frames = knownFrames ?? PredictedFrames(logits);
        int segment = Math.Min(Math.Max((int)(frames / 2.0 - 1), o.SlmAdversarialMinFrames / 2), o.SlmAdversarialMaxFrames / 2);
        int realFrames = p.Mel.Shape[1] / 2;
        if (frames <= segment || realFrames <= segment) return null;
        var alignment = Losses.DifferentiableAlignment(logits, frames, o.SlmDurationSigma);
        Tensor<T> text;
        using (new NoGradScope<T>()) text = TextEncoderNetwork.Forward(ToTensor(s.Ids), false, random);
        var asr = Aligned(text, alignment);
        var encoded = PredictorNetwork.EncodeText(dEn, prosodic, true, random);
        var features = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(encoded), alignment), new[] { 1, encoded.Shape[1], frames });
        // Both passes draw the crops, so every later draw matches between them.
        s.Frames = frames;
        s.Length = segment;
        s.PredictedStart = random.Next(0, frames - segment);
        s.RealStart = random.Next(0, realFrames - segment);
        s.Real = WaveCrop(p.PaddedWave, 2 * s.RealStart, 2 * segment);
        var (pitch, energy) = PredictorNetwork.Prosody(Crop(features, s.PredictedStart, s.Length), prosodic, true, random);
        var wave = DecoderNetwork.Forward(Crop(asr, s.PredictedStart, s.Length), pitch, energy, acoustic, false, random);
        return (wave, s);
    }

    private void SlmAdversarialStep(Tensor<T> tokens, JointStep p, StepDraws step)
    {
        var o = _options;
        var draws = step.Fork(4);
        // In-distribution text half the time (always, without out-of-distribution texts); then the utterance's own styles
        // half of those times.
        bool inDistribution = o.OutOfDistributionTexts is not { Count: > 0 } || draws.NextDouble() < 0.5;
        var s = new SlmStep
        {
            Ids = inDistribution ? p.Ids : o.OutOfDistributionTexts![draws.Next(o.OutOfDistributionTexts.Count)],
        };
        s.UseTarget = inDistribution && draws.NextDouble() < 0.5;
        (Tensor<T> Wave, SlmStep Step)? first;
        using (new NoGradScope<T>())
        {
            ApplyStageModes();
            first = SlmForward(p, s, step, null);
        }
        if (first is null) return;
        s.Generated = first.Value.Wave;

        // The generator's SLM loss, with the reference's gradient scaling: every gradient divided by the predictor's
        // gradient norm when that exceeds the threshold, then the duration head's, its LSTM's and the diffusion model's
        // scaled by SlmGradientScale.
        var predictorParameters = new HashSet<Tensor<T>>(ModuleOfParameter().Where(kv => kv.Value == "predictor").Select(kv => kv.Key),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        var damped = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(
            PredictorNetwork.DurationLayers.Concat(_diffusionLayers).Cast<ILayer<T>>().ToList(), -1), Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        void Transform(IReadOnlyList<Tensor<T>> parameters, Dictionary<Tensor<T>, Tensor<T>> gradients)
        {
            double norm = 0;
            foreach (var entry in gradients)
                if (predictorParameters.Contains(entry.Key))
                    for (int i = 0; i < entry.Value.Length; i++) norm += Math.Pow(NumOps.ToDouble(entry.Value[i]), 2);
            norm = Math.Sqrt(norm);
            foreach (var entry in gradients)
            {
                var gradient = entry.Value;
                double factor = (norm > o.SlmGradientThreshold ? 1.0 / norm : 1.0) * (damped.Contains(entry.Key) ? o.SlmGradientScale : 1.0);
                if (factor == 1.0) continue;
                for (int i = 0; i < gradient.Length; i++) gradient[i] = NumOps.FromDouble(NumOps.ToDouble(gradient[i]) * factor);
            }
        }
        RunStep(tokens, new[] { "bert_encoder", "bert", "predictor", "diffusion" }, discriminator: false,
            () => Losses.SlmGenerator(SlmNetwork, SlmDiscriminator, SlmForward(p, s, step, s.Frames)!.Value.Wave), Transform);

        // The SLM discriminator, every SlmDiscriminatorInterval-th update.
        if ((JointUpdates + 1) % o.SlmDiscriminatorInterval != 0) return;
        var choice = step.Fork(7);
        var generated = s.Generated;
        if (choice.Next(0, 2) == 0)
        {
            // Against the reconstruction (a different length): the scores of a crop must match the uncropped ones', and the
            // reconstruction's must match the recording's (ignoring reconstruction artifacts).
            var real = p.Reconstruction;
            int crop = Math.Min(real.Length, generated.Length);
            bool realLonger = real.Length > generated.Length;
            bool cropped = choice.Next(0, 2) == 0;
            RunStep(tokens, new[] { "wd" }, discriminator: true, () =>
            {
                var longer = realLonger ? real : generated;
                var head = Engine.TensorSlice(longer, new[] { 0 }, new[] { crop });
                var outCrop = Losses.SlmScores(SlmNetwork, SlmDiscriminator, head);
                var outFull = Losses.SlmScores(SlmNetwork, SlmDiscriminator, longer);
                var regularization = Losses.L1(outCrop, Engine.TensorSlice(outFull, new[] { 0, 0 }, new[] { 1, outCrop.Shape[1] }));
                var loss = realLonger
                    ? Losses.SlmDiscriminator(SlmNetwork, SlmDiscriminator, cropped ? head : real, generated)
                    : Losses.SlmDiscriminator(SlmNetwork, SlmDiscriminator, real, cropped ? head : generated);
                var artifacts = Losses.L1(Losses.SlmScores(SlmNetwork, SlmDiscriminator, p.Real),
                    Losses.SlmScores(SlmNetwork, SlmDiscriminator, p.Reconstruction));
                return Engine.TensorAdd(Engine.TensorAdd(loss, regularization), artifacts);
            });
        }
        else
        {
            RunStep(tokens, new[] { "wd" }, discriminator: true, () => Losses.SlmDiscriminator(SlmNetwork, SlmDiscriminator, s.Real, generated));
        }
    }
}
