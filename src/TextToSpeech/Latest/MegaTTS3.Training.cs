using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Latest;

public partial class MegaTTS3<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;

    /// <summary>A training recording as the transformer reads it: the aligner's phones, tones and frame-to-phone map, and
    /// the WaveVAE posterior's mean and standard deviation <c>[1, frames, channels]</c> over those frames.</summary>
    internal sealed record TrainingExample(int[] Phones, int[] Tones, int[] Mel2Ph, Tensor<T> Mean, Tensor<T> Std);

    /// <summary>One training step's draws: the alignment map the transformer reads (sparse anchors or every frame), the
    /// prompt's latent frames, the dropped conditions, the flow time, the source noise and the posterior's noise.</summary>
    internal sealed record TrainingDraws(int[] Mel2Ph, int PromptFrames, bool DropSpeaker, bool DropText, double Time, Tensor<T> Noise,
        Tensor<T> PosteriorNoise);

    private readonly Dictionary<Tensor<T>, TrainingExample> _examples = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The paper's objective on one recording: rectified flow (Jiang et al. 2025, Eq. 2) on the latent's target
    /// region after a prompt of U(0.1, 0.9) of it, the transformer reading one random anchor frame per phone (§3.2), the
    /// prompt dropped with probability 0.1 and then the text with 0.5 (Eq. 5).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var example = Example(sample);
        var draws = Draw(example, _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed));
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(example, draws), Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The objective with draws from <see cref="MegaTTS3Options.TrainingSeed"/>, the same on every call.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var example = Example(sample);
            return Loss(example, Draw(example, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed)))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // The recording padded as the release pads a prompt, aligned, and encoded (the frozen aligner and WaveVAE run once per
    // recording).
    internal TrainingExample Example(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        if (_examples.TryGetValue(audio, out var cached)) return cached;
        var o = _options;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var wav = Doubles(audio);
            int ws = o.FftSize;
            if (wav.Length % ws < ws - 1) Array.Resize(ref wav, wav.Length + ws - 1 - wav.Length % ws);
            Array.Resize(ref wav, wav.Length + 12000);
            var (phones, tones, mel2ph) = Aligner.Align(Audio.Resampling.SoxrResampler.LibrosaResample(wav, o.SampleRate, 16000));
            int frames = mel2ph.Length / VaeStride;
            var (mean, logVariance) = Vae.Encode(Wave(wav));
            if (mean.Shape[1] < frames) throw new ArgumentException("The recording's latent is shorter than its alignment.", nameof(sample));
            mean = Engine.TensorSlice(mean, new[] { 0, 0, 0 }, new[] { 1, frames, o.LatentChannels });
            var std = Engine.TensorExp(Engine.TensorMultiplyScalar(
                Engine.TensorSlice(logVariance, new[] { 0, 0, 0 }, new[] { 1, frames, o.LatentChannels }), NumOps.FromDouble(0.5)));
            var example = new TrainingExample(phones, tones, mel2ph, mean, std);
            if (_examples.Count > 64) _examples.Clear();
            return _examples[audio] = example;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    internal TrainingDraws Draw(TrainingExample example, Random random)
    {
        var o = _options;
        var map = example.Mel2Ph;
        if (o.SparseAlignment)
        {
            // One anchor per phone: a random frame among its own, the rest the mask (0).
            var anchors = new int[map.Length];
            for (int start = 0; start < map.Length;)
            {
                int end = start;
                while (end < map.Length && map[end] == map[start]) end++;
                if (map[start] > 0) anchors[start + random.Next(end - start)] = map[start];
                start = end;
            }
            map = anchors;
        }
        int frames = example.Mean.Shape[1];
        double fraction = o.PromptFractionMin + (o.PromptFractionMax - o.PromptFractionMin) * random.NextDouble();
        int promptFrames = Math.Min(Math.Max((int)(fraction * frames), 0), frames - 1);
        bool dropSpeaker = random.NextDouble() < o.SpeakerDropProbability;
        bool dropText = dropSpeaker && random.NextDouble() < o.TextDropProbability;
        double time = random.NextDouble();
        var shape = new[] { 1, frames, o.LatentChannels };
        return new TrainingDraws(map, promptFrames, dropSpeaker, dropText, time, Gaussian(shape, random), Gaussian(shape, random));
    }

    /// <summary>The rectified-flow loss: the squared error between the predicted velocity and z₁ − z₀ at
    /// z_t = (1 − t) z₀ + t z₁, averaged over the target region's frames and channels.</summary>
    internal Tensor<T> Loss(TrainingExample example, TrainingDraws draws)
    {
        var o = _options;
        int frames = example.Mean.Shape[1], channels = o.LatentChannels, l = example.Phones.Length;
        var z1 = Engine.TensorAdd(example.Mean, Engine.TensorMultiply(example.Std, draws.PosteriorNoise));
        var phones = new int[1, l];
        var tones = new int[1, l];
        for (int j = 0; j < l; j++)
        {
            int tone = example.Tones[j];
            phones[0, j] = draws.DropText ? o.DitPhoneVocabulary - 1 : example.Phones[j];
            tones[0, j] = draws.DropText ? o.DitToneVocabulary - 1 : tone == 4 || (tone >= 11 && tone <= 15) || tone == 0 ? tone : 3;
        }
        var map = new int[1, draws.Mel2Ph.Length];
        for (int f = 0; f < draws.Mel2Ph.Length; f++) map[0, f] = draws.Mel2Ph[f];
        var mask = new Tensor<T>(new[] { 1, frames, 1 });
        var keep = new Tensor<T>(new[] { 1, frames, channels });
        for (int f = 0; f < frames; f++)
        {
            if (f < draws.PromptFrames) mask[0, f, 0] = NumOps.One;
            else
                for (int c = 0; c < channels; c++) keep[0, f, c] = NumOps.One;
        }
        var promptLatent = draws.DropSpeaker
            ? new Tensor<T>(new[] { 1, frames, channels })
            : Engine.TensorMultiply(z1, Engine.TensorBroadcastTo(mask, new[] { 1, frames, channels }));
        var noise = draws.Noise;
        var t = NumOps.FromDouble(draws.Time);
        var zt = Engine.TensorAdd(Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - draws.Time)), Engine.TensorMultiplyScalar(z1, t));
        var velocity = Dit.Velocity(zt, mask, Dit.LocalCondition(promptLatent, mask), Dit.LinguisticFeatures(phones, tones, map),
            Dit.TimeEmbedding(new[] { draws.Time }));
        var error = Engine.TensorMultiply(Engine.TensorSubtract(velocity, Engine.TensorSubtract(z1, noise)), keep);
        double count = (frames - draws.PromptFrames) * (double)channels;
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(error, error), new[] { 0, 1, 2 }, false), NumOps.FromDouble(1 / count));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2,
            // The paper's Adam: its betas held, and no gradient clipping (Appendix A.1 states none).
            UseAdaptiveBetas = false, EnableGradientClipping = false,
        };
        if (o.WarmupSteps > 0)
        {
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, o.WarmupSteps, 0, 0.0, LinearWarmupScheduler.DecayMode.Constant);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the diffusion transformer; the aligner, duration model, g2p model and WaveVAE are frozen
    /// (each is trained apart in the release).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_ditLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
