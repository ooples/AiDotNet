using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

public partial class DiTToTTS<T>
{
    private sealed record DiffusionExample(Tensor<T> Text, Tensor<T> Clean, Tensor<T> Noise, Tensor<T> Mask,
        double Time, bool DropText);
    private sealed record LengthExample(Tensor<T> Text, Tensor<T>? Prompt, int[] Labels);

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed(); RequirePaperModel();
        if (_optimizer is null) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (IsCodecStage) return TrainCodec(sample);
        return CurrentStage == DiTToTrainingStage.Diffusion
            ? TrainDiffusion(sample)
            : TrainLength(sample);
    }

    private T TrainDiffusion(TtsTrainingSample<T> sample)
    {
        var example = PrepareDiffusion(sample, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => DiffusionObjective(example), _optimizer!);
    }

    private T TrainLength(TtsTrainingSample<T> sample)
    {
        var example = PrepareLength(sample, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => LengthObjective(example), _optimizer!);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed(); RequirePaperModel();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var inferenceScope = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 97);
            if (IsCodecStage) return CodecObjective(sample, random, out _, out _, out _)[0];
            return CurrentStage == DiTToTrainingStage.Diffusion
                ? DiffusionObjective(PrepareDiffusion(sample, random))[0]
                : LengthObjective(PrepareLength(sample, random))[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    private (Tensor<T> Text, Tensor<T> Latent) FrozenTargets(TtsTrainingSample<T> sample)
    {
        if (string.IsNullOrWhiteSpace(sample.Text)) throw new ArgumentException("DiTTo training needs the utterance text.", nameof(sample));
        Tensor<T> mel = sample.Mel ?? (sample.Audio is not null ? ComputeMel(sample.Audio)
            : throw new ArgumentException("DiTTo training needs the recording or its mel spectrogram.", nameof(sample)));
        using var _ = new NoGradScope<T>();
        return (TextEncoder!.Forward(TokenizeText(sample.Text!)), MelVae!.Encode(mel));
    }

    private DiffusionExample PrepareDiffusion(TtsTrainingSample<T> sample, Random random)
    {
        var (text, clean) = FrozenTargets(sample);
        int frames = clean.Shape[0];
        var mask = new Tensor<T>(new[] { frames, 1 });
        if (random.NextDouble() < _options.FullMaskProbability)
        {
            for (int i = 0; i < frames; i++) mask[i, 0] = Ops.One;
        }
        else
        {
            double ratio = _options.MinimumMaskRatio + random.NextDouble() * (_options.MaximumMaskRatio - _options.MinimumMaskRatio);
            int count = MathPolyfill.Clamp((int)Math.Ceiling(frames * ratio), 1, frames);
            int start = random.Next(frames - count + 1);
            for (int i = start; i < start + count; i++) mask[i, 0] = Ops.One;
        }
        double time = (random.Next(_options.TrainingDiffusionSteps) + 0.5) / _options.TrainingDiffusionSteps;
        return new DiffusionExample(text, clean, Gaussian(frames, _options.LatentDim, random), mask, time,
            random.NextDouble() < _options.ClassifierFreeDropProbability);
    }

    private Tensor<T> DiffusionObjective(DiffusionExample example)
    {
        var (signal, noiseLevel) = NoiseLevel(example.Time);
        var noisy = Engine.TensorAdd(Engine.TensorMultiplyScalar(example.Clean, Ops.FromDouble(signal)),
            Engine.TensorMultiplyScalar(example.Noise, Ops.FromDouble(noiseLevel)));
        var mask = Engine.TensorTile(example.Mask, new[] { 1, _options.LatentDim });
        noisy = Engine.TensorAdd(Engine.TensorMultiply(mask, noisy),
            Engine.TensorMultiply(Engine.TensorAddScalar(Engine.TensorNegate(mask), Ops.One), example.Clean));
        // v = alpha * epsilon - sigma * x0 (v-prediction in §3.3).
        var target = Engine.TensorSubtract(Engine.TensorMultiplyScalar(example.Noise, Ops.FromDouble(signal)),
            Engine.TensorMultiplyScalar(example.Clean, Ops.FromDouble(noiseLevel)));
        var prediction = DiffusionTransformer!.Forward(noisy, example.Mask, example.Text, example.Time, example.DropText);
        var error = Engine.TensorMultiply(mask, Engine.TensorSubtract(prediction, target));
        int maskedFrames = Math.Max(1, Enumerable.Range(0, example.Mask.Shape[0]).Count(i => Ops.ToDouble(example.Mask[i, 0]) > 0.5));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(error, error), new[] { 0, 1 }, keepDims: true),
            Ops.FromDouble(1.0 / (maskedFrames * (double)_options.LatentDim)));
    }

    private LengthExample PrepareLength(TtsTrainingSample<T> sample, Random random)
    {
        var (text, latent) = FrozenTargets(sample);
        int promptLength = latent.Shape[0] <= 1 ? 0 : random.Next(latent.Shape[0]);
        Tensor<T>? prompt = promptLength == 0 ? null : Engine.TensorSlice(latent, new[] { 0, 0 }, new[] { promptLength, latent.Shape[1] });
        return new LengthExample(text, prompt, DiTToLengthPredictor<T>.RemainingLabels(latent.Shape[0], promptLength));
    }

    private Tensor<T> LengthObjective(LengthExample example) =>
        LengthPredictor!.CrossEntropy(LengthPredictor.Forward(example.Text, example.Prompt), example.Labels);
}
