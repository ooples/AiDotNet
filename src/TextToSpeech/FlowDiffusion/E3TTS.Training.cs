using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

public partial class E3TTS<T>
{
    private sealed record TrainingExample(int[] TokenIds, Tensor<T> Clean, Tensor<T> Noise, Tensor<T> Weights,
        double AlphaBar, int SpeakerId);

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        RequirePaperModel();
        if (_optimizer is null) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var example = PrepareTrainingExample(sample, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => DiffusionObjective(example), _optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        RequirePaperModel();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 101);
            return DiffusionObjective(PrepareTrainingExample(sample, random))[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    private TrainingExample PrepareTrainingExample(TtsTrainingSample<T> sample, Random random)
    {
        if (sample.Audio is null) throw new ArgumentException("E3 training requires the target raw waveform.", nameof(sample));
        int[] ids = !string.IsNullOrWhiteSpace(sample.Text)
            ? TokenizeText(sample.Text!)
            : FrontEnd.TtsTokenIds.Read(sample.Tokens, _options.BertVocabularySize,
                Math.Min(_options.MaxTextLength, _options.BertMaxPositions), "E3 BERT");
        if (ids.Length == 0) throw new ArgumentException("E3 training requires text tokens.", nameof(sample));

        int total = _options.TrainingWaveformSamples;
        int valid = Math.Min(sample.Audio.Length, total);
        var clean = new Tensor<T>(new[] { total });
        var weights = new Tensor<T>(new[] { total });
        for (int i = 0; i < total; i++)
        {
            if (i < valid) { clean[i] = sample.Audio.GetFlat(i); weights[i] = Ops.One; }
            else weights[i] = Ops.FromDouble(_options.PaddingLossWeight);
        }

        // Equation 4 samples alpha-bar from an interval between adjacent schedule entries.
        int interval = random.Next(_options.NumDiffusionSteps);
        double left = TrainingAlphaBar(interval), right = TrainingAlphaBar(interval + 1);
        double alphaBar = left + random.NextDouble() * (right - left);
        int speaker = sample.SpeakerId ?? _options.DefaultSpeakerId;
        if (speaker < 0 || speaker >= _options.SpeakerCount) throw new ArgumentOutOfRangeException(nameof(sample), "SpeakerId is outside E3's speaker table.");
        return new TrainingExample(ids, clean, Gaussian(total, random), weights, alphaBar, speaker);
    }

    private Tensor<T> DiffusionObjective(TrainingExample example)
    {
        double signal = Math.Sqrt(example.AlphaBar), noiseLevel = Math.Sqrt(1 - example.AlphaBar);
        var noisy = Engine.TensorAdd(Engine.TensorMultiplyScalar(example.Clean, Ops.FromDouble(signal)),
            Engine.TensorMultiplyScalar(example.Noise, Ops.FromDouble(noiseLevel)));
        Tensor<T> text;
        if (_options.FreezeBert)
        {
            using var _ = new NoGradScope<T>();
            text = TextEncoder!.Forward(example.TokenIds);
        }
        else text = TextEncoder!.Forward(example.TokenIds);
        var (prediction, logVariance) = WaveformUnet!.Forward(noisy, text, signal, example.SpeakerId);
        return VarianceWeightedNoiseLoss(prediction, example.Noise, example.Weights, logVariance);
    }

    internal Tensor<T> VarianceWeightedNoiseLoss(Tensor<T> prediction, Tensor<T> noise,
        Tensor<T> weights, Tensor<T> logVariance)
    {
        var error = Engine.TensorSubtract(prediction, noise);
        var weightedSquared = Engine.TensorMultiply(weights, Engine.TensorMultiply(error, error));
        double normalization = weights.ToArray().Sum(Ops.ToDouble);
        var mse = Engine.TensorMultiplyScalar(Engine.ReduceSum(weightedSquared, new[] { 0 }, keepDims: true),
            Ops.FromDouble(1.0 / Math.Max(normalization, 1e-12)));
        mse = Engine.Reshape(mse, new[] { 1, 1 });
        // Equation 6: L2 / omega(alpha) + ln omega(alpha), with logVariance = ln omega.
        return Engine.TensorAdd(Engine.TensorMultiply(Engine.TensorExp(Engine.TensorNegate(logVariance)), mse), logVariance);
    }
}
