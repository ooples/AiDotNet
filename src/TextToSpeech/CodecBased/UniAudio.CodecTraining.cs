using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class UniAudio<T>
{
    private readonly List<LayerBase<T>> _codecLayers = new(), _languageLayers = new();
    private UniAudioCodecGan<T>? _codecGan;
    private AdamWOptimizer<T, Tensor<T>, Tensor<T>>? _codecGeneratorOptimizer, _codecDiscriminatorOptimizer;
    private Random? _codecRandom;
    private bool _codecDiscriminatorTurn;

    /// <summary>The last universal-codec generator loss, including commitment.</summary>
    public double LastCodecGeneratorLoss { get; private set; }
    /// <summary>The last universal-codec discriminator loss.</summary>
    public double LastCodecDiscriminatorLoss { get; private set; }

    /// <inheritdoc />
    public override double MaxGradNormValue => _options is not null &&
        _options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec ? 0 : base.MaxGradNormValue;

    /// <summary>Sets the completed codec epoch and applies the released epoch-wise exponential scheduler.</summary>
    public void SetCodecTrainingEpoch(int completedEpochs)
    {
        ThrowIfDisposed();
        if (completedEpochs < 0) throw new ArgumentOutOfRangeException(nameof(completedEpochs));
        _options.CodecTraining.Epoch = completedEpochs;
        ApplyCodecLearningRate();
    }

    private double CodecLearningRate => _options.CodecTraining.LearningRate *
        Math.Pow(_options.CodecTraining.EpochDecay, _options.CodecTraining.Epoch);

    private void ApplyCodecLearningRate()
    {
        _codecGeneratorOptimizer?.SetScheduledLearningRate(CodecLearningRate);
        _codecDiscriminatorOptimizer?.SetScheduledLearningRate(CodecLearningRate);
    }

    private void EnsureCodecGan()
    {
        if (_codecGan is not null) return;
        var recipe = _options.CodecTraining;
        if (recipe.Bandwidths.Length == 0 || recipe.Bandwidths.Any(value => double.IsNaN(value) || double.IsInfinity(value) || value <= 0) ||
            recipe.SegmentSize <= 0 || recipe.Epoch < 0 || recipe.CompletedSteps < 0 || recipe.AdversarialStartStep < 0 ||
            recipe.LearningRate <= 0 || double.IsNaN(recipe.LearningRate) || double.IsInfinity(recipe.LearningRate) ||
            recipe.EpochDecay <= 0 || recipe.EpochDecay > 1 || double.IsNaN(recipe.EpochDecay))
            throw new ArgumentException("Invalid universal-codec training recipe.");
        _codecGan = new UniAudioCodecGan<T>(Engine, _options.SampleRate, recipe);
        ComponentLayers.AddRange(_codecGan.Layers);
        _options.IncludeCodecDiscriminators = true;
        _codecRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        AdamWOptimizer<T, Tensor<T>, Tensor<T>> Optimizer() => new(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = CodecLearningRate, Beta1 = .8, Beta2 = .99, Epsilon = 1e-6,
                WeightDecay = recipe.WeightDecay, EnableGradientClipping = false,
            });
        _codecGeneratorOptimizer = Optimizer();
        _codecDiscriminatorOptimizer = Optimizer();
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperPipeline) return parameters;
        var layers = _options.TrainingComponent == UniAudioTrainingComponent.LanguageModel ? _languageLayers :
            _codecDiscriminatorTurn ? _codecGan!.Layers : _codecLayers;
        var selected = new HashSet<Tensor<T>>(AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(
            layers.Cast<ILayer<T>>().ToList(), -1), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private Tensor<T> PrepareCodecAudio(Tensor<T> audio, bool randomCrop)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int length = Math.Min(audio.Length, _options.CodecTraining.SegmentSize);
        length -= length % _codec!.HopLength;
        if (length <= 0) throw new ArgumentException("Codec pretraining requires at least one complete codec frame.", nameof(audio));
        int start = randomCrop && audio.Length > length ? _codecRandom!.Next(audio.Length - length + 1) : 0;
        return Engine.TensorSlice(Engine.Reshape(audio, new[] { audio.Length }), new[] { start }, new[] { length });
    }

    private int DrawCodecQuantizers()
    {
        double bitsPerFrame = _options.CodecFrameRate * Math.Log(_options.CodebookSize, 2);
        double bandwidth = _options.CodecTraining.Bandwidths[_codecRandom!.Next(_options.CodecTraining.Bandwidths.Length)];
        int quantizers = Math.Max(1, (int)Math.Floor(1000 * bandwidth / bitsPerFrame));
        if (quantizers > _options.CodecTotalQuantizers)
            throw new ArgumentException("The codec training bandwidth exceeds its configured codebook count.");
        return quantizers;
    }

    private Tensor<T> CodecGeneratorObjective(Tensor<T> audio, int quantizers)
    {
        var output = _codec!.TrainingForward(audio, quantizers);
        bool adversarial = _options.CodecTraining.CompletedSteps > _options.CodecTraining.AdversarialStartStep;
        var loss = adversarial ? _codecGan!.GeneratorObjective(audio, output.Waveform) :
            _codecGan!.ReconstructionObjective(audio, output.Waveform);
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(output.Commitment,
            NumOps.FromDouble(_options.CodecTraining.CommitmentWeight)));
    }

    private T TrainCodecSample(TtsTrainingSample<T> sample)
    {
        EnsureCodecGan();
        ApplyCodecLearningRate();
        var audio = PrepareCodecAudio(sample.Audio!, randomCrop: true);
        int quantizers = DrawCodecQuantizers();
        // The released loop updates G first, then reconstructs again before updating D.
        T generator = TrainWithCustomObjective(audio, audio, (input, _) => CodecGeneratorObjective(input, quantizers),
            _codecGeneratorOptimizer!);
        _codecDiscriminatorTurn = true;
        try
        {
            T discriminator = TrainWithCustomObjective(audio, audio, (input, _) =>
            {
                Tensor<T> generated;
                using (new NoGradScope<T>()) generated = _codec!.TrainingForward(input, quantizers).Waveform;
                return _codecGan!.DiscriminatorObjective(input, generated);
            }, _codecDiscriminatorOptimizer!);
            LastCodecDiscriminatorLoss = NumOps.ToDouble(discriminator);
        }
        finally { _codecDiscriminatorTurn = false; }
        _options.CodecTraining.CompletedSteps++;
        LastCodecGeneratorLoss = NumOps.ToDouble(generator);
        LastLoss = generator;
        return generator;
    }

    /// <summary>Loads the original universal-codec checkpoint's trained mel discriminators.</summary>
    public void LoadAuthorCodecDiscriminatorWeights(string path)
    {
        RequirePipeline();
        EnsureCodecGan();
        using var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(path, strictAxes: true,
            name => name.StartsWith("mfd.discriminators.", StringComparison.Ordinal));
        _codecGan!.LoadTorchWeights(file);
    }
}
