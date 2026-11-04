using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

public partial class DiTToTTS<T>
{
    private DiTToCodecTrainer<T>? _codecTrainer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _codecGeneratorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _codecDiscriminatorOptimizer;
    private bool _codecDiscriminatorTurn;
    private bool IsCodecStage => CurrentStage is DiTToTrainingStage.MelVaePretraining or DiTToTrainingStage.MelVaeSemanticFinetuning;
    internal DiTToCodecTrainer<T>? CodecTrainer => _codecTrainer;

    private void EnsureCodecTraining()
    {
        if (_codecTrainer is null)
        {
            _codecTrainer = new DiTToCodecTrainer<T>(Engine, _options.CodecTraining);
            Layers.AddRange(_codecTrainer.Critics.Layers);
            AdamOptimizer<T, Tensor<T>, Tensor<T>> Optimizer() => new(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                    { InitialLearningRate = _options.CodecTraining.LearningRate, Beta1 = .9, Beta2 = .999 });
            _codecGeneratorOptimizer = Optimizer(); _codecDiscriminatorOptimizer = Optimizer();
        }
        if (CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning && _codecTrainer.SemanticDecoder is null)
        {
            _codecTrainer.InitializeSemanticDecoder(_options.LatentDim,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 111));
            Layers.AddRange(_codecTrainer.ProjectionLayers); Layers.AddRange(_codecTrainer.DecoderLayers);
            Freeze(_codecTrainer.DecoderLayers);
        }
    }

    /// <summary>Loads the frozen ByT5 decoder used only by Mel-VAE++ semantic codec finetuning.</summary>
    /// <remarks>This does not replace the SpeechT5 text encoder of the English diffusion model. Bytes use
    /// ByT5's +3 offset and EOS=1. The native decoder remains differentiable with respect to projected speech latents.</remarks>
    public void LoadCodecSemanticDecoder(string checkpoint)
    {
        ThrowIfDisposed(); RequirePaperModel(); EnsureCodecTraining();
        if (_codecTrainer!.SemanticDecoder is null)
        {
            _codecTrainer.InitializeSemanticDecoder(_options.LatentDim,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 111));
            Layers.AddRange(_codecTrainer.ProjectionLayers); Layers.AddRange(_codecTrainer.DecoderLayers);
            Freeze(_codecTrainer.DecoderLayers);
        }
        EnsureLayerRandomSeedsWired();
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint);
        _codecTrainer.SemanticDecoder!.LoadHuggingFaceWeights((name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"ByT5 checkpoint is missing '{name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException($"ByT5 tensor '{name}' has an incompatible shape.");
            return tensor.ToArray().Select(value => (double)value).ToArray();
        });
        _codecTrainer.DecoderWeightsLoaded = true;
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!IsCodecStage) return parameters;
        EnsureCodecTraining();
        IEnumerable<LayerBase<T>> selectedLayers = _codecDiscriminatorTurn ? _codecTrainer!.Critics.Layers : _codecLayers;
        if (!_codecDiscriminatorTurn && CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning)
            selectedLayers = selectedLayers.Concat(_codecTrainer!.ProjectionLayers);
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(selectedLayers.Cast<ILayer<T>>().ToList()),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private Tensor<T> CodecObjective(TtsTrainingSample<T> sample, Random random,
        out Tensor<T> target, out Tensor<T> generated, out int[] starts)
    {
        EnsureCodecTraining();
        target = sample.Mel ?? (sample.Audio is not null ? ComputeMel(sample.Audio)
            : throw new ArgumentException("Codec training requires the recording or its mel spectrogram.", nameof(sample)));
        if (target.Rank != 2 || target.Shape[1] != _options.MelChannels
            || target.Shape[0] < _options.MelVaeDownsampleFactor || target.Shape[0] % _options.MelVaeDownsampleFactor != 0)
            throw new ArgumentException("Codec training mel must be [frames, mel channels] with a positive whole number of codec groups.", nameof(sample));
        int frames = target.Shape[0];
        starts = _options.CodecTraining.DiscriminatorWindows.Select(window => random.Next(Math.Max(1, frames - window + 1))).ToArray();
        var latents = MelVae!.Encode(target);
        var quantized = MelVae.Quantizer.TrainingForward(latents);
        generated = MelVae.Decode(quantized.Quantized, quantize: false);
        var loss = _codecTrainer!.Generator(target, generated, quantized.CommitmentLoss, quantized.VariationalLoss, starts);
        if (CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning)
        {
            if (string.IsNullOrWhiteSpace(sample.Text)) throw new ArgumentException("Semantic codec finetuning requires the transcript.", nameof(sample));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(_codecTrainer.SemanticLoss(latents, sample.Text!),
                Ops.FromDouble(_options.CodecTraining.SemanticWeight)));
        }
        return loss;
    }

    private T TrainCodec(TtsTrainingSample<T> sample)
    {
        EnsureCodecTraining();
        Tensor<T>? target = null, generation = null; int[]? starts = null;
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
        {
            var objective = CodecObjective(sample, _trainingRandom, out var real, out var fake, out var crops);
            target = new Tensor<T>(real.Shape.ToArray(), new Vector<T>(real.ToArray()));
            generation = new Tensor<T>(fake.Shape.ToArray(), new Vector<T>(fake.ToArray())); starts = crops;
            return objective;
        }, _codecGeneratorOptimizer!);
        _codecDiscriminatorTurn = true;
        try
        {
            TrainWithCustomObjective(sample.Tokens, sample.Tokens,
                (_, _) => _codecTrainer!.Discriminator(target!, generation!, starts!), _codecDiscriminatorOptimizer!);
        }
        finally { _codecDiscriminatorTurn = false; }
        return loss;
    }
}
