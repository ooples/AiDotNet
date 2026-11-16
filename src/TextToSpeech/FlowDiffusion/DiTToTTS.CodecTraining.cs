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
        using (var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(checkpoint, strictAxes: true))
            _codecTrainer.SemanticDecoder!.LoadHuggingFaceWeights(file);
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

    private Tensor<T> CodecObjective(TtsTrainingSample<T> sample, Random random)
    {
        var (target, starts) = CodecStep(sample, random);
        return CodecObjective(sample, target, starts);
    }

    // The step's mel and its critics' crop starts: drawn once, on the host, before any objective runs, so a compiled
    // plan's probe and trace see the same crops the eager step does.
    private (Tensor<T> Target, int[] Starts) CodecStep(TtsTrainingSample<T> sample, Random random)
    {
        EnsureCodecTraining();
        var target = sample.Mel ?? (sample.Audio is not null ? ComputeMel(sample.Audio)
            : throw new ArgumentException("Codec training requires the recording or its mel spectrogram.", nameof(sample)));
        if (target.Rank != 2 || target.Shape[1] != _options.MelChannels
            || target.Shape[0] < _options.MelVaeDownsampleFactor || target.Shape[0] % _options.MelVaeDownsampleFactor != 0)
            throw new ArgumentException("Codec training mel must be [frames, mel channels] with a positive whole number of codec groups.", nameof(sample));
        int frames = target.Shape[0];
        var starts = _options.CodecTraining.DiscriminatorWindows.Select(window => random.Next(Math.Max(1, frames - window + 1))).ToArray();
        return (target, starts);
    }

    // The Mel-VAE's reconstruction of target through the straight-through quantizer.
    private (Tensor<T> Generated, Tensor<T> Commitment, Tensor<T> Variational) CodecReconstruction(Tensor<T> target)
    {
        var quantized = MelVae!.Quantizer.TrainingForward(MelVae.Encode(target));
        return (MelVae.Decode(quantized.Quantized, quantize: false), quantized.CommitmentLoss, quantized.VariationalLoss);
    }

    private Tensor<T> CodecObjective(TtsTrainingSample<T> sample, Tensor<T> target, int[] starts)
    {
        var latents = MelVae!.Encode(target);
        var quantized = MelVae.Quantizer.TrainingForward(latents);
        var generated = MelVae.Decode(quantized.Quantized, quantize: false);
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
        var (target, starts) = CodecStep(sample, _trainingRandom);
        // The critics' update scores the reconstruction the generator update starts from. It is computed here,
        // without gradients, before that update: a compiled plan does not run the objective with real values, so the
        // objective cannot hand its own reconstruction out. The Mel-VAE forward is deterministic, so this is the
        // value the generator objective computes.
        Tensor<T> generation;
        using (new NoGradScope<T>())
            generation = CodecReconstruction(target).Generated.Clone();
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens,
            (_, _) => CodecObjective(sample, target, starts), _codecGeneratorOptimizer!);
        _codecDiscriminatorTurn = true;
        try
        {
            TrainWithCustomObjective(sample.Tokens, sample.Tokens,
                (_, _) => _codecTrainer!.Discriminator(target, generation, starts), _codecDiscriminatorOptimizer!);
        }
        finally { _codecDiscriminatorTurn = false; }
        return loss;
    }
}
