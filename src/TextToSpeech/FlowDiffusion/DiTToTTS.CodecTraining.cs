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
    internal int CodecGeneratorForwardCount { get; private set; }

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
        var codecTrainer = _codecTrainer ?? throw new InvalidOperationException("DiTTo codec trainer is not initialized.");
        if (codecTrainer.SemanticDecoder is null)
        {
            codecTrainer.InitializeSemanticDecoder(_options.LatentDim,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 111));
            Layers.AddRange(codecTrainer.ProjectionLayers); Layers.AddRange(codecTrainer.DecoderLayers);
            Freeze(codecTrainer.DecoderLayers);
        }
        EnsureLayerRandomSeedsWired();
        var semanticDecoder = codecTrainer.SemanticDecoder ?? throw new InvalidOperationException("DiTTo semantic decoder is not initialized.");
        using (var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(checkpoint, strictAxes: true))
            semanticDecoder.LoadHuggingFaceWeights(file);
        codecTrainer.DecoderWeightsLoaded = true;
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!IsCodecStage) return parameters;
        EnsureCodecTraining();
        var codecTrainer = _codecTrainer ?? throw new InvalidOperationException("DiTTo codec trainer is not initialized.");
        IEnumerable<LayerBase<T>> selectedLayers = _codecDiscriminatorTurn ? codecTrainer.Critics.Layers : _codecLayers;
        if (!_codecDiscriminatorTurn && CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning)
            selectedLayers = selectedLayers.Concat(codecTrainer.ProjectionLayers);
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(selectedLayers.Cast<ILayer<T>>().ToList()),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private Tensor<T> CodecObjective(TtsTrainingSample<T> sample, Random random)
    {
        var (target, starts) = CodecStep(sample, random);
        return CodecObjective(sample, target, starts, retainGeneration: false);
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

    private Tensor<T> CodecObjective(TtsTrainingSample<T> sample, Tensor<T> target, int[] starts, bool retainGeneration)
    {
        var melVae = MelVae ?? throw new InvalidOperationException("DiTTo Mel-VAE is not initialized.");
        var codecTrainer = _codecTrainer ?? throw new InvalidOperationException("DiTTo codec trainer is not initialized.");
        if (!AiDotNet.Tensors.Engines.Compilation.GraphMode.IsProbe && !Training.CompiledStepRandom<T>.InNonStepForward)
        {
            CodecGeneratorForwardCount++;
        }
        var latents = melVae.Encode(target);
        var quantized = melVae.Quantizer.TrainingForward(latents);
        var generated = melVae.Decode(quantized.Quantized, quantize: false);
        if (retainGeneration && GradientRecording.IsRecording<T>())
        {
            GradientRecording.RetainOutput("ditto-codec-generation", Engine.StopGradient(generated));
        }
        var loss = codecTrainer.Generator(target, generated, quantized.CommitmentLoss, quantized.VariationalLoss, starts);
        if (CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning)
        {
            string? text = sample.Text;
            if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Semantic codec finetuning requires the transcript.", nameof(sample));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(codecTrainer.SemanticLoss(latents, text),
                Ops.FromDouble(_options.CodecTraining.SemanticWeight)));
        }
        return loss;
    }

    private T TrainCodec(TtsTrainingSample<T> sample)
    {
        var (target, starts) = CodecStep(sample, _trainingRandom);
        var codecTrainer = _codecTrainer ?? throw new InvalidOperationException("DiTTo codec trainer is not initialized.");
        var generatorOptimizer = _codecGeneratorOptimizer ?? throw new InvalidOperationException("DiTTo codec generator optimizer is not initialized.");
        var discriminatorOptimizer = _codecDiscriminatorOptimizer ?? throw new InvalidOperationException("DiTTo codec discriminator optimizer is not initialized.");
        Tensor<T>? generation = null;
        T loss;
        using (var outputs = RetainedOutputCapture<T>.Begin())
        {
            loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens,
                (_, _) => CodecObjective(sample, target, starts, retainGeneration: true), generatorOptimizer);
            if (TryCompleteRetainedOutputs(outputs))
            {
                generation = outputs.Get("ditto-codec-generation");
            }
        }

        if (generation is not null)
        {
            _codecDiscriminatorTurn = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens,
                    (_, _) => codecTrainer.Discriminator(target, generation, starts), discriminatorOptimizer);
            }
            finally { _codecDiscriminatorTurn = false; }
        }
        return loss;
    }

    private static bool TryCompleteRetainedOutputs(RetainedOutputCapture<T> outputs)
    {
        try
        {
            outputs.Complete();
            return true;
        }
        catch (InvalidOperationException ex) when (ex.Message == "No successful training step published retained outputs.")
        {
            return false;
        }
    }
}
