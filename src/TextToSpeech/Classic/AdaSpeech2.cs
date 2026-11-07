using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// AdaSpeech 2: adaptive TTS that adapts to a new voice from untranscribed speech, through a mel-spectrogram encoder
/// aligned to the phoneme encoder's output space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "AdaSpeech 2: Adaptive Text to Speech with Untranscribed Data" (Yan et al., ICASSP 2021).</para>
/// <para>
/// The TTS pipeline is AdaSpeech's (§2.1, <see cref="AdaSpeech{T}"/>): phoneme encoder, acoustic condition modeling,
/// variance adaptor and a mel decoder with conditional layer normalization. AdaSpeech 2 adds a mel-spectrogram encoder
/// of 4 feed-forward Transformer blocks ("considering the symmetry of the system") and a four-step pipeline (§2, Fig. 2),
/// selected with <see cref="CurrentStep"/>:
/// </para>
/// <list type="number">
/// <item><b>Source model training</b>: AdaSpeech training on transcribed multi-speaker data
/// (<see cref="AdaSpeech{T}.CurrentPhase"/> selects AdaSpeech's own phases).</item>
/// <item><b>Mel encoder aligning</b> (§2.2): the source model is frozen and only the mel encoder trains, with an L2 loss
/// between its output and the phoneme encoder's hidden sequence expanded by the phoneme durations.</item>
/// <item><b>Untranscribed speech adaptation</b> (§2.3): speech is reconstructed through the mel encoder and the mel
/// decoder (<see cref="TrainUntranscribed(Tensor{T}, int, double[], double[])"/>), adapting only the parameters of the
/// conditional layer normalizations (with the speaker embedding, as AdaSpeech adapts).</item>
/// <item><b>Inference</b> (§2.4): the unadapted phoneme encoder with the adapted decoder — the ordinary synthesis path.</item>
/// </list>
/// <para>
/// The paper does not say how the mel encoder reads 80-bin frames into its 256-wide blocks; a linear projection and the
/// sinusoidal positions of the phoneme encoder's input do it. In reconstruction the speaker embedding, the utterance-level
/// vector and the frame pitch and energy embeddings are added as in synthesis; the phoneme-level vectors are not, since
/// untranscribed speech has no phoneme alignment to average frames over.
/// </para>
/// <para><b>For Beginners:</b> AdaSpeech 2 can learn a new voice from recordings that have no transcript: it learns to
/// read a spectrogram into the same internal representation it reads text into, then practises reproducing the new
/// speaker's recordings.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "AdaSpeech 2: Adaptive Text to Speech with Untranscribed Data",
    "https://arxiv.org/abs/2104.09715",
    Year = 2021,
    Authors = "Yan et al."
)]
// [PaperOptimizer] is inherited from AdaSpeech: Yan et al. 2021, Sec. 3.1 state the same Adam (beta1 0.9, beta2 0.98,
// epsilon 1e-9) and, like AdaSpeech, no learning rate. A second declaration would be a duplicate recipe (AIDN103).
public partial class AdaSpeech2<T> : AdaSpeech<T>
{
    private readonly List<LayerBase<T>> _melEncoder = new();

    /// <summary>Loads an ONNX AdaSpeech 2.</summary>
    public AdaSpeech2(NeuralNetworkArchitecture<T> architecture, string modelPath, AdaSpeech2Options? options = null)
        : base(architecture, modelPath, options ?? new AdaSpeech2Options()) { }

    /// <summary>Creates a trainable AdaSpeech 2.</summary>
    public AdaSpeech2(
        NeuralNetworkArchitecture<T> architecture,
        AdaSpeech2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new AdaSpeech2Options(), optimizer) { }

    /// <summary>The step of the adaptation pipeline (§2, Fig. 2) that training runs.</summary>
    public AdaSpeech2TrainingStep CurrentStep { get; set; } = AdaSpeech2TrainingStep.SourceModel;

    /// <summary>The mel-spectrogram encoder's layers: input projection, positions, FFT blocks.</summary>
    internal IReadOnlyList<LayerBase<T>> MelEncoderLayers => _melEncoder;

    private protected override object TrainingPhaseKey =>
        CurrentStep == AdaSpeech2TrainingStep.SourceModel ? base.TrainingPhaseKey : CurrentStep;

    protected override void InitializeLayers()
    {
        base.InitializeLayers();
        if (!HasAcousticConditioning)
            return;
        var options = (AdaSpeech2Options)AdaOptions;
        int hidden = options.HiddenDim;
        _melEncoder.Add(new DenseLayer<T>(hidden, new IdentityActivation<T>() as IActivationFunction<T>));
        _melEncoder.Add(new PositionalEncodingLayer<T>(options.MaxMelLength, hidden));
        for (int i = 0; i < options.NumMelEncoderLayers; i++)
            _melEncoder.Add(new FeedForwardTransformerBlock<T>(hidden, options.NumHeads, options.FftFilterSize,
                options.FftKernelSizes[0], options.FftKernelSizes[1], options.DropoutRate));
        ComponentLayers.AddRange(_melEncoder);
    }

    private Tensor<T> RunMelEncoder(Tensor<T> mel)
    {
        var x = mel;
        foreach (var layer in _melEncoder)
            x = layer.Forward(x);
        return x;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        switch (CurrentStep)
        {
            case AdaSpeech2TrainingStep.SourceModel:
                return base.TrainOnSample(sample);
            case AdaSpeech2TrainingStep.MelEncoderAligning:
            {
                var (mel, objective) = AligningObjective(sample);
                return TrainWithCustomObjective(sample.Tokens, mel, objective, TrainingOptimizer);
            }
            default:
                throw new InvalidOperationException(
                    $"Untranscribed adaptation trains on speech alone; call {nameof(TrainUntranscribed)}.");
        }
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        if (CurrentStep != AdaSpeech2TrainingStep.MelEncoderAligning)
            return base.EvaluateTrainingObjective(sample);
        var (mel, objective) = AligningObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            return objective(sample.Tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Mel encoder aligning (§2.2): the L2 distance between the mel encoder's output and the phoneme encoder's hidden
    /// sequence expanded by the durations. The source model is frozen, so the phoneme side carries no gradient.
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) AligningObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        var durations = sample.Durations ?? throw new ArgumentException(
            $"Mel encoder aligning expands the phoneme hidden sequence by its durations; set {nameof(sample.Durations)}.",
            nameof(sample));
        var mel = DeriveAcousticTargets(sample).Mel;
        if (durations.Sum() != mel.Shape[0])
            throw new ArgumentException(
                $"The durations sum to {durations.Sum()} frames but the mel spectrogram has {mel.Shape[0]}.", nameof(sample));

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> target)
        {
            Tensor<T> expanded;
            using (new NoGradScope<T>())
            {
                var phonemeHidden = RunEncoder(tokens);
                expanded = LengthRegulator.Expand(phonemeHidden, durations);
            }
            var label = new Tensor<T>(expanded._shape, expanded.ToVector());
            return MeanSquaredError(RunMelEncoder(target), label);
        }

        return (mel, Objective);
    }

    /// <summary>
    /// One untranscribed-speech adaptation step (§2.3): reconstructs <paramref name="mel"/> through the mel encoder and
    /// the mel decoder and updates only the conditional layer normalizations and the speaker embedding.
    /// </summary>
    /// <param name="mel">The target speaker's speech as a mel spectrogram, <c>[frames, melChannels]</c>.</param>
    /// <param name="speakerId">The speaker's row in the speaker embedding table.</param>
    /// <param name="pitch">F0 per frame (Hz, 0 unvoiced); required when the model predicts pitch.</param>
    /// <param name="energy">Energy per frame; required when the model predicts energy.</param>
    /// <returns>The reconstruction loss of the step.</returns>
    public T TrainUntranscribed(Tensor<T> mel, int speakerId, double[]? pitch = null, double[]? energy = null)
    {
        Guard.NotNull(mel);
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (CurrentStep != AdaSpeech2TrainingStep.UntranscribedAdaptation)
            throw new InvalidOperationException(
                $"Set {nameof(CurrentStep)} to {AdaSpeech2TrainingStep.UntranscribedAdaptation} before adapting on untranscribed speech.");
        return TrainWithCustomObjective(mel, mel, ReconstructionObjective(mel, speakerId, pitch, energy), TrainingOptimizer);
    }

    /// <summary>One untranscribed-speech adaptation step on a recording, deriving its mel spectrogram, WORLD pitch and
    /// frame energy as the source model's training does.</summary>
    public T TrainUntranscribed(Tensor<T> audio, int speakerId)
    {
        Guard.NotNull(audio);
        var targets = DeriveAcousticTargets(new TtsTrainingSample<T> { Tokens = new Tensor<T>(new[] { 0 }), Audio = audio });
        return TrainUntranscribed(targets.Mel, speakerId, targets.Pitch, targets.Energy);
    }

    /// <summary>The untranscribed reconstruction loss on <paramref name="mel"/>, without updating the model.</summary>
    public T EvaluateUntranscribed(Tensor<T> mel, int speakerId, double[]? pitch = null, double[]? energy = null)
    {
        Guard.NotNull(mel);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return ReconstructionObjective(mel, speakerId, pitch, energy)(mel, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> ReconstructionObjective(
        Tensor<T> mel, int speakerId, double[]? pitch, double[]? energy)
    {
        if (mel.Rank != 2 || mel.Shape[1] != AdaOptions.MelChannels)
            throw new ArgumentException(
                $"Expected a mel spectrogram [frames, {AdaOptions.MelChannels}], got [{string.Join(", ", mel.Shape)}].", nameof(mel));
        if (VarianceAdaptor.UsePitch && pitch is null)
            throw new ArgumentException("The model embeds frame pitch; supply it, or the recording.", nameof(pitch));
        if (VarianceAdaptor.UseEnergy && energy is null)
            throw new ArgumentException("The model embeds frame energy; supply it, or the recording.", nameof(energy));

        return (input, target) =>
        {
            var speaker = SpeakerEmbedding(speakerId);
            var frames = RunMelEncoder(input);
            frames = Engine.TensorAdd(frames, Expand(speaker, frames));
            frames = Engine.TensorAdd(frames, Expand(UtteranceEncoder.Forward(input), frames));
            frames = VarianceAdaptor.AddFrameVariance(frames, pitch, energy);
            return MeanAbsoluteError(RunDecoderLayers(frames, speaker), target);
        };
    }

    /// <inheritdoc />
    /// <remarks>Source model training trains what AdaSpeech's phase trains, never the mel encoder; aligning trains only
    /// the mel encoder; untranscribed adaptation only the conditional layer normalizations and the speaker embedding.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasAcousticConditioning)
            return parameters;
        var melEncoder = new HashSet<Tensor<T>>(
            Training.TapeTrainingStep<T>.CollectParameters(_melEncoder.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        switch (CurrentStep)
        {
            case AdaSpeech2TrainingStep.SourceModel:
                return base.SelectTrainableParametersForTraining(parameters).Where(p => !melEncoder.Contains(p)).ToList();
            case AdaSpeech2TrainingStep.MelEncoderAligning:
                return parameters.Where(melEncoder.Contains).ToList();
            default:
            {
                var adaptive = new HashSet<Tensor<T>>(
                    Training.TapeTrainingStep<T>.CollectParameters(AdaptationLayers(), -1),
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                return parameters.Where(adaptive.Contains).ToList();
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = base.GetModelMetadata();
        m.Name = m.Name.Replace("AdaSpeech", "AdaSpeech2");
        m.Description = "AdaSpeech 2: Adaptive Text to Speech with Untranscribed Data (Yan et al., 2021)";
        m.AdditionalInfo["Architecture"] = "AdaSpeech2";
        return m;
    }
}

/// <summary>The steps of AdaSpeech 2's adaptation pipeline (Yan et al. 2021, §2, Fig. 2) that train the model.</summary>
public enum AdaSpeech2TrainingStep
{
    /// <summary>Source model training: AdaSpeech training on transcribed data.</summary>
    SourceModel = 1,

    /// <summary>Mel encoder aligning: only the mel encoder trains, toward the expanded phoneme hidden sequence.</summary>
    MelEncoderAligning = 2,

    /// <summary>Untranscribed speech adaptation: reconstruction through the mel encoder and decoder, adapting only the
    /// conditional layer normalizations and the speaker embedding.</summary>
    UntranscribedAdaptation = 3,
}
