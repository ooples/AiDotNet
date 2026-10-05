using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// VITS2: single-stage text-to-speech that improves VITS with an adversarially trained stochastic duration predictor,
/// noise-scaled monotonic alignment search, a Transformer block in the normalizing flows and a speaker-conditioned text
/// encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial
/// Learning and Architecture Design" (Kong et al., Interspeech 2023). The paper has no official code; what it leaves open
/// follows the community reference implementation (p0p4k/vits2_pytorch), as listed on <see cref="VITS2Options"/>.</para>
/// <para>
/// Training first fits the waveform networks as VITS does, without its duration model: the reconstruction, KL,
/// adversarial and feature-matching losses, with Gaussian noise ε = std(P) · N(0, 1) · s on the alignment scores while
/// s = 0.01 − 2·10⁻⁶ · step is positive (§2.2), the mel spectrogram as the posterior encoder's input and a Transformer
/// block in every flow coupling (§2.3). After <see cref="VITS2Options.AcousticTrainingSteps"/> steps the duration
/// predictor G(z_d, h_text) trains on its own (§2.1, "separately trained as the last training step") against the
/// alignment's log-durations d with L_mse (Eq. 3) + L_adv(G) (Eq. 2), and the time-step-wise discriminator
/// D(d, h_text) with L_adv(D) (Eq. 1). With several speakers the speaker embedding also enters the text encoder before its
/// third block (§2.4). Synthesis draws z_d, predicts log-durations, expands the prior, samples it and inverts the flow.
/// </para>
/// <para><b>For Beginners:</b> VITS2 goes straight from text to a waveform like VITS, but learns how long each sound
/// lasts by competing against a judge that tells real durations from predicted ones, which makes speech sound more
/// natural.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design",
    "https://arxiv.org/abs/2307.16430",
    Year = 2023,
    Authors = "Kong et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, ReferenceBatchSize = 256,
                Source = "Kong et al. 2023, Sec. 3: AdamW with beta1 0.8, beta2 0.99 and weight decay 0.01, an initial "
                        + "learning rate of 2e-4 decayed by 0.999^(1/8) every epoch, 256 training instances per step.")]
public partial class VITS2<T> : VitsTtsModelBase<T>
{
    private Vits2DurationPredictor<T>? _durationPredictor;
    private Vits2DurationDiscriminator<T>? _durationDiscriminator;

    /// <summary>Creates a VITS2 model that runs an exported ONNX graph.</summary>
    public VITS2(NeuralNetworkArchitecture<T> architecture, string modelPath, VITS2Options? options = null)
        : base(architecture, modelPath, options ?? new VITS2Options())
    {
    }

    /// <summary>Creates a trainable VITS2 model.</summary>
    public VITS2(NeuralNetworkArchitecture<T> architecture, VITS2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VITS2Options(), optimizer)
    {
    }

    private VITS2Options PaperOptions => (VITS2Options)VitsOptions;

    /// <summary>Whether the next training step trains the duration predictor (after the waveform networks).</summary>
    public bool TrainsDurationPredictor => TrainingSteps >= PaperOptions.AcousticTrainingSteps;

    /// <inheritdoc />
    protected override bool PosteriorReadsMel => true;

    /// <inheritdoc />
    protected override double AlignmentNoiseScale(long step)
        => Math.Max(0.0, PaperOptions.AlignmentNoiseScale - PaperOptions.AlignmentNoiseDecay * step);

    /// <inheritdoc />
    protected override int SpeakerConditionedEncoderBlock => PaperOptions.SpeakerConditionedEncoderBlock;

    /// <inheritdoc />
    protected override (int Layers, int Heads, int KernelSize, double Dropout) FlowTransformer
        => (PaperOptions.FlowTransformerLayers, PaperOptions.FlowTransformerHeads, PaperOptions.FlowTransformerKernelSize, PaperOptions.FlowTransformerDropout);

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>> CreateDurationModel(int hidden, int speakerChannels, int languageChannels)
    {
        var o = PaperOptions;
        _durationPredictor = new Vits2DurationPredictor<T>(Engine, hidden, o.DurationPredictorFilterChannels,
            o.DurationPredictorKernelSize, o.DurationPredictorDropout, speakerChannels, languageChannels);
        _durationDiscriminator = new Vits2DurationDiscriminator<T>(Engine, hidden, hidden, o.DurationDiscriminatorKernelSize);
        return _durationPredictor.Layers.Concat(_durationDiscriminator.Layers);
    }

    /// <inheritdoc />
    /// <remarks>None: the duration predictor trains after the waveform networks.</remarks>
    protected override IEnumerable<LayerBase<T>> JointDurationLayers => Array.Empty<LayerBase<T>>();

    /// <inheritdoc />
    protected override Tensor<T>? JointDurationLoss(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, int[] durations, Random random) => null;

    /// <inheritdoc />
    protected override double[] PredictDurations(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, Random random)
    {
        var noise = Gaussian(new[] { 1, 1, hidden.Shape[2] }, random, PaperOptions.DurationNoiseScale);
        var logDurations = _durationPredictor!.Forward(hidden, speaker, language, noise);
        var durations = new double[logDurations.Length];
        for (int i = 0; i < durations.Length; i++) durations[i] = Math.Exp(NumOps.ToDouble(logDurations[i]));
        return durations;
    }

    /// <inheritdoc />
    protected override void TrainUtterance(VitsUtterance data, VitsTrainingDraw draw)
    {
        if (!TrainsDurationPredictor)
        {
            TrainAcoustic(data, draw);
            return;
        }
        var target = DurationTarget(data, draw);

        Tensor<T> generated;
        using (new NoGradScope<T>())
        {
            var g = _durationPredictor!.Forward(target.Hidden, target.Speaker, target.Language, target.Noise);
            generated = new Tensor<T>(g._shape, g.ToVector());
        }
        TrainLayers(_durationDiscriminator!.Layers.Cast<ILayer<T>>().ToList(), data, () =>
        {
            var features = _durationDiscriminator.Encode(target.Hidden);
            return LeastSquaresDiscriminatorLoss(_durationDiscriminator.Score(features, target.LogDurations),
                _durationDiscriminator.Score(features, generated));
        }, "durationDiscriminator");
        TrainLayers(_durationPredictor.Layers.Cast<ILayer<T>>().ToList(), data, () => DurationGeneratorLoss(target, adversarial: true), "duration");
    }

    /// <inheritdoc />
    /// <remarks>Before the duration phase, the reconstruction and KL terms; in it, L_mse (Eq. 3) — the adversarial term
    /// is measured against a discriminator trained at the same time.</remarks>
    protected override Tensor<T> Objective(VitsUtterance data, VitsTrainingDraw draw)
        => TrainsDurationPredictor ? DurationGeneratorLoss(DurationTarget(data, draw), adversarial: false) : base.Objective(data, draw);

    private sealed record DurationTraining(Tensor<T> Hidden, Tensor<T>? Speaker, Tensor<T>? Language, Tensor<T> LogDurations, Tensor<T> Noise);

    // The stop-gradient text encoding and speaker, the alignment's log-durations log(d + 1e-6) and the noise z_d.
    private DurationTraining DurationTarget(VitsUtterance data, VitsTrainingDraw draw)
    {
        VitsAlignment alignment;
        using (new NoGradScope<T>()) alignment = Align(data, draw, training: true);
        int tokens = alignment.Durations.Length;
        var logDurations = new Tensor<T>(new[] { 1, 1, tokens });
        for (int i = 0; i < tokens; i++) logDurations[0, 0, i] = NumOps.FromDouble(Math.Log(alignment.Durations[i] + 1e-6));
        var hidden = new Tensor<T>(alignment.Hidden._shape, alignment.Hidden.ToVector());
        var speaker = alignment.Speaker is null ? null : new Tensor<T>(alignment.Speaker._shape, alignment.Speaker.ToVector());
        var language = alignment.Language is null ? null : new Tensor<T>(alignment.Language._shape, alignment.Language.ToVector());
        var noise = Gaussian(new[] { 1, 1, tokens }, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(draw.DurationSeed));
        return new DurationTraining(hidden, speaker, language, logDurations, noise);
    }

    // L_mse (Eq. 3) + L_adv(G) (Eq. 2) for the duration predictor.
    private Tensor<T> DurationGeneratorLoss(DurationTraining target, bool adversarial)
    {
        var predicted = _durationPredictor!.Forward(target.Hidden, target.Speaker, target.Language, target.Noise);
        var error = Engine.TensorSubtract(predicted, target.LogDurations);
        var loss = Mean(Engine.TensorMultiply(error, error));
        if (!adversarial) return loss;
        var score = _durationDiscriminator!.Score(_durationDiscriminator.Encode(target.Hidden), predicted);
        return Engine.TensorAdd(loss, LeastSquaresGeneratorLoss(score));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "VITS2-ONNX" : "VITS2-Native",
            Description = "VITS2: Single-Stage TTS with Adversarial Learning and Architecture Design (Kong et al., 2023)",
            FeatureCount = PaperOptions.HiddenDim,
            Complexity = PaperOptions.NumEncoderLayers + PaperOptions.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "VITS2";
        m.AdditionalInfo["SampleRate"] = PaperOptions.SampleRate.ToString();
        return m;
    }
}
