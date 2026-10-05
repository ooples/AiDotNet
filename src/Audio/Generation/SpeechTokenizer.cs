using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// SpeechTokenizer: EnCodec's encoder–quantizer–decoder with a BiLSTM encoder, whose first residual codebook is distilled
/// toward a self-supervised teacher's representations so that it carries the content of speech and the remaining codebooks
/// the paralinguistic detail.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> "SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models" (Zhang et al., ICLR
/// 2024), with ZhangXInFD/SpeechTokenizer for what the paper leaves unstated.</para>
/// <para>
/// The encoder and decoder (§3.1, App. D) are EnCodec's SEANet (C = 32, strides 2, 4, 5, 8; residual kernels 3 and 1;
/// weight normalization; ELU) with the encoder's two-layer LSTM replaced by a BiLSTM; at 16 kHz the 1024-dimensional
/// latent runs at 50 frames per second. Eight EMA codebooks of 1024 entries quantize it as EnCodec's do. The first
/// codebook's quantized output Q1 is distilled (§3.2) toward the teacher's representations S (HuBERT's average layer) by
/// the "D-axis" loss <c>−(1/D) Σ_d log σ(cos(A Q1^(:,d), S^(:,d)))</c>, with A a learned projection.
/// </para>
/// <para>
/// Training (§3.3) adds a time L1 loss, EnCodec's multi-scale mel loss, hinge adversarial and relative feature-matching
/// losses against HiFi-Codec's discriminators (MS-STFT, multi-period, multi-scale) and the commitment loss. The paper states
/// no weights; they are the reference's (time 500, mel 45, adversarial 1, feature 1, commitment 10, distillation 120). The
/// teacher's representations are an input: the reference also trains on representations extracted beforehand.
/// </para>
/// <para><b>For Beginners:</b> Give it speech and it returns eight rows of codes; the first row is close to the words,
/// the rest to the voice. Decoding all eight rebuilds the speech.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models", "https://arxiv.org/abs/2308.16692",
    Year = 2024, Authors = "Zhang et al.")]
public partial class SpeechTokenizer<T> : NeuralAudioCodecBase<T>
{
    private SeanetEncoder<T>? _encoder;
    private SeanetDecoder<T>? _decoder;
    private ResidualVectorQuantizerLayer<T>? _quantizer;
    private DenseLayer<T>? _projection;
    private MultiScaleStftDiscriminator<T>? _stft;
    private HiFiGanDiscriminators<T>? _waveform;
    private List<CodecMelScale<T>>? _melScales;
    private Tensor<T>? _teacher;

    /// <summary>Creates a SpeechTokenizer that runs an exported ONNX graph.</summary>
    public SpeechTokenizer(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeechTokenizerOptions? options = null)
        : base(architecture, modelPath, options ?? new SpeechTokenizerOptions())
    {
    }

    /// <summary>Creates a trainable SpeechTokenizer.</summary>
    public SpeechTokenizer(NeuralNetworkArchitecture<T> architecture, SpeechTokenizerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpeechTokenizerOptions(), optimizer)
    {
    }

    private SpeechTokenizerOptions PaperOptions => (SpeechTokenizerOptions)CodecSettings;

    /// <inheritdoc />
    public override int HopLength => PaperOptions.Ratios.Aggregate(1, (a, b) => a * b);

    /// <summary>The semantic codes <c>[frames]</c> of speech: the first codebook's, which carry the content.</summary>
    public int[] EncodeSemantic(Tensor<T> audio)
    {
        var codes = Encode(audio);
        return Enumerable.Range(0, codes.GetLength(1)).Select(t => codes[0, t]).ToArray();
    }

    /// <inheritdoc />
    protected override IResidualVectorQuantizer<T> CreateCodec(List<LayerBase<T>> layers)
    {
        var o = PaperOptions;
        var config = new SeanetConfig
        {
            Channels = o.Channels,
            Dimension = o.Dimension,
            Filters = o.Filters,
            ResidualLayers = 1,
            Ratios = o.Ratios,
            Normalization = ConvolutionNormalization.Weight,
            ResidualKernelSizes = [3, 1],
            DilationBase = 2,
            Causal = false,
            Reflect = true,
            TrueSkip = false,
            Compress = 2,
            LstmLayers = o.LstmLayers,
            BidirectionalLstm = true,
        };
        _encoder = new SeanetEncoder<T>(Engine, config, layers);
        _quantizer = new ResidualVectorQuantizerLayer<T>(o.Dimension, o.NumQuantizers, o.CodebookSize, o.CodebookDecay, o.KMeansIterations,
            o.DeadCodeThreshold, o.CommitmentAsMean);
        layers.Add(_quantizer);
        _decoder = new SeanetDecoder<T>(Engine, config, layers);
        _projection = new DenseLayer<T>(o.SemanticDimension, new IdentityActivation<T>() as IActivationFunction<T>);
        layers.Add(_projection);
        _melScales = o.MelScales.Select(i => new CodecMelScale<T>(Engine, o.SampleRate, 1 << i, (1 << i) / 4, o.MelBins, 0.0, null,
            normalized: true)).ToList();
        return _quantizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        var layers = new List<LayerBase<T>>();
        _stft = new MultiScaleStftDiscriminator<T>(Engine, o.StftDiscriminatorWindows, o.StftDiscriminatorFilters, [1, 2, 4], 3, 9, 0.2, layers);
        _waveform = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 3, useScaleDiscriminator: true, widthDivisor: o.DiscriminatorWidthDivisor);
        layers.AddRange(_waveform.Layers);
        return layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> EncodeLatent(Tensor<T> audio) => _encoder!.Forward(audio);

    /// <inheritdoc />
    protected override Tensor<T> DecodeLatent(Tensor<T> latent) => _decoder!.Forward(latent);

    /// <inheritdoc />
    protected override Tensor<T> Reconstruct(Tensor<T> audio, int quantizers)
    {
        var output = base.Reconstruct(audio, quantizers);
        int length = audio.Shape[2];
        return output.Shape[2] > length ? Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { 1, output.Shape[1], length }) : output;
    }

    /// <inheritdoc />
    protected override (int Quantizers, int Bandwidth) SampleTrainingBandwidth(Random random) => (PaperOptions.NumQuantizers, 0);

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="expectedOutput"/> holds the teacher's representations <c>[frames, SemanticDimension]</c> of
    /// <paramref name="input"/>, one per 20 ms frame (HuBERT base at 16 kHz); a matching segment of both is drawn per step.
    /// SpeechTokenizer cannot train without them: the first codebook's role comes from the distillation (§3.2).
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (expectedOutput.Rank != 2 || expectedOutput.Shape[1] != PaperOptions.SemanticDimension)
            throw new NotSupportedException(
                $"SpeechTokenizer trains on speech and its semantic teacher's representations [frames, {PaperOptions.SemanticDimension}] " +
                "(HuBERT's average layer, §3.2); pass them as the expected output.");
        var audio = AsBatch(input);
        int hop = HopLength, frames = Math.Min(expectedOutput.Shape[0], audio.Shape[2] / hop);
        int segmentFrames = Math.Max(1, Math.Min(frames, CodecSettings.SegmentSize / hop));
        int start = frames > segmentFrames ? TrainingRandom.Next(0, frames - segmentFrames + 1) : 0;
        var source = Engine.TensorSlice(audio, new[] { 0, 0, start * hop }, new[] { 1, audio.Shape[1], segmentFrames * hop });
        _teacher = Engine.TensorSlice(expectedOutput, new[] { start, 0 }, new[] { segmentFrames, PaperOptions.SemanticDimension });
        try
        {
            TrainOnSegment(source, source);
        }
        finally
        {
            _teacher = null;
        }
    }

    // ---------------------------------------------------------------- losses

    // −(1/D) Σ_d log σ(cos(A Q1^(:,d), S^(:,d))): cosine similarity along time, per teacher dimension (§3.2, "D-axis").
    private Tensor<T> DistillationLoss(Tensor<T> firstQuantized, Tensor<T> teacher)
        => Engine.TensorNegate(Mean(Engine.TensorLog(Engine.Sigmoid(Cosines(firstQuantized, teacher)))));

    /// <summary>
    /// The mean over the teacher's dimensions of the cosine similarity along time between the projected first-codebook output
    /// A·Q1 and the teacher's representations — the agreement §3.2's distillation raises.
    /// </summary>
    /// <param name="audio">Speech <c>[samples]</c>.</param>
    /// <param name="teacher">The teacher's representations <c>[frames, SemanticDimension]</c>.</param>
    public double DistillationAgreement(Tensor<T> audio, Tensor<T> teacher)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            _quantizer!.ActiveQuantizers = PaperOptions.NumQuantizers;
            _quantizer.Forward(EncodeLatent(AsBatch(audio)));
            return NumOps.ToDouble(Mean(Cosines(_quantizer.FirstQuantized!, teacher))[0]);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // cos(A Q1^(:,d), S^(:,d)) for every teacher dimension d: [D].
    private Tensor<T> Cosines(Tensor<T> firstQuantized, Tensor<T> teacher)
    {
        int frames = Math.Min(firstQuantized.Shape[2], teacher.Shape[0]);
        var rows = Engine.TensorTranspose(Engine.Reshape(Engine.TensorSlice(firstQuantized, new[] { 0, 0, 0 }, new[] { 1, firstQuantized.Shape[1], frames }),
            new[] { firstQuantized.Shape[1], frames }));                                          // [T, D]
        var projected = _projection!.Forward(rows);                                                // [T, 768]
        var target = Engine.TensorSlice(teacher, new[] { 0, 0 }, new[] { frames, teacher.Shape[1] });
        var dot = Engine.ReduceSum(Engine.TensorMultiply(projected, target), new[] { 0 }, keepDims: false);
        var norms = Engine.TensorMultiply(
            Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(projected, projected), new[] { 0 }, keepDims: false), NumOps.FromDouble(1e-16)), NumOps.FromDouble(0.5)),
            Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(target, target), new[] { 0 }, keepDims: false), NumOps.FromDouble(1e-16)), NumOps.FromDouble(0.5)));
        return Engine.TensorDivide(dot, norms);
    }

    private Tensor<T> TimeLoss(Tensor<T> real, Tensor<T> generated) => Mean(Engine.TensorAbs(Engine.TensorSubtract(real, generated)));

    // EnCodec's multi-scale mel loss with mean reductions: per scale mean|d| + mean d², summed over the scales.
    private Tensor<T> FrequencyLoss(Tensor<T> real, Tensor<T> generated)
    {
        var x = Flat(real);
        var y = Flat(generated);
        var terms = new List<Tensor<T>>();
        foreach (var scale in _melScales!)
        {
            Tensor<T> target;
            using (new NoGradScope<T>()) target = Detached(scale.Forward(x));
            var d = Engine.TensorSubtract(target, scale.Forward(y));
            terms.Add(Engine.TensorAdd(Mean(Engine.TensorAbs(d)), Mean(Engine.TensorMultiply(d, d))));
        }
        return Sum(terms);
    }

    private List<(Tensor<T> Logits, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
    {
        var flat = Flat(audio);
        var outputs = _stft!.Forward(flat);
        foreach (var (score, features) in _waveform!.Forward(flat)) outputs.Add((score, features));
        return outputs;
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var o = PaperOptions;
        var fake = Discriminate(generated);
        List<(Tensor<T> Logits, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = Discriminate(real);
        var adversarial = Engine.TensorMultiplyScalar(
            Sum(fake.Select(f => Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(f.Logits), NumOps.One))))), NumOps.FromDouble(1.0 / fake.Count));
        var featureTerms = new List<Tensor<T>>();
        for (int k = 0; k < fake.Count; k++)
            for (int l = 0; l < fake[k].Features.Count; l++)
            {
                var r = Detached(realOut[k].Features[l]);
                double scale = NumOps.ToDouble(Mean(Engine.TensorAbs(r))[0]);
                featureTerms.Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(r, fake[k].Features[l]))),
                    NumOps.FromDouble(1.0 / Math.Max(scale, 1e-12))));
            }
        var feature = Engine.TensorMultiplyScalar(Sum(featureTerms), NumOps.FromDouble(1.0 / featureTerms.Count));
        var terms = new List<Tensor<T>>
        {
            Engine.TensorMultiplyScalar(TimeLoss(real, generated), NumOps.FromDouble(o.TimeLossWeight)),
            Engine.TensorMultiplyScalar(FrequencyLoss(real, generated), NumOps.FromDouble(o.FrequencyLossWeight)),
            Engine.TensorMultiplyScalar(adversarial, NumOps.FromDouble(o.AdversarialLossWeight)),
            Engine.TensorMultiplyScalar(feature, NumOps.FromDouble(o.FeatureLossWeight)),
        };
        if (_quantizer!.CommitmentLoss is { } commitment) terms.Add(Engine.TensorMultiplyScalar(commitment, NumOps.FromDouble(o.CommitmentLossWeight)));
        if (_teacher is not null && _quantizer.FirstQuantized is { } first)
            terms.Add(Engine.TensorMultiplyScalar(DistillationLoss(first, _teacher), NumOps.FromDouble(o.DistillationLossWeight)));
        return Sum(terms);
    }

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var r = Discriminate(real);
        var g = Discriminate(generated);
        return Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(
                Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(r[k].Logits), NumOps.One))),
                Mean(Engine.ReLU(Engine.TensorAddScalar(g[k].Logits, NumOps.One)))))),
            NumOps.FromDouble(1.0 / r.Count));
    }

    /// <inheritdoc />
    /// <remarks>λ_t ℓ_t + λ_f ℓ_f.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated)
        => Engine.TensorAdd(Engine.TensorMultiplyScalar(TimeLoss(real, generated), NumOps.FromDouble(PaperOptions.TimeLossWeight)),
            Engine.TensorMultiplyScalar(FrequencyLoss(real, generated), NumOps.FromDouble(PaperOptions.FrequencyLossWeight)));

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        int perEpoch = Math.Max(1, o.UpdatesPerEpoch);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate, step => Math.Pow(o.LearningRateDecay, step / perEpoch));
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                UseAdaptiveBetas = false,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            });
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "SpeechTokenizer-ONNX" : "SpeechTokenizer-Native",
            Description = "SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models (Zhang et al., 2024)",
            FeatureCount = o.Channels,
            Complexity = o.NumQuantizers,
        };
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        m.AdditionalInfo["FrameRate"] = TokenFrameRate.ToString();
        return m;
    }
}
