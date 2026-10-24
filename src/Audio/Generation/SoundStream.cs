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
/// SoundStream: a fully convolutional, causal encoder–decoder whose embeddings are compressed by a residual vector
/// quantizer trained with quantizer dropout, so one model serves every bitrate up to its maximum, trained adversarially
/// against a wave-based and an STFT-based discriminator, with optional denoising through FiLM conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> "SoundStream: An End-to-End Neural Audio Codec" (Zeghidour et al., 2021). No official
/// implementation exists; where the paper is silent this follows EnCodec's reimplementation of SoundStream (Défossez et al.
/// 2022, App. A.2), which used EnCodec's training setup.</para>
/// <para>
/// The encoder (§III-B) is a causal k = 7 convolution to C = 32 channels, four blocks of three residual units (dilations 1,
/// 3, 9; kernels 7 then 1) and a strided convolution (kernel 2S; strides 2, 4, 5, 8) doubling the channels, and a k = 3
/// convolution to D; the decoder mirrors it (k = 7 to 16C, transposed convolutions, k = 7 to the waveform). ELU, no
/// normalization. The residual vector quantizer (§III-C) has N_q codebooks of 1024 entries with EMA updates, k-means
/// initialization and dead-code replacement; each training example uses n_q ~ U[1, N_q] of them. FiLM layers (§III-F)
/// scale and shift the bottleneck from a one-hot denoising flag.
/// </para>
/// <para>
/// Training (§III-E) minimizes λ_adv L_adv + λ_feat L_feat + λ_rec L_rec (1, 100, 1; Eq. 6): the hinge adversarial loss
/// averaged over the STFT discriminator and three MelGAN wave discriminators (Eq. 1–2), the mean L1 distance between their
/// internal features (Eq. 3), and the multi-scale mel loss over windows 2⁶ … 2¹¹ (Eq. 4–5).
/// </para>
/// <para><b>For Beginners:</b> SoundStream squeezes audio into a short stream of codes and rebuilds it. Train it once and
/// choose the bitrate later by using more or fewer codebooks; it can also clean up background noise as it compresses.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SoundStream: An End-to-End Neural Audio Codec", "https://arxiv.org/abs/2107.03312", Year = 2021, Authors = "Zeghidour et al.")]
public partial class SoundStream<T> : NeuralAudioCodecBase<T>
{
    private SeanetEncoder<T>? _encoder;
    private SeanetDecoder<T>? _decoder;
    private DenseLayer<T>? _film;
    private MelGanDiscriminators<T>? _wave;
    private SoundStreamStftDiscriminator<T>? _stft;
    private List<CodecMelScale<T>>? _melScales;
    private bool _denoise;

    /// <summary>Creates a SoundStream that runs an exported ONNX graph.</summary>
    public SoundStream(NeuralNetworkArchitecture<T> architecture, string modelPath, SoundStreamOptions? options = null)
        : base(architecture, modelPath, options ?? new SoundStreamOptions())
    {
    }

    /// <summary>Creates a trainable SoundStream.</summary>
    public SoundStream(NeuralNetworkArchitecture<T> architecture, SoundStreamOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SoundStreamOptions(), optimizer)
    {
        _denoise = PaperOptions.Denoise;
    }

    private SoundStreamOptions PaperOptions => (SoundStreamOptions)CodecSettings;

    /// <inheritdoc />
    public override int HopLength => PaperOptions.Ratios.Aggregate(1, (a, b) => a * b);

    /// <summary>Gets or sets whether coding applies the denoising conditioning (§III-F).</summary>
    public bool Denoise
    {
        get => _denoise;
        set => _denoise = value;
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
            ResidualLayers = o.ResidualLayers,
            Ratios = o.Ratios,
            Normalization = ConvolutionNormalization.None,
            KernelSize = 7,
            LastKernelSize = 7,
            EncoderLastKernelSize = 3,
            ResidualKernelSizes = o.ResidualKernelSizes,
            DilationBase = o.DilationBase,
            Causal = true,
            Reflect = o.ReflectPadding,
            TrueSkip = true,
            Compress = 1,
            LstmLayers = 0,
        };
        _encoder = new SeanetEncoder<T>(Engine, config, layers);
        if (o.FilmPosition != SoundStreamFilmPosition.None)
        {
            _film = new DenseLayer<T>(2 * o.Dimension, new IdentityActivation<T>() as IActivationFunction<T>);
            layers.Add(_film);
            InitializeFilmAsIdentity(o.Dimension);
        }
        var quantizer = new ResidualVectorQuantizerLayer<T>(o.Dimension, o.NumQuantizers, o.CodebookSize, o.CodebookDecay, o.KMeansIterations,
            o.DeadCodeThreshold);
        layers.Add(quantizer);
        _decoder = new SeanetDecoder<T>(Engine, config, layers);
        _melScales = o.MelScales.Select(i => new CodecMelScale<T>(Engine, o.SampleRate, 1 << i, (1 << i) / 4, o.MelBins, 0.0, null,
            normalized: false)).ToList();
        return quantizer;
    }

    // γ = 1, β = 0 for both modes: the paper does not state FiLM's initialization; starting from the identity leaves an
    // untrained codec unconditioned.
    private void InitializeFilmAsIdentity(int dim)
    {
        using (new NoGradScope<T>())
            _film!.Forward(new Tensor<T>(new[] { 1, 2 }));
        var weights = _film!.GetWeights();                                                      // [2, 2D]
        var bias = _film.GetBiases();
        for (int mode = 0; mode < 2; mode++)
            for (int j = 0; j < 2 * dim; j++) weights[mode, j] = j < dim ? NumOps.One : NumOps.Zero;
        for (int j = 0; j < bias.Length; j++) bias[j] = NumOps.Zero;
        Engine.InvalidatePersistentTensor(weights);
        Engine.InvalidatePersistentTensor(bias);
    }

    // a' = γ a + β per channel, with (γ, β) from the one-hot denoising flag (Eq. 7).
    private Tensor<T> Film(Tensor<T> latent)
    {
        int dim = PaperOptions.Dimension;
        var flag = new Tensor<T>(new[] { 1, 2 });
        flag[0, _denoise ? 1 : 0] = NumOps.One;
        var coefficients = _film!.Forward(flag);                                                // [1, 2D]
        var gamma = Engine.Reshape(Engine.TensorSlice(coefficients, new[] { 0, 0 }, new[] { 1, dim }), new[] { 1, dim, 1 });
        var beta = Engine.Reshape(Engine.TensorSlice(coefficients, new[] { 0, dim }, new[] { 1, dim }), new[] { 1, dim, 1 });
        return Engine.TensorAdd(Engine.TensorMultiply(latent, Engine.TensorBroadcastTo(gamma, latent._shape)), Engine.TensorBroadcastTo(beta, latent._shape));
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        var layers = new List<LayerBase<T>>();
        _wave = new MelGanDiscriminators<T>(Engine, o.WaveDiscriminatorScales, o.WaveDiscriminatorChannels, o.WaveDiscriminatorWidthDivisor);
        layers.AddRange(_wave.Layers);
        _stft = new SoundStreamStftDiscriminator<T>(Engine, o.StftWindow, o.StftHop, o.StftChannels, layers);
        return layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> EncodeLatent(Tensor<T> audio)
    {
        var latent = _encoder!.Forward(audio);
        return _film is not null && PaperOptions.FilmPosition == SoundStreamFilmPosition.Encoder ? Film(latent) : latent;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeLatent(Tensor<T> latent)
        => _decoder!.Forward(_film is not null && PaperOptions.FilmPosition == SoundStreamFilmPosition.Decoder ? Film(latent) : latent);

    /// <inheritdoc />
    /// <remarks>Trimmed to the input's length.</remarks>
    protected override Tensor<T> Reconstruct(Tensor<T> audio, int quantizers)
    {
        var output = base.Reconstruct(audio, quantizers);
        int length = audio.Shape[2];
        return output.Shape[2] > length ? Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { 1, output.Shape[1], length }) : output;
    }

    /// <inheritdoc />
    /// <remarks>Quantizer dropout: n_q ~ U[1, N_q] for each example (§III-C).</remarks>
    protected override (int Quantizers, int Bandwidth) SampleTrainingBandwidth(Random random)
        => (PaperOptions.QuantizerDropout ? random.Next(1, PaperOptions.NumQuantizers + 1) : PaperOptions.NumQuantizers, 0);

    /// <summary>
    /// One training step on a (inputs, targets, denoise) tuple (§III-F): with <paramref name="denoise"/> false the target is
    /// the input itself; with it true, the clean component of a noisy input.
    /// </summary>
    public void TrainDenoising(Tensor<T> input, Tensor<T> target, bool denoise)
    {
        bool previous = _denoise;
        _denoise = denoise;
        try
        {
            Train(input, target);
        }
        finally
        {
            _denoise = previous;
        }
    }

    // ---------------------------------------------------------------- losses

    private List<(Tensor<T> Score, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
    {
        var flat = Flat(audio);
        var outputs = _wave!.Forward(flat);
        outputs.Insert(0, _stft!.Forward(flat));
        return outputs;
    }

    private Tensor<T> Hinge(Tensor<T> logits, double sign)
        => Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(logits, NumOps.FromDouble(sign)), NumOps.One)));

    // L_rec (Eq. 4-5): per scale s, Σ_t ||S_t(x) − S_t(x̂)||₁ + α_s Σ_t ||log S_t(x) − log S_t(x̂)||₂, α_s = √(s/2).
    private Tensor<T> ReconstructionLoss(Tensor<T> real, Tensor<T> generated)
    {
        var o = PaperOptions;
        var x = Flat(real);
        var y = Flat(generated);
        var terms = new List<Tensor<T>>();
        for (int i = 0; i < _melScales!.Count; i++)
        {
            Tensor<T> target;
            using (new NoGradScope<T>()) target = Detached(_melScales[i].Forward(x));
            var mel = _melScales[i].Forward(y);                                                    // [frames, mels]
            var l1 = Total(Engine.TensorAbs(Engine.TensorSubtract(target, mel)));
            var logDiff = Engine.TensorSubtract(LogMel(target), LogMel(mel));
            var perFrame = Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(logDiff, logDiff), new[] { 1 }, keepDims: false),
                NumOps.FromDouble(1e-24)), NumOps.FromDouble(0.5));
            double alpha = Math.Sqrt((1 << o.MelScales[i]) / 2.0);
            terms.Add(Engine.TensorAdd(l1, Engine.TensorMultiplyScalar(Total(perFrame), NumOps.FromDouble(alpha))));
        }
        return Sum(terms);
    }

    // log(max(S, 1e-5)) = log(1e-5 + relu(S − 1e-5)): the paper does not state the floor; 1e-5 as the mel losses elsewhere.
    private Tensor<T> LogMel(Tensor<T> mel)
        => Engine.TensorLog(Engine.TensorAddScalar(Engine.ReLU(Engine.TensorAddScalar(mel, NumOps.FromDouble(-1e-5))), NumOps.FromDouble(1e-5)));

    /// <inheritdoc />
    /// <remarks>λ_adv L_adv + λ_feat L_feat + λ_rec L_rec (Eq. 6), plus a commitment term when its weight is set.</remarks>
    protected override Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var o = PaperOptions;
        var fake = Discriminate(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = Discriminate(real);
        var adversarial = Engine.TensorMultiplyScalar(Sum(fake.Select(f => Hinge(f.Score, -1))), NumOps.FromDouble(1.0 / fake.Count));
        var featureTerms = new List<Tensor<T>>();
        for (int k = 0; k < fake.Count; k++)
            for (int l = 0; l < fake[k].Features.Count; l++)
                featureTerms.Add(Mean(Engine.TensorAbs(Engine.TensorSubtract(Detached(realOut[k].Features[l]), fake[k].Features[l]))));
        var feature = Engine.TensorMultiplyScalar(Sum(featureTerms), NumOps.FromDouble(1.0 / featureTerms.Count));
        var loss = Sum(new[]
        {
            Engine.TensorMultiplyScalar(adversarial, NumOps.FromDouble(o.AdversarialLossWeight)),
            Engine.TensorMultiplyScalar(feature, NumOps.FromDouble(o.FeatureLossWeight)),
            Engine.TensorMultiplyScalar(ReconstructionLoss(real, generated), NumOps.FromDouble(o.ReconstructionLossWeight)),
        });
        if (o.CommitmentLossWeight > 0 && Quantizer!.CommitmentLoss is { } commitment)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(commitment, NumOps.FromDouble(o.CommitmentLossWeight)));
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>L_D (Eq. 1): the hinge loss averaged over the discriminators and over time.</remarks>
    protected override Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var r = Discriminate(real);
        var g = Discriminate(generated);
        return Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(Hinge(r[k].Score, -1), Hinge(g[k].Score, 1)))),
            NumOps.FromDouble(1.0 / r.Count));
    }

    /// <inheritdoc />
    /// <remarks>L_rec (Eq. 4–5) per sample of the segment.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated)
        => Engine.TensorMultiplyScalar(ReconstructionLoss(real, generated), NumOps.FromDouble(1.0 / real.Length));

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
        => new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                Beta1 = PaperOptions.Beta1,
                Beta2 = PaperOptions.Beta2,
                UseAdaptiveBetas = false,
            });

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "SoundStream-ONNX" : "SoundStream-Native",
            Description = "SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)",
            FeatureCount = o.Channels,
            Complexity = o.NumQuantizers,
        };
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        m.AdditionalInfo["FrameRate"] = TokenFrameRate.ToString();
        m.AdditionalInfo["Bandwidth"] = o.TargetBandwidthKbps.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }
}
