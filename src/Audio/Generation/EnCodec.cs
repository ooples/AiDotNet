using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// EnCodec: a streaming convolutional encoder–decoder with an LSTM, residual vector quantization of its latent, and
/// adversarial training against a multi-scale STFT discriminator with a loss balancer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> "High Fidelity Neural Audio Compression" (Défossez et al., 2022), with facebookresearch/encodec
/// (the released model) and facebookresearch/audiocraft (the authors' training code) for what the paper leaves unstated.</para>
/// <para>
/// The encoder (§3.1) is a k = 7 convolution to C = 32 channels, four blocks of a residual unit and a strided convolution
/// (kernel 2S; strides 2, 4, 5, 8) doubling the channels, a two-layer LSTM and a k = 7 convolution to D = 128; ELU
/// throughout; the decoder mirrors it with transposed convolutions. The streamable model pads causally and uses weight
/// normalization; the non-streamable one splits the padding and layer-normalizes. The latent is quantized by up to 32
/// residual codebooks of 1024 entries (§3.2) with EMA updates (decay 0.99), dead-code replacement and a straight-through
/// gradient; a bandwidth (1.5 … 24 kbps) is drawn per step and selects the codebooks and a dedicated discriminator.
/// </para>
/// <para>
/// Training (§3.4) balances, on the output x̂, the time L1 loss, the multi-scale mel L1 + L2 loss (scales 2⁵ … 2¹¹, 64
/// bins), the hinge adversarial loss and the relative feature-matching loss of the MS-STFT discriminator with weights
/// 0.1, 1, 3, 3 (Eq. 5, R = 1, β = 0.999), and adds the commitment loss outside the balancer. The discriminator is
/// updated with probability 2/3 with the hinge loss; Adam (3e-4, β = 0.5, 0.9) trains both (§4.3).
/// </para>
/// <para><b>For Beginners:</b> EnCodec compresses audio into a few numbers per 13 ms and rebuilds the audio from them.
/// More codebooks per frame means a higher bitrate and better quality.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("High Fidelity Neural Audio Compression", "https://arxiv.org/abs/2210.13438", Year = 2022, Authors = "Défossez et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.5, Beta2 = 0.9, ReferenceBatchSize = 64,
                Source = "Défossez et al. 2022, Sec. 4.3: Adam with a learning rate of 3e-4, beta1 0.5 and beta2 0.9, batches of 64 one-second examples.")]
public partial class EnCodec<T> : NeuralAudioCodecBase<T>
{
    private SeanetEncoder<T>? _encoder;
    private SeanetDecoder<T>? _decoder;
    private ResidualVectorQuantizerLayer<T>? _quantizer;
    private readonly List<(MultiScaleStftDiscriminator<T> Discriminator, List<LayerBase<T>> Layers)> _discriminators = new();
    private List<CodecMelScale<T>>? _melScales;
    private LossBalancer<T>? _balancer;
    private double? _reported;

    /// <summary>Creates an EnCodec that runs an exported ONNX graph.</summary>
    public EnCodec(NeuralNetworkArchitecture<T> architecture, string modelPath, EnCodecOptions? options = null)
        : base(architecture, modelPath, options ?? new EnCodecOptions())
    {
    }

    /// <summary>Creates a trainable EnCodec.</summary>
    public EnCodec(NeuralNetworkArchitecture<T> architecture, EnCodecOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new EnCodecOptions(), optimizer)
    {
    }

    private EnCodecOptions PaperOptions => (EnCodecOptions)CodecSettings;

    /// <inheritdoc />
    public override int HopLength => PaperOptions.Ratios.Aggregate(1, (a, b) => a * b);

    private SeanetConfig Config()
    {
        var o = PaperOptions;
        return new SeanetConfig
        {
            Channels = o.Channels,
            Dimension = o.Dimension,
            Filters = o.Filters,
            ResidualLayers = o.ResidualLayers,
            Ratios = o.Ratios,
            Normalization = o.Causal ? ConvolutionNormalization.Weight : ConvolutionNormalization.None,
            TimeGroupNorm = !o.Causal,
            KernelSize = o.KernelSize,
            LastKernelSize = o.LastKernelSize,
            ResidualKernelSizes = o.ResidualKernelSizes,
            DilationBase = o.DilationBase,
            Causal = o.Causal,
            Reflect = o.ReflectPadding,
            TrueSkip = o.TrueSkip,
            Compress = o.Compress,
            LstmLayers = o.LstmLayers,
        };
    }

    /// <inheritdoc />
    protected override IResidualVectorQuantizer<T> CreateCodec(List<LayerBase<T>> layers)
    {
        var o = PaperOptions;
        var config = Config();
        _encoder = new SeanetEncoder<T>(Engine, config, layers);
        var quantizer = new ResidualVectorQuantizerLayer<T>(o.Dimension, o.NumQuantizers, o.CodebookSize, o.CodebookDecay, o.KMeansIterations,
            o.DeadCodeThreshold, o.CommitmentAsMean);
        layers.Add(quantizer);
        _quantizer = quantizer;
        _decoder = new SeanetDecoder<T>(Engine, config, layers);
        _melScales = o.MelScales.Select(i => new CodecMelScale<T>(Engine, o.SampleRate, 1 << i, (1 << i) / 4, o.MelBins, o.MelMinFrequency, null,
            normalized: true)).ToList();
        var weights = new Dictionary<string, double>
        {
            ["t"] = o.TimeLossWeight,
            ["f"] = o.FrequencyLossWeight,
            ["g"] = o.AdversarialLossWeight,
            ["feat"] = o.FeatureLossWeight,
        };
        _balancer = new LossBalancer<T>(Engine, weights, o.BalancerTotalNorm, o.BalancerDecay);
        return quantizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        int count = o.DiscriminatorPerBandwidth ? o.TargetBandwidths.Length : 1;
        var all = new List<LayerBase<T>>();
        for (int k = 0; k < count; k++)
        {
            var layers = new List<LayerBase<T>>();
            var d = new MultiScaleStftDiscriminator<T>(Engine, o.DiscriminatorWindows, o.DiscriminatorFilters, o.DiscriminatorDilations,
                o.DiscriminatorKernelTime, o.DiscriminatorKernelFrequency, o.DiscriminatorSlope, layers);
            _discriminators.Add((d, layers));
            all.AddRange(layers);
        }
        return all;
    }

    private int DiscriminatorIndex(int bandwidth) => PaperOptions.DiscriminatorPerBandwidth ? bandwidth : 0;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> DiscriminatorLayersFor(int bandwidth) => _discriminators[DiscriminatorIndex(bandwidth)].Layers;

    /// <inheritdoc />
    protected override string DiscriminatorGroup(int bandwidth) => $"discriminator{DiscriminatorIndex(bandwidth)}";

    /// <inheritdoc />
    protected override double DiscriminatorUpdateProbability => PaperOptions.DiscriminatorUpdateProbability;

    /// <inheritdoc />
    protected override Tensor<T> EncodeLatent(Tensor<T> audio) => _encoder!.Forward(audio);

    /// <inheritdoc />
    protected override Tensor<T> DecodeLatent(Tensor<T> latent) => _decoder!.Forward(latent);

    // The chunk's scale: 1e-8 + the RMS of its channel mean (reference EncodecModel._encode_frame), a constant.
    private double VolumeScale(Tensor<T> audio)
    {
        int channels = audio.Shape[1], length = audio.Shape[2];
        double sum = 0;
        for (int t = 0; t < length; t++)
        {
            double mono = 0;
            for (int c = 0; c < channels; c++) mono += NumOps.ToDouble(audio[0, c, t]);
            mono /= channels;
            sum += mono * mono;
        }
        return 1e-8 + Math.Sqrt(sum / Math.Max(1, length));
    }

    /// <inheritdoc />
    /// <remarks>Volume-normalized when configured (divided by the chunk's scale before encoding, multiplied back after
    /// decoding), and trimmed to the input's length.</remarks>
    protected override Tensor<T> Reconstruct(Tensor<T> audio, int quantizers)
    {
        double scale = PaperOptions.NormalizeVolume ? VolumeScale(audio) : 1.0;
        var input = scale == 1.0 ? audio : Engine.TensorMultiplyScalar(audio, NumOps.FromDouble(1.0 / scale));
        var output = base.Reconstruct(input, quantizers);
        if (scale != 1.0) output = Engine.TensorMultiplyScalar(output, NumOps.FromDouble(scale));
        int length = audio.Shape[2];
        return output.Shape[2] > length ? Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { 1, output.Shape[1], length }) : output;
    }

    /// <inheritdoc />
    protected override (int Quantizers, int Bandwidth) SampleTrainingBandwidth(Random random)
    {
        var bandwidths = PaperOptions.TargetBandwidths;
        int k = random.Next(bandwidths.Length);
        return (QuantizersForBandwidth(bandwidths[k]), k);
    }

    // ---------------------------------------------------------------- losses

    private IEnumerable<Tensor<T>> Channels(Tensor<T> audio)
    {
        var batch = AsBatch(audio);
        for (int c = 0; c < batch.Shape[1]; c++)
            yield return Engine.Reshape(Engine.TensorSlice(batch, new[] { 0, c, 0 }, new[] { 1, 1, batch.Shape[2] }), new[] { batch.Shape[2] });
    }

    /// <summary>ℓ_t = ‖x − x̂‖₁.</summary>
    private Tensor<T> TimeLoss(Tensor<T> real, Tensor<T> generated) => Total(Engine.TensorAbs(Engine.TensorSubtract(real, generated)));

    /// <summary>ℓ_f (Eq. 1): per scale i, ‖S_i(x) − S_i(x̂)‖₁ + α_i ‖S_i(x) − S_i(x̂)‖₂ over the mel spectrogram's elements,
    /// averaged over the scales (1 / (|α| · |s|)); each channel separately, summed.</summary>
    private Tensor<T> FrequencyLoss(IReadOnlyList<IReadOnlyList<Tensor<T>>> realMels, Tensor<T> generated)
    {
        var o = PaperOptions;
        var channels = Channels(generated).ToList();
        var terms = new List<Tensor<T>>();
        for (int c = 0; c < channels.Count; c++)
            for (int i = 0; i < _melScales!.Count; i++)
            {
                var d = Engine.TensorSubtract(realMels[c][i], _melScales[i].Forward(channels[c]));
                double count = d.Length;
                var l1 = Total(Engine.TensorAbs(d));
                var l2 = Engine.TensorPow(Engine.TensorAddScalar(Total(Engine.TensorMultiply(d, d)), NumOps.FromDouble(1e-24)), NumOps.FromDouble(0.5));
                terms.Add(Engine.TensorMultiplyScalar(Engine.TensorAdd(l1, Engine.TensorMultiplyScalar(l2, NumOps.FromDouble(o.MelL2Coefficient))),
                    NumOps.FromDouble(1.0 / (count * _melScales.Count))));
            }
        return Sum(terms);
    }

    private List<IReadOnlyList<Tensor<T>>> RealMels(Tensor<T> real)
    {
        using var _ = new NoGradScope<T>();
        return Channels(real).Select(ch => (IReadOnlyList<Tensor<T>>)_melScales!.Select(m => Detached(m.Forward(ch))).ToList()).ToList();
    }

    /// <summary>ℓ_g = (1/K) Σ_k max(0, 1 − D_k(x̂)), per channel, averaged over channels.</summary>
    private Tensor<T> AdversarialLoss(MultiScaleStftDiscriminator<T> discriminator, Tensor<T> generated)
    {
        var terms = new List<Tensor<T>>();
        foreach (var channel in Channels(generated))
        {
            var outputs = discriminator.Forward(channel);
            terms.Add(Engine.TensorMultiplyScalar(Sum(outputs.Select(d => Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(d.Logits), NumOps.One))))),
                NumOps.FromDouble(1.0 / outputs.Count)));
        }
        return Engine.TensorMultiplyScalar(Sum(terms), NumOps.FromDouble(1.0 / terms.Count));
    }

    /// <summary>ℓ_feat (Eq. 2) = (1/KL) Σ_k Σ_l mean|D_k^l(x) − D_k^l(x̂)| / mean|D_k^l(x)|, per channel, averaged.</summary>
    private Tensor<T> FeatureLoss(MultiScaleStftDiscriminator<T> discriminator, IReadOnlyList<List<(Tensor<T> Logits, List<Tensor<T>> Features)>> real,
        Tensor<T> generated)
    {
        var terms = new List<Tensor<T>>();
        int c = 0;
        foreach (var channel in Channels(generated))
        {
            var fake = discriminator.Forward(channel);
            var layerTerms = new List<Tensor<T>>();
            for (int k = 0; k < fake.Count; k++)
                for (int l = 0; l < fake[k].Features.Count; l++)
                {
                    var r = real[c][k].Features[l];
                    double scale = NumOps.ToDouble(Mean(Engine.TensorAbs(r))[0]);
                    layerTerms.Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(r, fake[k].Features[l]))),
                        NumOps.FromDouble(1.0 / Math.Max(scale, 1e-12))));
                }
            terms.Add(Engine.TensorMultiplyScalar(Sum(layerTerms), NumOps.FromDouble(1.0 / layerTerms.Count)));
            c++;
        }
        return Engine.TensorMultiplyScalar(Sum(terms), NumOps.FromDouble(1.0 / terms.Count));
    }

    private List<List<(Tensor<T> Logits, List<Tensor<T>> Features)>> RealFeatures(MultiScaleStftDiscriminator<T> discriminator, Tensor<T> real)
    {
        using var _ = new NoGradScope<T>();
        return Channels(real).Select(ch => discriminator.Forward(ch)
            .Select(d => (Detached(d.Logits), d.Features.Select(Detached).ToList())).ToList()).ToList();
    }

    /// <inheritdoc />
    /// <remarks>The balanced gradient of λ_t ℓ_t, λ_f ℓ_f, λ_g ℓ_g and λ_feat ℓ_feat on x̂ (Eq. 5), plus λ_w ℓ_w (Eq. 3)
    /// outside the balancer.</remarks>
    protected override Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var o = PaperOptions;
        var discriminator = _discriminators[DiscriminatorIndex(bandwidth)].Discriminator;
        var realMels = RealMels(real);
        var realFeatures = RealFeatures(discriminator, real);
        var losses = new Dictionary<string, Func<Tensor<T>, Tensor<T>>>
        {
            ["t"] = x => TimeLoss(real, x),
            ["f"] = x => FrequencyLoss(realMels, x),
            ["g"] = x => AdversarialLoss(discriminator, x),
            ["feat"] = x => FeatureLoss(discriminator, realFeatures, x),
        };
        var commitment = Engine.TensorMultiplyScalar(Quantizer!.CommitmentLoss ?? new Tensor<T>(new[] { 1 }), NumOps.FromDouble(o.CommitmentLossWeight));
        double commitmentValue = NumOps.ToDouble(commitment[0]);
        if (o.UseBalancer)
        {
            var surrogate = _balancer!.Surrogate(generated, losses);
            _reported = _balancer.LastWeightedLoss + commitmentValue;
            return Engine.TensorAdd(surrogate, commitment);
        }
        var weights = new Dictionary<string, double>
        {
            ["t"] = o.TimeLossWeight,
            ["f"] = o.FrequencyLossWeight,
            ["g"] = o.AdversarialLossWeight,
            ["feat"] = o.FeatureLossWeight,
        };
        var total = Engine.TensorAdd(Sum(losses.Select(l => Engine.TensorMultiplyScalar(l.Value(generated), NumOps.FromDouble(weights[l.Key])))), commitment);
        _reported = NumOps.ToDouble(total[0]);
        return total;
    }

    /// <inheritdoc />
    protected override double? ReportedGeneratorLoss => _reported;

    /// <inheritdoc />
    /// <remarks>L_d = (1/K) Σ_k max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂)) for the step's bandwidth, per channel, averaged.</remarks>
    protected override Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var discriminator = _discriminators[DiscriminatorIndex(bandwidth)].Discriminator;
        var realChannels = Channels(real).ToList();
        var fakeChannels = Channels(generated).ToList();
        var terms = new List<Tensor<T>>();
        for (int c = 0; c < realChannels.Count; c++)
        {
            var r = discriminator.Forward(realChannels[c]);
            var g = discriminator.Forward(fakeChannels[c]);
            terms.Add(Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(
                    Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(r[k].Logits), NumOps.One))),
                    Mean(Engine.ReLU(Engine.TensorAddScalar(g[k].Logits, NumOps.One)))))),
                NumOps.FromDouble(1.0 / r.Count)));
        }
        return Engine.TensorMultiplyScalar(Sum(terms), NumOps.FromDouble(1.0 / terms.Count));
    }

    /// <inheritdoc />
    /// <remarks>λ_t ℓ_t / samples + λ_f ℓ_f: the reconstruction terms of Eq. 4, the time term per sample.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated)
    {
        var o = PaperOptions;
        var time = Engine.TensorMultiplyScalar(TimeLoss(real, generated), NumOps.FromDouble(o.TimeLossWeight / real.Length));
        return Engine.TensorAdd(time, Engine.TensorMultiplyScalar(FrequencyLoss(RealMels(real), generated), NumOps.FromDouble(o.FrequencyLossWeight)));
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                Beta1 = PaperOptions.Beta1,
                Beta2 = PaperOptions.Beta2,
                // Adam's options adapt the betas during training by default; the paper's stay fixed.
                UseAdaptiveBetas = false,
            }));

    // ---------------------------------------------------------------- chunked (non-streamable) coding

    /// <summary>One encoded chunk: its codes <c>[n_q, frames]</c> and its volume scale (1 when not normalized).</summary>
    public sealed record EncodedFrame(int[,] Codes, double Scale);

    private bool Chunked => PaperOptions.ChunkSeconds is not null;

    private (int Length, int Stride) ChunkGeometry()
    {
        int length = (int)Math.Round(PaperOptions.ChunkSeconds!.Value * PaperOptions.SampleRate);
        return (length, Math.Max(1, (int)((1 - PaperOptions.ChunkOverlap) * length)));
    }

    /// <summary>Encodes audio chunk by chunk (§3.1 "Non-streamable": 1 s chunks with a 10 ms overlap, each normalized),
    /// or as one chunk when no chunk length is configured.</summary>
    public IReadOnlyList<EncodedFrame> EncodeFrames(Tensor<T> audio)
    {
        ThrowIfDisposed();
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        var x = AsBatch(audio);
        int total = x.Shape[2], quantizers = QuantizersForBandwidth(PaperOptions.TargetBandwidthKbps);
        var (length, stride) = Chunked ? ChunkGeometry() : (total, total);
        var frames = new List<EncodedFrame>();
        for (int offset = 0; offset < total; offset += stride)
        {
            int n = Math.Min(length, total - offset);
            var chunk = Engine.TensorSlice(x, new[] { 0, 0, offset }, new[] { 1, x.Shape[1], n });
            double scale = PaperOptions.NormalizeVolume ? VolumeScale(chunk) : 1.0;
            if (scale != 1.0) chunk = Engine.TensorMultiplyScalar(chunk, NumOps.FromDouble(1.0 / scale));
            frames.Add(new EncodedFrame(Quantizer!.Encode(EncodeLatent(chunk), quantizers), scale));
        }
        return frames;
    }

    /// <summary>Decodes chunks and joins them by linear overlap-add with triangular weights (reference
    /// <c>_linear_overlap_add</c>).</summary>
    public Tensor<T> DecodeFrames(IReadOnlyList<EncodedFrame> frames)
    {
        ThrowIfDisposed();
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        var decoded = frames.Select(f =>
        {
            var y = DecodeLatent(Quantizer!.Decode(f.Codes));
            return f.Scale == 1.0 ? y : Engine.TensorMultiplyScalar(y, NumOps.FromDouble(f.Scale));
        }).ToList();
        if (decoded.Count == 1) return decoded[0];
        int stride = Chunked ? ChunkGeometry().Stride : decoded[0].Shape[2];
        int channels = decoded[0].Shape[1];
        int totalLength = stride * (decoded.Count - 1) + decoded[^1].Shape[2];
        var sum = new double[channels, totalLength];
        var weightSum = new double[totalLength];
        for (int f = 0; f < decoded.Count; f++)
        {
            int n = decoded[f].Shape[2], offset = f * stride;
            for (int t = 0; t < n; t++)
            {
                // torch.linspace(0, 1, n + 2)[1:-1]; weight = 0.5 − |t − 0.5|.
                double u = (t + 1) / (double)(n + 1);
                double w = 0.5 - Math.Abs(u - 0.5);
                weightSum[offset + t] += w;
                for (int c = 0; c < channels; c++) sum[c, offset + t] += w * NumOps.ToDouble(decoded[f][0, c, t]);
            }
        }
        var output = new Tensor<T>(new[] { 1, channels, totalLength });
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < totalLength; t++) output[0, c, t] = NumOps.FromDouble(sum[c, t] / weightSum[t]);
        return output;
    }

    /// <inheritdoc />
    /// <remarks>A chunked, volume-normalized model needs each chunk's scale to decode: use <see cref="EncodeFrames"/>.</remarks>
    public override int[,] Encode(Tensor<T> audio)
    {
        if (Chunked || PaperOptions.NormalizeVolume)
            throw new NotSupportedException("This EnCodec configuration codes 1 s chunks with per-chunk volume scales; use EncodeFrames and DecodeFrames.");
        return base.Encode(audio);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(int[,] tokens)
    {
        if (Chunked || PaperOptions.NormalizeVolume)
            throw new NotSupportedException("This EnCodec configuration codes 1 s chunks with per-chunk volume scales; use EncodeFrames and DecodeFrames.");
        return base.Decode(tokens);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "EnCodec-ONNX" : "EnCodec-Native",
            Description = "High Fidelity Neural Audio Compression (Défossez et al., 2022)",
            FeatureCount = o.Channels,
            Complexity = o.NumQuantizers,
        };
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        m.AdditionalInfo["FrameRate"] = TokenFrameRate.ToString();
        m.AdditionalInfo["Bandwidth"] = o.TargetBandwidthKbps.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }
}
