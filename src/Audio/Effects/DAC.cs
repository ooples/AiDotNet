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
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// DAC, the Descript Audio Codec: a Snake-activated convolutional encoder–decoder with a residual vector quantizer of
/// factorized, L2-normalized codes, trained with quantizer dropout against multi-period and multi-band multi-scale STFT
/// discriminators and a multi-scale mel loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> "High-Fidelity Audio Compression with Improved RVQGAN" (Kumar et al., 2023), with
/// descriptinc/descript-audio-codec for what the paper leaves unstated.</para>
/// <para>
/// The encoder and decoder (§3.1, §4.3) are SoundStream's fully convolutional design with Snake activations: strides
/// (2, 4, 8, 8) to a 1024-dimensional latent (hop 512, about 86 frames per second at 44.1 kHz) and rates (8, 8, 4, 2) from a
/// 1536-wide decoder to a tanh waveform. The quantizer (§3.2, App. A) projects each residual to 8 dimensions, picks the
/// nearest of 1024 codes by cosine similarity, and projects back; its codebooks learn by the VQ-VAE codebook and commitment
/// losses. Quantizer dropout (§3.3) draws n_q ~ U[1, 9] for each example.
/// </para>
/// <para>
/// Training (§3.4–3.5, §4.3) minimizes 15 · the L1 distance of log-mel spectrograms over seven windows (32 … 2048 samples,
/// 5 … 320 mel bins) + 2 · feature matching + 1 · the hinge adversarial loss + 1 · codebook + 0.25 · commitment, against
/// five period discriminators (2, 3, 5, 7, 11) and complex STFT discriminators at windows 2048, 1024 and 512 split into five
/// frequency bands. AdamW (1e-4, β = 0.8, 0.9) trains both, its rate decaying by 0.999996 each step.
/// </para>
/// <para><b>For Beginners:</b> DAC compresses any kind of audio — speech, music, sound effects — into about 86 frames of
/// codes per second and reconstructs it with very little loss; using fewer codebooks lowers the bitrate.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("High-Fidelity Audio Compression with Improved RVQGAN", "https://arxiv.org/abs/2306.06546", Year = 2023, Authors = "Kumar et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.9, WeightDecay = 0.01, DecayRate = 0.999996,
                ReferenceBatchSize = 72,
                Source = "Kumar et al. 2023, Sec. 4.3: AdamW with a learning rate of 1e-4, beta1 0.8 and beta2 0.9 for the generator and "
                        + "the discriminator, decayed by 0.999996 every step, batch 72.")]
public partial class DAC<T> : NeuralAudioCodecBase<T>
{
    private DacEncoder<T>? _encoder;
    private DacDecoder<T>? _decoder;
    private FactorizedVectorQuantizerLayer<T>? _quantizer;
    private HiFiGanDiscriminators<T>? _periods;
    private readonly List<DacBandDiscriminator<T>> _bands = new();
    private List<(CenteredComplexStft<T> Stft, Tensor<T> Filterbank)>? _melScales;

    /// <summary>Creates a DAC that runs an exported ONNX graph.</summary>
    public DAC(NeuralNetworkArchitecture<T> architecture, string modelPath, DACOptions? options = null)
        : base(architecture, modelPath, options ?? new DACOptions())
    {
    }

    /// <summary>Creates a trainable DAC.</summary>
    public DAC(NeuralNetworkArchitecture<T> architecture, DACOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new DACOptions(), optimizer)
    {
    }

    private DACOptions PaperOptions => (DACOptions)CodecSettings;

    /// <inheritdoc />
    public override int HopLength => PaperOptions.EncoderRates.Aggregate(1, (a, b) => a * b);

    private int Latent => PaperOptions.LatentDim > 0 ? PaperOptions.LatentDim : PaperOptions.EncoderDim << PaperOptions.EncoderRates.Length;

    /// <inheritdoc />
    protected override IResidualVectorQuantizer<T> CreateCodec(List<LayerBase<T>> layers)
    {
        var o = PaperOptions;
        _encoder = new DacEncoder<T>(Engine, o.EncoderDim, o.EncoderRates, Latent, layers);
        _quantizer = new FactorizedVectorQuantizerLayer<T>(Latent, o.NumQuantizers, o.CodebookSize, o.CodebookDim, o.CodebookLossesOnRawVectors);
        layers.Add(_quantizer);
        _decoder = new DacDecoder<T>(Engine, Latent, o.DecoderDim, o.DecoderRates, layers);
        if (o.MelWindows.Length != o.MelBins.Length) throw new ArgumentException("Every mel window needs a mel bin count.");
        _melScales = new List<(CenteredComplexStft<T>, Tensor<T>)>();
        for (int i = 0; i < o.MelWindows.Length; i++)
        {
            int w = o.MelWindows[i];
            var stft = new CenteredComplexStft<T>(Engine, w, w / 4, w);
            var basis = new TacotronSpectrogram(o.SampleRate, w, w / 4, w, o.MelBins[i], 0.0, o.SampleRate / 2.0).MelBasis;   // [mels, bins]
            var fb = new Tensor<T>(new[] { stft.Bins, o.MelBins[i] });
            for (int m = 0; m < o.MelBins[i]; m++)
                for (int k = 0; k < stft.Bins; k++) fb[k, m] = NumOps.FromDouble(basis[m, k]);
            _melScales.Add((stft, fb));
        }
        return _quantizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        var layers = new List<LayerBase<T>>();
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false, widthDivisor: o.DiscriminatorWidthDivisor,
            spectralFirstScale: false, slope: 0.1, padFullPeriod: true);
        layers.AddRange(_periods.Layers);
        foreach (int w in o.DiscriminatorWindows)
            _bands.Add(new DacBandDiscriminator<T>(Engine, w, o.DiscriminatorBands, o.DiscriminatorChannels, layers));
        return layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> EncodeLatent(Tensor<T> audio) => _encoder!.Forward(audio);

    /// <inheritdoc />
    protected override Tensor<T> DecodeLatent(Tensor<T> latent) => _decoder!.Forward(latent);

    /// <inheritdoc />
    /// <remarks>Zero-padded on the right to whole hops (reference <c>preprocess</c>) and trimmed back to the input's length.</remarks>
    protected override Tensor<T> Reconstruct(Tensor<T> audio, int quantizers)
    {
        int length = audio.Shape[2], hop = HopLength, padded = (length + hop - 1) / hop * hop;
        var input = padded == length ? audio : Seanet.Pad(Engine, audio, 0, padded - length, reflect: false);
        var output = base.Reconstruct(input, quantizers);
        return output.Shape[2] > length ? Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { 1, output.Shape[1], length }) : output;
    }

    /// <inheritdoc />
    /// <remarks>Quantizer dropout (§3.3): with probability p, n_q ~ U[1, N_q]; otherwise every codebook.</remarks>
    protected override (int Quantizers, int Bandwidth) SampleTrainingBandwidth(Random random)
    {
        int n = PaperOptions.NumQuantizers;
        return (random.NextDouble() < PaperOptions.QuantizerDropout ? random.Next(1, n + 1) : n, 0);
    }

    // ---------------------------------------------------------------- losses

    // The reference discriminators' input normalization: y − mean(y), then 0.8 · y / (max|y| + 1e-9).
    private Tensor<T> Normalize(Tensor<T> audio)
    {
        var flat = Flat(audio);
        var centred = Engine.TensorSubtract(flat, Engine.TensorBroadcastTo(Engine.Reshape(Mean(flat), new[] { 1 }), flat._shape));
        var peak = Engine.ReduceMax(Engine.TensorAbs(centred), new[] { 0 }, keepDims: true);                   // [1]
        var scaled = Engine.TensorDivide(centred, Engine.TensorBroadcastTo(Engine.TensorAddScalar(peak, NumOps.FromDouble(1e-9)), centred._shape));
        return Engine.TensorMultiplyScalar(scaled, NumOps.FromDouble(0.8));
    }

    private List<(Tensor<T> Logits, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
    {
        var x = Normalize(audio);
        var outputs = _periods!.Forward(x);
        foreach (var band in _bands) outputs.Add(band.Forward(x));
        return outputs;
    }

    // Σ over windows of the mean |log10 max(mel(x), 1e-5) − log10 max(mel(x̂), 1e-5)| (magnitude STFT, centred, Hann; Slaney mel).
    private Tensor<T> MelLoss(Tensor<T> real, Tensor<T> generated)
    {
        var x = Flat(real);
        var y = Flat(generated);
        var terms = new List<Tensor<T>>();
        foreach (var (stft, filterbank) in _melScales!)
        {
            Tensor<T> target;
            using (new NoGradScope<T>()) target = Detached(LogMel(stft, filterbank, x));
            terms.Add(Mean(Engine.TensorAbs(Engine.TensorSubtract(target, LogMel(stft, filterbank, y)))));
        }
        return Sum(terms);
    }

    private Tensor<T> LogMel(CenteredComplexStft<T> stft, Tensor<T> filterbank, Tensor<T> audio)
    {
        var (re, im) = stft.Forward(audio);                                                       // [1, bins, frames]
        var magnitude = Engine.TensorPow(Engine.TensorAddScalar(Engine.TensorAdd(Engine.TensorMultiply(re, re), Engine.TensorMultiply(im, im)),
            NumOps.FromDouble(1e-12)), NumOps.FromDouble(0.5));
        var rows = Engine.TensorTranspose(Engine.Reshape(magnitude, new[] { magnitude.Shape[1], magnitude.Shape[2] }));   // [frames, bins]
        var mel = Engine.TensorMatMul(rows, filterbank);
        var clamped = Engine.TensorAddScalar(Engine.ReLU(Engine.TensorAddScalar(mel, NumOps.FromDouble(-1e-5))), NumOps.FromDouble(1e-5));
        return Engine.TensorMultiplyScalar(Engine.TensorLog(clamped), NumOps.FromDouble(1.0 / Math.Log(10.0)));
    }

    /// <inheritdoc />
    /// <remarks>15 · mel + 2 · feature matching + 1 · hinge adversarial + 1 · codebook + 0.25 · commitment (§3.5).</remarks>
    protected override Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var o = PaperOptions;
        var fake = Discriminate(generated);
        List<(Tensor<T> Logits, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = Discriminate(real);
        // HingeGAN (Lim and Ye 2017): the generator minimizes −D(G(z)).
        var adversarial = Sum(fake.Select(f => Engine.TensorNegate(Mean(f.Logits))));
        var featureTerms = new List<Tensor<T>>();
        for (int k = 0; k < fake.Count; k++)
            for (int l = 0; l < fake[k].Features.Count - 1; l++)
                featureTerms.Add(Mean(Engine.TensorAbs(Engine.TensorSubtract(fake[k].Features[l], Detached(realOut[k].Features[l])))));
        var terms = new List<Tensor<T>>
        {
            Engine.TensorMultiplyScalar(MelLoss(real, generated), NumOps.FromDouble(o.MelLossWeight)),
            Engine.TensorMultiplyScalar(Sum(featureTerms), NumOps.FromDouble(o.FeatureLossWeight)),
            Engine.TensorMultiplyScalar(adversarial, NumOps.FromDouble(o.AdversarialLossWeight)),
        };
        if (_quantizer!.CodebookLoss is { } codebook) terms.Add(Engine.TensorMultiplyScalar(codebook, NumOps.FromDouble(o.CodebookLossWeight)));
        if (_quantizer.CommitmentLoss is { } commitment) terms.Add(Engine.TensorMultiplyScalar(commitment, NumOps.FromDouble(o.CommitmentLossWeight)));
        return Sum(terms);
    }

    /// <inheritdoc />
    /// <remarks>The hinge loss Σ_k mean(max(0, 1 − D_k(x))) + mean(max(0, 1 + D_k(x̂))).</remarks>
    protected override Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth)
    {
        var r = Discriminate(real);
        var g = Discriminate(generated);
        return Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(
            Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(r[k].Logits), NumOps.One))),
            Mean(Engine.ReLU(Engine.TensorAddScalar(g[k].Logits, NumOps.One))))));
    }

    /// <inheritdoc />
    /// <remarks>The multi-scale mel distance (§4.4's "Mel distance").</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated) => MelLoss(real, generated);

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate, step => Math.Pow(o.LearningRateDecay, step));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                WeightDecay = o.WeightDecay,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "DAC-ONNX" : "DAC-Native",
            Description = "High-Fidelity Audio Compression with Improved RVQGAN (Kumar et al., 2023)",
            FeatureCount = o.Channels,
            Complexity = o.NumQuantizers,
        };
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        m.AdditionalInfo["FrameRate"] = TokenFrameRate.ToString();
        m.AdditionalInfo["Bandwidth"] = o.TargetBandwidthKbps.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }
}
