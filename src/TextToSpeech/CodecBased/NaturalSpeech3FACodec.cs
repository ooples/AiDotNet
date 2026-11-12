using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The factorized codes and embeddings FACodec makes of an utterance.</summary>
/// <param name="Codes">The codes <c>[codebooks, frames]</c>: prosody, then content, then acoustic detail.</param>
/// <param name="Groups">Each subspace's quantized vectors z_p, z_c, z_d <c>[1, dim, frames]</c>.</param>
/// <param name="Speaker">The timbre vector h_t <c>[1, dim]</c>.</param>
internal sealed record FACodecFactors<T>(int[,] Codes, Tensor<T>[] Groups, Tensor<T> Speaker)
{
    public Tensor<T> Sum(IEngine engine) => Groups.Aggregate(engine.TensorAdd);
}

/// <summary>
/// FACodec (Ju et al. 2024, §3.2; Amphion <c>FACodecEncoder</c>/<c>FACodecDecoder</c>, their v2 and the redecoder): the
/// speech encoder; three factorized vector quantizers for prosody, content and acoustic detail (8-dimensional codes); a
/// Transformer timbre extractor giving a global vector; supervision heads (normalized F0 and voicing from z_p, phones from
/// z_c) and gradient-reversal heads; and a decoder that sums the subspaces, applies timbre by conditional LayerNorm and
/// reconstructs the waveform. v2 quantizes prosody from the first 20 mel bins; the redecoder re-synthesizes codes in
/// another speaker's timbre.
/// </summary>
internal sealed class FACodecModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly FACodecConfiguration _c;
    private double[,]? _melBasis;
    private double[]? _melWindow;

    public FACodecModel(IEngine engine, FACodecConfiguration c, Random random, List<LayerBase<T>> encoderLayers, List<LayerBase<T>> decoderLayers,
        List<LayerBase<T>> headLayers, List<LayerBase<T>> redecoderLayers)
    {
        _engine = engine;
        _c = c;
        int d = c.Dim;
        Encoder = new FACodecEncoder<T>(engine, encoderLayers, c);
        int bins = c.CodebookSize;
        Quantizers = new[] { c.ProsodyQuantizers, c.ContentQuantizers, c.DetailQuantizers }.Select(n =>
        {
            var q = new FactorizedVectorQuantizerLayer<T>(d, n, bins, c.CodebookDim, lossesOnRawVectors: true);
            decoderLayers.Add(q);
            return q;
        }).ToArray();
        TimbreEncoder = new Ns2TransformerEncoder<T>(engine, decoderLayers, random, 0, d, c.TimbreHeads, c.TimbreFilter, c.TimbreKernel, c.TimbreLayers,
            c.TimbreDropout, false, NaturalSpeech2Positions.AmphionBatchIndexed);
        TimbreLinear = StyleLinear(engine, decoderLayers, random, d);
        Decoder = new FACodecDecoderStack<T>(engine, decoderLayers, d, c.DecoderInitial, c.DecoderRates);
        if (c.MelProsody)
        {
            MelLinear = IndexTtsOps.Linear(engine, decoderLayers, random, c.ProsodyMels, d);
            MelEncoder = new Ns2TransformerEncoder<T>(engine, decoderLayers, random, 0, d, c.TimbreHeads, c.TimbreFilter, c.TimbreKernel, c.TimbreLayers,
                c.TimbreDropout, false, NaturalSpeech2Positions.AmphionBatchIndexed);
        }

        F0Predictor = new FACodecPredictor<T>(engine, headLayers, random, d, 1, 2);
        PhonePredictor = new FACodecPredictor<T>(engine, headLayers, random, d, c.Phones, 1);
        if (c.DetailF0Reversal) DetailF0Predictor = new FACodecPredictor<T>(engine, headLayers, random, d, 1, 2);
        if (c.DetailPhoneReversal) DetailPhonePredictor = new FACodecPredictor<T>(engine, headLayers, random, d, c.Phones, 1);
        if (c.ContentF0Reversal) ContentF0Predictor = new FACodecPredictor<T>(engine, headLayers, random, d, 1, 2);
        if (c.ProsodyPhoneReversal) ProsodyPhonePredictor = new FACodecPredictor<T>(engine, headLayers, random, d, c.Phones, 1);
        if (c.TimbreReversal) TimbrePredictor = new FACodecPredictor<T>(engine, headLayers, random, d, c.Speakers, 1, global: true);

        if (c.Redecoder)
        {
            RedecoderEmbeddings = Enumerable.Range(0, c.Codebooks).Select(_ => IndexTtsOps.Embedding(redecoderLayers, random, bins, d, 1e-5)).ToArray();
            RedecoderProsody = new Ns2TransformerEncoder<T>(engine, redecoderLayers, random, 0, d, c.TimbreHeads, c.TimbreFilter, c.TimbreKernel, c.TimbreLayers,
                c.TimbreDropout, true, NaturalSpeech2Positions.AmphionBatchIndexed);
            RedecoderTimbreLinear = StyleLinear(engine, redecoderLayers, random, d);
            RedecoderStack = new FACodecDecoderStack<T>(engine, redecoderLayers, d, c.RedecoderInitial, c.DecoderRates);
        }
    }

    // nn.Linear(d, 2d) with the bias set to gains of 1 and shifts of 0.
    private static DenseLayer<T> StyleLinear(IEngine engine, List<LayerBase<T>> layers, Random random, int d) =>
        Ns2Ops.Linear(engine, layers, random, d, 2 * d, IndexTtsOps.Uniform(random, 2 * d * d, d), Enumerable.Range(0, 2 * d).Select(i => i < d ? 1.0 : 0.0).ToArray());

    public FACodecConfiguration Configuration => _c;

    /// <summary>The codebooks of subspace <paramref name="group"/> (0 prosody, 1 content, 2 detail).</summary>
    public int GroupSize(int group) => group switch { 0 => _c.ProsodyQuantizers, 1 => _c.ContentQuantizers, _ => _c.DetailQuantizers };
    public FACodecEncoder<T> Encoder { get; }
    public FactorizedVectorQuantizerLayer<T>[] Quantizers { get; }
    public Ns2TransformerEncoder<T> TimbreEncoder { get; }
    public DenseLayer<T> TimbreLinear { get; }
    public FACodecDecoderStack<T> Decoder { get; }
    public DenseLayer<T>? MelLinear { get; }
    public Ns2TransformerEncoder<T>? MelEncoder { get; }
    public FACodecPredictor<T> F0Predictor { get; }
    public FACodecPredictor<T> PhonePredictor { get; }
    public FACodecPredictor<T>? DetailF0Predictor { get; }
    public FACodecPredictor<T>? DetailPhonePredictor { get; }
    public FACodecPredictor<T>? ContentF0Predictor { get; }
    public FACodecPredictor<T>? ProsodyPhonePredictor { get; }
    public FACodecPredictor<T>? TimbrePredictor { get; }
    public TiedEmbeddingLayer<T>[]? RedecoderEmbeddings { get; }
    public Ns2TransformerEncoder<T>? RedecoderProsody { get; }
    public DenseLayer<T>? RedecoderTimbreLinear { get; }
    public FACodecDecoderStack<T>? RedecoderStack { get; }

    /// <summary>
    /// The factors of an encoder latent <c>[1, dim, frames]</c> (<c>quantize</c> and the timbre extractor): z_p from the
    /// latent (v2: from the prosody features <c>[1, mels, frames]</c>), z_c from the latent, z_d from the latent minus
    /// stop(z_p + z_c); h_t the mean over frames of the timbre Transformer on the latent.
    /// </summary>
    public FACodecFactors<T> Factorize(Tensor<T> latent, Tensor<T>? prosodyFeature, bool training, Random? random)
    {
        var prosodyInput = latent;
        if (_c.MelProsody)
        {
            if (prosodyFeature is null) throw new ArgumentNullException(nameof(prosodyFeature), "FACodec v2 quantizes prosody from mel features.");
            var rows = MelLinear!.Forward(Ns2Ops.Rows(_engine, prosodyFeature));
            prosodyInput = Ns2Ops.Channels(_engine, MelEncoder!.Forward(rows, null, training, random));
        }
        var zp = Quantizers[0].Forward(prosodyInput);
        var zc = Quantizers[1].Forward(latent);
        var detailInput = _engine.TensorSubtract(latent, Ns2Ops.Detach(_engine.TensorAdd(zp, zc)));
        var zd = Quantizers[2].Forward(detailInput);
        int frames = latent.Shape[2], codebooks = _c.Codebooks;
        var codes = new int[codebooks, frames];
        int row = 0;
        foreach (var q in Quantizers)
        {
            var c = q.LastCodes!;
            for (int k = 0; k < c.GetLength(0); k++, row++)
                for (int f = 0; f < frames; f++) codes[row, f] = c[k, f];
        }
        var timbre = TimbreEncoder.Forward(Ns2Ops.Rows(_engine, latent), null, training, random);
        var speaker = _engine.TensorMultiplyScalar(_engine.ReduceSum(timbre, new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / frames));
        return new FACodecFactors<T>(codes, new[] { zp, zc, zd }, speaker);
    }

    /// <summary>The sum of the code vectors of codes <c>[codebooks, frames]</c> (<c>vq2emb</c>), with or without the detail
    /// codes.</summary>
    public Tensor<T> Embed(int[,] codes, bool detail = true)
    {
        Tensor<T>? sum = null;
        int row = 0;
        for (int g = 0; g < Quantizers.Length; g++)
        {
            if (g == 2 && !detail) break;
            int n = GroupSize(g);
            var part = new int[n, codes.GetLength(1)];
            for (int k = 0; k < n; k++, row++)
                for (int f = 0; f < codes.GetLength(1); f++) part[k, f] = codes[row, f];
            var y = Quantizers[g].Decode(part);
            sum = sum is null ? y : _engine.TensorAdd(sum, y);
        }
        return sum!;
    }

    /// <summary>The waveform <c>[1, 1, samples]</c> of a latent <c>[1, dim, frames]</c> in a timbre <c>[1, dim]</c>
    /// (<c>inference</c>): the non-affine LayerNorm over channels, scaled and shifted by <c>timbre_linear(h_t)</c>.</summary>
    public Tensor<T> Decode(Tensor<T> latent, Tensor<T> speaker) => Decoder.Forward(Condition(latent, speaker, TimbreLinear));

    private Tensor<T> Condition(Tensor<T> latent, Tensor<T> speaker, DenseLayer<T> linear)
    {
        int d = _c.Dim;
        var style = linear.Forward(speaker);                                                               // [1, 2d]
        var gamma = _engine.TensorNarrow(style, 1, 0, d);
        var beta = _engine.TensorNarrow(style, 1, d, d);
        var rows = Ns2Ops.NormalizeLastAxis(_engine, Ns2Ops.Rows(_engine, latent), 1e-5);
        rows = _engine.TensorAdd(_engine.TensorMultiply(rows, _engine.TensorBroadcastTo(gamma, rows._shape)), _engine.TensorBroadcastTo(beta, rows._shape));
        return Ns2Ops.Channels(_engine, rows);
    }

    /// <summary>The redecoder's embedding of codes <c>[codebooks, frames]</c> in a timbre (<c>FACodecRedecoder.vq2emb</c>):
    /// its own prosody embeddings through a Transformer style-adaptive on the timbre, plus the content (and detail)
    /// embeddings.</summary>
    public Tensor<T> RedecoderEmbed(int[,] codes, Tensor<T> speaker, bool detail, bool training = false, Random? random = null)
    {
        if (RedecoderEmbeddings is null) throw new InvalidOperationException("This FACodec has no redecoder.");
        int frames = codes.GetLength(1), p = _c.ProsodyQuantizers, c = _c.ContentQuantizers;
        Tensor<T> Row(int k) => RedecoderEmbeddings[k].Forward(IndexTtsOps.Ids<T>(Enumerable.Range(0, frames).Select(f => codes[k, f]).ToArray()));
        Tensor<T>? prosody = null;
        for (int k = 0; k < p; k++)
        {
            // The release re-encodes the running sum after each prosody codebook.
            prosody = prosody is null ? Row(k) : _engine.TensorAdd(prosody, Row(k));
            prosody = RedecoderProsody!.Forward(prosody, speaker, training, random);
        }
        var sum = prosody!;
        for (int k = p; k < p + c; k++) sum = _engine.TensorAdd(sum, Row(k));
        if (detail) for (int k = p + c; k < _c.Codebooks; k++) sum = _engine.TensorAdd(sum, Row(k));
        return Ns2Ops.Channels(_engine, sum);
    }

    /// <summary>The redecoder's waveform of its embedding in a timbre.</summary>
    public Tensor<T> Redecode(Tensor<T> latent, Tensor<T> speaker) =>
        (RedecoderStack ?? throw new InvalidOperationException("This FACodec has no redecoder.")).Forward(Condition(latent, speaker, RedecoderTimbreLinear!));

    // ---------------------------------------------------------------- v2's prosody features

    /// <summary>
    /// v2's prosody features (<c>get_prosody_feature</c>): the first <see cref="FACodecConfiguration.ProsodyMels"/> bins of
    /// an 80-band log-mel spectrogram (n_fft 1024, hop 200, window 800 Hann, 0-8 kHz Slaney mel, reflect-padded by 412,
    /// uncentred; magnitude √(re² + im² + 1e-9), log(max(·, 1e-5))) as <c>[1, mels, frames]</c>.
    /// </summary>
    public Tensor<T> ProsodyFeature(IReadOnlyList<double> wave)
    {
        const int nFft = 1024, hop = 200, win = 800, mels = 80;
        _melBasis ??= Slaney(nFft, mels);
        _melWindow ??= Enumerable.Range(0, win).Select(n => 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / win)).ToArray();
        int pad = (nFft - hop) / 2, length = wave.Count + 2 * pad, frames = (length - nFft) / hop + 1, binsCount = nFft / 2 + 1;
        double Sample(int i)
        {
            int j = i - pad;
            if (j < 0) j = -j;
            if (j >= wave.Count) j = 2 * (wave.Count - 1) - j;
            return wave[j];
        }
        var feature = new Tensor<T>(new[] { 1, _c.ProsodyMels, frames });
        var magnitude = new double[binsCount];
        int offset = (nFft - win) / 2;
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < binsCount; k++)
            {
                double re = 0, im = 0;
                for (int n = 0; n < win; n++)
                {
                    double v = Sample(f * hop + offset + n) * _melWindow[n], angle = -2 * Math.PI * k * (offset + n) / nFft;
                    re += v * Math.Cos(angle);
                    im += v * Math.Sin(angle);
                }
                magnitude[k] = Math.Sqrt(re * re + im * im + 1e-9);
            }
            for (int m = 0; m < _c.ProsodyMels; m++)
            {
                double s = 0;
                for (int k = 0; k < binsCount; k++) s += _melBasis[m, k] * magnitude[k];
                feature[0, m, f] = NumOps.FromDouble(Math.Log(Math.Max(s, 1e-5)));
            }
        }
        return feature;
    }

    private double[,] Slaney(int nFft, int mels) =>
        new TacotronSpectrogram(_c.SampleRate, nFft, 200, 800, mels, 0.0, 8000.0).MelBasis;

    internal void SetMel(double[] basis, double[] window)
    {
        int mels = 80, bins = 513;
        _melBasis = new double[mels, bins];
        for (int m = 0; m < mels; m++)
            for (int k = 0; k < bins; k++) _melBasis[m, k] = basis[m * bins + k];
        _melWindow = window;
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads the release's encoder state dict.</summary>
    public void LoadEncoder(Func<string, double[]?> read) => LoadEncoder(TorchTensorSource.From(read));

    /// <summary>Loads the release's encoder state dict from <paramref name="source"/>.</summary>
    internal void LoadEncoder(TorchTensorSource source)
    {
        Encoder.Load(source, "");
        if (_c.MelProsody && source.TryRead("mel_transform.mel_basis") is { } basis && source.TryRead("mel_transform.hann_window") is { } window) SetMel(basis, window);
    }

    /// <summary>Loads the release's decoder state dict (v1 or v2).</summary>
    public void LoadDecoder(Func<string, double[]?> read) => LoadDecoder(TorchTensorSource.From(read));

    /// <summary>Loads the release's decoder state dict from <paramref name="source"/>.</summary>
    internal void LoadDecoder(TorchTensorSource source)
    {
        int d = _c.Dim;
        for (int g = 0; g < Quantizers.Length; g++)
        {
            var q = Quantizers[g];
            int n = GroupSize(g);
            for (int k = 0; k < n; k++)
            {
                string p = $"quantizer.{g}.layers.{k}.";
                FACodecWeights.Conv(source, q.InProjection(k), p + "in_proj");
                FACodecWeights.Conv(source, q.OutProjection(k), p + "out_proj");
                var book = q.Codebook(k);
                source.CopyTo(p + "_codebook.weight", book.Shape.ToArray(), book);
                _engine.InvalidatePersistentTensor(book);
            }
        }
        TimbreEncoder.Load(source, "timbre_encoder.");
        TorchParameters.Linear(_engine, TimbreLinear, d, 2 * d, source, "timbre_linear.weight", "timbre_linear.bias");
        Decoder.Load(source, "model.");
        if (MelLinear is not null)
        {
            TorchParameters.Linear(_engine, MelLinear, _c.ProsodyMels, d, source, "melspec_linear.weight", "melspec_linear.bias");
            MelEncoder!.Load(source, "melspec_encoder.");
        }
        F0Predictor.Load(source, "f0_predictor.");
        PhonePredictor.Load(source, "phone_predictor.");
        DetailF0Predictor?.Load(source, "res_f0_predictor.1.");
        DetailPhonePredictor?.Load(source, "res_phone_predictor.1.");
        ContentF0Predictor?.Load(source, "content_f0_predictor.1.");
        ProsodyPhonePredictor?.Load(source, "prosody_phone_predictor.1.");
        TimbrePredictor?.Load(source, "x_timbre_predictor.1.");
    }

    /// <summary>Loads the release's redecoder state dict.</summary>
    public void LoadRedecoder(Func<string, double[]?> read) => LoadRedecoder(TorchTensorSource.From(read));

    /// <summary>Loads the release's redecoder state dict from <paramref name="source"/>.</summary>
    internal void LoadRedecoder(TorchTensorSource source)
    {
        int d = _c.Dim, p = _c.ProsodyQuantizers, c = _c.ContentQuantizers;
        for (int k = 0; k < _c.Codebooks; k++)
        {
            string name = k < p ? $"prosody_embs.{k}" : k < p + c ? $"content_embs.{k - p}" : $"residual_embs.{k - p - c}";
            RedecoderEmbeddings![k].LoadLeadingRows(source, name + ".weight");
        }
        RedecoderProsody!.Load(source, "timbre_cond_prosody_enc.");
        TorchParameters.Linear(_engine, RedecoderTimbreLinear!, d, 2 * d, source, "timbre_linear.weight", "timbre_linear.bias");
        RedecoderStack!.Load(source, "model.");
    }
}
