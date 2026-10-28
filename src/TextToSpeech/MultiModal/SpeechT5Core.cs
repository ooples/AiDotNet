using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpeechT5 (Ao et al. 2022) as the official <c>T5TransformerModel</c> composes it: the shared encoder-decoder with the
/// text encoder pre-net, the speech encoder pre-net, the speech decoder pre-net and post-net, the text decoder pre-net and
/// post-net (one shared embedding), the HuBERT post-net and the cross-modal quantizer; text to speech with the speech
/// decoder, and the joint pre-training objectives.
/// </summary>
internal sealed class SpeechT5Core<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public SpeechT5Core(IEngine engine, List<LayerBase<T>> layers, List<LayerBase<T>> pretrainLayers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        Configuration = c;
        int start = layers.Count;
        // fairseq's Embedding: N(0, d^-1/2) with the padding row zero.
        Embedding = new TiedEmbeddingLayer<T>(c.TextVocabulary, c.Hidden);
        layers.Add(Embedding);
        double std = Math.Pow(c.Hidden, -0.5);
        Embedding.Reinitialize(() => std * IndexTtsOps.Gaussian(random), c.PadId);
        TextPositions = new SpeechT5ScaledPositions<T>(engine, layers, c.Hidden);
        Encoder = new SpeechT5Encoder<T>(engine, layers, c, random);
        TextEncoderLayers = layers.Skip(start).ToList();
        Decoder = new SpeechT5Decoder<T>(engine, layers, c, random);
        SpeechPrenet = new SpeechT5SpeechDecoderPrenet<T>(engine, layers, c, random);
        SpeechPostnet = new SpeechT5SpeechDecoderPostnet<T>(engine, layers, c, random);
        SpeechEncoderPrenet = new SpeechT5SpeechEncoderPrenet<T>(engine, pretrainLayers, c, random);
        Hubert = new SpeechT5HubertPostnet<T>(engine, pretrainLayers, c, random);
        Quantizer = new SpeechT5Quantizer<T>(engine, pretrainLayers, random, c.Hidden, c.LatentGroups, c.LatentVars);
    }

    public SpeechT5Configuration Configuration { get; }

    /// <summary>The text encoder pre-net's and the shared encoder's layers (what the TTS recipe freezes).</summary>
    public List<LayerBase<T>> TextEncoderLayers { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public SpeechT5ScaledPositions<T> TextPositions { get; }
    public SpeechT5Encoder<T> Encoder { get; }
    public SpeechT5Decoder<T> Decoder { get; }
    public SpeechT5SpeechDecoderPrenet<T> SpeechPrenet { get; }
    public SpeechT5SpeechDecoderPostnet<T> SpeechPostnet { get; }
    public SpeechT5SpeechEncoderPrenet<T> SpeechEncoderPrenet { get; }
    public SpeechT5HubertPostnet<T> Hubert { get; }
    public SpeechT5Quantizer<T> Quantizer { get; }

    // ---------------------------------------------------------------- text to speech

    /// <summary>The encoder states <c>[N, d]</c> of text tokens (the text encoder pre-net: the shared embedding plus the
    /// scaled positional encoding).</summary>
    public Tensor<T> EncodeText(IReadOnlyList<int> tokens, bool training = false, Random? random = null)
    {
        var x = TextPositions.Forward(Embedding.Forward(IndexTtsOps.Ids<T>(tokens)), 0, Configuration.PositionalDropout, training, random);
        return Encoder.Forward(x, training, random);
    }

    /// <summary>The teacher-forced decoder inputs of a target mel <c>[F, mel]</c>: every r-th frame (the last of each group),
    /// shifted right behind a zero frame.</summary>
    public Tensor<T> DecoderInputs(Tensor<T> mel)
    {
        int r = Configuration.ReductionFactor, steps = mel.Shape[0] / r, mels = Configuration.MelChannels;
        var rows = new Tensor<int>(new[] { Math.Max(steps - 1, 0) });
        for (int i = 0; i < steps - 1; i++) rows[i] = i * r + r - 1;
        var zero = new Tensor<T>(new[] { 1, mels });
        return steps <= 1 ? zero : _engine.TensorConcatenate(new[] { zero, _engine.TensorIndexSelect(mel, rows, 0) }, 0);
    }

    internal sealed record SpeechOutputs(Tensor<T> Before, Tensor<T> After, Tensor<T> Logits, List<Tensor<T>> CrossWeights);

    /// <summary>The speech decoder on decoder inputs <c>[L, mel]</c> over encoder states, with the x-vector.</summary>
    public SpeechOutputs DecodeSpeech(Tensor<T> memory, Tensor<T> inputs, Tensor<T> speaker, bool training = false, Random? random = null)
    {
        var x = SpeechPrenet.Forward(inputs, speaker, 0, training, random);
        var (states, weights) = Decoder.Forward(x, memory, null, training, random);
        var (before, logits) = SpeechPostnet.Project(states);
        return new SpeechOutputs(before, SpeechPostnet.Refine(before, training, random), logits, weights);
    }

    /// <summary>
    /// The official <c>generate_speech</c>: from a zero frame, each step decodes the pre-net of the frames so far (its
    /// dropout drawn from <paramref name="random"/>), emits r frames and r stop probabilities, and feeds back the last
    /// frame, until a probability reaches <paramref name="threshold"/> or <c>⌊N · 20 / r⌋</c> steps; then the post-net.
    /// With <paramref name="summedStop"/>, the step's r probabilities summed must reach it instead (the Hugging Face port).
    /// </summary>
    public (Tensor<T> Mel, double[] Probabilities) Generate(IReadOnlyList<int> tokens, Tensor<T> speaker, Random random, double threshold = 0.5,
        double maxRatio = 20.0, bool summedStop = false)
    {
        var c = Configuration;
        var memory = EncodeText(tokens);
        int maxSteps = (int)(memory.Shape[0] * maxRatio / c.ReductionFactor);
        var state = Decoder.Begin(memory);
        var frames = new List<Tensor<T>> { new Tensor<T>(new[] { 1, c.MelChannels }) };
        var outputs = new List<Tensor<T>>();
        var probabilities = new List<double>();
        for (int step = 1; ; step++)
        {
            var all = _engine.TensorConcatenate(frames.ToArray(), 0);
            var prenet = SpeechPrenet.Forward(all, speaker, 0, false, random);
            var (z, _) = Decoder.Forward(_engine.TensorNarrow(prenet, 0, step - 1, 1), memory, state);
            var (emitted, logits) = SpeechPostnet.Project(z);
            outputs.Add(emitted);
            bool stop = false;
            double sum = 0;
            for (int i = 0; i < c.ReductionFactor; i++)
            {
                double p = 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(logits[i])));
                probabilities.Add(p);
                sum += p;
                stop |= p >= threshold;
            }
            if (summedStop) stop = sum >= threshold;
            frames.Add(_engine.TensorNarrow(emitted, 0, c.ReductionFactor - 1, 1));
            if (stop || step >= maxSteps) break;
        }
        var mel = _engine.TensorConcatenate(outputs.ToArray(), 0);
        return (SpeechPostnet.Refine(mel, false, null), probabilities.ToArray());
    }

    /// <summary>
    /// The speech decoder's loss (the official <c>TexttoSpeechLoss.compute_loss</c>): on the target cut to a multiple of r
    /// with a stop label at its last frame, L1 of the frames before and after the post-net, binary cross-entropy of the
    /// stop logits with positive weight 5, and, when <paramref name="guidedHeads"/> &gt; 0, the guided attention loss
    /// (σ = 0.4) on that many heads of every layer's cross-attention.
    /// </summary>
    public Tensor<T> SpeechLoss(SpeechOutputs outputs, Tensor<T> mel, int sourceLength, int guidedHeads, double sigma = 0.4, double positiveWeight = 5.0)
    {
        var c = Configuration;
        int frames = mel.Shape[0], olen = frames - frames % c.ReductionFactor;
        var target = _engine.TensorNarrow(mel, 0, 0, olen);
        double scale = 1.0 / (olen * c.MelChannels);
        var l1 = _engine.TensorMultiplyScalar(_engine.TensorAdd(
            _engine.ReduceSum(_engine.TensorAbs(_engine.TensorSubtract(outputs.After, target)), null, false),
            _engine.ReduceSum(_engine.TensorAbs(_engine.TensorSubtract(outputs.Before, target)), null, false)), NumOps.FromDouble(scale));
        var labels = new Tensor<T>(new[] { olen });
        labels[olen - 1] = NumOps.One;
        if (frames - 1 < olen) labels[frames - 1] = NumOps.One;
        // BCEWithLogits(pos_weight w): (1 − y)·x + (1 + (w − 1)·y)·softplus(−x).
        var x = outputs.Logits;
        var weight = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(labels, NumOps.FromDouble(positiveWeight - 1)), NumOps.One);
        var bce = _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorSubtract(_engine.TensorAddScalar(new Tensor<T>(labels.Shape.ToArray()), NumOps.One), labels), x),
            _engine.TensorMultiply(weight, _engine.Softplus(_engine.TensorNegate(x))));
        var loss = _engine.TensorAdd(l1, _engine.TensorMultiplyScalar(_engine.ReduceSum(bce, null, false), NumOps.FromDouble(1.0 / olen)));
        if (guidedHeads <= 0) return loss;
        int steps = frames / c.ReductionFactor;
        var mask = new Tensor<T>(new[] { guidedHeads, steps, sourceLength });
        for (int o = 0; o < steps; o++)
            for (int i = 0; i < sourceLength; i++)
            {
                double value = 1.0 - Math.Exp(-Math.Pow((double)i / sourceLength - (double)o / steps, 2) / (2 * sigma * sigma));
                for (int h = 0; h < guidedHeads; h++) mask[h, o, i] = NumOps.FromDouble(value);
            }
        Tensor<T>? guided = null;
        foreach (var w in outputs.CrossWeights)
        {
            var heads = _engine.TensorNarrow(w, 0, 0, guidedHeads);
            var term = _engine.ReduceSum(_engine.TensorMultiply(heads, mask), null, false);
            guided = guided is null ? term : _engine.TensorAdd(guided, term);
        }
        double count = (double)outputs.CrossWeights.Count * guidedHeads * steps * sourceLength;
        return _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(guided!, NumOps.FromDouble(1.0 / count)));
    }

    // ---------------------------------------------------------------- pre-training

    /// <summary>The encoder states with <c>⌊T · p⌋</c> of their rows (the first of <paramref name="permutation"/>)
    /// replaced by their quantized vectors, and the quantizer's perplexity.</summary>
    public (Tensor<T> Mixed, Tensor<T> Perplexity) MixCodebook(Tensor<T> states, IReadOnlyList<int> permutation, double probability, bool training,
        Random? random)
    {
        int t = states.Shape[0], d = Configuration.Hidden;
        var (quantized, perplexity) = Quantizer.Forward(states, training, random);
        var choose = new Tensor<T>(new[] { t, d });
        var keep = new Tensor<T>(new[] { t, d });
        var chosen = new HashSet<int>(permutation.Take((int)(t * probability)));
        for (int i = 0; i < t; i++)
            for (int k = 0; k < d; k++) (chosen.Contains(i) ? choose : keep)[i, k] = NumOps.One;
        return (_engine.TensorAdd(_engine.TensorMultiply(quantized, choose), _engine.TensorMultiply(states, keep)), perplexity);
    }

    internal sealed record SpeechPretrainTerms(Tensor<T> Total, Tensor<T> Masked, Tensor<T> FeaturePenalty, Tensor<T> Perplexity, Tensor<T> Decoder, int Count);

    /// <summary>
    /// The speech pre-training loss (the official <c>SpeechPretrainCriterion</c>): HuBERT's masked prediction (the sum of
    /// cross-entropies over the masked frames), plus 10 × the feature penalty and 0.1 × the codebook diversity, each times
    /// the masked count S, plus S × the speech decoder's reconstruction loss (L1 + BCE) on the log-mel target from the
    /// codebook-mixed encoder states.
    /// </summary>
    public SpeechPretrainTerms SpeechPretrainLoss(Tensor<T> wave, IReadOnlyList<int[]> labels, Tensor<T> mel, Tensor<T> speaker, bool[] mask,
        IReadOnlyList<int> permutation, double codebookProbability, double labelRate, int sampleRate, bool training, Random? random,
        double penaltyWeight = 10, double diversityWeight = 0.1)
    {
        var c = Configuration;
        // forward_targets: the label of each feature frame at ⌊t · label_rate · Π stride / sample_rate⌋, the features cut
        // to the frames the shortest label sequence covers.
        double ratio = labelRate * c.FeatureLayers.Aggregate(1, (a, l) => a * l.Stride) / sampleRate;
        int frames = SpeechEncoderPrenet.Frames(wave.Length), shortest = labels.Min(l => l.Length);
        if (ratio * frames > shortest) frames = (int)(shortest / ratio);
        var (inputs, penalty) = SpeechEncoderPrenet.Forward(wave, mask, training, random, frames);
        var targets = labels.Select(l => Enumerable.Range(0, frames).Select(t => l[(int)(t * ratio)]).ToArray()).ToList();
        var states = Encoder.Forward(inputs, training, random);
        var logits = Hubert.MaskedLogits(states, mask, targets);
        Tensor<T>? masked = null;
        foreach (var l in logits)
        {
            var term = _engine.TensorNegate(_engine.ReduceSum(_engine.TensorNarrow(_engine.TensorLogSoftmax(l, 1), 1, 0, 1), null, false));
            masked = masked is null ? term : _engine.TensorAdd(masked, term);
        }
        int count = mask.Count(m => m);
        var (mixed, perplexity) = MixCodebook(states, permutation, codebookProbability, training, random);
        var speech = DecodeSpeech(mixed, DecoderInputs(mel), speaker, training, random);
        var decoder = SpeechLoss(speech, mel, frames, 0);
        var s = NumOps.FromDouble(count);
        var total = _engine.TensorAdd(masked!, _engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(penaltyWeight * count)));
        total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(Quantizer.Diversity(perplexity), NumOps.FromDouble(diversityWeight * count)));
        total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(decoder, s));
        return new SpeechPretrainTerms(total, masked!, penalty, perplexity, decoder, count);
    }

    /// <summary>The text decoder's logits <c>[L, V]</c> of previous tokens (the shared embedding plus fairseq's sinusoidal
    /// positions from 2, the decoder, the tied output projection).</summary>
    public Tensor<T> TextLogits(Tensor<T> memory, IReadOnlyList<int> previous, bool training = false, Random? random = null)
    {
        var c = Configuration;
        var x = _engine.TensorAdd(Embedding.Forward(IndexTtsOps.Ids<T>(previous)), SpeechT5Ops.FairseqTable<T>(previous.Count, c.Hidden, c.PadId + 1));
        x = SpeechT5Ops.Drop(_engine, x, c.Dropout, training, random);
        var (states, _) = Decoder.Forward(x, memory, null, training, random);
        return Embedding.Logits(states);
    }

    /// <summary>
    /// The text pre-training loss (the official <c>TextPretrainCriterion</c>): the summed cross-entropy of reconstructing
    /// the original tokens from the noised ones (BART's text infilling), plus 0.1 × the codebook diversity times the token
    /// count.
    /// </summary>
    public Tensor<T> TextPretrainLoss(IReadOnlyList<int> noised, IReadOnlyList<int> target, IReadOnlyList<int> permutation, double codebookProbability,
        bool training, Random? random, double diversityWeight = 0.1)
    {
        var c = Configuration;
        var states = EncodeText(noised, training, random);
        var (mixed, perplexity) = MixCodebook(states, permutation, codebookProbability, training, random);
        var previous = new List<int> { c.EosId };
        previous.AddRange(target.Take(target.Count - 1));
        var logits = TextLogits(mixed, previous, training, random);
        var logp = _engine.Reshape(_engine.TensorLogSoftmax(logits, 1), new[] { target.Count * c.TextVocabulary, 1 });
        var picks = Enumerable.Range(0, target.Count).Where(i => target[i] != c.PadId).Select(i => i * c.TextVocabulary + target[i]).ToArray();
        var nll = _engine.TensorNegate(_engine.ReduceSum(_engine.TensorIndexSelect(logp, new Tensor<int>(new[] { picks.Length }, new Vector<int>(picks)), 0), null, false));
        return _engine.TensorAdd(nll, _engine.TensorMultiplyScalar(Quantizer.Diversity(perplexity), NumOps.FromDouble(diversityWeight * target.Count)));
    }

    /// <summary>
    /// The official pre-training's <c>--bert-init</c> (fairseq's <c>init_bert_params</c>): every linear layer's weight
    /// drawn from N(0, 0.02) and its bias zeroed, and every embedding table (the text embedding with its padding row zero,
    /// the relative positions) drawn from N(0, 0.02).
    /// </summary>
    public void BertInitialize(IEnumerable<LayerBase<T>> layers, Random random)
    {
        foreach (var dense in layers.OfType<DenseLayer<T>>())
        {
            var w = dense.GetWeights();
            for (int i = 0; i < w.Length; i++) w.SetFlat(i, NumOps.FromDouble(0.02 * IndexTtsOps.Gaussian(random)));
            _engine.InvalidatePersistentTensor(w);
            var b = dense.GetBiases();
            for (int i = 0; i < b.Length; i++) b.SetFlat(i, NumOps.Zero);
            _engine.InvalidatePersistentTensor(b);
        }
        Embedding.Reinitialize(() => 0.02 * IndexTtsOps.Gaussian(random), Configuration.PadId);
        Encoder.RelativePositions.Reinitialize(() => 0.02 * IndexTtsOps.Gaussian(random));
    }

    /// <summary>
    /// The Hugging Face name (<c>microsoft/speecht5_tts</c>) of an official checkpoint tensor this model reads, or null
    /// for one the text-to-speech checkpoint does not hold.
    /// </summary>
    public static string? HuggingFaceName(string name)
    {
        string Layer(string rest) => rest.Replace("fc1", "feed_forward.intermediate_dense").Replace("fc2", "feed_forward.output_dense");
        var m = System.Text.RegularExpressions.Regex.Match(name, @"^encoder\.layers\.(\d+)\.(.+)$");
        if (m.Success)
        {
            string rest = m.Groups[2].Value;
            rest = rest.StartsWith("self_attn_layer_norm.", StringComparison.Ordinal) ? "layer_norm." + rest.Substring("self_attn_layer_norm.".Length)
                : rest.StartsWith("self_attn.", StringComparison.Ordinal) ? "attention." + rest.Substring("self_attn.".Length) : Layer(rest);
            return $"speecht5.encoder.wrapped_encoder.layers.{m.Groups[1].Value}.{rest}";
        }
        m = System.Text.RegularExpressions.Regex.Match(name, @"^decoder\.layers\.(\d+)\.(.+)$");
        if (m.Success) return $"speecht5.decoder.wrapped_decoder.layers.{m.Groups[1].Value}.{Layer(m.Groups[2].Value)}";
        m = System.Text.RegularExpressions.Regex.Match(name, @"^speech_decoder_prenet\.decoder_prenet\.0\.0\.prenet\.(\d+)\.0\.(.+)$");
        if (m.Success) return $"speecht5.decoder.prenet.layers.{m.Groups[1].Value}.{m.Groups[2].Value}";
        m = System.Text.RegularExpressions.Regex.Match(name, @"^speech_decoder_postnet\.postnet\.postnet\.(\d+)\.([01])\.(.+)$");
        if (m.Success) return $"speech_decoder_postnet.layers.{m.Groups[1].Value}.{(m.Groups[2].Value == "0" ? "conv" : "batch_norm")}.{m.Groups[3].Value}";
        switch (name)
        {
            case "text_encoder_prenet.encoder_prenet.0.weight": return "speecht5.encoder.prenet.embed_tokens.weight";
            case "text_encoder_prenet.encoder_prenet.1.alpha": return "speecht5.encoder.prenet.encode_positions.alpha";
            case "encoder.layer_norm.weight": return "speecht5.encoder.wrapped_encoder.layer_norm.weight";
            case "encoder.layer_norm.bias": return "speecht5.encoder.wrapped_encoder.layer_norm.bias";
            case "encoder.pos_emb.pe_k.weight": return "speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight";
            case "speech_decoder_prenet.decoder_prenet.1.alpha": return "speecht5.decoder.prenet.encode_positions.alpha";
        }
        if (name.StartsWith("speech_decoder_prenet.decoder_prenet.0.1.", StringComparison.Ordinal))
            return "speecht5.decoder.prenet.final_layer." + name.Substring("speech_decoder_prenet.decoder_prenet.0.1.".Length);
        if (name.StartsWith("speech_decoder_prenet.spkembs_layer.0.", StringComparison.Ordinal))
            return "speecht5.decoder.prenet.speaker_embeds_layer." + name.Substring("speech_decoder_prenet.spkembs_layer.0.".Length);
        if (name.StartsWith("speech_decoder_postnet.feat_out.", StringComparison.Ordinal) || name.StartsWith("speech_decoder_postnet.prob_out.", StringComparison.Ordinal))
            return name;
        return null;
    }

    /// <summary>Loads the official checkpoint's names (<c>encoder.*</c>, <c>decoder.*</c>, the pre- and post-nets,
    /// <c>hubert_layer.*</c>, <c>quantizer.*</c>); the pre-training modules only when <paramref name="pretraining"/>.</summary>
    public void Load(Func<string, int[], double[]> read, bool pretraining)
    {
        var c = Configuration;
        Embedding.LoadTable(read("text_encoder_prenet.encoder_prenet.0.weight", new[] { c.TextVocabulary, c.Hidden }));
        TextPositions.Load(read, "text_encoder_prenet.encoder_prenet.1.alpha");
        Encoder.Load(read, "encoder.");
        Decoder.Load(read, "decoder.");
        SpeechPrenet.Load(read, "speech_decoder_prenet.");
        SpeechPostnet.Load(read, "speech_decoder_postnet.");
        if (!pretraining) return;
        SpeechEncoderPrenet.Load(read, "speech_encoder_prenet.");
        Hubert.Load(read, "hubert_layer.");
        Quantizer.Load(read, "quantizer.");
    }
}
