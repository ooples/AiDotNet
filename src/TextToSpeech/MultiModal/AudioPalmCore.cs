using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// AudioPaLM's machinery, shared by <see cref="AudioPaLM{T}"/> (speech and text out) and its speech-recognition task
/// variant: the decoder over the joint text-and-audio vocabulary (§3.2), the speech tokenizer (§3.1), the text tokenizer,
/// the detokenizer (§3.3), task prefixes (§3.4), generation and the loss masked to the outputs (§3.6).
/// </summary>
internal sealed class AudioPalmCore<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly AudioPaLMOptions _o;
    private readonly List<LayerBase<T>> _w2vBert2Layers = new();
    private SentencePieceModel? _tokenizer;

    public AudioPalmCore(IEngine engine, AudioPaLMOptions options, List<LayerBase<T>> decoderLayers, int seed, bool withDetokenizer)
    {
        _engine = engine;
        _o = options;
        Validate(options);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        Decoder = new PalmDecoder<T>(engine, decoderLayers, DecoderConfiguration(options), random);
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
        { RandomSeed = seed };
        int embeddingDim;
        if (options.SpeechTokenizer == AudioPalmSpeechTokenizer.W2vBert)
        {
            SpeechEncoder = new W2vBert<T>(architecture, new W2vBertOptions(options.W2vBert));
            embeddingDim = options.W2vBert.ModelDim;
        }
        else
        {
            var config = new Wav2Vec2BertConfiguration();
            W2vBert2Features = new Wav2Vec2BertFeatures<T>(engine, config);
            W2vBert2 = new Wav2Vec2BertNetwork<T>(engine, _w2vBert2Layers, config, options.W2vBert2Layer);
            embeddingDim = config.HiddenSize;
        }
        // The k-means centroids [AudioTokens, d], held as parameters so a clone or a saved model keeps the fitted
        // tokenizer; all zero until fitted or set.
        Codebook = new BiasFreeLinearLayer<T>(1, options.AudioTokens * embeddingDim);
        _codebookLayers.Add(Codebook);
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(Codebook, 1, options.AudioTokens * embeddingDim, new double[options.AudioTokens * embeddingDim]);
        _embeddingDim = embeddingDim;
        if (!withDetokenizer) return;
        if (options.Detokenizer == AudioPalmDetokenizer.SoundStorm)
            SoundStorm = new SoundStorm<T>(architecture, new SoundStormOptions(options.SoundStorm) { IncludeTextToSemantic = false });
        else
            AudioLM = new AudioLM<T>(architecture, options.AudioLM);
    }

    private readonly List<LayerBase<T>> _codebookLayers = new();
    private readonly int _embeddingDim;

    /// <summary>The audio tokens' k-means centroids, flattened <c>[AudioTokens * d]</c>.</summary>
    public BiasFreeLinearLayer<T> Codebook { get; }

    /// <summary>
    /// The layers of every component besides the decoder, in a fixed order: the codebook, the speech encoder's, and the
    /// detokenizer's with its codec's. A model registers them as component layers so its parameters, clones and saved
    /// state carry the fitted tokenizer and the components' weights; none of them trains with the decoder.
    /// </summary>
    public IEnumerable<LayerBase<T>> ComponentLayers()
    {
        foreach (var layer in _codebookLayers) yield return layer;
        if (SpeechEncoder is not null)
            foreach (var layer in SpeechEncoder.Layers.OfType<LayerBase<T>>()) yield return layer;
        foreach (var layer in _w2vBert2Layers) yield return layer;
        if (SoundStorm is not null)
        {
            foreach (var layer in SoundStorm.Layers.OfType<LayerBase<T>>()) yield return layer;
            foreach (var layer in SoundStorm.Codec.Layers.OfType<LayerBase<T>>()) yield return layer;
        }
        if (AudioLM is not null)
        {
            foreach (var layer in AudioLM.Layers.OfType<LayerBase<T>>()) yield return layer;
            foreach (var layer in AudioLM.Codec.Layers.OfType<LayerBase<T>>()) yield return layer;
        }
    }

    internal static void Validate(AudioPaLMOptions o)
    {
        if (o.NumHeads % o.KeyValueHeads != 0) throw new ArgumentException("NumHeads must be a multiple of KeyValueHeads.");
        if (o.AudioTokens < 1) throw new ArgumentException("AudioTokens must be positive.");
        if (o.BeginId < 0 || o.BeginId >= o.VocabSize || o.EndId < 0 || o.EndId >= o.VocabSize)
            throw new ArgumentException("BeginId and EndId must be text tokens.");
        if (o.VoicePromptSeconds <= 0) throw new ArgumentException("VoicePromptSeconds must be positive.");
        if (o.SampleRate != 16000) throw new ArgumentException("The speech tokenizers read 16 kHz audio.");
        int tokenRate = o.SpeechTokenizer == AudioPalmSpeechTokenizer.W2vBert ? 25 : 50;
        int detokenizerRate = o.Detokenizer == AudioPalmDetokenizer.SoundStorm ? o.SoundStorm.SemanticFrameRate : o.AudioLM.SemanticFrameRate;
        if (tokenRate != detokenizerRate)
            throw new ArgumentException($"The audio tokens come at {tokenRate} Hz; set the detokenizer's SemanticFrameRate to match.");
        int detokenizerCodes = o.Detokenizer == AudioPalmDetokenizer.SoundStorm ? o.SoundStorm.SemanticCodes : o.AudioLM.SemanticCodes;
        if (detokenizerCodes != o.AudioTokens)
            throw new ArgumentException($"The detokenizer reads {detokenizerCodes} semantic codes; AudioTokens is {o.AudioTokens}.");
    }

    internal static PalmConfiguration DecoderConfiguration(AudioPaLMOptions o) => new()
    {
        Kind = o.Decoder, VocabSize = o.VocabSize + o.AudioTokens, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.NumHeads,
        KeyValueHeads = o.KeyValueHeads, HeadDim = o.HeadDim, Intermediate = o.IntermediateSize, NormEpsilon = o.NormEpsilon,
        RopeTheta = o.RopeTheta, Dropout = o.DropoutRate,
    };

    public AudioPaLMOptions Options => _o;
    public PalmDecoder<T> Decoder { get; }
    public W2vBert<T>? SpeechEncoder { get; }
    public Wav2Vec2BertFeatures<T>? W2vBert2Features { get; }
    public Wav2Vec2BertNetwork<T>? W2vBert2 { get; }
    public IReadOnlyList<LayerBase<T>> W2vBert2Layers => _w2vBert2Layers;
    public SoundStorm<T>? SoundStorm { get; }
    public AudioLM<T>? AudioLM { get; }

    /// <summary>The first audio token's id: the text vocabulary's size.</summary>
    public int AudioOffset => _o.VocabSize;
    public int Vocabulary => _o.VocabSize + _o.AudioTokens;

    // ---------------------------------------------------------------- text

    public SentencePieceModel Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = _o.SentencePieceModelPath
                ?? throw new InvalidOperationException("Text needs the decoder's SentencePiece model; set AudioPaLMOptions.SentencePieceModelPath.");
            var tokenizer = new SentencePieceModel(File.ReadAllBytes(path));
            if (tokenizer.Count > _o.VocabSize)
                throw new InvalidOperationException($"The SentencePiece model has {tokenizer.Count} pieces; VocabSize is {_o.VocabSize}.");
            return _tokenizer = tokenizer;
        }
    }

    public void ResetTokenizer() => _tokenizer = null;

    public int[] TokenizeText(string text) => Tokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)));

    public string DecodeText(IEnumerable<int> ids) =>
        Tokenizer.Decode(ids.Where(id => id < _o.VocabSize && id != _o.EndId && id != _o.BeginId));

    /// <summary>The decoder ids of a task: the start token, the tag's text tokens, then the input's ids (§3.4).</summary>
    public int[] TaskPrefix(string tag, IReadOnlyList<int> input)
    {
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        if (input is null) throw new ArgumentNullException(nameof(input));
        return new[] { _o.BeginId }.Concat(TokenizeText(tag)).Concat(input).ToArray();
    }

    public int[] AudioIds(IReadOnlyList<int> codes) => codes.Select(c => AudioOffset + c).ToArray();

    public int[] AudioCodes(IReadOnlyList<int> ids) => ids.Where(id => id >= AudioOffset).Select(id => id - AudioOffset).ToArray();

    // ---------------------------------------------------------------- audio tokens

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    /// <summary>The embeddings <c>[frames, d]</c> k-means clusters: the w2v-BERT semantic block output (AudioLM's
    /// layer, not normalized, section 3.1) or the w2v-BERT 2.0 hidden state W2vBert2Layer.</summary>
    private double[,] Embeddings(Tensor<T> recording)
    {
        using var _ = new NoGradScope<T>();
        Tensor<T> hidden;
        var wave = recording.Rank == 1 ? recording : _engine.Reshape(recording, new[] { recording.Length });
        if (SpeechEncoder is not null)
        {
            SpeechEncoder.SetTrainingMode(false);
            hidden = SpeechEncoder.ExtractLayerFeatures(wave, _o.W2vBert.ContrastiveLayers + _o.W2vBert.SemanticLayer - 1);
        }
        else
        {
            foreach (var layer in _w2vBert2Layers) layer.SetTrainingMode(false);
            hidden = W2vBert2!.HiddenStates(W2vBert2Features!.Forward(Doubles(wave)))[_o.W2vBert2Layer];
        }
        int frames = hidden.Shape[hidden.Rank - 2], d = hidden.Shape[hidden.Rank - 1];
        hidden = _engine.Reshape(hidden, new[] { frames, d });
        var rows = new double[frames, d];
        for (int t = 0; t < frames; t++)
            for (int i = 0; i < d; i++) rows[t, i] = Ops.ToDouble(hidden[t, i]);
        return rows;
    }

    /// <summary>Fits the audio tokens' k-means codebook (w2v-BERT's tokenizer on its unnormalized embeddings, §3.1, or
    /// k-means over w2v-BERT 2.0's embeddings).</summary>
    public void FitSpeechTokenizer(IEnumerable<Tensor<T>> recordings)
    {
        if (recordings is null) throw new ArgumentNullException(nameof(recordings));
        var all = recordings.Select(Embeddings).ToList();
        if (all.Count == 0) throw new ArgumentException("No recordings.", nameof(recordings));
        int d = all[0].GetLength(1), frames = all.Sum(r => r.GetLength(0)), row = 0;
        if (frames < _o.AudioTokens) throw new ArgumentException($"{frames} frames cannot form {_o.AudioTokens} clusters.", nameof(recordings));
        var data = new Matrix<T>(frames, d);
        foreach (var r in all)
            for (int t = 0; t < r.GetLength(0); t++, row++)
                for (int i = 0; i < d; i++) data[row, i] = Ops.FromDouble(r[t, i]);
        var clusters = new KMeans<T>(new KMeansOptions<T> { NumClusters = _o.AudioTokens, Seed = _o.TrainingSeed });
        clusters.Train(data, new Vector<T>(data.Rows));
        var centers = clusters.ClusterCenters ?? throw new InvalidOperationException("k-means produced no centers.");
        var codebook = new double[centers.Rows, centers.Columns];
        for (int k = 0; k < centers.Rows; k++)
            for (int i = 0; i < centers.Columns; i++) codebook[k, i] = Ops.ToDouble(centers[k, i]);
        LoadSpeechTokenizerCodebook(codebook);
    }

    /// <summary>Sets the k-means centroids <c>[AudioTokens, d]</c>.</summary>
    public void LoadSpeechTokenizerCodebook(double[,] centroids)
    {
        if (centroids is null) throw new ArgumentNullException(nameof(centroids));
        if (centroids.GetLength(0) != _o.AudioTokens || centroids.GetLength(1) != _embeddingDim)
            throw new ArgumentException($"Give {_o.AudioTokens} centroids of {_embeddingDim} values.", nameof(centroids));
        var flat = new double[_o.AudioTokens * _embeddingDim];
        for (int k = 0; k < _o.AudioTokens; k++)
            for (int i = 0; i < _embeddingDim; i++) flat[k * _embeddingDim + i] = centroids[k, i];
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(Codebook, 1, flat.Length, flat);
    }

    /// <summary>The audio tokens of a 16 kHz recording, codes in [0, AudioTokens).</summary>
    public int[] AudioTokensOf(Tensor<T> recording)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var values = Codebook.GetParameters();
        int k = _o.AudioTokens, d = _embeddingDim;
        var codebook = new double[k, d];
        bool fitted = false;
        for (int c = 0; c < k; c++)
            for (int i = 0; i < d; i++) fitted |= (codebook[c, i] = Ops.ToDouble(values[c * d + i])) != 0;
        if (!fitted) throw new InvalidOperationException("Fit (FitSpeechTokenizer) or set (LoadSpeechTokenizerCodebook) the codebook first.");
        var rows = Embeddings(recording);
        int frames = rows.GetLength(0);
        var tokens = new int[frames];
        for (int t = 0; t < frames; t++)
        {
            double best = double.PositiveInfinity;
            for (int c = 0; c < k; c++)
            {
                double distance = 0;
                for (int i = 0; i < d; i++)
                {
                    double diff = rows[t, i] - codebook[c, i];
                    distance += diff * diff;
                }
                if (distance < best)
                {
                    best = distance;
                    tokens[t] = c;
                }
            }
        }
        return tokens;
    }

    // ---------------------------------------------------------------- generation

    /// <summary>Continues <paramref name="prefix"/> until the end token or MaxNewTokens, among audio tokens or text
    /// tokens (and the end after the first); greedy, or sampled at the temperature. The new ids, without the end.</summary>
    public int[] Generate(IReadOnlyList<int> prefix, bool audioOutput) => Generate(prefix, audioOutput, out _);

    /// <summary>As <see cref="Generate(IReadOnlyList{int}, bool)"/>, with the probability (over the allowed tokens) of
    /// each token chosen, the end token included.</summary>
    public int[] Generate(IReadOnlyList<int> prefix, bool audioOutput, out double[] probabilities)
    {
        if (prefix is null) throw new ArgumentNullException(nameof(prefix));
        if (prefix.Count == 0) throw new ArgumentException("The prefix is empty.", nameof(prefix));
        using var _ = new NoGradScope<T>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_o.SamplingSeed);
        var cache = Decoder.NewCache();
        var hidden = Decoder.Forward(Decoder.Embed(prefix), cache);
        var output = new List<int>();
        var chosen = new List<double>();
        for (int step = 0; step < _o.MaxNewTokens && prefix.Count + output.Count < _o.MaxTextLength; step++)
        {
            var logits = Decoder.Logits(_engine.TensorNarrow(hidden, 0, hidden.Shape[0] - 1, 1));
            int next = Choose(logits, audioOutput, step > 0, random, out double probability);
            chosen.Add(probability);
            if (next == _o.EndId) break;
            output.Add(next);
            hidden = Decoder.Forward(Decoder.Embed(new[] { next }), cache);
        }
        probabilities = chosen.ToArray();
        return output.ToArray();
    }

    private int Choose(Tensor<T> logits, bool audioOutput, bool mayEnd, Random random, out double probability)
    {
        bool Allowed(int id) => (mayEnd && id == _o.EndId) || (audioOutput ? id >= _o.VocabSize : id < _o.VocabSize && id != _o.EndId && id != _o.BeginId);
        int vocabulary = Vocabulary, best = -1;
        double bestValue = double.NegativeInfinity;
        var values = new double[vocabulary];
        for (int id = 0; id < vocabulary; id++)
        {
            values[id] = Allowed(id) ? Ops.ToDouble(logits.GetFlat(id)) : double.NegativeInfinity;
            if (values[id] > bestValue)
            {
                bestValue = values[id];
                best = id;
            }
        }
        // The softmax over the allowed tokens at the sampling temperature (1 when greedy).
        double temperature = _o.Temperature > 0 ? _o.Temperature : 1.0, total = 0;
        var weights = new double[vocabulary];
        for (int id = 0; id < vocabulary; id++)
        {
            weights[id] = double.IsNegativeInfinity(values[id]) ? 0 : Math.Exp((values[id] - bestValue) / temperature);
            total += weights[id];
        }
        int pick = best;
        if (_o.Temperature > 0)
        {
            double u = random.NextDouble() * total;
            for (int id = 0; id < vocabulary; id++)
            {
                u -= weights[id];
                if (u < 0 && weights[id] > 0)
                {
                    pick = id;
                    break;
                }
            }
        }
        probability = weights[pick] / total;
        return pick;
    }

    // ---------------------------------------------------------------- audio

    /// <summary>The first VoicePromptSeconds of a recording, repeated when shorter (§3.3).</summary>
    private Tensor<T> VoicePrompt(Tensor<T> recording)
    {
        int samples = (int)Math.Round(_o.VoicePromptSeconds * _o.SampleRate);
        var source = Doubles(recording);
        if (source.Length == 0) throw new ArgumentException("The voice recording is empty.", nameof(recording));
        var prompt = new Tensor<T>(new[] { samples });
        for (int i = 0; i < samples; i++) prompt[i] = Ops.FromDouble(source[i % source.Length]);
        return prompt;
    }

    /// <summary>Renders audio token codes to a 16 kHz waveform, continuing the voice prompt of
    /// <paramref name="voiceRecording"/> (its audio tokens and its codec codes, §3.3).</summary>
    public Tensor<T> Detokenize(IReadOnlyList<int> codes, Tensor<T> voiceRecording)
    {
        if (codes is null) throw new ArgumentNullException(nameof(codes));
        if (voiceRecording is null) throw new ArgumentNullException(nameof(voiceRecording));
        if (codes.Count == 0) return new Tensor<T>(new[] { 0 });
        var prompt = VoicePrompt(voiceRecording);
        var promptTokens = AudioTokensOf(prompt);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_o.SamplingSeed);
        if (SoundStorm is not null)
            return SoundStorm.SemanticToAudio(codes, promptTokens, SoundStorm.Codec.Encode(prompt), random);
        var audioLm = AudioLM ?? throw new InvalidOperationException("This AudioPaLM has no detokenizer.");
        return audioLm.Decode(audioLm.GenerateAcoustic(codes, audioLm.Codec.Encode(prompt), promptTokens, _o.SamplingSeed));
    }

    // ---------------------------------------------------------------- the loss

    /// <summary>
    /// The loss of <c>prefix · output · end</c>: per target position (the outputs and the end), the cross-entropy plus
    /// ZLossWeight · log²Z (PaLM §5), averaged over those positions; the prefix is masked (§3.6).
    /// </summary>
    public Tensor<T> Loss(int[] prefix, int[] output, bool training, Random? random)
    {
        var sequence = prefix.Concat(output).Concat(new[] { _o.EndId }).ToArray();
        if (sequence.Length > _o.MaxTextLength) throw new ArgumentException($"The example's {sequence.Length} tokens exceed {_o.MaxTextLength}.");
        if (sequence.Any(id => id < 0 || id >= Vocabulary)) throw new ArgumentOutOfRangeException(nameof(output), $"Token ids lie in [0, {Vocabulary}).");
        var inputs = sequence.Take(sequence.Length - 1).ToArray();
        var hidden = Decoder.Forward(Decoder.Embed(inputs), null, training, random);
        int first = prefix.Length - 1, count = output.Length + 1, vocabulary = Vocabulary;
        var logits = Decoder.Logits(_engine.TensorNarrow(hidden, 0, first, count));                          // [count, V]
        var logSoftmax = _engine.TensorLogSoftmax(logits, axis: 1);
        // log Z = logit − log-probability, at any column.
        var logZ = _engine.TensorSubtract(_engine.TensorNarrow(logits, 1, 0, 1), _engine.TensorNarrow(logSoftmax, 1, 0, 1));    // [count, 1]
        var logProbabilities = _engine.Reshape(logSoftmax, new[] { count * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { count }, new Vector<int>(Enumerable.Range(0, count).Select(i => i * vocabulary + sequence[first + i + 1]).ToArray()));
        var nll = _engine.TensorNegate(_engine.ReduceSum(_engine.TensorIndexSelect(logProbabilities, picks, 0), null, false));
        var z = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(logZ, logZ), null, false), Ops.FromDouble(_o.ZLossWeight));
        return _engine.TensorMultiplyScalar(_engine.TensorAdd(nll, z), Ops.FromDouble(1.0 / count));
    }

    // ---------------------------------------------------------------- weights

    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors = new(StringComparer.Ordinal);

        public Checkpoint(IEnumerable<string> paths)
        {
            foreach (var path in paths)
            {
                if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
                foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path)) _tensors[pair.Key] = pair.Value;
            }
            Keys = new HashSet<string>(_tensors.Keys, StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }
        public HashSet<string> Used { get; } = new(StringComparer.Ordinal);

        public int[] Shape(string name) => _tensors.TryGetValue(name, out var t) ? t.Shape.ToArray() : throw new InvalidDataException($"No tensor '{name}'.");

        public Func<string, int[], double[]> Reader(string what) => (name, _) =>
        {
            if (!_tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The {what} checkpoint has no tensor '{name}'.");
            _tensors.Remove(name);
            Used.Add(name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        };

        public void RequireAllUsed(string what)
        {
            var left = Keys.Where(k => !Used.Contains(k)).ToList();
            if (left.Count > 0) throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
        }
    }

    /// <summary>Loads a Hugging Face Gemma checkpoint directory (one <c>model.safetensors</c> or its index's shards); its
    /// text rows, the audio rows zeroed (§3.2); its <c>tokenizer.model</c> unless a tokenizer is set.</summary>
    public void LoadGemma(string directory)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (_o.Decoder != PalmDecoderKind.Gemma) throw new InvalidOperationException("Gemma weights need AudioPaLMOptions.Gemma2B() (Decoder = Gemma).");
        string index = Path.Combine(directory, "model.safetensors.index.json");
        var files = File.Exists(index)
            ? Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(index))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList()
            : new List<string> { Path.Combine(directory, "model.safetensors") };
        var checkpoint = new Checkpoint(files);
        int rows = checkpoint.Shape("model.embed_tokens.weight")[0];
        if (rows != _o.VocabSize) throw new InvalidDataException($"The checkpoint has {rows} text tokens; set VocabSize to it.");
        Decoder.LoadGemma(checkpoint.Reader("Gemma"), rows);
        checkpoint.RequireAllUsed("Gemma");
        string tokenizer = Path.Combine(directory, "tokenizer.model");
        if (_o.SentencePieceModelPath is null && File.Exists(tokenizer))
        {
            _o.SentencePieceModelPath = tokenizer;
            _tokenizer = null;
        }
    }

    /// <summary>Loads a lucidrains/PaLM-pytorch state dict into a PaLM decoder (its text rows; the audio rows zeroed).</summary>
    public void LoadPalmPytorch(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        var checkpoint = new Checkpoint(new[] { path });
        int rows = checkpoint.Shape("0.weight")[0];
        if (rows != _o.VocabSize) throw new InvalidDataException($"The checkpoint has {rows} text tokens; set VocabSize to it.");
        Decoder.LoadPalmPytorch(checkpoint.Reader("PaLM-pytorch"), rows);
        // PaLM-pytorch's shared output projection is the embedding under a second name.
        checkpoint.Used.Add($"{_o.Layers + 2}.weight");
        checkpoint.RequireAllUsed("PaLM-pytorch");
    }

    /// <summary>Loads w2v-BERT 2.0 (<c>facebook/w2v-bert-2.0</c>'s <c>model.safetensors</c>); its layers above the one
    /// clustered are not read.</summary>
    public void LoadW2vBert2(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        var network = W2vBert2 ?? throw new InvalidOperationException("w2v-BERT 2.0 weights need SpeechTokenizer = W2vBert2.");
        network.LoadTorchWeights(new Checkpoint(new[] { path }).Reader("w2v-BERT 2.0"));
    }
}
