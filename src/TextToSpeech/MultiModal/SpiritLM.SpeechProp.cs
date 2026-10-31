using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpiritLM<T>
{
    private EnCodec<T>? _speechPropCodec;
    private SonarExpressive<T>? _sonar;
    private DenseLayer<T>? _styleHead;
    private SonarTextEncoder<T>? _sonarText;
    private SonarTextTokenizer? _sonarTokenizer;
    private double[,]? _styleCentroids;
    private readonly List<LayerBase<T>> _speechPropLayers = new();
    private readonly List<LayerBase<T>> _sonarDecoderLayers = new();
    private readonly List<LayerBase<T>> _sonarTextLayers = new();

    private bool UsesSpeechProp => IsExpressive && _options.StyleTokenizer == SpiritLmStyleTokenizer.SpeechProp;

    /// <summary>SONAR Expressive's 25 Hz EnCodec (with <see cref="SpiritLmStyleTokenizer.SpeechProp"/>; otherwise null): load or
    /// train it as a codec model of its own.</summary>
    public EnCodec<T>? SpeechPropCodec => _speechPropCodec;

    private EnCodec<T> RequireCodec() => _speechPropCodec ?? throw new InvalidOperationException("The SpeechProp codec exists with StyleTokenizer = SpeechProp.");

    internal SonarExpressive<T> Sonar => _sonar ?? throw new InvalidOperationException("SONAR Expressive exists with StyleTokenizer = SpeechProp.");

    private void BuildSpeechProp(Random random)
    {
        var o = _options;
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.SpeechPropCodec);
        _speechPropCodec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
        {
            RandomSeed = Architecture.RandomSeed,
        }, codecOptions);
        int codebooks = _speechPropCodec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps);
        _sonar = new SonarExpressive<T>(Engine, _speechPropLayers, _sonarDecoderLayers, new SonarExpressiveConfiguration
        {
            Codebooks = codebooks, CodebookSize = codecOptions.CodebookSize, Width = o.SpeechPropWidth, Heads = o.SpeechPropHeads,
            FeedForward = o.SpeechPropFeedForward, EncoderLayers = o.SpeechPropEncoderLayers, DecoderLayers = o.SpeechPropDecoderLayers,
            SemanticDim = o.SonarWidth, SpeechPropDim = o.SpeechPropDim, Dropout = o.DropoutRate,
        }, random);
        _styleHead = new DenseLayer<T>(o.ExpressoStyles, (IActivationFunction<T>)new IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(o.SpeechPropDim);
        TorchParameters.Linear(Engine, _styleHead, o.SpeechPropDim, o.ExpressoStyles,
            Enumerable.Range(0, o.SpeechPropDim * o.ExpressoStyles).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray(),
            Enumerable.Range(0, o.ExpressoStyles).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray());
        _speechPropLayers.Add(_styleHead);
        if (o.IncludeSonarTextEncoder)
            _sonarText = new SonarTextEncoder<T>(Engine, _sonarTextLayers, new SonarTextEncoderConfiguration
            {
                VocabSize = o.SonarVocabSize, Width = o.SonarWidth, Layers = o.SonarLayers, Heads = o.SonarHeads, FeedForward = o.SonarFeedForward,
            });
        _styleCentroids = new double[o.StyleTokens, o.SpeechPropDim];
    }

    // ---------------------------------------------------------------- style features and tokens

    private int[,] CodecCodes(IReadOnlyList<double> audio)
    {
        var tensor = new Tensor<T>(new[] { audio.Count });
        for (int i = 0; i < audio.Count; i++) tensor[i] = Ops.FromDouble(audio[i]);
        return RequireCodec().Encode(tensor);
    }

    /// <summary>
    /// The SpeechProp style features <c>[seconds, speechprop]</c> of 16 kHz audio (§3.2): SpeechProp's encoder states over
    /// the 25 Hz EnCodec units, averaged over each second (25 frames, the last window shorter) and projected.
    /// </summary>
    internal Tensor<T> StyleFeatures(int[,] codes)
    {
        var states = Sonar.SpeechPropStates(codes);
        int frames = states.Shape[0], width = states.Shape[1], windows = (frames + 24) / 25;
        var pooled = new List<Tensor<T>>();
        for (int w = 0; w < windows; w++)
        {
            int start = w * 25, count = Math.Min(25, frames - start);
            pooled.Add(Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorSlice(states, new[] { start, 0 }, new[] { count, width }), new[] { 0 }, keepDims: true),
                Ops.FromDouble(1.0 / count)));
        }
        return Sonar.SpeechPropProjection.Forward(pooled.Count == 1 ? pooled[0] : Engine.TensorConcatenate(pooled.ToArray(), 0));
    }

    /// <summary>The SpeechProp style features of 16 kHz audio, without gradients.</summary>
    public Tensor<T> SpeechPropStyleFeatures(Tensor<T> recording)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        return StyleFeatures(CodecCodes(Samples(recording)));
    }

    // The style token of each second: the nearest of the k-means centroids.
    private int[] SpeechPropStyleTokens(IReadOnlyList<double> audio)
    {
        Tensor<T> features;
        using (new NoGradScope<T>()) features = StyleFeatures(CodecCodes(audio));
        var centroids = _styleCentroids!;
        int windows = features.Shape[0], width = features.Shape[1], k = centroids.GetLength(0);
        var tokens = new int[windows];
        for (int w = 0; w < windows; w++)
        {
            double best = double.PositiveInfinity;
            for (int c = 0; c < k; c++)
            {
                double distance = 0;
                for (int i = 0; i < width; i++)
                {
                    double d = Ops.ToDouble(features[w, i]) - centroids[c, i];
                    distance += d * d;
                }
                if (distance < best) { best = distance; tokens[w] = c; }
            }
        }
        return tokens;
    }

    /// <summary>The style token of each second of audio, by the configured tokenizer.</summary>
    internal int[] StyleTokensOf(IReadOnlyList<double> audio) => UsesSpeechProp ? SpeechPropStyleTokens(audio) : StyleEncoder.Tokens(audio);

    /// <summary>
    /// Fits the style k-means (§3.2: 100 clusters of the normalized features) on the SpeechProp style features of
    /// <paramref name="recordings"/>: k-means++ seeding, then Lloyd's iterations until no assignment changes or
    /// <paramref name="iterations"/> have run.
    /// </summary>
    public void FitStyleCentroids(IEnumerable<Tensor<T>> recordings, int iterations = 300)
    {
        ThrowIfDisposed();
        var rows = new List<double[]>();
        foreach (var recording in recordings)
        {
            var features = SpeechPropStyleFeatures(recording);
            for (int w = 0; w < features.Shape[0]; w++)
                rows.Add(Enumerable.Range(0, features.Shape[1]).Select(i => Ops.ToDouble(features[w, i])).ToArray());
        }
        int k = _options.StyleTokens, width = _options.SpeechPropDim;
        if (rows.Count < k) throw new ArgumentException($"{rows.Count} seconds of speech cannot fit {k} centroids.", nameof(recordings));
        var random = TrainingRandom;
        double Distance(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; } return s; }
        var centers = new List<double[]> { (double[])rows[random.Next(rows.Count)].Clone() };
        var nearest = rows.Select(r => Distance(r, centers[0])).ToArray();
        while (centers.Count < k)
        {
            double draw = random.NextDouble() * nearest.Sum();
            int pick = rows.Count - 1;
            for (int i = 0; i < rows.Count; i++) { draw -= nearest[i]; if (draw <= 0) { pick = i; break; } }
            centers.Add((double[])rows[pick].Clone());
            for (int i = 0; i < rows.Count; i++) nearest[i] = Math.Min(nearest[i], Distance(rows[i], centers[^1]));
        }
        var assignment = Enumerable.Repeat(-1, rows.Count).ToArray();
        for (int round = 0; round < iterations; round++)
        {
            bool changed = false;
            for (int i = 0; i < rows.Count; i++)
            {
                int best = 0;
                double bestDistance = double.PositiveInfinity;
                for (int c = 0; c < k; c++) { double d = Distance(rows[i], centers[c]); if (d < bestDistance) { bestDistance = d; best = c; } }
                if (assignment[i] != best) { assignment[i] = best; changed = true; }
            }
            if (!changed) break;
            for (int c = 0; c < k; c++)
            {
                var members = Enumerable.Range(0, rows.Count).Where(i => assignment[i] == c).ToList();
                if (members.Count == 0) continue;
                var mean = new double[width];
                foreach (int i in members) for (int j = 0; j < width; j++) mean[j] += rows[i][j];
                for (int j = 0; j < width; j++) mean[j] /= members.Count;
                centers[c] = mean;
            }
        }
        var table = new double[k, width];
        for (int c = 0; c < k; c++) for (int j = 0; j < width; j++) table[c, j] = centers[c][j];
        _styleCentroids = table;
    }

    // ---------------------------------------------------------------- training

    /// <summary>
    /// One step of the style fine-tuning (§3.2): the SpeechProp features of each second of a recording classified into
    /// Expresso's styles by a linear head, cross-entropy against <paramref name="style"/>; SpeechProp's encoder trains with
    /// the head.
    /// </summary>
    public T TrainSpeechPropStyle(Tensor<T> recording, int style)
    {
        ThrowIfDisposed();
        if (style < 0 || style >= _options.ExpressoStyles) throw new ArgumentOutOfRangeException(nameof(style), $"Styles lie in [0, {_options.ExpressoStyles}).");
        var codes = CodecCodes(Samples(recording));
        _trainedGroup = "speechprop";
        int styles = _options.ExpressoStyles;
        return TrainWithCustomObjective(recording, recording, (_, _) =>
        {
            var logits = _styleHead!.Forward(StyleFeatures(codes));
            int windows = logits.Shape[0];
            var log = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { windows * styles, 1 });
            var picks = new Tensor<int>(new[] { windows }, new Vector<int>(Enumerable.Range(0, windows).Select(w => w * styles + style).ToArray()));
            return Engine.TensorMultiplyScalar(Engine.TensorNegate(Engine.ReduceSum(Engine.TensorIndexSelect(log, picks, 0), null, false)), Ops.FromDouble(1.0 / windows));
        }, SpeechPropOptimizer("speechprop"));
    }

    /// <summary>
    /// One step of SONAR Expressive (§4.1) on a recording and the SONAR embedding <paramref name="semantic"/>
    /// (<c>[1024]</c>, of its transcript or translation): the decoders' cross-entropy on its EnCodec units. Pre-training
    /// (<paramref name="withSpeechProp"/> false) trains the decoders on the semantic embedding alone; fine-tuning trains
    /// SpeechProp's encoder on random crops of the units together with only the decoders' cross-attention.
    /// </summary>
    public T TrainSonarExpressive(Tensor<T> recording, Tensor<T> semantic, bool withSpeechProp)
    {
        ThrowIfDisposed();
        if (semantic is null) throw new ArgumentNullException(nameof(semantic));
        if (semantic.Length != _options.SonarWidth) throw new ArgumentException($"A SONAR embedding has {_options.SonarWidth} values.", nameof(semantic));
        var codes = CodecCodes(Samples(recording));
        if (codes.GetLength(1) == 0) throw new ArgumentException("The recording gives no units.", nameof(recording));
        _trainedGroup = withSpeechProp ? "sonar finetune" : "sonar pretrain";
        var random = TrainingRandom;
        return TrainWithCustomObjective(recording, recording, (_, _) => Sonar.Loss(codes, semantic, withSpeechProp, random),
            SpeechPropOptimizer(_trainedGroup));
    }

    /// <summary>The step of <see cref="TrainSonarExpressive(Tensor{T}, Tensor{T}, bool)"/> with the semantic embedding of
    /// <paramref name="transcript"/> in <paramref name="language"/> (e.g. <c>eng_Latn</c>) by SONAR's text encoder (needs
    /// <see cref="SpiritLMOptions.IncludeSonarTextEncoder"/> and <see cref="LoadSonarTextEncoder"/>).</summary>
    public T TrainSonarExpressive(Tensor<T> recording, string transcript, string language, bool withSpeechProp) =>
        TrainSonarExpressive(recording, SonarEmbedding(transcript, language), withSpeechProp);

    /// <summary>SONAR's sentence embedding <c>[1024]</c> of text.</summary>
    public Tensor<T> SonarEmbedding(string text, string language)
    {
        ThrowIfDisposed();
        var encoder = _sonarText ?? throw new InvalidOperationException("SONAR's text encoder is built with IncludeSonarTextEncoder.");
        if (_sonarTokenizer is null)
        {
            string directory = _options.SonarTokenizerDirectory ?? throw new InvalidOperationException("Set SpiritLMOptions.SonarTokenizerDirectory.");
            _sonarTokenizer = SonarTextTokenizer.FromDirectory(directory);
        }
        return encoder.EmbedText(_sonarTokenizer, text, language);
    }

    /// <summary>Speech of a SONAR embedding and a SpeechProp embedding (from <paramref name="styleReference"/>, or none):
    /// SONAR Expressive's units (top-k 10, guidance 3), decoded by its EnCodec.</summary>
    public Tensor<T> SonarExpressiveSpeech(Tensor<T> semantic, Tensor<T>? styleReference = null, int maxFrames = 500)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        Tensor<T>? prop = styleReference is null ? null : Sonar.SpeechProp(CodecCodes(Samples(styleReference)));
        var codes = Sonar.Generate(semantic, prop, maxFrames, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue));
        return codes.GetLength(1) == 0 ? new Tensor<T>(new[] { 0 }) : RequireCodec().Decode(codes);
    }

    /// <summary>Loads SONAR's text encoder (transformers' <c>M2M100Encoder</c>, e.g. cointegrated/SONAR_200_text_encoder)
    /// and, unless set, its tokenizer from the same directory.</summary>
    public void LoadSonarTextEncoder(string directory)
    {
        ThrowIfDisposed();
        var encoder = _sonarText ?? throw new InvalidOperationException("SONAR's text encoder is built with IncludeSonarTextEncoder.");
        string path = File.Exists(Path.Combine(directory, "model.safetensors")) ? Path.Combine(directory, "model.safetensors") : Path.Combine(directory, "pytorch_model.bin");
        var tensors = ReadAll(new[] { path });
        encoder.LoadTorchWeights(name => tensors.TryGetValue(name, out var t) ? Values(t) : null);
        if (_options.SonarTokenizerDirectory is null && File.Exists(Path.Combine(directory, "sentencepiece.bpe.model")))
        {
            _options.SonarTokenizerDirectory = directory;
            _sonarTokenizer = null;
        }
    }

    private AiDotNet.Interfaces.IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> SpeechPropOptimizer(string group)
    {
        if (_optimizers.TryGetValue(group, out var existing)) return existing;
        var optimizer = new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.SpeechPropLearningRate, Beta1 = 0.9, Beta2 = 0.98, WeightDecay = 0.01,
        });
        return _optimizers[group] = optimizer;
    }

    /// <summary>The layers a SpeechProp-side step trains.</summary>
    private IEnumerable<LayerBase<T>>? SpeechPropTrainedLayers(string group) => group switch
    {
        "speechprop" => _speechPropLayers,
        "sonar pretrain" => _sonarDecoderLayers,
        "sonar finetune" => _speechPropLayers.Where(l => !ReferenceEquals(l, _styleHead)).Concat(Sonar.CrossAttentionLayers),
        _ => null,
    };

    /// <summary>Declares the style centroids as model state.</summary>
    private void DeclareSpeechPropState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        if (!UsesSpeechProp) return;
        int k = _options.StyleTokens, width = _options.SpeechPropDim;
        state.DeclareDoubleVector("StyleCentroids", () =>
        {
            var flat = new Vector<double>(k * width);
            for (int c = 0; c < k; c++) for (int i = 0; i < width; i++) flat[c * width + i] = _styleCentroids![c, i];
            return flat;
        }, flat =>
        {
            if (flat is null) return;
            if (flat.Length != k * width) throw new InvalidDataException($"State 'StyleCentroids' holds {flat.Length} values.");
            var table = new double[k, width];
            for (int c = 0; c < k; c++) for (int i = 0; i < width; i++) table[c, i] = flat[c * width + i];
            _styleCentroids = table;
        });
    }
}
