using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The paper's English-only, uncased BERT-base text feature extractor.</summary>
internal sealed class E3BertTextEncoder<T>
{
    public E3BertTextEncoder(IEngine engine, List<LayerBase<T>> layers, E3TTSOptions options, Random random)
    {
        Network = new BertNetwork<T>(engine, layers, new BertConfiguration
        {
            VocabSize = options.BertVocabularySize,
            Hidden = options.BertHiddenDim,
            Layers = options.BertLayers,
            Heads = options.BertHeads,
            Intermediate = options.BertIntermediateDim,
            MaxPositions = options.BertMaxPositions,
            Dropout = options.BertDropout,
        });
        Network.InitializeLikeReference(random);
    }

    public BertNetwork<T> Network { get; }

    public Tensor<T> Forward(IReadOnlyList<int> tokenIds)
    {
        var ids = new int[1, tokenIds.Count];
        for (int i = 0; i < tokenIds.Count; i++) ids[0, i] = tokenIds[i];
        return Network.Encode(ids);
    }

    public void LoadHuggingFace(Func<string, int[], double[]> read, string prefix = "bert.") =>
        Network.LoadTorchWeights(read, prefix);
}

/// <summary>
/// One E3 DBlock or mirrored UBlock. Both convolutions use softmax kernel mixtures when Table 1 supplies an
/// adaptive-kernel count; other blocks receive the speaker/timestep condition through FiLM. The lowest five pairs add
/// self-attention and BERT cross-attention.
/// </summary>
internal sealed class E3WaveBlock<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _inputChannels;
    private readonly int _outputChannels;

    public E3WaveBlock(IEngine engine, List<LayerBase<T>> layers, Random random, int inputChannels, int outputChannels,
        int kernel, int stride, int adaptiveKernels, int attentionHeads, int textDim, int conditionDim,
        int requestedGroups, double epsilon, bool transposed)
    {
        _engine = engine;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        KernelSize = kernel;
        Stride = stride;
        AdaptiveKernelCount = adaptiveKernels;
        AttentionHeads = attentionHeads;
        IsTransposed = transposed;

        NormedConv1DLayer<T> Conv(int input, int output, int k, int s, bool transpose)
        {
            var conv = new NormedConv1DLayer<T>(input, output, k, s, 1, 1, (k - 1) / 2, transpose,
                ConvolutionNormalization.None, outputPadding: transpose ? s - 1 : 0);
            layers.Add(conv);
            double std = Math.Sqrt(2.0 / Math.Max(1, input * k));
            conv.Reinitialize(() => std * IndexTtsOps.Gaussian(random), bias: 0.0);
            return conv;
        }

        int groups = ChooseGroups(outputChannels, requestedGroups);
        int experts = Math.Max(1, adaptiveKernels);
        // Imagen-style UBlocks resize first and then convolve. Besides matching that design, this avoids requiring a
        // ConvTranspose output-padding inverse for the paper's odd kernels and even strides.
        for (int i = 0; i < experts; i++) FirstConvolutions.Add(Conv(inputChannels, outputChannels, kernel, transposed ? 1 : stride, false));
        FirstNorm = new GroupNormalizationLayer<T>(groups, outputChannels, epsilon); layers.Add(FirstNorm);
        for (int i = 0; i < experts; i++) SecondConvolutions.Add(Conv(outputChannels, outputChannels, kernel, 1, false));
        SecondNorm = new GroupNormalizationLayer<T>(groups, outputChannels, epsilon); layers.Add(SecondNorm);
        if (inputChannels != outputChannels || stride != 1)
            Shortcut = Conv(inputChannels, outputChannels, 1, transposed ? 1 : stride, false);

        if (adaptiveKernels > 0)
            AdaptiveLogits = IndexTtsOps.Linear(engine, layers, random, conditionDim, adaptiveKernels);
        else
            Film = IndexTtsOps.Linear(engine, layers, random, conditionDim, 4 * outputChannels);

        if (attentionHeads > 0)
        {
            SelfNorm = IndexTtsOps.LayerNorm(layers, outputChannels, epsilon);
            SelfQuery = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
            SelfKey = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
            SelfValue = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
            SelfOutput = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
            CrossNorm = IndexTtsOps.LayerNorm(layers, outputChannels, epsilon);
            CrossQuery = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
            CrossKey = IndexTtsOps.Linear(engine, layers, random, textDim, outputChannels);
            CrossValue = IndexTtsOps.Linear(engine, layers, random, textDim, outputChannels);
            CrossOutput = IndexTtsOps.Linear(engine, layers, random, outputChannels, outputChannels);
        }
    }

    public int KernelSize { get; }
    public int Stride { get; }
    public int AdaptiveKernelCount { get; }
    public int AttentionHeads { get; }
    public bool IsTransposed { get; }
    public List<NormedConv1DLayer<T>> FirstConvolutions { get; } = new();
    public GroupNormalizationLayer<T> FirstNorm { get; }
    public List<NormedConv1DLayer<T>> SecondConvolutions { get; } = new();
    public NormedConv1DLayer<T> SecondConvolution => SecondConvolutions[0];
    public GroupNormalizationLayer<T> SecondNorm { get; }
    public NormedConv1DLayer<T>? Shortcut { get; }
    public DenseLayer<T>? AdaptiveLogits { get; }
    public DenseLayer<T>? Film { get; }
    public LayerNormalizationLayer<T>? SelfNorm { get; }
    public DenseLayer<T>? SelfQuery { get; }
    public DenseLayer<T>? SelfKey { get; }
    public DenseLayer<T>? SelfValue { get; }
    public DenseLayer<T>? SelfOutput { get; }
    public LayerNormalizationLayer<T>? CrossNorm { get; }
    public DenseLayer<T>? CrossQuery { get; }
    public DenseLayer<T>? CrossKey { get; }
    public DenseLayer<T>? CrossValue { get; }
    public DenseLayer<T>? CrossOutput { get; }

    private static int ChooseGroups(int channels, int requested)
    {
        for (int groups = Math.Min(channels, requested); groups > 1; groups--)
            if (channels % groups == 0) return groups;
        return 1;
    }

    private Tensor<T> Mixture(IReadOnlyList<NormedConv1DLayer<T>> convolutions, Tensor<T> x, Tensor<T> condition)
    {
        if (AdaptiveLogits is null) return convolutions[0].Forward(x);
        var weights = _engine.Softmax(AdaptiveLogits.Forward(condition), axis: 1);
        Tensor<T>? mixed = null;
        for (int i = 0; i < convolutions.Count; i++)
        {
            var branch = convolutions[i].Forward(x);
            var weight = _engine.TensorTile(_engine.Reshape(_engine.TensorNarrow(weights, 1, i, 1), new[] { 1, 1, 1 }),
                new[] { 1, branch.Shape[1], branch.Shape[2] });
            var weighted = _engine.TensorMultiply(branch, weight);
            mixed = mixed is null ? weighted : _engine.TensorAdd(mixed, weighted);
        }
        return mixed!;
    }

    private Tensor<T> Upsample(Tensor<T> x)
    {
        if (!IsTransposed || Stride == 1) return x;
        int channels = x.Shape[1], time = x.Shape[2];
        return _engine.Reshape(_engine.TensorTile(_engine.Reshape(x, new[] { 1, channels, time, 1 }),
            new[] { 1, 1, 1, Stride }), new[] { 1, channels, time * Stride });
    }

    private Tensor<T> FilmNorm(Tensor<T> normalized, Tensor<T>? film, int pair)
    {
        if (film is null) return normalized;
        int c = _outputChannels;
        var shift = _engine.Reshape(_engine.TensorNarrow(film, 1, pair * 2 * c, c), new[] { 1, c, 1 });
        var scale = _engine.Reshape(_engine.TensorNarrow(film, 1, (pair * 2 + 1) * c, c), new[] { 1, c, 1 });
        shift = _engine.TensorTile(shift, new[] { 1, 1, normalized.Shape[2] });
        scale = _engine.TensorTile(scale, new[] { 1, 1, normalized.Shape[2] });
        return _engine.TensorAdd(_engine.TensorMultiply(normalized, _engine.TensorAddScalar(scale, Ops.One)), shift);
    }

    private Tensor<T> Rows(Tensor<T> channels) =>
        _engine.Reshape(_engine.TensorPermute(channels, new[] { 0, 2, 1 }), new[] { channels.Shape[2], channels.Shape[1] });

    // GroupNormalizationLayer's rank-3 contract is [C, H, W] (no batch dimension), while Conv1D uses [B, C, T].
    // E3 synthesizes one waveform at a time, so expose its singleton batch as the height axis and restore it afterwards.
    private Tensor<T> Normalize(GroupNormalizationLayer<T> norm, Tensor<T> channels) =>
        _engine.Reshape(norm.Forward(_engine.Reshape(channels, new[] { channels.Shape[1], 1, channels.Shape[2] })),
            channels.Shape.ToArray());

    private Tensor<T> Channels(Tensor<T> rows) =>
        _engine.TensorPermute(_engine.Reshape(rows, new[] { 1, rows.Shape[0], rows.Shape[1] }), new[] { 0, 2, 1 });

    private Tensor<T> Attention(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        int dk = _outputChannels / AttentionHeads;
        return IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine,
            IndexTtsOps.SplitHeads(_engine, query, AttentionHeads),
            IndexTtsOps.SplitHeads(_engine, key, AttentionHeads),
            IndexTtsOps.SplitHeads(_engine, value, AttentionHeads), 1.0 / Math.Sqrt(dk), null));
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> condition, Tensor<T> text)
    {
        x = Upsample(x);
        var film = Film?.Forward(condition);
        var h = _engine.Swish(FilmNorm(Normalize(FirstNorm, Mixture(FirstConvolutions, x, condition)), film, 0));
        h = _engine.Swish(FilmNorm(Normalize(SecondNorm, Mixture(SecondConvolutions, h, condition)), film, 1));
        h = _engine.TensorAdd(h, Shortcut is null ? x : Shortcut.Forward(x));
        if (AttentionHeads == 0) return h;

        var rows = Rows(h);
        var selfInput = SelfNorm!.Forward(rows);
        rows = _engine.TensorAdd(rows, SelfOutput!.Forward(Attention(SelfQuery!.Forward(selfInput),
            SelfKey!.Forward(selfInput), SelfValue!.Forward(selfInput))));
        var crossInput = CrossNorm!.Forward(rows);
        var memory = _engine.Reshape(text, new[] { text.Shape[1], text.Shape[2] });
        rows = _engine.TensorAdd(rows, CrossOutput!.Forward(Attention(CrossQuery!.Forward(crossInput),
            CrossKey!.Forward(memory), CrossValue!.Forward(memory))));
        return Channels(rows);
    }

    public void Load(Func<string, int[], double[]> read, string prefix, int textDim, int conditionDim)
    {
        for (int i = 0; i < FirstConvolutions.Count; i++)
            IndexTtsOps.LoadConv(read, FirstConvolutions[i], $"{prefix}first.{i}", _outputChannels);
        for (int i = 0; i < SecondConvolutions.Count; i++)
            IndexTtsOps.LoadConv(read, SecondConvolutions[i], AdaptiveKernelCount > 0 ? $"{prefix}second.{i}" : prefix + "second", _outputChannels);
        if (Shortcut is not null) IndexTtsOps.LoadConv(read, Shortcut, prefix + "shortcut", _outputChannels);
        StyleTtsOps.LoadAffine(_engine, FirstNorm.GetGammaTensor(), FirstNorm.GetBetaTensor(),
            read(prefix + "norm1.weight", new[] { _outputChannels }), read(prefix + "norm1.bias", new[] { _outputChannels }));
        StyleTtsOps.LoadAffine(_engine, SecondNorm.GetGammaTensor(), SecondNorm.GetBetaTensor(),
            read(prefix + "norm2.weight", new[] { _outputChannels }), read(prefix + "norm2.bias", new[] { _outputChannels }));
        if (AdaptiveLogits is not null) IndexTtsOps.LoadLinear(_engine, read, AdaptiveLogits, prefix + "adaptive_logits", conditionDim, AdaptiveKernelCount);
        if (Film is not null) IndexTtsOps.LoadLinear(_engine, read, Film, prefix + "film", conditionDim, 4 * _outputChannels);
        if (AttentionHeads == 0) return;
        IndexTtsOps.LoadNorm(_engine, read, SelfNorm!, prefix + "self.norm", _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, SelfQuery!, prefix + "self.q", _outputChannels, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, SelfKey!, prefix + "self.k", _outputChannels, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, SelfValue!, prefix + "self.v", _outputChannels, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, SelfOutput!, prefix + "self.out", _outputChannels, _outputChannels);
        IndexTtsOps.LoadNorm(_engine, read, CrossNorm!, prefix + "cross.norm", _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, CrossQuery!, prefix + "cross.q", _outputChannels, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, CrossKey!, prefix + "cross.k", textDim, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, CrossValue!, prefix + "cross.v", textDim, _outputChannels);
        IndexTtsOps.LoadLinear(_engine, read, CrossOutput!, prefix + "cross.out", _outputChannels, _outputChannels);
    }
}

/// <summary>The raw-waveform 1D U-Net from E3 Table 1, including the learned timestep loss variance.</summary>
internal sealed class E3WaveformUnet<T>
{
    private readonly IEngine _engine;

    public E3WaveformUnet(IEngine engine, List<LayerBase<T>> layers, E3TTSOptions options, Random random)
    {
        _engine = engine;
        Options = options;
        TimeInput = IndexTtsOps.Linear(engine, layers, random, options.TimeEmbeddingDim, options.ConditionDim);
        TimeOutput = IndexTtsOps.Linear(engine, layers, random, options.ConditionDim, options.ConditionDim);
        SpeakerEmbedding = IndexTtsOps.Embedding(layers, random, options.SpeakerCount, options.ConditionDim, 0.02);
        VarianceHead = IndexTtsOps.Linear(engine, layers, random, options.ConditionDim, 1);

        int channels = 1;
        for (int i = 0; i < options.BlockChannels.Length; i++)
        {
            var block = new E3WaveBlock<T>(engine, layers, random, channels, options.BlockChannels[i],
                options.BlockKernelSizes[i], options.BlockStrides[i], options.AdaptiveKernelCounts[i], options.AttentionHeads[i],
                options.BertHiddenDim, options.ConditionDim, options.GroupNormGroups, options.GroupNormEpsilon, false);
            DownBlocks.Add(block);
            channels = options.BlockChannels[i];
            var residuals = new List<E3WaveBlock<T>>();
            for (int j = 1; j < options.BlockRepetitions[i]; j++)
                residuals.Add(new E3WaveBlock<T>(engine, layers, random, channels, channels,
                    options.BlockKernelSizes[i], 1, options.AdaptiveKernelCounts[i], options.AttentionHeads[i],
                    options.BertHiddenDim, options.ConditionDim, options.GroupNormGroups, options.GroupNormEpsilon, false));
            DownResiduals.Add(residuals);
        }
        for (int i = options.BlockChannels.Length - 1; i >= 0; i--)
        {
            var residuals = new List<E3WaveBlock<T>>();
            for (int j = 1; j < options.BlockRepetitions[i]; j++)
                residuals.Add(new E3WaveBlock<T>(engine, layers, random, channels, channels,
                    options.BlockKernelSizes[i], 1, options.AdaptiveKernelCounts[i], options.AttentionHeads[i],
                    options.BertHiddenDim, options.ConditionDim, options.GroupNormGroups, options.GroupNormEpsilon, false));
            UpResiduals.Add(residuals);
            int output = i == 0 ? 1 : options.BlockChannels[i - 1];
            var block = new E3WaveBlock<T>(engine, layers, random, channels, output,
                options.BlockKernelSizes[i], options.BlockStrides[i], options.AdaptiveKernelCounts[i], options.AttentionHeads[i],
                options.BertHiddenDim, options.ConditionDim, options.GroupNormGroups, options.GroupNormEpsilon, true);
            UpBlocks.Add(block);
            channels = output;
        }
        Output = new NormedConv1DLayer<T>(1, 1, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None);
        layers.Add(Output);
        Output.Reinitialize(() => Math.Sqrt(2.0 / 3.0) * IndexTtsOps.Gaussian(random), bias: 0.0);
    }

    public E3TTSOptions Options { get; }
    public DenseLayer<T> TimeInput { get; }
    public DenseLayer<T> TimeOutput { get; }
    public TiedEmbeddingLayer<T> SpeakerEmbedding { get; }
    public DenseLayer<T> VarianceHead { get; }
    public List<E3WaveBlock<T>> DownBlocks { get; } = new();
    public List<E3WaveBlock<T>> UpBlocks { get; } = new();
    internal List<List<E3WaveBlock<T>>> DownResiduals { get; } = new();
    internal List<List<E3WaveBlock<T>>> UpResiduals { get; } = new();
    public NormedConv1DLayer<T> Output { get; }

    internal static Tensor<TValue> TimestepEmbedding<TValue>(double noiseLevel, int dimension)
    {
        var ops = MathHelper.GetNumericOperations<TValue>();
        var result = new Tensor<TValue>(new[] { 1, dimension });
        int half = dimension / 2;
        for (int i = 0; i < half; i++)
        {
            double frequency = Math.Exp(-Math.Log(10_000.0) * i / Math.Max(half - 1, 1));
            result[0, i] = ops.FromDouble(Math.Sin(noiseLevel * frequency));
            if (i + half < dimension) result[0, i + half] = ops.FromDouble(Math.Cos(noiseLevel * frequency));
        }
        return result;
    }

    private (Tensor<T> Time, Tensor<T> Joint) Condition(double noiseLevel, int speakerId)
    {
        if (speakerId < 0 || speakerId >= Options.SpeakerCount) throw new ArgumentOutOfRangeException(nameof(speakerId));
        var time = TimeOutput.Forward(_engine.Swish(TimeInput.Forward(TimestepEmbedding<T>(noiseLevel, Options.TimeEmbeddingDim))));
        var speaker = SpeakerEmbedding.Forward(IndexTtsOps.Ids<T>(new[] { speakerId }));
        return (time, _engine.TensorAdd(time, speaker));
    }

    public (Tensor<T> Epsilon, Tensor<T> LogVariance) Forward(Tensor<T> waveform, Tensor<T> text, double noiseLevel, int speakerId)
    {
        var (time, condition) = Condition(noiseLevel, speakerId);
        int factor = Options.BlockStrides.Aggregate(1, (a, b) => checked(a * b));
        int paddedLength = checked((waveform.Length + factor - 1) / factor * factor);
        var padded = _engine.Reshape(waveform, new[] { waveform.Length });
        if (paddedLength != waveform.Length)
            padded = _engine.TensorConcatenate(new[] { padded, new Tensor<T>(new[] { paddedLength - waveform.Length }) }, 0);
        var x = _engine.Reshape(padded, new[] { 1, 1, paddedLength });
        var skips = new List<Tensor<T>>(DownBlocks.Count);
        for (int i = 0; i < DownBlocks.Count; i++)
        {
            skips.Add(x);
            x = DownBlocks[i].Forward(x, condition, text);
            foreach (var residual in DownResiduals[i]) x = residual.Forward(x, condition, text);
        }
        for (int i = 0; i < UpBlocks.Count; i++)
        {
            foreach (var residual in UpResiduals[i]) x = residual.Forward(x, condition, text);
            x = UpBlocks[i].Forward(x, condition, text);
            var skip = skips[skips.Count - 1 - i];
            if (!x.Shape.ToArray().SequenceEqual(skip.Shape.ToArray()))
                throw new InvalidOperationException($"E3 U-Net skip shape mismatch: [{string.Join(",", x.Shape)}] and [{string.Join(",", skip.Shape)}].");
            x = _engine.TensorAdd(x, skip);
        }
        var epsilon = _engine.Reshape(Output.Forward(x), new[] { paddedLength });
        if (paddedLength != waveform.Length) epsilon = _engine.TensorNarrow(epsilon, 0, 0, waveform.Length);
        return (epsilon, VarianceHead.Forward(time));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        var o = Options;
        IndexTtsOps.LoadLinear(_engine, read, TimeInput, prefix + "time.0", o.TimeEmbeddingDim, o.ConditionDim);
        IndexTtsOps.LoadLinear(_engine, read, TimeOutput, prefix + "time.2", o.ConditionDim, o.ConditionDim);
        var speaker = read(prefix + "speaker_embedding.weight", new[] { o.SpeakerCount, o.ConditionDim });
        SpeakerEmbedding.SetParameters(new Vector<T>(speaker.Select(MathHelper.GetNumericOperations<T>().FromDouble)));
        IndexTtsOps.LoadLinear(_engine, read, VarianceHead, prefix + "variance", o.ConditionDim, 1);
        for (int i = 0; i < DownBlocks.Count; i++) DownBlocks[i].Load(read, $"{prefix}down.{i}.", o.BertHiddenDim, o.ConditionDim);
        for (int i = 0; i < UpBlocks.Count; i++) UpBlocks[i].Load(read, $"{prefix}up.{i}.", o.BertHiddenDim, o.ConditionDim);
        for (int i = 0; i < DownResiduals.Count; i++)
            for (int j = 0; j < DownResiduals[i].Count; j++)
                DownResiduals[i][j].Load(read, $"{prefix}down.{i}.residual.{j}.", o.BertHiddenDim, o.ConditionDim);
        for (int i = 0; i < UpResiduals.Count; i++)
            for (int j = 0; j < UpResiduals[i].Count; j++)
                UpResiduals[i][j].Load(read, $"{prefix}up.{i}.residual.{j}.", o.BertHiddenDim, o.ConditionDim);
        IndexTtsOps.LoadConv(read, Output, prefix + "output", 1);
    }
}
