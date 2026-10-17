using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Generation;

/// <summary>The sizes of AudioLM's token streams.</summary>
/// <param name="SemanticCodes">Semantic token vocabulary (1024 w2v-BERT k-means clusters).</param>
/// <param name="Codebooks">Q, the SoundStream RVQ levels (12).</param>
/// <param name="CoarseCodebooks">Q′, the levels the coarse stage models (4).</param>
/// <param name="CodebookSize">N, codes per level (1024).</param>
internal sealed record AudioLmTokens(int SemanticCodes, int Codebooks, int CoarseCodebooks, int CodebookSize)
{
    /// <summary>The semantic stage's vocabulary: the semantic codes, then BOS and EOS.</summary>
    public int SemanticVocabulary => SemanticCodes + 2;

    /// <summary>The coarse stage's vocabulary: the semantic codes, the Q′ coarse levels offset by q·N, then BOS,
    /// the separator and EOS.</summary>
    public int CoarseVocabulary => SemanticCodes + CoarseCodebooks * CodebookSize + 3;

    /// <summary>The fine stage's vocabulary: all Q levels offset by q·N, then BOS.</summary>
    public int FineVocabulary => Codebooks * CodebookSize + 1;
}

/// <summary>Which of AudioLM's three Transformers a training step updates.</summary>
public enum AudioLMStage
{
    /// <summary>Semantic modeling: p(z_t | z_&lt;t).</summary>
    Semantic,

    /// <summary>Coarse acoustic modeling: the first Q′ levels given the semantic tokens.</summary>
    CoarseAcoustic,

    /// <summary>Fine acoustic modeling: the remaining levels given the coarse ones, on 3-second chunks.</summary>
    FineAcoustic,
}

/// <summary>
/// AudioLM's three decoder-only Transformers and their token sequences (Borsos et al. 2022, §III-C, §III-D), shared by
/// <see cref="AudioLM{T}"/> and SPEAR-TTS.
/// </summary>
/// <remarks>
/// <para>
/// Acoustic tokens are flattened row-major, frame by frame and coarse to fine within a frame, each level's codes offset
/// by q·N so the levels have distinct ids (§III-B). The semantic stage reads <c>BOS z₁ … z_T EOS</c>; the coarse stage
/// <c>BOS z₁ … z_T SEP y₁¹ … y₁^Q′ y₂¹ … EOS</c>, predicting the acoustic tokens and the end; the fine stage
/// <c>BOS</c>, a chunk's coarse tokens, then its fine tokens, which it predicts. The semantic tokens of the first two
/// stages have consecutive repetitions removed (§IV-B).
/// </para>
/// <para>
/// The paper does not give the special tokens; BOS, the separator and EOS follow lucidrains/audiolm-pytorch, which
/// starts each stage's sequence, separates the semantic and acoustic streams and appends an end token. Sampling is
/// restricted to the ids that may come next: semantic codes or the end in the first stage, the current level's codes
/// (and, at a frame boundary, the end) in the second, the current level's codes in the third.
/// </para>
/// </remarks>
internal sealed class AudioLmCore<T>
{
    public AudioLmCore(IEngine engine, List<LayerBase<T>> semanticLayers, List<LayerBase<T>> coarseLayers,
        List<LayerBase<T>> fineLayers, AudioLmTokens tokens, Func<int, T5Configuration> configuration, Random initialization)
    {
        Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        if (tokens.CoarseCodebooks < 1 || tokens.CoarseCodebooks >= tokens.Codebooks)
            throw new ArgumentException("The coarse levels must be at least one and fewer than all levels.", nameof(tokens));
        Semantic = new T5DecoderOnly<T>(engine, semanticLayers, configuration(tokens.SemanticVocabulary));
        Coarse = new T5DecoderOnly<T>(engine, coarseLayers, configuration(tokens.CoarseVocabulary));
        Fine = new T5DecoderOnly<T>(engine, fineLayers, configuration(tokens.FineVocabulary));
        Semantic.InitializeLikeHuggingFace(initialization);
        Coarse.InitializeLikeHuggingFace(initialization);
        Fine.InitializeLikeHuggingFace(initialization);
    }

    public AudioLmTokens Tokens { get; }
    public T5DecoderOnly<T> Semantic { get; }
    public T5DecoderOnly<T> Coarse { get; }
    public T5DecoderOnly<T> Fine { get; }

    private int SemanticBegin => Tokens.SemanticCodes;
    private int SemanticEnd => Tokens.SemanticCodes + 1;
    private int CoarseOffset => Tokens.SemanticCodes;
    private int CoarseBegin => Tokens.SemanticCodes + Tokens.CoarseCodebooks * Tokens.CodebookSize;
    private int CoarseSeparator => CoarseBegin + 1;
    private int CoarseEnd => CoarseBegin + 2;
    private int FineBegin => Tokens.Codebooks * Tokens.CodebookSize;

    /// <summary>Removes consecutive repetitions (§IV-B, following GSLM).</summary>
    public static int[] Deduplicate(IReadOnlyList<int> tokens)
    {
        var result = new List<int>(tokens.Count);
        foreach (int token in tokens)
            if (result.Count == 0 || result[result.Count - 1] != token) result.Add(token);
        return result.ToArray();
    }

    private void CheckSemantic(IReadOnlyList<int> semantic)
    {
        foreach (int token in semantic)
            if (token < 0 || token >= Tokens.SemanticCodes)
                throw new ArgumentOutOfRangeException(nameof(semantic), $"Semantic tokens lie in [0, {Tokens.SemanticCodes}).");
    }

    private void CheckCodes(int[,] codes, int firstLevel, int levels)
    {
        if (codes.GetLength(0) < firstLevel + levels)
            throw new ArgumentException($"Expected at least {firstLevel + levels} levels of codes.", nameof(codes));
        for (int q = firstLevel; q < firstLevel + levels; q++)
            for (int t = 0; t < codes.GetLength(1); t++)
                if (codes[q, t] < 0 || codes[q, t] >= Tokens.CodebookSize)
                    throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {Tokens.CodebookSize}).");
    }

    // ---------------------------------------------------------------- sequences

    private List<int> CoarsePrefix(IReadOnlyList<int> semantic)
    {
        var ids = new List<int> { CoarseBegin };
        ids.AddRange(Deduplicate(semantic));
        ids.Add(CoarseSeparator);
        return ids;
    }

    private void AppendCoarse(List<int> ids, int[,] codes, int from, int to)
    {
        int n = Tokens.CodebookSize;
        for (int t = from; t < to; t++)
            for (int q = 0; q < Tokens.CoarseCodebooks; q++) ids.Add(CoarseOffset + q * n + codes[q, t]);
    }

    private List<int> FineSequence(int[,] codes, int from, int to, bool withFine)
    {
        int n = Tokens.CodebookSize;
        var ids = new List<int> { FineBegin };
        for (int t = from; t < to; t++)
            for (int q = 0; q < Tokens.CoarseCodebooks; q++) ids.Add(q * n + codes[q, t]);
        if (withFine)
            for (int t = from; t < to; t++)
                for (int q = Tokens.CoarseCodebooks; q < Tokens.Codebooks; q++) ids.Add(q * n + codes[q, t]);
        return ids;
    }

    // ---------------------------------------------------------------- losses

    /// <summary>The semantic stage's loss on <c>BOS z EOS</c> (repetitions removed).</summary>
    public Func<Tensor<T>> SemanticObjective(IReadOnlyList<int> semantic, bool training, Random random)
    {
        CheckSemantic(semantic);
        var ids = new List<int> { SemanticBegin };
        ids.AddRange(Deduplicate(semantic));
        ids.Add(SemanticEnd);
        return () => Semantic.Loss(ids, 1, training, random);
    }

    /// <summary>The coarse stage's loss on its acoustic tokens and end, given the semantic tokens.</summary>
    public Func<Tensor<T>> CoarseObjective(IReadOnlyList<int> semantic, int[,] codes, bool training, Random random)
    {
        CheckSemantic(semantic);
        CheckCodes(codes, 0, Tokens.CoarseCodebooks);
        var ids = CoarsePrefix(semantic);
        int first = ids.Count;
        AppendCoarse(ids, codes, 0, codes.GetLength(1));
        ids.Add(CoarseEnd);
        return () => Coarse.Loss(ids, first, training, random);
    }

    /// <summary>The fine stage's loss on a chunk's fine tokens, given its coarse ones.</summary>
    public Func<Tensor<T>> FineObjective(int[,] codes, bool training, Random random)
    {
        CheckCodes(codes, 0, Tokens.Codebooks);
        int frames = codes.GetLength(1);
        var ids = FineSequence(codes, 0, frames, withFine: true);
        int first = 1 + frames * Tokens.CoarseCodebooks;
        return () => Fine.Loss(ids, first, training, random);
    }

    // ---------------------------------------------------------------- generation (§III-D)

    /// <summary>Continues the prompt's semantic tokens (repetitions removed) with at most <paramref name="maxTokens"/>
    /// more; returns the new tokens.</summary>
    public int[] GenerateSemantic(IReadOnlyList<int> prompt, int maxTokens, double temperature, Random random)
    {
        CheckSemantic(prompt);
        var prefix = new List<int> { SemanticBegin };
        prefix.AddRange(Deduplicate(prompt));
        return Semantic.Generate(prefix, temperature, maxTokens, (_, v) => v < Tokens.SemanticCodes || v == SemanticEnd,
            SemanticEnd, random).ToArray();
    }

    /// <summary>
    /// Continues the prompt's coarse codes <c>[Q′+, P]</c> by at most <paramref name="frames"/> frames given all the
    /// semantic tokens (prompt and continuation); returns the new frames' coarse codes <c>[Q′, F]</c>.
    /// </summary>
    public int[,] GenerateCoarse(IReadOnlyList<int> semantic, int[,] prompt, int frames, double temperature, Random random)
    {
        CheckSemantic(semantic);
        CheckCodes(prompt, 0, prompt.GetLength(1) == 0 ? 0 : Tokens.CoarseCodebooks);
        var prefix = CoarsePrefix(semantic);
        AppendCoarse(prefix, prompt, 0, prompt.GetLength(1));
        int levels = Tokens.CoarseCodebooks, n = Tokens.CodebookSize;
        var tokens = Coarse.Generate(prefix, temperature, frames * levels, (step, v) =>
        {
            int level = step % levels;
            // The end token may close a frame, but not before the first: ending at once rendered no audio at all.
            return (v >= CoarseOffset + level * n && v < CoarseOffset + (level + 1) * n) || (level == 0 && step > 0 && v == CoarseEnd);
        }, CoarseEnd, random);
        int generated = tokens.Count / levels;
        var codes = new int[levels, generated];
        for (int t = 0; t < generated; t++)
            for (int q = 0; q < levels; q++) codes[q, t] = tokens[t * levels + q] - CoarseOffset - q * n;
        return codes;
    }

    /// <summary>Fills levels Q′…Q−1 of <paramref name="codes"/> <c>[Q, frames]</c> from frame
    /// <paramref name="from"/> on, chunk by chunk of <paramref name="chunkFrames"/> non-overlapping frames.</summary>
    public void GenerateFine(int[,] codes, int from, int chunkFrames, double temperature, Random random)
    {
        int frames = codes.GetLength(1), n = Tokens.CodebookSize, coarse = Tokens.CoarseCodebooks, fine = Tokens.Codebooks - coarse;
        CheckCodes(codes, 0, coarse);
        for (int start = from; start < frames; start += chunkFrames)
        {
            int end = Math.Min(frames, start + chunkFrames);
            var prefix = FineSequence(codes, start, end, withFine: false);
            var tokens = Fine.Generate(prefix, temperature, (end - start) * fine, (step, v) =>
            {
                int level = coarse + step % fine;
                return v >= level * n && v < (level + 1) * n;
            }, stopToken: -1, random);
            for (int t = start; t < end; t++)
                for (int q = coarse; q < Tokens.Codebooks; q++)
                    codes[q, t] = tokens[(t - start) * fine + (q - coarse)] - q * n;
        }
    }
}
