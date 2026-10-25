using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// The token layout both KaniTTS releases share (kani-tts's <c>NemoAudioPlayer</c> and the training pipelines'
/// <c>TrainDataPreProcessor</c>): the text tokenizer's ids, ten control tokens after them, then each NanoCodec
/// codebook's codes; the prompt, the training sequence and the speech span.
/// </summary>
internal sealed class KaniLayout
{
    public KaniLayout(int textVocabulary, int endOfText, int codebooks, int codebookSize)
    {
        TextVocabulary = textVocabulary;
        EndOfText = endOfText;
        Codebooks = codebooks;
        CodebookSize = codebookSize;
    }

    public int TextVocabulary { get; }
    public int EndOfText { get; }
    public int Codebooks { get; }
    public int CodebookSize { get; }

    public int StartOfSpeech => TextVocabulary + 1;
    public int EndOfSpeech => TextVocabulary + 2;
    public int StartOfHuman => TextVocabulary + 3;
    public int EndOfHuman => TextVocabulary + 4;
    public int StartOfAi => TextVocabulary + 5;
    public int EndOfAi => TextVocabulary + 6;
    public int AudioTokensStart => TextVocabulary + 10;

    /// <summary>The ids the vocabulary must hold: the text tokens, the control tokens and every codebook's codes.</summary>
    public int RequiredVocabulary => AudioTokensStart + Codebooks * CodebookSize;

    /// <summary><c>get_input_ids</c>: start-of-human, the tokenizer's ids of "prefix: text" (or the text) with its start
    /// token, end-of-text, end-of-human.</summary>
    public int[] PromptIds(ByteLevelBpeTokenizer tokenizer, string text, string? prefix)
    {
        string prompt = prefix is null ? text : $"{prefix.Trim()}: {text}";
        return new[] { StartOfHuman }.Concat(tokenizer.EncodeWithSpecialTokens(prompt)).Concat(new[] { EndOfText, EndOfHuman }).ToArray();
    }

    /// <summary>
    /// <c>add_codes</c> then <c>create_input_ids</c>: start-of-human, the tokenizer's ids of "prefix: text" (the prefix
    /// lowercased) with its start token, end-of-text, end-of-human, start-of-AI, start-of-speech, the frames' codes offset
    /// by codebook (a frame equal to the one before dropped when <paramref name="removeRepeatedFrames"/>), end-of-speech,
    /// end-of-AI.
    /// </summary>
    public int[] TrainingIds(ByteLevelBpeTokenizer tokenizer, string text, string? prefix, int[,] codes, bool removeRepeatedFrames)
    {
        int q = codes.GetLength(0), frames = codes.GetLength(1);
        if (q != Codebooks) throw new ArgumentException($"Codes are [{Codebooks}, frames].", nameof(codes));
        var flattened = new List<int>();
        int[]? previous = null;
        for (int t = 0; t < frames; t++)
        {
            var frame = new int[q];
            for (int c = 0; c < q; c++)
            {
                if (codes[c, t] < 0 || codes[c, t] >= CodebookSize)
                    throw new ArgumentOutOfRangeException(nameof(codes), $"Code {codes[c, t]} is outside [0, {CodebookSize}).");
                frame[c] = codes[c, t] + c * CodebookSize;
            }
            if (removeRepeatedFrames && previous is not null && frame.SequenceEqual(previous)) continue;
            previous = frame;
            flattened.AddRange(frame.Select(code => code + AudioTokensStart));
        }
        if (flattened.Count == 0) throw new ArgumentException("The recording has no codec frames.", nameof(codes));
        string prompt = prefix is null ? text : $"{prefix.ToLowerInvariant()}: {text}";
        return new[] { StartOfHuman }.Concat(tokenizer.EncodeWithSpecialTokens(prompt))
            .Concat(new[] { EndOfText, EndOfHuman, StartOfAi, StartOfSpeech })
            .Concat(flattened).Concat(new[] { EndOfSpeech, EndOfAi }).ToArray();
    }

    /// <summary>The frame of <see cref="Codebooks"/> tokens at <paramref name="start"/> as codes (each codebook's offset
    /// removed), or null when one is not a code.</summary>
    public int[]? Frame(IReadOnlyList<int> tokens, int start)
    {
        var frame = new int[Codebooks];
        for (int q = 0; q < frame.Length; q++)
        {
            frame[q] = tokens[start + q] - q * CodebookSize - AudioTokensStart;
            if (frame[q] < 0) return null;
        }
        return frame;
    }

    /// <summary>Whole frames of tokens as codes <c>[codebooks, frames]</c>, or null when a token is not a code.</summary>
    public int[,]? Codes(IReadOnlyList<int> tokens, int frames)
    {
        var codes = new int[Codebooks, frames];
        for (int f = 0; f < frames; f++)
        {
            var frame = Frame(tokens, f * Codebooks);
            if (frame is null) return null;
            for (int q = 0; q < frame.Length; q++) codes[q, f] = frame[q];
        }
        return codes;
    }

    /// <summary><c>get_waveform</c>'s checks: the codes between the first start-of-speech and the first end-of-speech,
    /// which must both be present and hold whole frames of codes.</summary>
    public int[,] SpanCodes(IReadOnlyList<int> sequence)
    {
        int start = IndexOf(sequence, StartOfSpeech), end = IndexOf(sequence, EndOfSpeech);
        if (start < 0 || end < 0)
            throw new InvalidOperationException("The model wrote no speech span (start-of-speech and end-of-speech tokens).");
        if (start >= end) throw new InvalidOperationException("The model's speech span ends before it starts.");
        var tokens = sequence.Skip(start + 1).Take(end - start - 1).ToList();
        if (tokens.Count % Codebooks != 0)
            throw new InvalidOperationException($"The model's speech span has {tokens.Count} codes, not whole frames of {Codebooks}.");
        if (tokens.Count == 0) throw new InvalidOperationException("The model's speech span is empty.");
        return Codes(tokens, tokens.Count / Codebooks)
            ?? throw new InvalidOperationException("The model's speech span holds tokens that are not audio codes.");
    }

    private static int IndexOf(IReadOnlyList<int> sequence, int token)
    {
        for (int i = 0; i < sequence.Count; i++) if (sequence[i] == token) return i;
        return -1;
    }
}
