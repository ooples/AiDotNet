using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Vocabulary;

namespace AiDotNet.NER.Preprocessing;

/// <summary>
/// Converts raw text into the packed integer indices consumed by
/// <see cref="AiDotNet.NeuralNetworks.Layers.WordCharEmbeddingLayer{T}"/> /
/// <see cref="AiDotNet.NER.SequenceLabeling.WordCharBiLSTMCRF{T}"/>, and builds the word and
/// character vocabularies. This is the data-layer counterpart to the model, mirroring the standard
/// NER pipeline (a separate vocabulary/tokenizer feeding a model that owns its embeddings).
/// </summary>
/// <remarks>
/// <para>
/// Responsibilities:
/// <list type="bullet">
/// <item><b>Tokenization:</b> a regex word tokenizer that splits punctuation off words, so "Paris,"
/// becomes ["Paris", ","] instead of an out-of-vocabulary blob — fixing the recall loss a naive
/// whitespace split causes.</item>
/// <item><b>Vocabulary construction:</b> a case-folded word vocabulary (words are lowercased for the
/// word-embedding lookup, matching GloVe) and a case-preserving character vocabulary (so capitalization
/// is captured by the character encoder, not discarded).</item>
/// <item><b>Encoding:</b> a sentence becomes a <c>[sequenceLength, 1 + maxWordLength]</c> integer
/// matrix: column 0 is the word id, the rest are the word's character ids (zero-padded).</item>
/// </list>
/// </para>
/// <para>
/// Index 0 is the padding id and index 1 is the unknown id in both vocabularies, so unseen words/chars
/// map deterministically to a stable "[UNK]" embedding rather than hash-derived noise.
/// </para>
/// <para>
/// <b>For Beginners:</b> models work on numbers, not text. This helper assigns every word and every
/// letter a fixed number, splits punctuation correctly, and packs each sentence into a grid of numbers
/// the model can read. "Unknown" words it never saw still get a consistent number, so the model behaves
/// the same every run.
/// </para>
/// </remarks>
public sealed class NerTextEncoder
{
    /// <summary>The padding token, reserved at index 0 in both vocabularies.</summary>
    public const string PadToken = "[PAD]";

    /// <summary>The unknown token, reserved at index 1 in both vocabularies.</summary>
    public const string UnkToken = "[UNK]";

    private static readonly Regex TokenPattern =
        new(@"[A-Za-z0-9]+|[^\sA-Za-z0-9]", RegexOptions.Compiled);

    private readonly Vocabulary _wordVocab;
    private readonly Vocabulary _charVocab;

    /// <summary>Gets the maximum number of characters encoded per word; longer words are truncated.</summary>
    public int MaxWordLength { get; }

    /// <summary>Gets the word vocabulary size (including [PAD] and [UNK]).</summary>
    public int WordVocabSize => _wordVocab.Size;

    /// <summary>Gets the character vocabulary size (including [PAD] and [UNK]).</summary>
    public int CharVocabSize => _charVocab.Size;

    /// <summary>Gets the word vocabulary (word ids are looked up case-folded).</summary>
    public Vocabulary WordVocabulary => _wordVocab;

    /// <summary>Gets the character vocabulary (case-preserving).</summary>
    public Vocabulary CharVocabulary => _charVocab;

    private NerTextEncoder(Vocabulary wordVocab, Vocabulary charVocab, int maxWordLength)
    {
        _wordVocab = wordVocab;
        _charVocab = charVocab;
        MaxWordLength = maxWordLength;
    }

    /// <summary>
    /// Splits a sentence into word/punctuation tokens, separating trailing punctuation from words.
    /// </summary>
    /// <param name="sentence">The raw sentence text.</param>
    /// <returns>The ordered list of token strings.</returns>
    public static string[] Tokenize(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) return [];
        var matches = TokenPattern.Matches(sentence);
        var tokens = new string[matches.Count];
        for (int i = 0; i < matches.Count; i++) tokens[i] = matches[i].Value;
        return tokens;
    }

    /// <summary>
    /// Builds a <see cref="NerTextEncoder"/> from already-tokenized training sentences, populating the
    /// word and character vocabularies. [PAD] and [UNK] are reserved at indices 0 and 1.
    /// </summary>
    /// <param name="tokenizedSentences">The training sentences, each a token array.</param>
    /// <param name="maxWordLength">Maximum characters encoded per word.</param>
    /// <returns>A configured encoder.</returns>
    public static NerTextEncoder Build(IEnumerable<string[]> tokenizedSentences, int maxWordLength = 20)
    {
        if (tokenizedSentences is null) throw new ArgumentNullException(nameof(tokenizedSentences));
        if (maxWordLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxWordLength));

        var wordSeed = new Dictionary<string, int> { [PadToken] = 0, [UnkToken] = 1 };
        var charSeed = new Dictionary<string, int> { [PadToken] = 0, [UnkToken] = 1 };
        var wordVocab = new Vocabulary(wordSeed, UnkToken);
        var charVocab = new Vocabulary(charSeed, UnkToken);

        foreach (var sentence in tokenizedSentences)
        {
            if (sentence is null) continue;
            foreach (var token in sentence)
            {
                if (string.IsNullOrEmpty(token)) continue;
                wordVocab.AddToken(token.ToLowerInvariant());
                foreach (var ch in token)
                    charVocab.AddToken(ch.ToString());
            }
        }

        return new NerTextEncoder(wordVocab, charVocab, maxWordLength);
    }

    /// <summary>
    /// Encodes a tokenized sentence into a packed <c>[sequenceLength, 1 + MaxWordLength]</c> integer
    /// matrix as <see cref="double"/> values: column 0 is the (case-folded) word id, columns 1.. are
    /// the (case-preserving) character ids, zero-padded.
    /// </summary>
    /// <param name="tokens">The sentence tokens.</param>
    /// <param name="sequenceLength">Fixed output sequence length; the sentence is padded/truncated to this.</param>
    /// <returns>A flat row-major <see cref="double"/> array of length <c>sequenceLength * (1 + MaxWordLength)</c>.</returns>
    public double[] EncodePacked(string[] tokens, int sequenceLength)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (sequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(sequenceLength));

        int width = 1 + MaxWordLength;
        var packed = new double[sequenceLength * width];
        int count = Math.Min(tokens.Length, sequenceLength);
        for (int t = 0; t < count; t++)
        {
            string token = tokens[t] ?? string.Empty;
            int rowBase = t * width;
            packed[rowBase] = _wordVocab.GetTokenId(token.ToLowerInvariant());
            int charCount = Math.Min(token.Length, MaxWordLength);
            for (int c = 0; c < charCount; c++)
                packed[rowBase + 1 + c] = _charVocab.GetTokenId(token[c].ToString());
        }
        return packed;
    }

    /// <summary>
    /// Reconstructs an encoder from already-built vocabularies — used when deserializing a
    /// <see cref="AiDotNet.NER.SequenceLabeling.WordCharBiLSTMCRF{T}"/> so the restored model maps
    /// tokens/characters back to the exact same embedding-row ids it was trained with.
    /// </summary>
    /// <param name="wordVocab">The word vocabulary (must reserve [PAD]=0, [UNK]=1).</param>
    /// <param name="charVocab">The character vocabulary (must reserve [PAD]=0, [UNK]=1).</param>
    /// <param name="maxWordLength">Maximum characters encoded per word.</param>
    /// <returns>An encoder backed by the supplied vocabularies.</returns>
    internal static NerTextEncoder FromVocabularies(Vocabulary wordVocab, Vocabulary charVocab, int maxWordLength)
    {
        if (wordVocab is null) throw new ArgumentNullException(nameof(wordVocab));
        if (charVocab is null) throw new ArgumentNullException(nameof(charVocab));
        if (maxWordLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxWordLength));
        return new NerTextEncoder(wordVocab, charVocab, maxWordLength);
    }
}
