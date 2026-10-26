using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// MegaTTS 3's grapheme-to-phoneme model (<c>frontend_function.g2p</c>): a Qwen2 language model continues
/// <c>&lt;BOT&gt;</c> + text + <c>&lt;BOS&gt;</c> and the phone-start token with phone tokens (offset by the speech-start
/// token <c>&lt;Reserved_TTS_0&gt;</c>), greedily (the release samples with top-k 1) until the end-of-phones token or 256
/// tokens; the tokens from the phone-start on, without the last, are split into phones and tones.
/// </summary>
internal sealed class MegaTts3G2p<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The phone that starts every phone sequence (offset by the speech start).</summary>
    public const int PhoneStart = 145;

    /// <summary>The end-of-phones token (offset by the speech start).</summary>
    public const int PhoneEnd = 801;

    /// <summary>The release's Qwen2-0.5B (<c>g2p/config.json</c>).</summary>
    public static LlamaConfiguration ReleaseConfiguration => new()
    {
        VocabSize = 168896, Hidden = 896, Intermediate = 4864, Layers = 24, Heads = 14, KeyValueHeads = 2, RmsEpsilon = 1e-6,
        RopeTheta = 1000000, AttentionBias = true, TieWordEmbeddings = true,
    };

    public MegaTts3G2p(IEngine engine, List<LayerBase<T>> layers, LlamaConfiguration configuration, ByteLevelBpeTokenizer? tokenizer, int speechStart)
    {
        Network = new LlamaNetwork<T>(engine, layers, configuration);
        Tokenizer = tokenizer;
        SpeechStart = speechStart;
    }

    public LlamaNetwork<T> Network { get; }

    /// <summary>The checkpoint's tokenizer; the release reads <c>&lt;Reserved_TTS_0&gt;</c>'s id from it as the speech start.</summary>
    public ByteLevelBpeTokenizer? Tokenizer { get; }

    /// <summary>The id the phone tokens are offset by.</summary>
    public int SpeechStart { get; }

    /// <summary>The most tokens generated (the release's 256).</summary>
    public int MaxNewTokens { get; set; } = 256;

    /// <summary>Builds the release's model from its checkpoint folder's <c>tokenizer.json</c>.</summary>
    public static MegaTts3G2p<T> FromTokenizer(IEngine engine, List<LayerBase<T>> layers, LlamaConfiguration configuration, string tokenizerJson)
    {
        var tokenizer = ByteLevelBpeTokenizer.FromJson(tokenizerJson);
        return new MegaTts3G2p<T>(engine, layers, configuration, tokenizer, tokenizer.Encode("<Reserved_TTS_0>")[0]);
    }

    /// <summary>The release's prompt: the tokenized <c>&lt;BOT&gt;</c> + text + <c>&lt;BOS&gt;</c>.</summary>
    public int[] TextTokens(string text)
    {
        if (Tokenizer is null) throw new InvalidOperationException("The model has no tokenizer.");
        return Tokenizer.EncodeWithSpecialTokens("<BOT>" + text + "<BOS>");
    }

    /// <summary>Generation from the text tokens and the phone-start token: the whole sequence (prompt included).</summary>
    public List<int> Generate(IReadOnlyList<int> textTokens)
    {
        using var _ = new NoGradScope<T>();
        var ids = new List<int>(textTokens) { PhoneStart + SpeechStart };
        var cache = new LlamaCache<T>();
        var logits = Network.Forward(Network.Embed(Row(ids)), null, null, cache, lastLogitsOnly: true).Logits!;
        int end = PhoneEnd + SpeechStart;
        for (int step = 0; step < MaxNewTokens; step++)
        {
            int best = 0;
            for (int v = 1; v < logits.Length; v++)
                if (NumOps.GreaterThan(logits[v], logits[best])) best = v;
            ids.Add(best);
            if (best == end || step == MaxNewTokens - 1) break;
            logits = Network.Forward(Network.Embed(Row(new[] { best })), null, null, cache, lastLogitsOnly: true).Logits!;
        }
        return ids;
    }

    /// <summary>The phones and tones of generated tokens: from the phone-start token, without the last, less the speech
    /// start, through <see cref="MegaTts3Aligner{T}.SplitPhone"/>.</summary>
    public (int[] Phones, int[] Tones) Split(IReadOnlyList<int> generated, int textLength)
    {
        var phones = new List<int>();
        var tones = new List<int>();
        for (int i = textLength; i < generated.Count - 1; i++)
        {
            var (phone, tone) = MegaTts3Aligner<T>.SplitPhone(generated[i] - SpeechStart);
            phones.Add(phone);
            tones.Add(tone);
        }
        return (phones.ToArray(), tones.ToArray());
    }

    /// <summary><c>g2p</c>: the phones and tones of a text.</summary>
    public (int[] Phones, int[] Tones) Convert(string text)
    {
        var tokens = TextTokens(text);
        return Split(Generate(tokens), tokens.Length);
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }
}
