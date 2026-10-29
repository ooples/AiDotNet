using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// GLM-4's tokenizer as THUDM's <c>tokenization_chatglm.py</c> builds it (<c>ChatGLM4Tokenizer</c>): a tiktoken
/// byte-level BPE over cl100k's pre-tokenizer pattern (<c>tokenizer.model</c>), the added tokens of
/// <c>tokenizer_config.json</c> numbered after the ranks (the chat roles, <c>&lt;|begin_of_audio|&gt;</c> and
/// GLM-4-Voice's <c>&lt;|audio_N|&gt;</c> speech tokens) matched wherever they occur, and the <c>[gMASK]&lt;sop&gt;</c>
/// prefix that <c>build_inputs_with_special_tokens</c> adds.
/// </summary>
internal sealed class GlmTokenizer
{
    private readonly TiktokenEncoding _encoding;

    public GlmTokenizer(TiktokenEncoding encoding)
    {
        _encoding = encoding ?? throw new ArgumentNullException(nameof(encoding));
        Prefix = new[] { TokenId("[gMASK]"), TokenId("<sop>") };
    }

    /// <summary>Reads <c>tokenizer.model</c> and <c>tokenizer_config.json</c>'s added tokens from a model directory.</summary>
    public static GlmTokenizer FromDirectory(string directory)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        Dictionary<byte[], int> ranks;
        using (var reader = new StreamReader(Path.Combine(directory, "tokenizer.model")))
            ranks = TiktokenEncoding.ReadRanks(reader);
        var added = (JObject)(JObject.Parse(File.ReadAllText(Path.Combine(directory, "tokenizer_config.json")))["added_tokens_decoder"]
            ?? throw new InvalidDataException("tokenizer_config.json has no added_tokens_decoder."));
        var tokens = added.Properties().Select(p => (Id: int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture), Content: (string)p.Value["content"]!))
            .OrderBy(t => t.Id).ToList();
        for (int i = 0; i < tokens.Count; i++)
            if (tokens[i].Id != ranks.Count + i)
                throw new InvalidDataException($"The added tokens must follow the {ranks.Count} ranks without gaps; '{tokens[i].Content}' is {tokens[i].Id}.");
        return new GlmTokenizer(new TiktokenEncoding(ranks, tokens.Select(t => t.Content).ToList(), TiktokenPattern.Cl100k));
    }

    /// <summary><c>[gMASK]</c> and <c>&lt;sop&gt;</c>, which begin every encoded input.</summary>
    public int[] Prefix { get; }

    /// <summary>The number of ids.</summary>
    public int VocabularySize => _encoding.VocabularySize;

    /// <summary>The id of an added token such as <c>&lt;|user|&gt;</c> or <c>&lt;|audio_0|&gt;</c>.</summary>
    public int TokenId(string token) => _encoding.SpecialTokenId(token);

    /// <summary>The ids of <paramref name="text"/>, added tokens matched where they occur; with
    /// <paramref name="addPrefix"/>, after <c>[gMASK]&lt;sop&gt;</c> (the tokenizer's call with special tokens).</summary>
    public int[] Encode(string text, bool addPrefix = true)
    {
        var ids = _encoding.Encode(text, allowSpecial: true);
        return addPrefix ? Prefix.Concat(ids).ToArray() : ids;
    }

    /// <summary>The text of ids, added tokens written out.</summary>
    public string Decode(IEnumerable<int> ids) => _encoding.Decode(ids);
}
