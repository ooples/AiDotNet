using System.Globalization;
using System.Text;
using AiDotNet.ActivationFunctions;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Dicta's Hebrew diacritizer (dicta-onnx 1.0.9 <c>Dicta.add_diacritics</c> over DictaBERT-large-char-menaked, Shmidman
/// et al. 2024): every Hebrew letter gets the niqqud — and a shin its dot — that a character-level BERT-large predicts.
/// </summary>
/// <remarks>
/// <para>The text's niqqud (U+05B0–U+05BD, U+05C1, U+05C2, U+05C7) is removed; the model's tokenizer normalizes it (NFKC,
/// lower case, combining marks removed, runs of non-Hebrew, non-ASCII characters to <c>[UNK]</c>) and reads it a character
/// at a time; the 24-layer encoder's states go through the nikud (29 classes) and shin (2 classes) heads, and each Hebrew
/// letter that is one token gets its arg-max classes, a matres lectionis kept bare.</para>
/// <para>The weights are <c>dicta-il/dictabert-large-char-menaked</c> (CC BY 4.0); <see cref="DefaultModelDirectory"/>
/// downloads them from the Hugging Face Hub once.</para>
/// </remarks>
internal sealed class DictaDiacritizer
{
    /// <summary>The Hugging Face repository and the revision the port was checked against.</summary>
    public const string Repository = "dicta-il/dictabert-large-char-menaked";
    private const string Revision = "4dfc20b94565d7868e1c8c00bf039d38c071a062";
    private const string MatresLectionisToken = "<MAT_LECT>";
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    private readonly BertNetwork<float> _bert;
    private readonly DenseLayer<float> _nikud, _shin;
    private readonly string[] _nikudClasses, _shinClasses;
    private readonly Dictionary<string, int> _vocabulary;
    private readonly int _unknown, _start, _end;

    private DictaDiacritizer(BertNetwork<float> bert, DenseLayer<float> nikud, DenseLayer<float> shin, string[] nikudClasses,
        string[] shinClasses, Dictionary<string, int> vocabulary)
    {
        _bert = bert;
        _nikud = nikud;
        _shin = shin;
        _nikudClasses = nikudClasses;
        _shinClasses = shinClasses;
        _vocabulary = vocabulary;
        _unknown = vocabulary["[UNK]"];
        _start = vocabulary["[CLS]"];
        _end = vocabulary["[SEP]"];
    }

    /// <summary>Loads a model directory (<c>config.json</c>, <c>tokenizer.json</c>, <c>model.safetensors</c>).</summary>
    public static DictaDiacritizer Load(string modelDirectory)
    {
        var config = JObject.Parse(File.ReadAllText(Path.Combine(modelDirectory, "config.json")));
        var configuration = new BertConfiguration
        {
            VocabSize = (int)config["vocab_size"]!, Hidden = (int)config["hidden_size"]!, Layers = (int)config["num_hidden_layers"]!,
            Heads = (int)config["num_attention_heads"]!, Intermediate = (int)config["intermediate_size"]!,
            MaxPositions = (int)config["max_position_embeddings"]!, TypeVocabSize = (int)config["type_vocab_size"]!,
            LayerNormEpsilon = (double)config["layer_norm_eps"]!, Dropout = 0,
        };
        var nikudClasses = config["nikud_classes"]!.Values<string>().Select(v => v ?? "").ToArray();
        var shinClasses = config["shin_classes"]!.Values<string>().Select(v => v ?? "").ToArray();
        var layers = new List<LayerBase<float>>();
        var bert = new BertNetwork<float>(Engine, layers, configuration);
        var nikud = new DenseLayer<float>(nikudClasses.Length, (IActivationFunction<float>)new IdentityActivation<float>());
        var shin = new DenseLayer<float>(shinClasses.Length, (IActivationFunction<float>)new IdentityActivation<float>());
        using (var file = TorchCheckpointFile.Open(Path.Combine(modelDirectory, "model.safetensors")))
        {
            bert.LoadTorchWeights(file, "bert.");
            int h = configuration.Hidden;
            TorchParameters.Linear(Engine, nikud, h, nikudClasses.Length, file, "menaked.nikud_cls.weight", "menaked.nikud_cls.bias");
            TorchParameters.Linear(Engine, shin, h, shinClasses.Length, file, "menaked.shin_cls.weight", "menaked.shin_cls.bias");
        }
        foreach (var layer in layers.Append(nikud).Append(shin)) layer.SetTrainingMode(false);
        var tokenizer = JObject.Parse(File.ReadAllText(Path.Combine(modelDirectory, "tokenizer.json"), Encoding.UTF8));
        var vocabulary = ((JObject)tokenizer["model"]!["vocab"]!).Properties().ToDictionary(p => p.Name, p => (int)p.Value, StringComparer.Ordinal);
        return new DictaDiacritizer(bert, nikud, shin, nikudClasses, shinClasses, vocabulary);
    }

    /// <summary>The model's directory in the Hugging Face cache, downloading it there first if needed.</summary>
    public static string DefaultModelDirectory()
    {
        using var loader = new HuggingFaceModelLoader<float>();
        loader.DownloadModelAsync(Repository, Revision, new[] { "config.json", "tokenizer.json", "model.safetensors" })
            .GetAwaiter().GetResult();
        return loader.GetCachePath(Repository, Revision);
    }

    private static bool IsNiqqud(char c) => (c >= (char)0x05B0 && c <= (char)0x05BD) || c == (char)0x05C1 || c == (char)0x05C2 || c == (char)0x05C7;
    private static bool IsHebrewLetter(string s) => s.Length == 1 && s[0] >= (char)0x05D0 && s[0] <= (char)0x05EA;
    private static bool IsMatres(string s) => s is "\u05D0" or "\u05D5" or "\u05D9";

    // The normalizer's allowed characters: Hebrew, ASCII, U+200C–U+203F, U+20A0–U+20BF, U+2200–U+22FF,
    // U+2150–U+218B, U+FB00–U+FB4F; every other run becomes [UNK].
    private static bool Allowed(int c) => (c >= 0x0590 && c <= 0x05FF) || c <= 0x7F || (c >= 0x200C && c <= 0x203F)
        || (c >= 0x20A0 && c <= 0x20BF) || (c >= 0x2200 && c <= 0x22FF) || (c >= 0x2150 && c <= 0x218B) || (c >= 0xFB00 && c <= 0xFB4F);

    /// <summary>The model's tokens of <paramref name="sentence"/> and each one's span in it (the tokenizer's normalizer and
    /// per-character pre-tokenizer, with the alignment Hugging Face tokenizers keep).</summary>
    internal (int[] Ids, (int Start, int End)[] Offsets) Tokenize(string sentence)
    {
        // Normalized code points, each with the span of the original text it came from.
        var normalized = new List<(string Text, int Start, int End)>();
        for (int i = 0; i < sentence.Length;)
        {
            // A starter and the combining marks after it normalize together.
            int end = i + (char.IsSurrogatePair(sentence, i) ? 2 : 1);
            while (end < sentence.Length && IsMark(sentence, end)) end += char.IsSurrogatePair(sentence, end) ? 2 : 1;
            var source = sentence.Substring(i, end - i);
            var nfkc = ChineseText.Codepoints(source.Normalize(NormalizationForm.FormKC));
            var original = ChineseText.Codepoints(source);
            if (nfkc.Count == original.Count)
            {
                int at = i;
                foreach (var (n, o) in nfkc.Zip(original, (n, o) => (n, o)))
                {
                    normalized.Add((n, at, at + o.Length));
                    at += o.Length;
                }
            }
            else
            {
                foreach (var n in nfkc) normalized.Add((n, i, end));
            }
            i = end;
        }
        // Lowercase (U+0130 to i̇), then StripAccents (combining marks removed).
        var lowered = new List<(string Text, int Start, int End)>();
        foreach (var (text, start, end) in normalized)
        {
            string lower = text == "\u0130" ? "i\u0307" : text.ToLowerInvariant();
            foreach (var c in ChineseText.Codepoints(lower))
                if (!IsMark(c, 0)) lowered.Add((c, start, end));
        }
        // Replace("<foreign>", "[UNK]"), then each run of disallowed characters to one [UNK].
        var pieces = new List<(string Text, int Start, int End)>();
        for (int i = 0; i < lowered.Count;)
        {
            if (i + 9 <= lowered.Count && string.Concat(lowered.Skip(i).Take(9).Select(p => p.Text)) == "<foreign>")
            {
                pieces.Add(("[UNK]", lowered[i].Start, lowered[i + 8].End));
                i += 9;
                continue;
            }
            if (!Allowed(char.ConvertToUtf32(lowered[i].Text, 0)))
            {
                int j = i;
                while (j < lowered.Count && !Allowed(char.ConvertToUtf32(lowered[j].Text, 0))) j++;
                pieces.Add(("[UNK]", lowered[i].Start, lowered[j - 1].End));
                i = j;
                continue;
            }
            pieces.Add(lowered[i]);
            i++;
        }
        var ids = new List<int> { _start };
        var offsets = new List<(int, int)> { (0, 0) };
        foreach (var (text, start, end) in pieces)
        {
            ids.Add(_vocabulary.TryGetValue(text, out int id) ? id : _unknown);
            offsets.Add((start, end));
        }
        ids.Add(_end);
        offsets.Add((0, 0));
        return (ids.ToArray(), offsets.ToArray());
    }

    private static bool IsMark(string s, int index) => CharUnicodeInfo.GetUnicodeCategory(s, index)
        is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;

    /// <summary><c>Dicta.add_diacritics(text)</c>: the text with its predicted niqqud.</summary>
    public string AddDiacritics(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var sentence = new string(text.Where(c => !IsNiqqud(c)).ToArray());
        var (ids, offsets) = Tokenize(sentence);
        if (ids.Length > _bert.Configuration.MaxPositions)
            throw new ArgumentException($"Dicta reads at most {_bert.Configuration.MaxPositions - 2} characters at a time.", nameof(text));
        var input = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) input[0, i] = ids[i];
        int[] nikud, shin;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<float>())
        {
            var states = Engine.Reshape(_bert.Encode(input), new[] { ids.Length, _bert.Configuration.Hidden });
            nikud = ArgMax(_nikud.Forward(states));
            shin = ArgMax(_shin.Forward(states));
        }

        var output = new StringBuilder();
        int previous = 0;
        for (int index = 0; index < offsets.Length; index++)
        {
            var (start, end) = offsets[index];
            if (start > previous) output.Append(sentence, previous, start - previous);
            if (end - start != 1) continue;
            string character = sentence.Substring(start, 1);
            previous = end;
            if (!IsHebrewLetter(character))
            {
                output.Append(character);
                continue;
            }
            string mark = _nikudClasses[nikud[index]];
            string dot = character == "\u05E9" ? _shinClasses[shin[index]] : "";
            if (mark == MatresLectionisToken)
            {
                if (!IsMatres(character))
                {
                    mark = "";
                }
                else
                {
                    output.Append(character);
                    continue;
                }
            }
            output.Append(character).Append(dot).Append(mark);
        }
        output.Append(sentence, previous, sentence.Length - previous);
        return output.ToString();
    }

    // The arg-max of each row (the first on ties, as numpy's).
    private static int[] ArgMax(Tensor<float> logits)
    {
        int rows = logits.Shape[0], classes = logits.Shape[1];
        var values = logits.ToArray();
        var result = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int k = 1; k < classes; k++)
                if (values[r * classes + k] > values[r * classes + best]) best = k;
            result[r] = best;
        }
        return result;
    }
}
