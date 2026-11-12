using System.IO.Compression;
using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>The released Kaldi text-to-phone lexicon followed by UniAudio's alignment-phone tokenizer.</summary>
internal static class UniAudioReleasedTokenizer
{
    private sealed class Vocabulary
    {
        internal readonly Dictionary<string, int> Phones = new(StringComparer.Ordinal);
        internal readonly Dictionary<string, string[]> Lexicon = new(StringComparer.Ordinal);
        internal readonly Dictionary<int, string> Uppercase = new();
        internal readonly int Unknown;

        internal Vocabulary()
        {
            using (var reader = Read("uniaudio-phones.txt.gz"))
            {
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    string phone = line.Trim();
                    if (phone.Length > 0 && !Phones.ContainsKey(phone)) Phones.Add(phone, Phones.Count);
                }
            }
            // The author's default unknown symbol is distinct from the table's lowercase <unk>.
            if (!Phones.ContainsKey("<UNK>")) Phones.Add("<UNK>", Phones.Count);
            Unknown = Phones["<UNK>"];
            using (var reader = Read("uniaudio-uppercase.tsv.gz"))
            {
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    int tab = line.IndexOf('\t');
                    Uppercase.Add(int.Parse(line.Substring(0, tab), NumberStyles.HexNumber, CultureInfo.InvariantCulture),
                        line.Substring(tab + 1));
                }
            }
            using (var reader = Read("uniaudio-align-lexicon.txt.gz"))
            {
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    var fields = Split(line);
                    if (fields.Length >= 3) Lexicon[fields[0]] = fields.Skip(2).ToArray();
                }
            }
        }
    }

    private static readonly Lazy<Vocabulary> Data = new(() => new Vocabulary());

    internal static IReadOnlyDictionary<string, int> PhoneInventory => Data.Value.Phones;

    internal static int[] Tokenize(string text, int maximum)
    {
        // Preserve the original method's replacement order and punctuation behavior.
        string normalized = text.Replace("\"", "").Replace("|", " ").Replace("[", "").Replace("]", "")
            .Replace("--", " ").Replace("-", " ").Replace(".", "").Replace(",", " ,").Replace("?", "");
        var phones = new List<string> { "SIL" };
        foreach (string word in Split(normalized))
        {
            string upper = Upper(word);
            if (Data.Value.Lexicon.TryGetValue(upper, out var pronunciation)) phones.AddRange(pronunciation);
            else if (upper is "." or "," or ":" or ";" or "!" or "?") phones.Add("SIL");
            else phones.AddRange(Data.Value.Lexicon["<UNK>"]);
        }
        return Encode(phones, maximum);
    }

    internal static int[] TokenizePhones(string phones, int maximum) => Encode(Split(phones), maximum);

    private static int[] Encode(IEnumerable<string> phones, int maximum)
    {
        if (maximum <= 0) throw new ArgumentOutOfRangeException(nameof(maximum));
        var result = new List<int>();
        foreach (string phone in phones)
        {
            int id = Data.Value.Phones.TryGetValue(phone, out int found) ? found : Data.Value.Unknown;
            if (result.Count == 0 || result[result.Count - 1] != id) result.Add(id);
            if (result.Count == maximum) break;
        }
        return result.ToArray();
    }

    private static readonly char[] Whitespace = Enumerable.Range(0, char.MaxValue + 1)
        .Select(code => (char)code).Where(PythonText.IsSpace).ToArray();

    private static string[] Split(string text) => text.Split(Whitespace, StringSplitOptions.RemoveEmptyEntries);

    private static string Upper(string word)
    {
        var result = new StringBuilder(word.Length);
        for (int i = 0; i < word.Length; i++)
        {
            int length = char.IsHighSurrogate(word[i]) && i + 1 < word.Length && char.IsLowSurrogate(word[i + 1]) ? 2 : 1;
            int code = length == 2 ? char.ConvertToUtf32(word[i], word[i + 1]) : word[i];
            result.Append(Data.Value.Uppercase.TryGetValue(code, out string? mapped) ? mapped : word.Substring(i, length));
            i += length - 1;
        }
        return result.ToString();
    }

    private static StreamReader Read(string name)
    {
        var stream = typeof(UniAudioReleasedTokenizer).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd." + name)
            ?? throw new InvalidOperationException("Missing UniAudio frontend resource: " + name);
        return new StreamReader(new GZipStream(stream, CompressionMode.Decompress));
    }
}
