using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Cutlet's default Hepburn word readings and token joining over UniDic-lite, used by XTTS.</summary>
internal sealed class Cutlet
{
    private sealed record Entry(string Reading, string Pos1, string Pos2, bool Foreign);
    private sealed class Token
    {
        internal string Reading = "";
        internal bool Space;
    }
    private readonly Entry[] _entries;
    private readonly Dictionary<string, string> _kana, _exceptions;
    private static readonly Lazy<Cutlet> Shared = new(() => new Cutlet());
    internal static Cutlet Default => Shared.Value;

    private Cutlet()
    {
        using var input = typeof(Cutlet).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.cutlet_unidic.json.gz")
            ?? throw new InvalidDataException("The embedded Cutlet/UniDic readings are missing.");
        using var gzip = new GZipStream(input, CompressionMode.Decompress);
        using var document = JsonDocument.Parse(gzip);
        var root = document.RootElement;
        _entries = root.GetProperty("entries").EnumerateArray().Select(x => new Entry(x[0].GetString()!, x[1].GetString()!, x[2].GetString()!, x[3].GetBoolean())).ToArray();
        _kana = root.GetProperty("kana").EnumerateObject().ToDictionary(x => x.Name, x => x.Value.GetString()!, StringComparer.Ordinal);
        _exceptions = root.GetProperty("exceptions").EnumerateObject().ToDictionary(x => x.Name, x => x.Value.GetString()!, StringComparer.Ordinal);
    }

    private static bool Ascii(string s) => s.All(c => c < 128);
    private static bool Digits(string s) => s.Length > 0 && s.All(char.IsDigit);
    private string MapKana(string text)
    {
        text = new string(text.Select(c => c >= 'ァ' && c <= 'ヶ' ? (char)(c - 0x60) : c).ToArray());
        var result = new StringBuilder();
        const string small = "ゃゅょぁぃぅぇぉ";
        for (int i = 0; i < text.Length; i++)
        {
            string c = text[i].ToString(), previous = i > 0 ? text[i - 1].ToString() : "", next = i + 1 < text.Length ? text[i + 1].ToString() : "";
            string value;
            if (previous.Length > 0 && _kana.TryGetValue(previous + c, out var pair)) value = pair;
            else if (next.Length > 0 && _kana.ContainsKey(c + next)) value = "";
            else if (next.Length > 0 && small.Contains(next))
            {
                string baseValue = _kana.TryGetValue(c, out var b) ? b : "?";
                value = c == "っ" ? "" : baseValue.Substring(0, Math.Max(0, baseValue.Length - 1)) + _kana[next];
            }
            else if (small.Contains(c)) value = "";
            else if (c == "ー") value = _kana.TryGetValue(previous, out var p) && p.Length > 0 ? p.Substring(p.Length - 1) : "-";
            else if (c == "っ") value = next.Length == 0 ? "っ" : next == "ち" ? "t" : "あいうえおっ".Contains(next) ? "-" : _kana.TryGetValue(next, out var n) && n.Length > 0 ? n.Substring(0, 1) : "";
            else if (c == "ん") value = next.Length > 0 && "あいうえおやゆよ".Contains(next) ? "n'" : "n";
            else value = _kana.TryGetValue(c, out var v) ? v : Ascii(c) ? c : "?";
            result.Append(value);
        }
        return result.ToString();
    }

    internal string Romanize(string text)
    {
        text = text.Normalize(NormalizationForm.FormKC);
        var words = MeCab.UnidicLite.ParseCutlet(text);
        var entries = words.Select(w => w.Token >= 0 ? _entries[w.Token] : new Entry("", "", "", false)).ToArray();
        var whitespace = new bool[words.Count];
        int cursor = 0;
        for (int i = 0; i < words.Count; i++)
        {
            int at = text.IndexOf(words[i].Surface, cursor, StringComparison.Ordinal);
            whitespace[i] = at > cursor;
            cursor = Math.Max(cursor, at) + words[i].Surface.Length;
        }
        var output = new List<Token>();
        for (int i = 0; i < words.Count; i++)
        {
            var word = words[i]; var entry = entries[i];
            bool hasNext = i + 1 < words.Count;
            string next = hasNext ? words[i + 1].Surface : "";
            var nextEntry = hasNext ? entries[i + 1] : null;
            Token? previous = output.Count > 0 ? output[output.Count - 1] : null;
            if (word.Surface == "'" && hasNext && Ascii(next) && next.Any(char.IsLetter) && !whitespace[i] && !whitespace[i + 1])
            {
                if (previous is not null) previous.Space = false;
                output.Add(new Token { Reading = "'" });
                continue;
            }
            string reading;
            if (_exceptions.TryGetValue(word.Surface, out var special)) reading = special;
            else if (Ascii(word.Surface) || Digits(word.Surface)) reading = word.Surface;
            else if (word.Surface == "€" || word.Surface == "¥") reading = "";
            else if (word.Token >= 0) reading = entry.Reading;
            else if (word.Surface.All(c => c >= 'ぁ' && c <= 'ヿ')) reading = MapKana(word.Surface);
            else reading = new string('?', word.Surface.Length);
            if (reading.StartsWith("\0CutletKeyError:", StringComparison.Ordinal))
                throw new ArgumentException($"Upstream Cutlet cannot romanize dictionary entry '{word.Surface}': {reading.Substring(16)}");
            if (reading.Length > 0 && previous is not null && previous.Reading.EndsWith("っ", StringComparison.Ordinal))
                previous.Reading = previous.Reading.Substring(0, previous.Reading.Length - 1) + reading[0];
            var token = new Token { Reading = reading };
            if ("「『".Contains(word.Surface) || reading == "(" || reading == "[")
            {
                if (previous is not null) previous.Space = true;
                output.Add(token); continue;
            }
            if (reading == "/") { output.Add(token); continue; }
            if (Ascii(word.Surface) && hasNext && Ascii(next))
            {
                output.Add(new Token { Reading = word.Surface, Space = whitespace[i + 1] }); continue;
            }
            output.Add(token);
            if (entry.Pos1 == "接頭辞" || nextEntry?.Pos1 == "補助記号" || nextEntry?.Pos1 == "接尾辞" || next == ","
                || entry.Foreign && reading.EndsWith("-", StringComparison.Ordinal) || nextEntry?.Pos2 == "接続助詞"
                || Digits(word.Surface) && Digits(next)
                || hasNext && (entry.Pos1 == "動詞" || entry.Pos1 == "助動詞" || entry.Pos1 == "形容詞") && nextEntry?.Pos1 == "助動詞" && next != "です") continue;
            token.Space = true;
        }
        return string.Concat(output.Select(t => t.Reading.Replace("っ", "") + (t.Space ? " " : ""))).Trim().ToLowerInvariant();
    }
}
