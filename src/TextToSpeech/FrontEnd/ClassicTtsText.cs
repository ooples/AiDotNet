using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Explicit English token inventories for the classic acoustic models. Ids never depend on a character modulus.</summary>
internal static class ClassicTtsText
{
    private static readonly string[] Consonants = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".Split(' ');
    private static readonly string[] Vowels = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".Split(' ');
    internal static readonly string[] FastSpeechSymbols = new[] { "<pad>", "<eos>", "<unk>", " ", "!", "'", ",", "-", ".", ":", ";", "?" }
        .Concat(Consonants).Concat(Vowels).ToArray();
    internal static readonly string[] FastSpeech2Symbols = new[] { "<pad>", "<eos>", " ", ",", ".", "!", "?" }
        .Concat(Consonants).Concat(Vowels.SelectMany(v => new[] { v + "0", v + "1", v + "2" })).ToArray();
    internal static readonly string[] AlignSymbols = "_~ !'(),-.:;?abcdefghijklmnopqrstuvwxyz".Select(c => c.ToString()).ToArray();

    internal static int[] Phonemes(string text, bool stress, string[]? symbols, int vocabulary, int maximum)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must have pronounceable content.", nameof(text));
        symbols ??= stress ? FastSpeech2Symbols : FastSpeechSymbols;
        var phones = G2pEn.Default.Word(text);
        return Encode(phones.Select(p => !stress && p.Length > 1 && char.IsDigit(p[p.Length - 1])
            ? p.Substring(0, p.Length - 1) : p).Concat(new[] { "<eos>" }), symbols, vocabulary, maximum);
    }

    internal static int[] Characters(string text, string[]? symbols, int vocabulary, int maximum)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must have pronounceable content.", nameof(text));
        text = G2pEnNumbers.Normalize(text.ToLowerInvariant());
        text = new string(text.Normalize(NormalizationForm.FormD)
            .Where(c => CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark).ToArray());
        text = Regex.Replace(text, @"\s+", " ").Trim();
        // AlignTTS section 4.1 uses a leading space and sentence-final period.
        text = " " + text.TrimEnd('.') + ".";
        return Encode(text.Select(c => c.ToString()), symbols ?? AlignSymbols, vocabulary, maximum);
    }

    private static int[] Encode(IEnumerable<string> tokens, string[] symbols, int vocabulary, int maximum)
    {
        if (symbols.Distinct(StringComparer.Ordinal).Count() != symbols.Length)
            throw new ArgumentException("The text symbol inventory must contain unique symbols.", nameof(symbols));
        var ids = symbols.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i, StringComparer.Ordinal);
        var result = new List<int>();
        foreach (string token in tokens)
        {
            if (!ids.TryGetValue(token, out int id))
            {
                // g2p_en retains punctuation; inventories without that punctuation use a word boundary.
                if (token.Length == 1 && char.IsPunctuation(token[0]) && ids.TryGetValue(" ", out int space)) id = space;
                else throw new ArgumentException($"Text symbol '{token}' is not present in the model's training vocabulary.");
            }
            if (id >= vocabulary) throw new ArgumentException($"Text symbol '{token}' has id {id}, outside vocabulary size {vocabulary}. Supply the matching TextSymbols inventory.");
            result.Add(id);
            if (result.Count == maximum) break;
        }
        return result.ToArray();
    }
}
