using System.Collections.Generic;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Renders an ARPAbet pronunciation in the IPA symbols espeak-ng writes for American English (its <c>en-us</c> voice,
/// unstressed, without ties), following espeak's own conventions for the sounds a dictionary transcription leaves
/// implicit.
/// </summary>
/// <remarks>
/// Each convention below was checked against espeak-ng 1.52's output for words that show it:
/// a vowel and a following R that no vowel follows become one r-coloured vowel (<c>ɑːɹ</c>, <c>ɔːɹ</c>, <c>ɪɹ</c>,
/// <c>ɛɹ</c>, <c>ʊɹ</c>); unstressed AH is <c>ə</c> (<c>ɐ</c> word-initially before a written "a") and stressed AH
/// <c>ʌ</c>; ER is <c>ɚ</c> unstressed and <c>ɜː</c> stressed, with a linking <c>ɹ</c> before a vowel; unstressed IY
/// is <c>i</c> at the end of a word and <c>ɪ</c> inside it; the inflections -ed and -es after t, d or a sibilant, the
/// -ity ending and the be-, re- and de- prefixes reduce to <c>ᵻ</c>; a final AH L is the syllabic <c>əl</c>; t between
/// a vowel and an unstressed vowel is the flap <c>ɾ</c> (flapping d as well measured worse: espeak keeps most of them),
/// and a final -tten is a glottal stop and a syllabic n
/// (<c>ʔ n ̩</c>); n before k or g is <c>ŋ</c>; stressed AO before NG, S, F, TH or N is <c>ɔ</c> and before G <c>ɑː</c>.
/// </remarks>
internal static class EspeakArpabet
{
    private static readonly Dictionary<string, string> Base = new(System.StringComparer.Ordinal)
    {
        ["AA"] = "ɑː", ["AE"] = "æ", ["AO"] = "ɔː", ["AW"] = "aʊ", ["AY"] = "aɪ", ["EH"] = "ɛ", ["EY"] = "eɪ",
        ["IH"] = "ɪ", ["IY"] = "iː", ["OW"] = "oʊ", ["OY"] = "ɔɪ", ["UH"] = "ʊ", ["UW"] = "uː", ["B"] = "b",
        ["CH"] = "tʃ", ["D"] = "d", ["DH"] = "ð", ["F"] = "f", ["G"] = "ɡ", ["HH"] = "h", ["JH"] = "dʒ", ["K"] = "k",
        ["L"] = "l", ["M"] = "m", ["N"] = "n", ["NG"] = "ŋ", ["NX"] = "ŋ", ["P"] = "p", ["R"] = "ɹ", ["S"] = "s",
        ["SH"] = "ʃ", ["T"] = "t", ["TH"] = "θ", ["V"] = "v", ["W"] = "w", ["WH"] = "w", ["Y"] = "j", ["Z"] = "z",
        ["ZH"] = "ʒ",
    };

    private static readonly Dictionary<string, string> RColoured = new(System.StringComparer.Ordinal)
    {
        ["AA"] = "ɑːɹ", ["AO"] = "ɔːɹ", ["IH"] = "ɪɹ", ["IY"] = "ɪɹ", ["EH"] = "ɛɹ", ["EY"] = "ɛɹ", ["AE"] = "ɛɹ",
        ["UH"] = "ʊɹ", ["UW"] = "ʊɹ", ["OW"] = "ɔːɹ",
    };

    private static readonly HashSet<string> VowelBases = new(System.StringComparer.Ordinal)
    {
        "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    };

    private static readonly HashSet<string> SibilantsAndAlveolarStops = new(System.StringComparer.Ordinal)
    {
        "T", "D", "S", "Z", "SH", "ZH", "CH", "JH",
    };

    /// <summary>Splits an ARPAbet phone into its base and its stress digit ("" when it has none).</summary>
    internal static (string Base, string Stress) Split(string phone)
    {
        int end = phone.Length;
        while (end > 0 && char.IsDigit(phone[end - 1])) end--;
        return (phone.Substring(0, end), phone.Substring(end));
    }

    /// <summary>Whether an ARPAbet phone is a vowel.</summary>
    internal static bool IsVowel(string phone) => VowelBases.Contains(Split(phone).Base);

    private static bool IsUnstressed(string stress) => stress.Length == 0 || stress == "0";

    /// <summary>The espeak-style IPA phones of one word's ARPAbet pronunciation; <paramref name="word"/> is its
    /// lower-case spelling, which the a-, ex-, be-, re- and de- conventions read.</summary>
    public static List<string> ToIpa(IReadOnlyList<string> arpa, string word)
    {
        var output = new List<string>(arpa.Count + 2);
        int n = arpa.Count;
        int i = 0;
        while (i < n)
        {
            var (phone, stress) = Split(arpa[i]);
            var (next, nextStress) = i + 1 < n ? Split(arpa[i + 1]) : ("", "");
            string? after = i + 2 < n ? arpa[i + 2] : null;
            bool unstressed = IsUnstressed(stress);
            bool nextIsReducedSchwa = (next == "AH" || next == "AX") && IsUnstressed(nextStress);

            if (RColoured.TryGetValue(phone, out var rColoured) && next == "R" && !(after is not null && IsVowel(after)))
            {
                output.Add(rColoured);
                i += 2;
                continue;
            }
            if ((phone == "UH" || phone == "UW") && next == "R" && after is not null && IsVowel(after))
            {
                output.Add("ʊɹ");
                output.Add("ɹ");
                i += 2;
                continue;
            }
            if (phone == "AO" && !unstressed && (next == "NG" || next == "S" || next == "F" || next == "TH" || next == "N"))
            {
                output.Add("ɔ");
                i++;
                continue;
            }
            if (phone == "AO" && !unstressed && next == "G")
            {
                output.Add("ɑː");
                i++;
                continue;
            }
            if (phone == "AY" && next == "ER")
            {
                output.Add("aɪɚ");
                i += 2;
                continue;
            }
            if (phone == "AY" && nextIsReducedSchwa)
            {
                output.Add("aɪə");
                i += 2;
                continue;
            }
            if (phone == "IY" && nextIsReducedSchwa && (i + 2 == n || unstressed))
            {
                output.Add("iə");
                i += 2;
                continue;
            }
            // Syllabic l: an unstressed AH L at the end of the word, or before a final Z, D or IY0.
            if ((phone == "AH" || phone == "AX") && unstressed && next == "L"
                && (i + 2 == n
                    || (i + 3 == n && (Split(arpa[i + 2]).Base == "Z" || Split(arpa[i + 2]).Base == "D"))
                    || (i + 3 == n && arpa[i + 2] == "IY0")))
            {
                output.Add("əl");
                i += 2;
                continue;
            }
            // -tten: T AH0 N closing the word after a vowel is a glottal stop and a syllabic n.
            if (phone == "T" && i > 0 && IsVowel(arpa[i - 1]) && nextIsReducedSchwa
                && after is not null && Split(after).Base == "N" && i + 3 == n)
            {
                output.Add("ʔ");
                output.Add("n");
                output.Add("̩");
                i += 3;
                continue;
            }

            if (phone == "AH" || phone == "AX")
            {
                if (unstressed && i == 0 && word.StartsWith("a", System.StringComparison.Ordinal))
                    output.Add("ɐ");
                else
                    output.Add(unstressed || phone == "AX" ? "ə" : "ʌ");
            }
            else if (phone == "ER")
            {
                output.Add(stress == "0" ? "ɚ" : "ɜː");
                if (VowelBases.Contains(next)) output.Add("ɹ");
            }
            else if (phone == "IY" && stress == "0")
            {
                bool final = i + 1 == n || (i + 2 == n && (next == "Z" || next == "D"));
                output.Add(final ? "i" : "ɪ");
            }
            else if (phone == "IH" && unstressed && IsReducedInflection(arpa, i))
            {
                output.Add("ᵻ");
            }
            else if ((phone == "IH" || phone == "IY") && unstressed && i == 1 && n > 3
                     && (word.StartsWith("be", System.StringComparison.Ordinal)
                         || word.StartsWith("re", System.StringComparison.Ordinal)
                         || word.StartsWith("de", System.StringComparison.Ordinal)))
            {
                output.Add("ᵻ");
            }
            else if (phone == "IH" && unstressed && i == 0 && word.StartsWith("ex", System.StringComparison.Ordinal))
            {
                output.Add("ɛ");
            }
            else if (phone == "N" && (next == "K" || next == "G"))
            {
                output.Add("ŋ");
            }
            else if (phone == "T" && i > 0 && (IsVowel(arpa[i - 1]) || Split(arpa[i - 1]).Base == "R")
                     && VowelBases.Contains(next) && nextStress == "0")
            {
                output.Add("ɾ");
            }
            else
            {
                output.Add(Base.TryGetValue(phone, out var ipa)
                    ? ipa
                    : throw new System.ArgumentException($"'{arpa[i]}' is not an ARPAbet phone.", nameof(arpa)));
            }
            i++;
        }
        return output;
    }

    // -ed or -es after t, d or a sibilant (IH0 D / IH0 Z closing the word), or the -ity ending (IH0 T IY0).
    private static bool IsReducedInflection(IReadOnlyList<string> arpa, int i)
    {
        int remaining = arpa.Count - i - 1;
        if (remaining == 1 && i > 0 && SibilantsAndAlveolarStops.Contains(Split(arpa[i - 1]).Base))
        {
            string last = Split(arpa[i + 1]).Base;
            return last == "D" || last == "Z";
        }
        return remaining == 2 && Split(arpa[i + 1]).Base == "T" && Split(arpa[i + 2]).Base == "IY";
    }
}
