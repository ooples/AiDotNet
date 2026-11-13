using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// An English grapheme-to-phoneme front end that writes text as espeak-ng's American English phonemes, in the symbol
/// sequence phonemizer produces for the TTS models trained on them (Pheme, VALL-E, and others built on USLM's symbol
/// table).
/// </summary>
/// <remarks>
/// <para>
/// Words are looked up in the CMU Pronouncing Dictionary; words it does not list go through the letter-to-sound rules
/// of NRL Report 7948 (Elovitz et al. 1976). The ARPAbet pronunciation is then written in espeak's symbols and with
/// espeak's conventions (<see cref="EspeakArpabet"/>), function words take espeak's weak forms, numbers are read as
/// espeak reads them, and across word boundaries "the" and "to" change before a vowel, a final r links to a following
/// vowel, and a final t flaps before an attached "of" or "a".
/// </para>
/// <para>
/// The output is the list phonemizer's espeak backend gives with the separators Pheme uses (phone "|", word "_"),
/// split as Pheme's <c>TextTokenizer.to_list</c> splits it: one entry per phone, "_" between words, and punctuation
/// marks as their own entries after the word they follow.
/// </para>
/// <para>
/// <b>How close it is to espeak.</b> Pronunciations come from a different dictionary than espeak's, so they differ where
/// the two dictionaries do. Measured against espeak-ng 1.52 (<c>tools/reference-data/espeak_g2p_reference.py</c>), the
/// phone error rate is 2.98 % on the 1132 CMU ARCTIC prompts its conventions were checked on, and 3.45 % on held-out
/// text (chapters II–IV of <i>Alice's Adventures in Wonderland</i>). Most of the remainder is espeak's sentence-level
/// weak forms of function words and individual dictionary differences.
/// </para>
/// <para><b>For Beginners:</b> Speech models read sounds, not letters: "though" and "tough" end alike in writing and
/// differently in speech. This turns English text into the list of sounds those models were trained on.</para>
/// </remarks>
public sealed class EnglishG2P
{
    /// <summary>The symbol written between words.</summary>
    public const string WordSeparator = "_";

    private static readonly System.Lazy<EnglishG2P> SharedInstance = new(() => new EnglishG2P());

    /// <summary>phonemizer's default punctuation marks, kept as symbols of their own.</summary>
    private static readonly HashSet<string> Punctuation = new(System.StringComparer.Ordinal)
    {
        ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", "(", ")", "{", "}", "[", "]",
    };

    private static readonly Dictionary<string, string[]> FunctionWords = new(System.StringComparer.Ordinal)
    {
        ["a"] = new[] { "ɐ" }, ["an"] = new[] { "ɐ", "n" }, ["the"] = new[] { "ð", "ə" }, ["to"] = new[] { "t", "ə" },
        ["and"] = new[] { "æ", "n", "d" }, ["was"] = new[] { "w", "ʌ", "z" }, ["of"] = new[] { "ʌ", "v" },
        ["from"] = new[] { "f", "ɹ", "ʌ", "m" }, ["than"] = new[] { "ð", "ɐ", "n" },
        ["into"] = new[] { "ɪ", "n", "t", "ʊ" }, ["does"] = new[] { "d", "ʌ", "z" }, ["her"] = new[] { "h", "ɜː" },
        ["are"] = new[] { "ɑːɹ" }, ["or"] = new[] { "ɔːɹ" }, ["for"] = new[] { "f", "ɔːɹ" },
        ["at"] = new[] { "æ", "t" }, ["his"] = new[] { "h", "ɪ", "z" }, ["them"] = new[] { "ð", "ɛ", "m" },
        ["can"] = new[] { "k", "æ", "n" }, ["have"] = new[] { "h", "æ", "v" }, ["has"] = new[] { "h", "æ", "z" },
        ["had"] = new[] { "h", "æ", "d" }, ["that"] = new[] { "ð", "æ", "t" }, ["some"] = new[] { "s", "ʌ", "m" },
        ["on"] = new[] { "ɔ", "n" }, ["your"] = new[] { "j", "ʊɹ" }, ["you're"] = new[] { "j", "ʊɹ" },
        ["could"] = new[] { "k", "ʊ", "d" }, ["would"] = new[] { "w", "ʊ", "d" }, ["should"] = new[] { "ʃ", "ʊ", "d" },
        ["with"] = new[] { "w", "ɪ", "ð" },
        // A number word whose dictionary vowel espeak writes differently.
        ["hundred"] = new[] { "h", "ʌ", "n", "d", "ɹ", "ɪ", "d" },
    };

    private static readonly Regex DigitGroupComma = new(@"(?<=\d),(?=\d{3})", RegexOptions.CultureInvariant);
    private static readonly Regex Decimal = new(@"(\d+)\.(\d+)", RegexOptions.CultureInvariant);
    private static readonly Regex Dollars = new(@"\$(\d+)", RegexOptions.CultureInvariant);
    private static readonly Regex Percent = new(@"(\d+)%", RegexOptions.CultureInvariant);
    private static readonly Regex Token = new(@"[A-Za-z']+(?:-[A-Za-z']+)*|\d+(?:st|nd|rd|th)?|[^\sA-Za-z'\d]",
        RegexOptions.CultureInvariant);
    private static readonly Regex OrdinalToken = new(@"^(\d+)(st|nd|rd|th)$",
        RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
    private static readonly Regex DigitsToken = new(@"^\d+$", RegexOptions.CultureInvariant);

    /// <summary>A shared instance (the front end holds no per-call state).</summary>
    public static EnglishG2P Default => SharedInstance.Value;

    /// <summary>The phone, word-separator and punctuation symbols of <paramref name="text"/>.</summary>
    public IReadOnlyList<string> Phonemize(string text)
    {
        if (text is null) throw new System.ArgumentNullException(nameof(text));
        var words = new List<(List<string> Phones, List<string> Punctuation)>();
        foreach (var token in Tokenize(text))
        {
            if (Punctuation.Contains(token))
            {
                if (words.Count > 0) words[words.Count - 1].Punctuation.Add(token);
                continue;
            }
            if (!ContainsLetterOrDigit(token)) continue;
            foreach (var piece in Expand(token))
            {
                // The parts of a hyphenated word (or of a number espeak writes as one word) form one word, so a final r
                // links to a vowel-initial part as it does between words.
                var phones = new List<string>();
                foreach (var part in piece.ToLowerInvariant().Trim('\'').Split('+', '-'))
                {
                    if (part.Length == 0) continue;
                    var partPhones = WordPhones(part);
                    if (StartsWithVowel(partPhones)) LinkR(phones);
                    phones.AddRange(partPhones);
                }
                words.Add((phones, new List<string>()));
            }
        }
        ApplyCrossWordConventions(words);

        var symbols = new List<string>();
        var kept = words.FindAll(w => w.Phones.Count > 0 || w.Punctuation.Count > 0);
        for (int k = 0; k < kept.Count; k++)
        {
            symbols.AddRange(kept[k].Phones);
            symbols.AddRange(kept[k].Punctuation);
            if (k < kept.Count - 1) symbols.Add(WordSeparator);
        }
        return symbols;
    }

    /// <summary>
    /// The phonemes as phonemizer writes them: phones joined by "|" within a word and words joined by "_", with
    /// punctuation after the word it follows (for example "h|ə|l|oʊ_w|ɜː|l|d.").
    /// </summary>
    public string PhonemizeToString(string text)
    {
        var builder = new StringBuilder();
        bool startOfWord = true;
        foreach (var symbol in Phonemize(text))
        {
            if (symbol == WordSeparator)
            {
                builder.Append(WordSeparator);
                startOfWord = true;
                continue;
            }
            if (!startOfWord && !Punctuation.Contains(symbol)) builder.Append('|');
            builder.Append(symbol);
            startOfWord = false;
        }
        return builder.ToString();
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        text = DigitGroupComma.Replace(text, string.Empty);
        text = Decimal.Replace(text, m => m.Groups[1].Value + " point " + string.Join(" ", m.Groups[2].Value.ToCharArray()));
        text = Dollars.Replace(text, "dollar $1");
        text = Percent.Replace(text, "$1 percent");
        foreach (Match match in Token.Matches(text))
            yield return match.Value;
    }

    private static bool ContainsLetterOrDigit(string token)
    {
        foreach (char c in token)
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || char.IsDigit(c)) return true;
        return false;
    }

    private static IEnumerable<string> Expand(string token)
    {
        var ordinal = OrdinalToken.Match(token);
        if (ordinal.Success) return EnglishNumberReader.Ordinal(ordinal.Groups[1].Value);
        if (DigitsToken.IsMatch(token)) return EnglishNumberReader.Cardinal(token);
        return new[] { token };
    }

    private static List<string> WordPhones(string word)
    {
        if (FunctionWords.TryGetValue(word, out var weak)) return new List<string>(weak);
        string[]? arpa = CmuPronouncingDictionary.Lookup(word);
        if (arpa is not null)
        {
            arpa = SpellingGuided(arpa, word);
        }
        else if (word.EndsWith("'s", System.StringComparison.Ordinal)
                 && CmuPronouncingDictionary.Lookup(word.Substring(0, word.Length - 2)) is { } stem)
        {
            arpa = new string[stem.Length + 1];
            stem.CopyTo(arpa, 0);
            arpa[stem.Length] = "Z";
        }
        arpa ??= NrlLetterToSound.Translate(word);
        return EspeakArpabet.ToIpa(arpa, word);
    }

    // Where the dictionary has a reduced AH0 and the spelling-driven NRL rules read the same vowel as IH (a written i or
    // e), espeak writes the reduced vowel as ɪ; aligned only when both give the same number of vowels.
    private static string[] SpellingGuided(string[] arpa, string word)
    {
        var guide = new List<string>();
        foreach (var phone in NrlLetterToSound.Translate(word))
            if (EspeakArpabet.IsVowel(phone)) guide.Add(phone);
        var vowelPositions = new List<int>();
        for (int k = 0; k < arpa.Length; k++)
            if (EspeakArpabet.IsVowel(arpa[k])) vowelPositions.Add(k);
        if (guide.Count != vowelPositions.Count) return arpa;
        string[]? result = null;
        for (int v = 0; v < vowelPositions.Count; v++)
        {
            int k = vowelPositions[v];
            if (arpa[k] == "AH0" && EspeakArpabet.Split(guide[v]).Base == "IH" && k + 1 < arpa.Length)
            {
                result ??= (string[])arpa.Clone();
                result[k] = "IH0";
            }
        }
        return result ?? arpa;
    }

    private static bool StartsWithVowel(List<string> phones) =>
        phones.Count > 0 && "aeiouæɑɐɔəɚɛɜɪʊʌ".IndexOf(phones[0][0]) >= 0;

    private static bool IsExactly(List<string> phones, params string[] expected)
    {
        if (phones.Count != expected.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (phones[i] != expected[i]) return false;
        return true;
    }

    private static void ApplyCrossWordConventions(List<(List<string> Phones, List<string> Punctuation)> words)
    {
        for (int k = 0; k < words.Count - 1; k++)
        {
            var current = words[k];
            var next = words[k + 1].Phones;
            bool joined = current.Punctuation.Count == 0;
            bool beforeVowel = StartsWithVowel(next);
            if (joined && beforeVowel && IsExactly(current.Phones, "ð", "ə"))
                current.Phones[1] = "ɪ";
            if (joined && beforeVowel && IsExactly(current.Phones, "t", "ə"))
                current.Phones[1] = "ʊ";
            // A final t after a vowel flaps before an attached "of" or "a", which espeak writes as part of the word.
            bool attachedOf = IsExactly(next, "ʌ", "v");
            if (joined && (attachedOf || IsExactly(next, "ɐ")) && current.Phones.Count > 1
                && current.Phones[current.Phones.Count - 1] == "t"
                && "aeiouæɑɐɔəɚɛɜɪʊʌ".IndexOf(current.Phones[current.Phones.Count - 2][0]) >= 0)
            {
                current.Phones[current.Phones.Count - 1] = "ɾ";
                current.Phones.Add("ə");
                if (attachedOf) current.Phones.Add("v");
                next.Clear();
            }
            if (joined && beforeVowel) LinkR(current.Phones);
        }
    }

    // Linking r before a vowel: ɚ and ɜː gain an ɹ, and an r-coloured vowel (ɑːɹ, ɪɹ, ...) splits into vowel and ɹ.
    private static void LinkR(List<string> phones)
    {
        if (phones.Count == 0) return;
        string last = phones[phones.Count - 1];
        if (last == "ɚ" || last == "ɜː")
        {
            phones.Add("ɹ");
        }
        else if (last.Length > 1 && last.EndsWith("ɹ", System.StringComparison.Ordinal))
        {
            phones[phones.Count - 1] = last.Substring(0, last.Length - 1);
            phones.Add("ɹ");
        }
    }
}
