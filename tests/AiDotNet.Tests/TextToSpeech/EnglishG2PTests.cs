using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The English G2P writes text as espeak-ng's American English phonemes, in the symbols Pheme-style models read.
/// </summary>
/// <remarks>
/// <c>ReferenceData/espeak_arctic_phonemes.json</c> is espeak-ng 1.52's phonemization of the 1132 CMU ARCTIC prompts
/// through phonemizer with Pheme's settings (<c>tools/reference-data/espeak_g2p_reference.py</c>). The expected
/// sequences in the examples below are espeak's own output for those phrases.
/// </remarks>
public class EnglishG2PTests
{
    private static readonly HashSet<string> Punctuation = new(StringComparer.Ordinal)
    {
        ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", "(", ")", "{", "}", "[", "]",
    };

    /// <summary>USLM's unique_text_tokens.k2symbols (Pheme's phoneme table), without &lt;eps&gt;.</summary>
    private static readonly HashSet<string> PhemeSymbols = new(StringComparer.Ordinal)
    {
        "!", "\"", "(", ")", ",", ".", ":", ";", "?", "_", "aɪ", "aɪə", "aɪɚ", "aɪʊ", "aɪʊɹ", "aʊ", "b", "d", "dʒ", "e",
        "enus", "es", "eɪ", "f", "fr", "h", "i", "iə", "iː", "j", "k", "l", "m", "n", "nʲ", "oʊ", "oː", "oːɹ", "p", "r",
        "s", "t", "tʃ", "uː", "v", "w", "x", "z", "æ", "ç", "ð", "ø", "ŋ", "ɐ", "ɑ", "ɑː", "ɑːɹ", "ɔ", "ɔɪ", "ɔː", "ɔːɹ",
        "ə", "əl", "ɚ", "ɛ", "ɛɹ", "ɛː", "ɜː", "ɡ", "ɡʲ", "ɣ", "ɪ", "ɪɹ", "ɫ", "ɬ", "ɲ", "ɹ", "ɾ", "ʃ", "ʊ", "ʊɹ", "ʌ",
        "ʒ", "ʔ", "̃", "̩", "θ", "ᵻ", "—",
    };

    private static JObject Reference()
    {
        const string fileName = "espeak_arctic_phonemes.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static List<string> PhonesOnly(IEnumerable<string> symbols) =>
        symbols.Where(s => s != EnglishG2P.WordSeparator && !Punctuation.Contains(s)).ToList();

    private static int EditDistance(IReadOnlyList<string> a, IReadOnlyList<string> b)
    {
        var previous = Enumerable.Range(0, b.Count + 1).ToArray();
        for (int i = 1; i <= a.Count; i++)
        {
            var current = new int[b.Count + 1];
            current[0] = i;
            for (int j = 1; j <= b.Count; j++)
                current[j] = Math.Min(Math.Min(previous[j] + 1, current[j - 1] + 1),
                    previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            previous = current;
        }
        return previous[b.Count];
    }

    [Fact(Timeout = 120000)]
    public async Task PhoneErrorRate_AgainstEspeak_OnTheArcticPrompts()
    {
        await Task.Yield();
        var g2p = EnglishG2P.Default;
        long errors = 0, total = 0;
        foreach (var item in Reference()["items"]!)
        {
            var expected = PhonesOnly(item["phones"]!.Select(p => (string)p!));
            var actual = PhonesOnly(g2p.Phonemize((string)item["text"]!));
            errors += EditDistance(actual, expected);
            total += expected.Count;
        }
        double rate = (double)errors / total;
        // Measured 2.98 % (1052 of 35,347 phones); the bound fails on any regression beyond rounding.
        Assert.True(rate <= 0.0300, $"Phone error rate against espeak-ng is {rate:P2} ({errors} of {total} phones).");
    }

    [Fact(Timeout = 120000)]
    public async Task EverySymbol_IsInPhemesPhonemeTable()
    {
        await Task.Yield();
        var g2p = EnglishG2P.Default;
        var outside = Reference()["items"]!
            .SelectMany(item => g2p.Phonemize((string)item["text"]!))
            .Where(s => !PhemeSymbols.Contains(s)).Distinct().ToList();
        Assert.Empty(outside);
    }

    [Theory(Timeout = 120000)]
    [InlineData("Hello world.", "h ə l oʊ _ w ɜː l d .")]
    [InlineData("the apple", "ð ɪ _ æ p əl")]
    [InlineData("the dog", "ð ə _ d ɑː ɡ")]
    [InlineData("1908", "n aɪ n t iː n h ʌ n d ɹ ɪ d _ eɪ t")]
    [InlineData("29th", "t w ɛ n t i _ n aɪ n θ")]
    [InlineData("far away", "f ɑː ɹ _ ɐ w eɪ")]
    [InlineData("never ever", "n ɛ v ɚ ɹ _ ɛ v ɚ")]
    [InlineData("little", "l ɪ ɾ əl")]
    [InlineData("out of", "aʊ ɾ ə v")]
    [InlineData("forgotten", "f ɚ ɡ ɑː ʔ n ̩")]
    [InlineData("affected", "ɐ f ɛ k t ᵻ d")]
    [InlineData("absurdity", "ɐ b s ɜː d ᵻ ɾ i")]
    [InlineData("fire", "f aɪɚ")]
    [InlineData("idea", "aɪ d iə")]
    [InlineData("along", "ɐ l ɔ ŋ")]
    [InlineData("engaged", "ɛ ŋ ɡ eɪ dʒ d")]
    [InlineData("curious", "k j ʊɹ ɹ iə s")]
    [InlineData("to it", "t ʊ _ ɪ t")]
    [InlineData("$5", "d ɑː l ɚ _ f aɪ v")]
    [InlineData("50%", "f ɪ f t i _ p ɚ s ɛ n t")]
    [InlineData("3.5", "θ ɹ iː _ p ɔɪ n t _ f aɪ v")]
    [InlineData("ten-year-old", "t ɛ n j ɪ ɹ oʊ l d")]
    [InlineData("It's an animal!", "ɪ t s _ ɐ n _ æ n ɪ m əl !")]
    public async Task Phonemize_MatchesEspeak(string text, string expected)
    {
        await Task.Yield();
        Assert.Equal(expected, string.Join(" ", EnglishG2P.Default.Phonemize(text)));
    }

    [Fact(Timeout = 120000)]
    public async Task PhonemizeToString_UsesPhonemizersSeparators()
    {
        await Task.Yield();
        Assert.Equal("h|ə|l|oʊ_w|ɜː|l|d.", EnglishG2P.Default.PhonemizeToString("Hello world."));
    }

    /// <summary>Words the dictionary does not list go through the NRL rules (Elovitz et al. 1976).</summary>
    [Theory(Timeout = 120000)]
    [InlineData("zorbleck", "Z AO1 R B L EH1 K")]
    [InlineData("quixotry", "K W IH1 K S AA1 T R IY1")]
    [InlineData("snerdle", "S N ER1 D AX0 L")]
    [InlineData("xylophonics", "K S AY1 L AA1 F AA1 N IH1 K S")]
    public async Task UnlistedWords_UseTheNrlLetterToSoundRules(string word, string expected)
    {
        await Task.Yield();
        Assert.Null(CmuPronouncingDictionary.Lookup(word));
        Assert.Equal(expected, string.Join(" ", NrlLetterToSound.Translate(word)));
    }

    [Fact(Timeout = 120000)]
    public async Task VeryLargeNumbers_AreReadDigitByDigit_AndAnyDecimalDigitIsANumber()
    {
        await Task.Yield();
        Assert.Equal(new[] { "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "one", "two", "three" },
            EnglishNumberReader.Cardinal("1234567890123"));
        Assert.Equal(new[] { "forty", "two" }, EnglishNumberReader.Cardinal("٤٢"));   // Arabic-Indic digits
    }
}
