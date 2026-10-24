using System.Collections.Generic;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// MeloTTS's symbol set and language tables (myshell-ai/MeloTTS <c>melo/text/symbols.py</c>): 219 symbols shared by its
/// languages, 16 tones (each language's tones offset by its start) and 8 language ids; and its <c>add_blank</c>.
/// </summary>
public static class MeloSymbols
{
    /// <summary>The symbols, in id order.</summary>
    public static IReadOnlyList<string> Symbols { get; } = new[]
    {
        "_", "\"", "(", ")", "*", "/", ":", "AA", "E", "EE", "En", "N",
        "OO", "Q", "V", "[", "\\", "]", "^", "a", "a:", "aa", "ae", "ah",
        "ai", "an", "ang", "ao", "aw", "ay", "b", "by", "c", "ch", "d", "dh",
        "dy", "e", "e:", "eh", "ei", "en", "eng", "er", "ey", "f", "g", "gy",
        "h", "hh", "hy", "i", "i0", "i:", "ia", "ian", "iang", "iao", "ie", "ih",
        "in", "ing", "iong", "ir", "iu", "iy", "j", "jh", "k", "ky", "l", "m",
        "my", "n", "ng", "ny", "o", "o:", "ong", "ou", "ow", "oy", "p", "py",
        "q", "r", "ry", "s", "sh", "t", "th", "ts", "ty", "u", "u:", "ua",
        "uai", "uan", "uang", "uh", "ui", "un", "uo", "uw", "v", "van", "ve", "vn",
        "w", "x", "y", "z", "zh", "zy", "~", "æ", "ç", "ð", "ø", "ŋ",
        "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ɕ", "ə", "ɛ", "ɜ", "ɡ", "ɣ", "ɥ",
        "ɦ", "ɪ", "ɫ", "ɬ", "ɭ", "ɯ", "ɲ", "ɵ", "ɸ", "ɹ", "ɾ", "ʁ",
        "ʃ", "ʊ", "ʌ", "ʎ", "ʏ", "ʑ", "ʒ", "ʝ", "ʲ", "ˈ", "ˌ", "ː",
        "̃", "̩", "β", "θ", "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄄ", "ᄅ", "ᄆ", "ᄇ",
        "ᄈ", "ᄉ", "ᄊ", "ᄋ", "ᄌ", "ᄍ", "ᄎ", "ᄏ", "ᄐ", "ᄑ", "ᄒ", "ᅡ",
        "ᅢ", "ᅣ", "ᅤ", "ᅥ", "ᅦ", "ᅧ", "ᅨ", "ᅩ", "ᅪ", "ᅫ", "ᅬ", "ᅭ",
        "ᅮ", "ᅯ", "ᅰ", "ᅱ", "ᅲ", "ᅳ", "ᅴ", "ᅵ", "ᆨ", "ᆫ", "ᆮ", "ᆯ",
        "ᆷ", "ᆸ", "ᆼ", "ㄸ", "!", "?", "…", ",", ".", "'", "-", "¿",
        "¡", "SP", "UNK",
    };

    private static readonly Dictionary<string, int> Ids = BuildIds();

    private static Dictionary<string, int> BuildIds()
    {
        var ids = new Dictionary<string, int>(System.StringComparer.Ordinal);
        for (int i = 0; i < Symbols.Count; i++) ids[Symbols[i]] = i;
        return ids;
    }

    /// <summary>The number of tones (6 Chinese, 1 Japanese, 4 English, 1 Korean, 1 Spanish, 1 French, plus the rest).</summary>
    public const int ToneCount = 16;

    /// <summary>The number of language ids.</summary>
    public const int LanguageCount = 8;

    /// <summary>Whether <paramref name="symbol"/> is one of MeloTTS's symbols.</summary>
    public static bool Contains(string symbol) => Ids.ContainsKey(symbol);

    /// <summary>The id of each language (<c>language_id_map</c>).</summary>
    public static int LanguageId(string language) => language switch
    {
        "ZH" => 0, "JP" => 1, "EN" => 2, "ZH_MIX_EN" => 3, "KR" => 4, "ES" or "SP" => 5, "FR" => 6,
        _ => throw new System.ArgumentException($"MeloTTS has no language {language}.", nameof(language)),
    };

    /// <summary>The first tone id of each language (<c>language_tone_start_map</c>).</summary>
    public static int ToneStart(string language) => language switch
    {
        "ZH" or "ZH_MIX_EN" => 0, "JP" => 6, "EN" => 7, "KR" => 11, "ES" or "SP" => 12, "FR" => 13,
        _ => throw new System.ArgumentException($"MeloTTS has no language {language}.", nameof(language)),
    };

    /// <summary><c>cleaned_text_to_sequence</c>: symbol ids, tones offset to the language, and the language id per symbol.</summary>
    public static (int[] Phones, int[] Tones, int[] Languages) ToSequence(IReadOnlyList<string> phones, IReadOnlyList<int> tones, string language)
    {
        var ids = new int[phones.Count];
        for (int i = 0; i < phones.Count; i++)
            ids[i] = Ids.TryGetValue(phones[i], out int id) ? id : throw new KeyNotFoundException($"'{phones[i]}' is not a MeloTTS symbol.");
        int start = ToneStart(language), lang = LanguageId(language);
        var t = new int[tones.Count];
        for (int i = 0; i < tones.Count; i++) t[i] = tones[i] + start;
        var languages = new int[phones.Count];
        for (int i = 0; i < languages.Length; i++) languages[i] = lang;
        return (ids, t, languages);
    }

    /// <summary><c>commons.intersperse(sequence, 0)</c>: a 0 before, between and after the items.</summary>
    public static int[] Intersperse(IReadOnlyList<int> sequence)
    {
        var result = new int[2 * sequence.Count + 1];
        for (int i = 0; i < sequence.Count; i++) result[2 * i + 1] = sequence[i];
        return result;
    }
}
