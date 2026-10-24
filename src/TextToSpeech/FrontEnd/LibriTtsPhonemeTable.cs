namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The phoneme table of the espeak-ng front end that lifeiteng/vall-e's LibriTTS recipe builds
/// (<c>unique_text_tokens.k2symbols</c>), as USLM publishes it (fnlp/USLM, USLM_libritts): espeak-ng's American English
/// symbols, the word separator <c>_</c>, punctuation and k2's <c>&lt;eps&gt;</c>. VALL-E's reproduction and Pheme's
/// checkpoints both read text through this table; <see cref="EnglishG2P"/> produces its symbols.
/// </summary>
internal static class LibriTtsPhonemeTable
{
    /// <summary>The table's symbols, in the file's order.</summary>
    public static readonly string[] Symbols =
    {
        "<eps>", "!", "\"", "(", ")", ",", ".", ":", ";", "?", "_", "aɪ", "aɪə", "aɪɚ", "aɪʊ", "aɪʊɹ", "aʊ", "b", "d",
        "dʒ", "e", "enus", "es", "eɪ", "f", "fr", "h", "i", "iə", "iː", "j", "k", "l", "m", "n", "nʲ", "oʊ", "oː", "oːɹ",
        "p", "r", "s", "t", "tʃ", "uː", "v", "w", "x", "z", "æ", "ç", "ð", "ø", "ŋ", "ɐ", "ɑ", "ɑː", "ɑːɹ", "ɔ", "ɔɪ",
        "ɔː", "ɔːɹ", "ə", "əl", "ɚ", "ɛ", "ɛɹ", "ɛː", "ɜː", "ɡ", "ɡʲ", "ɣ", "ɪ", "ɪɹ", "ɫ", "ɬ", "ɲ", "ɹ", "ɾ", "ʃ",
        "ʊ", "ʊɹ", "ʌ", "ʒ", "ʔ", "̃", "̩", "θ", "ᵻ", "—",
    };
}
