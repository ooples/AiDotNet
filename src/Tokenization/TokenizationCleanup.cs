namespace AiDotNet.Tokenization;

/// <summary>Text clean-up shared by the decoders that follow Hugging Face's tokenizers.</summary>
internal static class TokenizationCleanup
{
    /// <summary>
    /// Hugging Face's <c>clean_up_tokenization</c>: the space before punctuation and English contractions removed
    /// (" ." ".", " ?" "?", " !" "!", " ," ",", " ' " "'", " n't" "n't", " 'm" "'m", " 's" "'s", " 've" "'ve", " 're" "'re").
    /// </summary>
    public static string CleanUpSpaces(string text)
    {
        foreach (var (from, to) in new[] { (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
                     (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re") })
            text = text.Replace(from, to);
        return text;
    }
}
