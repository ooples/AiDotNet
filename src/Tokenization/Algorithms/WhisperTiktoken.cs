using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Whisper's multilingual text encoding (openai-whisper <c>whisper.tokenizer.get_encoding("multilingual")</c>): the
    /// released <c>multilingual.tiktoken</c> vocabulary (MIT, shipped with the library) and its special tokens.
    /// </summary>
    /// <remarks>
    /// The special tokens follow the vocabulary's 50,257 ranks in this order: <c>&lt;|endoftext|&gt;</c>,
    /// <c>&lt;|startoftranscript|&gt;</c>, one per language, <c>&lt;|translate|&gt;</c>, <c>&lt;|transcribe|&gt;</c>,
    /// <c>&lt;|startoflm|&gt;</c>, <c>&lt;|startofprev|&gt;</c>, <c>&lt;|nospeech|&gt;</c>, <c>&lt;|notimestamps|&gt;</c>
    /// and the 1,501 timestamps <c>&lt;|0.00|&gt;</c> … <c>&lt;|30.00|&gt;</c>. Every id after the language block
    /// therefore depends on the language count: 99 for Whisper up to large-v2, 100 (Cantonese added) for large-v3 and
    /// for CosyVoice's text tokenizer.
    /// </remarks>
    internal static class WhisperTiktoken
    {
        private const string Resource = "AiDotNet.Tokenization.whisper-multilingual.tiktoken.gz";

        /// <summary>Whisper's language codes in token order (<c>whisper.tokenizer.LANGUAGES</c>).</summary>
        public static readonly IReadOnlyList<string> LanguageCodes = new[]
        {
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
        };

        private static readonly Lazy<Dictionary<byte[], int>> Ranks = new(LoadRanks);
        private static readonly ConcurrentDictionary<int, TiktokenEncoding> Encodings = new();

        /// <summary>The encoding with the first <paramref name="numLanguages"/> language tokens (99 or 100 in released models).</summary>
        public static TiktokenEncoding Multilingual(int numLanguages = 99)
        {
            if (numLanguages < 1 || numLanguages > LanguageCodes.Count)
                throw new ArgumentOutOfRangeException(nameof(numLanguages), numLanguages, $"Whisper defines 1 … {LanguageCodes.Count} languages.");
            return Encodings.GetOrAdd(numLanguages, n => new TiktokenEncoding(Ranks.Value, SpecialTokens(n)));
        }

        /// <summary>The special tokens of the <paramref name="numLanguages"/>-language encoding, in id order.</summary>
        public static IReadOnlyList<string> SpecialTokens(int numLanguages)
        {
            var specials = new List<string> { "<|endoftext|>", "<|startoftranscript|>" };
            specials.AddRange(LanguageCodes.Take(numLanguages).Select(code => $"<|{code}|>"));
            specials.AddRange(new[] { "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>" });
            // f"<|{i * 0.02:.2f}|>". Each product is within 1e-15 of a multiple of 0.02, far from a rounding tie.
            for (int i = 0; i <= 1500; i++)
                specials.Add("<|" + (i * 0.02).ToString("0.00", CultureInfo.InvariantCulture) + "|>");
            return specials;
        }

        private static Dictionary<byte[], int> LoadRanks()
        {
            using var stream = typeof(WhisperTiktoken).Assembly.GetManifestResourceStream(Resource)
                ?? throw new System.IO.InvalidDataException($"The embedded resource {Resource} is missing.");
            using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
            using var reader = new System.IO.StreamReader(gzip, Encoding.ASCII);
            return TiktokenEncoding.ReadRanks(reader);
        }
    }
}
