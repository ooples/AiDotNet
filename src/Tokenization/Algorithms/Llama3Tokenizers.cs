using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Newtonsoft.Json;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// The Llama 3 tokenizer (shared by Llama 3, 3.1 and 3.2): byte-level BPE over 128000 tokens with 280147 merges
    /// (<c>ignore_merges</c>), Llama 3's split pattern, and its 256 special tokens at 128000–128255; models built on it add
    /// their own tokens after those.
    /// </summary>
    internal static class Llama3Tokenizers
    {
        /// <summary>Llama 3's pre-tokenizer pattern.</summary>
        public const string Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        /// <summary><c>&lt;|begin_of_text|&gt;</c>, which the post-processor puts first.</summary>
        public const int BeginOfText = 128000;

        /// <summary>The first id after the special tokens.</summary>
        public const int FirstAddedId = 128256;

        private static readonly Lazy<(Dictionary<string, int> Vocabulary, Dictionary<string, int> Merges, string[] Specials)> Parts =
            new(LoadParts, isThreadSafe: true);

        private static readonly Lazy<ByteLevelBpeTokenizer> BaseInstance = new(() => Create(Array.Empty<(string, int, bool)>()), isThreadSafe: true);

        /// <summary>Llama 3's own tokenizer.</summary>
        public static ByteLevelBpeTokenizer Base => BaseInstance.Value;

        /// <summary>Llama 3's tokenizer with <paramref name="added"/> tokens (each id at least <see cref="FirstAddedId"/>).</summary>
        public static ByteLevelBpeTokenizer Create(IEnumerable<(string Content, int Id, bool Special)> added)
        {
            var parts = Parts.Value;
            var all = parts.Specials.Select((content, i) => (content, BeginOfText + i, true)).ToList();
            foreach (var token in added ?? throw new ArgumentNullException(nameof(added)))
            {
                if (token.Id < FirstAddedId) throw new ArgumentException($"Added token '{token.Content}' has id {token.Id}, inside Llama 3's own range.", nameof(added));
                all.Add(token);
            }
            return new ByteLevelBpeTokenizer(parts.Vocabulary, parts.Merges, all, Pattern, nfc: false, ignoreMerges: true);
        }

        private static string ReadResource(string name)
        {
            using var stream = typeof(Llama3Tokenizers).Assembly.GetManifestResourceStream(name)
                ?? throw new InvalidDataException($"The embedded resource {name} is missing.");
            using var gzip = new GZipStream(stream, CompressionMode.Decompress);
            using var reader = new StreamReader(gzip, System.Text.Encoding.UTF8);
            return reader.ReadToEnd();
        }

        private static (Dictionary<string, int>, Dictionary<string, int>, string[]) LoadParts()
        {
            var vocabulary = JsonConvert.DeserializeObject<Dictionary<string, int>>(ReadResource("AiDotNet.Tokenization.llama3-vocab.json.gz"))
                ?? throw new InvalidDataException("The Llama 3 vocabulary is empty.");
            var merges = new Dictionary<string, int>(StringComparer.Ordinal);
            int rank = 0;
            foreach (var line in ReadResource("AiDotNet.Tokenization.llama3-merges.txt.gz").Split('\n'))
            {
                if (line.Length == 0) continue;
                int space = line.IndexOf(' ');
                string key = line.Substring(0, space) + "\u0001" + line.Substring(space + 1);
                if (!merges.ContainsKey(key)) merges[key] = rank;
                rank++;
            }
            var specials = JsonConvert.DeserializeObject<string[]>(ReadResource("AiDotNet.Tokenization.llama3-special-tokens.json.gz"))
                ?? throw new InvalidDataException("The Llama 3 special tokens are missing.");
            if (vocabulary.Count != BeginOfText || specials.Length != FirstAddedId - BeginOfText)
                throw new InvalidDataException("The embedded Llama 3 tokenizer is incomplete.");
            return (new Dictionary<string, int>(vocabulary, StringComparer.Ordinal), merges, specials);
        }
    }
}
