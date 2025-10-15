using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// Character-level tokenizer that treats each character as a token.
    /// Useful for character-level language models and languages with complex morphology.
    /// </summary>
    public class CharacterLevelTokenizer : TokenizerBase
    {
        private readonly bool _includePrintableAscii;
        private readonly bool _caseSensitive;
        private readonly HashSet<char> _allowedCharacters;

        /// <inheritdoc/>
        public override int MaxSequenceLength => 512;

        /// <summary>
        /// Initializes a new instance of the CharacterLevelTokenizer class
        /// </summary>
        /// <param name="includePrintableAscii">Include all printable ASCII characters</param>
        /// <param name="caseSensitive">Whether to treat uppercase and lowercase as different tokens</param>
        /// <param name="allowedCharacters">Optional set of allowed characters. If null, all characters are allowed</param>
        public CharacterLevelTokenizer(
            bool includePrintableAscii = true, 
            bool caseSensitive = true,
            HashSet<char>? allowedCharacters = null)
        {
            _includePrintableAscii = includePrintableAscii;
            _caseSensitive = caseSensitive;
            _allowedCharacters = allowedCharacters ?? new HashSet<char>();
        }

        /// <summary>
        /// Loads the character vocabulary
        /// </summary>
        protected override async Task LoadVocabularyAsync()
        {
            _vocabulary.Clear();
            _reverseVocabulary.Clear();

            int tokenId = 0;

            // Add special tokens first
            foreach (var specialToken in _specialTokens)
            {
                _vocabulary[specialToken.Key] = tokenId;
                _reverseVocabulary[tokenId] = specialToken.Key;
                tokenId++;
            }

            // Add printable ASCII characters if requested
            if (_includePrintableAscii)
            {
                for (char c = ' '; c <= '~'; c++)
                {
                    if (_allowedCharacters.Count == 0 || _allowedCharacters.Contains(c))
                    {
                        string charStr = c.ToString();
                        if (!_caseSensitive && char.IsLetter(c))
                        {
                            charStr = charStr.ToLowerInvariant();
                        }
                        
                        if (!_vocabulary.ContainsKey(charStr))
                        {
                            _vocabulary[charStr] = tokenId;
                            _reverseVocabulary[tokenId] = charStr;
                            tokenId++;
                        }
                    }
                }
            }

            // Add common unicode characters
            var commonUnicodeChars = new[] { 
                '\n', '\t', '\r', // Whitespace
                '€', '£', '¥', '©', '®', '™', // Common symbols
                'á', 'é', 'í', 'ó', 'ú', 'ñ', // Spanish
                'à', 'è', 'ì', 'ò', 'ù', // Italian/French
                'ä', 'ö', 'ü', 'ß', // German
                'ą', 'ę', 'ł', 'ń', 'ś', 'ź', 'ż', // Polish
                'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я' // Russian
            };

            foreach (var c in commonUnicodeChars)
            {
                if (_allowedCharacters.Count == 0 || _allowedCharacters.Contains(c))
                {
                    string charStr = c.ToString();
                    if (!_caseSensitive && char.IsLetter(c))
                    {
                        charStr = charStr.ToLowerInvariant();
                    }
                    
                    if (!_vocabulary.ContainsKey(charStr))
                    {
                        _vocabulary[charStr] = tokenId;
                        _reverseVocabulary[tokenId] = charStr;
                        tokenId++;
                    }
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Tokenizes text into individual characters
        /// </summary>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            var tokens = new List<string>();
            
            if (!_caseSensitive)
            {
                text = text.ToLowerInvariant();
            }

            foreach (char c in text)
            {
                if (_allowedCharacters.Count == 0 || _allowedCharacters.Contains(c))
                {
                    tokens.Add(c.ToString());
                }
                else
                {
                    // Replace unknown characters with a special unknown character token
                    tokens.Add("[UNK_CHAR]");
                }
            }

            return await Task.FromResult(tokens);
        }

        /// <summary>
        /// Post-processes tokens by concatenating characters
        /// </summary>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var sb = new StringBuilder();
            
            foreach (var token in tokens)
            {
                // Skip unknown character markers
                if (token != "[UNK_CHAR]")
                {
                    sb.Append(token);
                }
            }

            return await Task.FromResult(sb.ToString());
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            return await TokenizeInternalAsync(text);
        }

        /// <summary>
        /// Creates a character-level tokenizer with digits and letters only
        /// </summary>
        public static CharacterLevelTokenizer CreateAlphanumeric(bool caseSensitive = false)
        {
            var allowedChars = new HashSet<char>();
            
            // Add digits
            for (char c = '0'; c <= '9'; c++)
                allowedChars.Add(c);
            
            // Add letters
            for (char c = 'a'; c <= 'z'; c++)
            {
                allowedChars.Add(c);
                if (caseSensitive)
                    allowedChars.Add(char.ToUpper(c));
            }
            
            // Add space
            allowedChars.Add(' ');
            
            return new CharacterLevelTokenizer(false, caseSensitive, allowedChars);
        }

        /// <summary>
        /// Creates a character-level tokenizer for DNA sequences
        /// </summary>
        public static CharacterLevelTokenizer CreateDNATokenizer()
        {
            return new CharacterLevelTokenizer(false, true, new HashSet<char> { 'A', 'T', 'G', 'C', 'N' });
        }

        /// <summary>
        /// Creates a character-level tokenizer for protein sequences
        /// </summary>
        public static CharacterLevelTokenizer CreateProteinTokenizer()
        {
            var aminoAcids = new HashSet<char> { 
                'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
            };
            return new CharacterLevelTokenizer(false, true, aminoAcids);
        }
    }
}