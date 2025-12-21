using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AiDotNet.Helpers
{
    /// <summary>
    /// Provides text processing utilities for splitting and tokenizing text.
    /// </summary>
    public static class TextProcessingHelper
    {
        /// <summary>
        /// Splits text into sentences based on common sentence-ending punctuation.
        /// Handles periods, exclamation marks, and question marks followed by spaces or newlines.
        /// </summary>
        /// <param name="text">The text to split into sentences.</param>
        /// <returns>A list of sentences extracted from the text.</returns>
        public static List<string> SplitIntoSentences(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<string>();

            var sentences = new List<string>();
            var sentenceEndings = new[] { ". ", "! ", "? ", ".\n", "!\n", "?\n" };
            var currentSentence = new StringBuilder();
            var maxEndingLength = sentenceEndings.Max(e => e.Length);

            for (int i = 0; i < text.Length; i++)
            {
                currentSentence.Append(text[i]);

                // Only check for endings if we have enough characters
                if (currentSentence.Length >= maxEndingLength)
                {
                    var lastChars = text.Substring(Math.Max(0, i - maxEndingLength + 1), Math.Min(maxEndingLength, i + 1));
                    var matchedEnding = sentenceEndings.FirstOrDefault(ending => lastChars.EndsWith(ending));

                    if (matchedEnding != null)
                    {
                        sentences.Add(currentSentence.ToString().Trim());
                        currentSentence.Clear();
                    }
                }
            }

            if (currentSentence.Length > 0 && !string.IsNullOrWhiteSpace(currentSentence.ToString()))
            {
                sentences.Add(currentSentence.ToString().Trim());
            }

            return sentences;
        }

        /// <summary>
        /// Tokenizes text by splitting on whitespace and common punctuation marks.
        /// Converts text to lowercase and removes empty tokens.
        /// </summary>
        /// <param name="text">The text to tokenize.</param>
        /// <returns>A list of tokens extracted from the text.</returns>
        public static List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            return text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?' },
                       StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }
    }
}
