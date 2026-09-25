using System;
using System.Threading.Tasks;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics
{
    /// <summary>
    /// Verifies the OCR error-rate metrics against hand-computed values. Every expected number
    /// here is derived by hand from the metric definition, not from a previous run of this code.
    /// </summary>
    public class TextRecognitionMetricsTests
    {
        [Fact(Timeout = 60000)]
        public async Task LevenshteinDistance_KittenToSitting_IsThree()
        {
            await Task.Yield();

            // The textbook example: kitten -> sitten (substitute k/s), sitten -> sittin
            // (substitute e/i), sittin -> sitting (insert g).
            Assert.Equal(3, TextRecognitionMetrics.LevenshteinDistance("kitten", "sitting"));
        }

        [Fact(Timeout = 60000)]
        public async Task LevenshteinDistance_IsSymmetric()
        {
            await Task.Yield();

            Assert.Equal(
                TextRecognitionMetrics.LevenshteinDistance("recognition", "recogniton"),
                TextRecognitionMetrics.LevenshteinDistance("recogniton", "recognition"));
        }

        [Fact(Timeout = 60000)]
        public async Task LevenshteinDistance_AgainstEmpty_IsTheOtherLength()
        {
            await Task.Yield();

            Assert.Equal(3, TextRecognitionMetrics.LevenshteinDistance("", "abc"));
            Assert.Equal(3, TextRecognitionMetrics.LevenshteinDistance("abc", ""));
            Assert.Equal(0, TextRecognitionMetrics.LevenshteinDistance("", ""));
        }

        [Fact(Timeout = 60000)]
        public async Task LevenshteinDistance_TreatsNullAsEmpty()
        {
            await Task.Yield();

            Assert.Equal(3, TextRecognitionMetrics.LevenshteinDistance(null, "abc"));
            Assert.Equal(3, TextRecognitionMetrics.LevenshteinDistance("abc", null));
        }

        [Fact(Timeout = 60000)]
        public async Task CharacterErrorRate_ExactMatch_IsZero()
        {
            await Task.Yield();

            Assert.Equal(0.0, TextRecognitionMetrics.CharacterErrorRate("hello", "hello"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task CharacterErrorRate_OneSubstitutionInFive_IsOneFifth()
        {
            await Task.Yield();

            // hello -> hallo is a single substitution over a 5-character reference.
            Assert.Equal(0.2, TextRecognitionMetrics.CharacterErrorRate("hello", "hallo"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task CharacterErrorRate_CanExceedOne_WhenHypothesisIsMuchLonger()
        {
            await Task.Yield();

            // Reference "a" (length 1), hypothesis "abcd" needs 3 insertions -> 3 / 1 = 3.
            Assert.Equal(3.0, TextRecognitionMetrics.CharacterErrorRate("a", "abcd"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task CharacterErrorRate_Corpus_PoolsDistancesAndLengths()
        {
            await Task.Yield();

            // Distances 0 and 1 over reference lengths 2 and 2 -> 1 / 4.
            var references = new[] { "ab", "cd" };
            var hypotheses = new[] { "ab", "ce" };

            Assert.Equal(0.25, TextRecognitionMetrics.CharacterErrorRate(references, hypotheses), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task CharacterErrorRate_Corpus_IsNotTheMeanOfPerSampleRates()
        {
            await Task.Yield();

            // Per-sample rates are 1/1 = 1.0 and 0/10 = 0.0, whose mean is 0.5. The corpus rate
            // pools instead: 1 edit over 11 reference characters. This is the distinction that
            // makes short samples unable to dominate a benchmark number.
            var references = new[] { "a", "abcdefghij" };
            var hypotheses = new[] { "b", "abcdefghij" };

            Assert.Equal(1.0 / 11.0, TextRecognitionMetrics.CharacterErrorRate(references, hypotheses), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task WordErrorRate_OneWrongWordInThree_IsOneThird()
        {
            await Task.Yield();

            Assert.Equal(1.0 / 3.0, TextRecognitionMetrics.WordErrorRate("the cat sat", "the dog sat"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task WordErrorRate_IgnoresRepeatedWhitespace()
        {
            await Task.Yield();

            Assert.Equal(0.0, TextRecognitionMetrics.WordErrorRate("the cat", "the   cat"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task WordErrorRate_CountsAnInsertedWord()
        {
            await Task.Yield();

            // Reference has 2 tokens; the hypothesis adds one -> 1 / 2.
            Assert.Equal(0.5, TextRecognitionMetrics.WordErrorRate("the cat", "the big cat"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task NormalizedEditDistance_ExactMatch_IsOne()
        {
            await Task.Yield();

            Assert.Equal(1.0, TextRecognitionMetrics.NormalizedEditDistance("abc", "abc"), 12);
            Assert.Equal(1.0, TextRecognitionMetrics.NormalizedEditDistance("", ""), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task NormalizedEditDistance_DividesByTheLongerString()
        {
            await Task.Yield();

            // One substitution over max(3, 3) -> 1 - 1/3.
            Assert.Equal(1.0 - (1.0 / 3.0), TextRecognitionMetrics.NormalizedEditDistance("abc", "abd"), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task NormalizedEditDistance_StaysWithinZeroAndOne_EvenWhenLengthsDiffer()
        {
            await Task.Yield();

            // The same input that drives CER above 1 stays bounded here, which is the property
            // that makes 1-NED safe to average across samples.
            double value = TextRecognitionMetrics.NormalizedEditDistance("a", "abcd");

            Assert.InRange(value, 0.0, 1.0);
            Assert.Equal(1.0 - (3.0 / 4.0), value, 12);
        }

        [Fact(Timeout = 60000)]
        public async Task ExactMatchAccuracy_DefaultsToCaseInsensitiveAlphanumeric()
        {
            await Task.Yield();

            // The scene-text benchmark protocol: case folded, punctuation stripped.
            var references = new[] { "Hello!", "world" };
            var hypotheses = new[] { "hello", "WORLD" };

            Assert.Equal(1.0, TextRecognitionMetrics.ExactMatchAccuracy(references, hypotheses), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task ExactMatchAccuracy_RawComparison_RejectsThoseSamePairs()
        {
            await Task.Yield();

            var references = new[] { "Hello!", "world" };
            var hypotheses = new[] { "hello", "WORLD" };

            Assert.Equal(
                0.0,
                TextRecognitionMetrics.ExactMatchAccuracy(
                    references, hypotheses, caseSensitive: true, alphanumericOnly: false),
                12);
        }

        [Fact(Timeout = 60000)]
        public async Task ExactMatchAccuracy_CountsTheMatchingFraction()
        {
            await Task.Yield();

            var references = new[] { "cat", "dog", "bird", "fish" };
            var hypotheses = new[] { "cat", "dog", "bird", "fis" };

            Assert.Equal(0.75, TextRecognitionMetrics.ExactMatchAccuracy(references, hypotheses), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task Normalize_StripsNonAlphanumericAndFoldsCase()
        {
            await Task.Yield();

            Assert.Equal("abc123", TextRecognitionMetrics.Normalize("A-B c.1 2/3!"));
        }

        [Fact(Timeout = 60000)]
        public async Task CorpusMetrics_RejectMisalignedInput()
        {
            await Task.Yield();

            var references = new[] { "a", "b" };
            var hypotheses = new[] { "a" };

            Assert.Throws<ArgumentException>(
                () => TextRecognitionMetrics.CharacterErrorRate(references, hypotheses));
            Assert.Throws<ArgumentException>(
                () => TextRecognitionMetrics.WordErrorRate(references, hypotheses));
            Assert.Throws<ArgumentException>(
                () => TextRecognitionMetrics.NormalizedEditDistance(references, hypotheses));
            Assert.Throws<ArgumentException>(
                () => TextRecognitionMetrics.ExactMatchAccuracy(references, hypotheses));
        }

        [Fact(Timeout = 60000)]
        public async Task EmptyReference_WithNonEmptyHypothesis_IsUndefined()
        {
            await Task.Yield();

            // Dividing by a zero-length reference has no meaningful value, so the per-sample
            // overloads say so rather than returning a number that looks like a score.
            Assert.True(double.IsNaN(TextRecognitionMetrics.CharacterErrorRate("", "abc")));
            Assert.True(double.IsNaN(TextRecognitionMetrics.WordErrorRate("", "abc")));
            Assert.Equal(0.0, TextRecognitionMetrics.CharacterErrorRate("", ""), 12);
        }
    }
}
