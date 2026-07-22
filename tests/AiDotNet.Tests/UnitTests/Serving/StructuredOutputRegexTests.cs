using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Serving.StructuredOutput;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serving;

/// <summary>
/// Correctness tests for <see cref="RegexTokenConstraint"/>. The regex→NFA engine is validated by
/// exhaustive comparison against <see cref="System.Text.RegularExpressions.Regex"/> (acceptance) over a
/// small alphabet, plus targeted prefix-feasibility assertions (the property that actually matters for
/// guided decoding: a token is permitted only if the pattern can still complete).
/// </summary>
public class StructuredOutputRegexTests
{
    // A per-character "vocabulary": token id i -> alphabet[i] as a 1-char string; EOS is the last id.
    private sealed class CharVocab
    {
        public readonly string[] TokenText;
        public readonly int Eos;
        private readonly Dictionary<char, int> _index = new();
        public CharVocab(string alphabet)
        {
            TokenText = new string[alphabet.Length + 1];
            for (int i = 0; i < alphabet.Length; i++)
            {
                TokenText[i] = alphabet[i].ToString();
                _index[alphabet[i]] = i;
            }
            TokenText[alphabet.Length] = string.Empty; // EOS piece carries no chars
            Eos = alphabet.Length;
        }
        public int Id(char c) => _index[c];
    }

    // Drives the constraint character-by-character; returns true only if every char was permitted at its
    // step AND the constraint is in an accepting state at the end (i.e. a full match, mask-respecting).
    private static bool MatchesViaConstraint(string pattern, CharVocab v, string input)
    {
        var c = new RegexTokenConstraint(pattern, v.TokenText, v.Eos);
        var logits = new float[v.TokenText.Length];
        foreach (char ch in input)
        {
            Array.Clear(logits, 0, logits.Length);
            c.ApplyMask(logits);
            if (float.IsNegativeInfinity(logits[v.Id(ch)]))
            {
                return false; // this character is forbidden here -> not a mask-respecting match
            }
            c.Accept(v.Id(ch));
        }
        return c.IsComplete;
    }

    private static IEnumerable<string> AllStrings(string alphabet, int maxLen)
    {
        yield return string.Empty;
        var frontier = new List<string> { string.Empty };
        for (int len = 1; len <= maxLen; len++)
        {
            var next = new List<string>(frontier.Count * alphabet.Length);
            foreach (var s in frontier)
            {
                foreach (char ch in alphabet)
                {
                    var t = s + ch;
                    next.Add(t);
                    yield return t;
                }
            }
            frontier = next;
        }
    }

    [Theory]
    [InlineData(@"a*b")]
    [InlineData(@"a?b?c")]
    [InlineData(@"(a|b)c")]
    [InlineData(@"a{2,3}")]
    [InlineData(@"[ab]+c")]
    [InlineData(@"[^ab]c")]
    [InlineData(@"\dc")]
    [InlineData(@"a(bc)*")]
    [InlineData(@"(ab|a)b")]     // overlapping-prefix alternation (backtracking trap for naive matchers)
    [InlineData(@"a{2,}")]
    public void MatchesDotNetRegex_Exhaustively(string pattern)
    {
        const string alphabet = "ab01";
        var v = new CharVocab(alphabet);
        var net = new Regex("^(?:" + pattern + ")$", RegexOptions.None, TimeSpan.FromSeconds(1));

        foreach (var s in AllStrings(alphabet, 5))
        {
            bool mine = MatchesViaConstraint(pattern, v, s);
            bool theirs = net.IsMatch(s);
            Assert.True(mine == theirs, $"pattern '{pattern}' input '{s}': mine={mine} dotnet={theirs}");
        }
    }

    [Fact]
    public void DatePattern_ForbidsLettersAllowsDigits_AtEachStep()
    {
        // ISO-date shape. Prove prefix feasibility: only digits at digit positions, only '-' at separators.
        const string pattern = @"\d{4}-\d{2}-\d{2}";
        var v = new CharVocab("0123456789-abc");
        var c = new RegexTokenConstraint(pattern, v.TokenText, v.Eos);
        var logits = new float[v.TokenText.Length];

        void Step(char expected, char forbidden)
        {
            Array.Clear(logits, 0, logits.Length);
            c.ApplyMask(logits);
            Assert.False(float.IsNegativeInfinity(logits[v.Id(expected)]), $"'{expected}' should be allowed");
            Assert.True(float.IsNegativeInfinity(logits[v.Id(forbidden)]), $"'{forbidden}' should be forbidden");
            c.Accept(v.Id(expected));
        }

        Step('2', 'a'); Step('0', '-'); Step('2', 'b'); Step('6', '-'); // year: digits, not '-'
        Step('-', '0');                                                 // separator: '-', not a digit
        Step('0', '-'); Step('7', 'c');                                 // month
        Step('-', '9');
        Step('1', '-'); Step('6', 'a');                                 // day
        Assert.True(c.IsComplete);

        // At the accepting end, EOS is permitted and further digits are not.
        Array.Clear(logits, 0, logits.Length);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[v.Eos]));
        Assert.True(float.IsNegativeInfinity(logits[v.Id('0')]));
    }

    [Fact]
    public void MultiCharTokens_AreMaskedByFeasibility()
    {
        // Vocabulary with multi-character pieces: guided decoding must judge a whole token, not just a char.
        // Pattern (true|false): from the start, only tokens that begin a viable path are allowed.
        string[] vocab = { "true", "false", "tru", "fal", "xyz", "" };
        int eos = vocab.Length - 1;
        var c = new RegexTokenConstraint("(true|false)", vocab, eos);

        var logits = new float[vocab.Length];
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[0])); // "true"  — a complete match
        Assert.False(float.IsNegativeInfinity(logits[1])); // "false" — a complete match
        Assert.False(float.IsNegativeInfinity(logits[2])); // "tru"   — viable prefix of "true"
        Assert.False(float.IsNegativeInfinity(logits[3])); // "fal"   — viable prefix of "false"
        Assert.True(float.IsNegativeInfinity(logits[4]));  // "xyz"   — impossible

        c.Accept(0); // emit "true"
        Assert.True(c.IsComplete);
        Array.Clear(logits, 0, logits.Length);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[eos])); // done -> EOS allowed
        Assert.True(float.IsNegativeInfinity(logits[1]));    // no more tokens
    }
}
