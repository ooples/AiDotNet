using System;
using System.Collections.Generic;
using AiDotNet.Serving.StructuredOutput;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serving;

/// <summary>
/// Correctness tests for <see cref="JsonGrammarConstraint"/> (the unbounded-depth JSON pushdown automaton).
/// Acceptance is validated by exhaustive comparison against System.Text.Json.JsonDocument over a small JSON alphabet,
/// plus targeted deep-nesting and prefix-feasibility checks.
/// </summary>
public class JsonGrammarConstraintTests
{
    private sealed class CharVocab
    {
        public readonly string[] TokenText;
        public readonly int Eos;
        private readonly Dictionary<char, int> _index = new();
        public CharVocab(string alphabet)
        {
            TokenText = new string[alphabet.Length + 1];
            for (int i = 0; i < alphabet.Length; i++) { TokenText[i] = alphabet[i].ToString(); _index[alphabet[i]] = i; }
            TokenText[alphabet.Length] = string.Empty;
            Eos = alphabet.Length;
        }
        public bool Has(char c) => _index.ContainsKey(c);
        public int Id(char c) => _index[c];
    }

    // Drives the constraint char-by-char; true iff every char was permitted AND the value is complete.
    private static bool MatchesViaConstraint(CharVocab v, string input)
    {
        var c = new JsonGrammarConstraint(v.TokenText, v.Eos);
        var logits = new float[v.TokenText.Length];
        foreach (char ch in input)
        {
            if (!v.Has(ch)) return false;
            Array.Clear(logits, 0, logits.Length);
            c.ApplyMask(logits);
            if (float.IsNegativeInfinity(logits[v.Id(ch)])) return false;
            c.Accept(v.Id(ch));
        }
        return c.IsComplete;
    }

    // Strict RFC-8259 validation oracle. Newtonsoft is deliberately lenient (it accepts "00", trailing
    // content, etc.), so it cannot validate a strict grammar; System.Text.Json.JsonDocument is the strict
    // parser and is used here only as a test oracle (not for production serialization).
    private static bool IsValidJson(string s)
    {
        try { using var _ = System.Text.Json.JsonDocument.Parse(s); return true; }
        catch (System.Text.Json.JsonException) { return false; }
    }

    private static IEnumerable<string> AllStrings(string alphabet, int maxLen)
    {
        var frontier = new List<string> { string.Empty };
        for (int len = 1; len <= maxLen; len++)
        {
            var next = new List<string>();
            foreach (var s in frontier)
                foreach (char ch in alphabet)
                {
                    var t = s + ch;
                    next.Add(t);
                    yield return t;
                }
            frontier = next;
        }
    }

    [Fact]
    public void MatchesJsonDocument_Exhaustively()
    {
        // Structure + one string/key letter + a digit + sign. Exhaustive to length 4 keeps this fast while
        // covering objects, arrays, strings, numbers, nesting, and every malformed permutation.
        const string alphabet = "{}[]\":,a0-";
        var v = new CharVocab(alphabet);

        int checkedCount = 0;
        foreach (var s in AllStrings(alphabet, 4))
        {
            bool mine = MatchesViaConstraint(v, s);
            bool theirs = IsValidJson(s);
            Assert.True(mine == theirs, $"input '{s}': mine={mine} jsondoc={theirs}");
            checkedCount++;
        }
        // Prove AllStrings enumerated EVERY string through length 4 (a truncated enumeration would still
        // pass a loose '> 1000'): 10-char alphabet over lengths 1..4 = 10 + 100 + 1000 + 10000 = 11110.
        Assert.Equal(11110, checkedCount);
    }

    [Fact]
    public void Keywords_TrueFalseNull()
    {
        var v = new CharVocab("truefalsn[],");
        Assert.True(MatchesViaConstraint(v, "true"));
        Assert.True(MatchesViaConstraint(v, "false"));
        Assert.True(MatchesViaConstraint(v, "null"));
        Assert.True(MatchesViaConstraint(v, "[true,false,null]"));
        Assert.False(MatchesViaConstraint(v, "tru"));   // incomplete keyword
        Assert.False(MatchesViaConstraint(v, "trux"));  // not a keyword
    }

    [Fact]
    public void AcceptsDeeplyNestedJson_BeyondAnyFixedDepth()
    {
        var v = new CharVocab("{}[]\":,abc0");
        // Depth 20 nested arrays — impossible for a bounded-depth regex, fine for the pushdown automaton.
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < 20; i++) sb.Append('[');
        sb.Append('0');
        for (int i = 0; i < 20; i++) sb.Append(']');
        Assert.True(MatchesViaConstraint(v, sb.ToString()));

        // Nested objects to depth 10: {"a":{"a":...{"a":0}...}} (10 levels), built programmatically so it
        // actually reaches depth 10 rather than the depth-4 literal a hand-written string settled for.
        var obj = new System.Text.StringBuilder();
        for (int i = 0; i < 10; i++) obj.Append("{\"a\":");
        obj.Append('0');
        for (int i = 0; i < 10; i++) obj.Append('}');
        Assert.True(MatchesViaConstraint(v, obj.ToString()));
    }

    [Fact]
    public void RejectsMalformedJson()
    {
        var v = new CharVocab("{}[]\":,abc0 ");
        Assert.False(MatchesViaConstraint(v, "{\"a\":0,}"));   // trailing comma
        Assert.False(MatchesViaConstraint(v, "{\"a\":}"));      // missing value
        Assert.False(MatchesViaConstraint(v, "[0,]"));          // trailing comma in array
        Assert.False(MatchesViaConstraint(v, "{a:0}"));         // unquoted key
        Assert.False(MatchesViaConstraint(v, "[0"));            // unterminated
    }

    [Fact]
    public void EnforcesStructure_AtEachStep()
    {
        var v = new CharVocab("{}[]\":,abc0 ");
        var c = new JsonGrammarConstraint(v.TokenText, v.Eos);
        var logits = new float[v.TokenText.Length];

        // At the very start, an object/array/string/number/keyword may begin, but ':' or ',' may not.
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[v.Id('{')]));
        Assert.False(float.IsNegativeInfinity(logits[v.Id('[')]));
        Assert.True(float.IsNegativeInfinity(logits[v.Id(':')]));
        Assert.True(float.IsNegativeInfinity(logits[v.Id(',')]));
        Assert.False(c.IsComplete);

        // After '{', only a key-string or '}' — not a bare value.
        c.Accept(v.Id('{'));
        Array.Clear(logits, 0, logits.Length);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[v.Id('"')]));
        Assert.False(float.IsNegativeInfinity(logits[v.Id('}')]));
        Assert.True(float.IsNegativeInfinity(logits[v.Id('0')]));
    }
}
