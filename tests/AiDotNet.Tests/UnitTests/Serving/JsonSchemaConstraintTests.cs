using System;
using System.Collections.Generic;
using AiDotNet.Serving.StructuredOutput;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serving;

/// <summary>
/// Tests for <see cref="JsonSchemaConstraint"/>: a JSON Schema is compiled to a regex constraint that
/// forces compact JSON conforming to the schema (accepts valid instances, rejects malformed/mistyped ones).
/// </summary>
public class JsonSchemaConstraintTests
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

    // Characters that appear in the test JSON strings and schema literals.
    private const string Alphabet = "{}[]\":,-.0123456789abcdefghijklmnoprstuvwxyz";
    private static readonly CharVocab Vocab = new(Alphabet);

    private static bool AcceptsJson(string schemaJson, string json)
    {
        var c = JsonSchemaConstraint.FromSchema(schemaJson, Vocab.TokenText, Vocab.Eos);
        var logits = new float[Vocab.TokenText.Length];
        foreach (char ch in json)
        {
            Assert.True(Vocab.Has(ch), $"test char '{ch}' missing from alphabet");
            Array.Clear(logits, 0, logits.Length);
            c.ApplyMask(logits);
            if (float.IsNegativeInfinity(logits[Vocab.Id(ch)])) return false;
            c.Accept(Vocab.Id(ch));
        }
        return c.IsComplete;
    }

    [Fact]
    public void FlatObject_AcceptsValidCompactJson()
    {
        const string schema = @"{""type"":""object"",""properties"":{""name"":{""type"":""string""},""age"":{""type"":""integer""}}}";
        Assert.True(AcceptsJson(schema, @"{""name"":""bob"",""age"":42}"));
        Assert.True(AcceptsJson(schema, @"{""name"":"""",""age"":0}"));
        Assert.True(AcceptsJson(schema, @"{""name"":""a"",""age"":-7}"));
    }

    [Fact]
    public void FlatObject_RejectsMalformedOrMistyped()
    {
        const string schema = @"{""type"":""object"",""properties"":{""name"":{""type"":""string""},""age"":{""type"":""integer""}}}";
        Assert.False(AcceptsJson(schema, @"{""age"":42,""name"":""bob""}")); // wrong property order
        Assert.False(AcceptsJson(schema, @"{""name"":""bob""}"));             // missing required property
        Assert.False(AcceptsJson(schema, @"{""name"":bob,""age"":42}"));      // unquoted string
        Assert.False(AcceptsJson(schema, @"{""name"":""bob"",""age"":""42""}")); // integer given as string
        Assert.False(AcceptsJson(schema, @"{""name"":""bob"",""age"":4.2}"));  // non-integer
    }

    [Fact]
    public void Enum_RestrictsToLiterals()
    {
        const string schema = @"{""enum"":[""red"",""green"",""blue""]}";
        Assert.True(AcceptsJson(schema, @"""red"""));
        Assert.True(AcceptsJson(schema, @"""green"""));
        Assert.False(AcceptsJson(schema, @"""yellow"""));
        Assert.False(AcceptsJson(schema, @"red")); // must be a JSON string literal (quoted)
    }

    [Fact]
    public void StringEnumProperty_AndBoolean()
    {
        const string schema = @"{""type"":""object"",""properties"":{""status"":{""type"":""string"",""enum"":[""ok"",""bad""]},""done"":{""type"":""boolean""}}}";
        Assert.True(AcceptsJson(schema, @"{""status"":""ok"",""done"":true}"));
        Assert.True(AcceptsJson(schema, @"{""status"":""bad"",""done"":false}"));
        Assert.False(AcceptsJson(schema, @"{""status"":""meh"",""done"":true}")); // enum violation
        Assert.False(AcceptsJson(schema, @"{""status"":""ok"",""done"":yes}"));   // bad boolean
    }

    [Fact]
    public void ArrayOfIntegers()
    {
        const string schema = @"{""type"":""array"",""items"":{""type"":""integer""}}";
        Assert.True(AcceptsJson(schema, @"[]"));
        Assert.True(AcceptsJson(schema, @"[1]"));
        Assert.True(AcceptsJson(schema, @"[1,2,30]"));
        Assert.False(AcceptsJson(schema, @"[1,]"));   // trailing comma
        Assert.False(AcceptsJson(schema, @"[1,a]"));  // non-integer element
    }

    [Fact]
    public void AnyJsonObject_BoundedNesting()
    {
        var logits = new float[Vocab.TokenText.Length];
        // Fresh constraint per case (it is stateful) so each string is judged from a clean start.
        bool Feed(string json)
        {
            var c = JsonSchemaConstraint.AnyJsonObject(Vocab.TokenText, Vocab.Eos, maxDepth: 2);
            foreach (char ch in json)
            {
                Array.Clear(logits, 0, logits.Length);
                c.ApplyMask(logits);
                if (float.IsNegativeInfinity(logits[Vocab.Id(ch)])) return false;
                c.Accept(Vocab.Id(ch));
            }
            return c.IsComplete;
        }
        // Exercise BOTH sides of the bound so a constraint that ignores maxDepth cannot pass. AnyJsonObject
        // wraps a top-level object whose values may nest maxDepth (=2) further, so 1 + 2 = 3 nested objects
        // are accepted and a 4th level is rejected.
        Assert.True(Feed(@"{""a"":1}"));                             // 1 object - accepted
        Assert.True(Feed(@"{""a"":{""a"":{""a"":1}}}"));            // top + maxDepth(2) nested objects - accepted
        Assert.False(Feed(@"{""a"":{""a"":{""a"":{""a"":1}}}}"));   // one level beyond the bound - rejected
    }

    [Fact]
    public void InvalidSchema_Throws()
    {
        Assert.Throws<ArgumentException>(() => JsonSchemaConstraint.FromSchema("{ not json", Vocab.TokenText, Vocab.Eos));
    }
}
