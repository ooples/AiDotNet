using System.Text.Json.Nodes;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;
using static AiDotNet.Evolution.CSharp.Tests.CompilerTestSupport;

namespace AiDotNet.Evolution.CSharp.Tests;

public sealed class CSharpPatchCompilerTests
{
    [Fact]
    public void Every_emit_stream_write_path_enforces_the_same_image_size_limit()
    {
        using var stream = new CSharpPatchCompiler.BoundedImageStream();
        stream.Write(new byte[] { 1, 2 }, 0, 2);
        stream.Write(new byte[] { 3, 4 }.AsSpan());
        stream.WriteByte(5);
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5 }, stream.ToArray());
        stream.SetLength(CSharpPatchCompiler.MaximumImageBytes);
        stream.Position = CSharpPatchCompiler.MaximumImageBytes;
        Assert.ThrowsAny<IOException>(() => stream.WriteByte(1));
        Assert.ThrowsAny<IOException>(() => stream.Write(new byte[] { 1 }, 0, 1));
        Assert.ThrowsAny<IOException>(() => stream.Write(new byte[] { 1 }.AsSpan()));
        Assert.ThrowsAny<IOException>(() => stream.SetLength(CSharpPatchCompiler.MaximumImageBytes + 1));
        Assert.Equal(CSharpPatchCompiler.MaximumImageBytes, stream.Length);
    }

    [Fact]
    public void Compiling_a_throwing_method_does_not_run_it_and_catalog_limits_do_not_escape_their_bound()
    {
        var options = Options();
        options.MaxCatalogNodes = 1;
        var compiler = new CSharpPatchCompiler(options, Program());
        var prepared = compiler.Prepare(Parent(), default);
        Assert.Single(prepared.Targets);
        var target = prepared.Targets[0];
        var result = compiler.Apply(prepared, Patch(prepared.Parent.Id, Edit(target, "{ throw new System.Exception(); }")), default);
        Assert.NotNull(result.Candidate);
        Assert.True(result.CompilerInvoked);
        Assert.Contains("throw new System.Exception()", result.Candidate.Source);
    }

    [Fact]
    public void Source_and_edit_count_bounds_are_checked_independently_of_response_length()
    {
        var options = Options();
        options.MaxSourceChars = 256;
        options.MaxResponseChars = 4096;
        options.MaxEdits = 1;
        var compiler = new CSharpPatchCompiler(options, Program());
        var prepared = compiler.Prepare(Parent(), default);
        var target = prepared.Targets.Single(item => item.Kind == "NumericLiteralExpression");
        Assert.Null(compiler.Apply(prepared, Patch(prepared.Parent.Id, Edit(target, "2"), Edit(target, "3")), default).Candidate);
        Assert.Null(compiler.Apply(prepared, Patch(prepared, "2" + new string(' ', 254)), default).Candidate);
        Assert.Null(compiler.Apply(prepared, Patch(prepared, new string('x', 257)), default).Candidate);
        var plan = JsonNode.Parse(Patch(prepared))!;
        plan["hypothesis"] = new string('x', 1025);
        Assert.Null(compiler.Apply(prepared, plan.ToJsonString(), default).Candidate);
        Assert.Null(compiler.Apply(prepared, Patch(prepared).Replace("\"replacement\":\"2\"", "\"replacement\":\"\\ud800\"", StringComparison.Ordinal), default).Candidate);
    }

    [Fact]
    public void Valid_syntax_patch_emits_deterministic_PE_without_executing_the_candidate()
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var prepared = compiler.Prepare(Parent(), default);
        var result = compiler.Apply(prepared, Patch(prepared, "2"), default);
        Assert.NotNull(result.Candidate);
        Assert.Equal(Source.Replace("return 1", "return 2", StringComparison.Ordinal), result.Candidate.Source);
        Assert.True(result.CompilerInvoked);
        Assert.Matches("^[a-f0-9]{64}$", result.EmittedSha256);
        Assert.Equal(result.EmittedSha256, compiler.Apply(prepared, Patch(prepared, "2"), default).EmittedSha256);
        Assert.Equal(Source, prepared.Parent.Source);
        Assert.NotEqual(prepared.Parent.Id, result.Candidate.Id);
        Assert.False(string.IsNullOrWhiteSpace(result.Hypothesis));
        Assert.Single(compiler.ReferenceHashes);
        Assert.True(compiler.ReferenceBytes > 0);
    }

    [Theory]
    [InlineData("schemaVersion", "2")]
    [InlineData("parentId", "\"other\"")]
    [InlineData("hypothesis", "\" \"")]
    [InlineData("edits", "[]")]
    [InlineData("edits", "null")]
    [InlineData("unexpected", "true")]
    public void Invalid_schema_and_identity_are_rejected_before_emit(string property, string json)
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var prepared = compiler.Prepare(Parent(), default);
        var plan = JsonNode.Parse(Patch(prepared))!.AsObject();
        plan[property] = JsonNode.Parse(json);
        var result = compiler.Apply(prepared, plan.ToJsonString(), default);
        Assert.Null(result.Candidate);
        Assert.False(result.CompilerInvoked);
    }

    [Theory]
    [InlineData("start", "-1")]
    [InlineData("length", "999")]
    [InlineData("kind", "\"IdentifierName\"")]
    [InlineData("expectedSha256", "\"wrong\"")]
    [InlineData("replacement", "\"\"")]
    [InlineData("replacement", "\"1\"")]
    [InlineData("replacement", "\"2; public class Escape {}\"")]
    [InlineData("replacement", "\"#if true\\n2\\n#endif\"")]
    [InlineData("unexpected", "true")]
    public void Stale_spans_hashes_unchanged_source_and_syntax_escape_are_rejected(string property, string json)
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var prepared = compiler.Prepare(Parent(), default);
        var plan = JsonNode.Parse(Patch(prepared))!;
        plan["edits"]![0]![property] = JsonNode.Parse(json);
        var result = compiler.Apply(prepared, plan.ToJsonString(), default);
        Assert.Null(result.Candidate);
        Assert.False(result.CompilerInvoked);
    }

    [Fact]
    public void Duplicate_properties_overlapping_nodes_and_malformed_json_are_not_ambiguous_edits()
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var prepared = compiler.Prepare(Parent(), default);
        var literal = prepared.Targets.Single(item => item.Kind == "NumericLiteralExpression");
        var statement = prepared.Targets.Single(item => item.Kind == "ReturnStatement");
        foreach (string invalid in new[]
        {
            Patch(prepared).Replace("\"schemaVersion\":1", "\"schemaVersion\":1,\"schemaVersion\":1", StringComparison.Ordinal),
            Patch(prepared).Replace("\"start\":", "\"replacement\":\"2\",\"start\":", StringComparison.Ordinal),
            Patch(prepared.Parent.Id, Edit(literal, "2"), Edit(literal, "3")),
            Patch(prepared.Parent.Id, Edit(literal, "2"), Edit(statement, "return 3;")),
            "{", "null", "[]", new string('[', 9) + new string(']', 9)
        })
        {
            var result = compiler.Apply(prepared, invalid, default);
            Assert.Null(result.Candidate);
            Assert.False(result.CompilerInvoked);
        }
    }

    [Fact]
    public void Type_error_reaches_real_emit_but_feedback_contains_no_source_or_mapped_filename()
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var parent = Parent("#line 900 \"SECRET_PATH\"\n" + Source);
        var prepared = compiler.Prepare(parent, default);
        var result = compiler.Apply(prepared, Patch(prepared, "SECRET_IDENTIFIER"), default);
        Assert.Null(result.Candidate);
        Assert.NotNull(result.Proposed);
        Assert.True(result.CompilerInvoked);
        Assert.Contains("CS0103 at ", result.Feedback);
        Assert.DoesNotContain("SECRET", result.Feedback);
        Assert.DoesNotContain("900", result.Feedback);
    }

    [Theory]
    [InlineData("public static class C { public static int F() => 1; }")]
    [InlineData("public static class C { public static int F() { int s = 0; for(int i=0;i<3;i++){s+=i;} return 1; } }")]
    public void Expression_bodies_and_loop_statements_are_available_without_public_declarations(string source)
    {
        var compiler = new CSharpPatchCompiler(Options(), Program());
        var prepared = compiler.Prepare(Parent(source), default);
        Assert.DoesNotContain(prepared.Targets, item => item.Kind.Contains("Declaration", StringComparison.Ordinal) && !item.IsStatement);
        if (source.Contains("for(", StringComparison.Ordinal))
        {
            var result = compiler.Apply(prepared, Patch(prepared, "for(int i=0;i<3;i++){s=s+i;}", "ForStatement"), default);
            Assert.NotNull(result.Candidate);
        }
        else Assert.NotNull(compiler.Apply(prepared, Patch(prepared), default).Candidate);
    }

    [Fact]
    public void Multiple_protected_regions_preserve_exact_mixed_line_terminators_and_outside_text()
    {
        const string source = "public static class C { public static int F() {\r\n// EVOLVE-BLOCK-START\nint x = 1;\r// EVOLVE-BLOCK-END\r\nint y = 10;\n// EVOLVE-BLOCK-START\rreturn x;\n// EVOLVE-BLOCK-END\r\n} }";
        var compiler = new CSharpPatchCompiler(Options(), Program(enforce: true));
        var prepared = compiler.Prepare(Parent(source), default);
        Assert.DoesNotContain(prepared.Targets, item => item.Preview == "10");
        var literal = prepared.Targets.Single(item => item.Kind == "NumericLiteralExpression");
        var statement = prepared.Targets.Single(item => item.Kind == "ReturnStatement");
        var result = compiler.Apply(prepared, Patch(prepared.Parent.Id, Edit(literal, "200"), Edit(statement, "return x + y;")), default);
        Assert.NotNull(result.Candidate);
        Assert.Equal(source.Replace("int x = 1", "int x = 200", StringComparison.Ordinal).Replace("return x;", "return x + y;", StringComparison.Ordinal), result.Candidate.Source);
        Assert.Null(compiler.Apply(prepared, Patch(prepared, "2 /*\n// EVOLVE-BLOCK-END\n*/"), default).Candidate);
    }

    [Fact]
    public void Bounded_inputs_cancellation_and_invalid_parents_fail_before_acceptance()
    {
        var options = Options();
        options.MaxSourceChars = 256;
        options.MaxResponseChars = 512;
        var compiler = new CSharpPatchCompiler(options, Program());
        var prepared = compiler.Prepare(Parent(), default);
        Assert.Null(compiler.Apply(prepared, new string('x', 513), default).Candidate);
        Assert.Null(compiler.Apply(prepared, Patch(prepared, new string('x', 257)), default).Candidate);
        Assert.Null(compiler.Apply(prepared, Patch(prepared, "2" + new string(' ', 255)), default).Candidate);
        Assert.Throws<ArgumentException>(() => compiler.Prepare(Parent("class C {"), default));
        Assert.Throws<ArgumentException>(() => compiler.Prepare(Parent("public class C {}"), default));
        Assert.Throws<ArgumentException>(() => compiler.Prepare(Parent(Source + new string(' ', 257)), default));
        Assert.Throws<ArgumentException>(() => compiler.Prepare(new("x", ProgramLanguage.Python), default));
        Assert.Throws<ArgumentException>(() => new CSharpPatchCompiler(Options(), Program(true)).Prepare(Parent(), default));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => compiler.Apply(prepared, Patch(prepared), cancelled.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => compiler.Prepare(Parent(), cancelled.Token));
    }

    [Fact]
    public void Reference_content_and_declared_target_are_fingerprinted_and_caller_mutation_is_not_retained()
    {
        var options = Options();
        var compiler = new CSharpPatchCompiler(options, Program());
        string original = compiler.VersionHash;
        options.ReferencePaths[0] = "does-not-exist";
        options.MaxEdits = 1;
        Assert.Equal(original, compiler.VersionHash);
        Assert.NotNull(compiler.Apply(compiler.Prepare(Parent(), default), Patch(compiler.Prepare(Parent(), default)), default).Candidate);
        var changed = Options();
        changed.TargetIdentity += "changed";
        Assert.NotEqual(original, new CSharpPatchCompiler(changed, Program()).VersionHash);
        changed = Options();
        changed.ReferencePaths = new[] { typeof(CSharpPatchCompiler).Assembly.Location };
        Assert.NotEqual(original, new CSharpPatchCompiler(changed, Program()).VersionHash);
        changed.ReferencePaths = new[] { typeof(object).Assembly.Location, typeof(object).Assembly.Location };
        Assert.Throws<ArgumentException>(() => new CSharpPatchCompiler(changed, Program()));
    }
}
