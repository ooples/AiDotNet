using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Configuration;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace AiDotNet.Evolution.CSharp;

/// <summary>Applies syntax-addressed edits to an immutable snapshot and emits without loading or executing code.</summary>
internal sealed class CSharpPatchCompiler
{
    internal const int MaximumReferenceBytes = 128 * 1024 * 1024;
    internal const int MaximumImageBytes = 8 * 1024 * 1024;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private readonly CSharpProgramEvolutionOptions _options;
    private readonly ProgramEvolutionOptions _program;
    private readonly MetadataReference[] _references;
    private readonly CSharpParseOptions _parse = new(LanguageVersion.CSharp12, DocumentationMode.None, SourceCodeKind.Regular);

    internal CSharpPatchCompiler(CSharpProgramEvolutionOptions options, ProgramEvolutionOptions program)
    {
        _options = options.Snapshot();
        _program = program.Clone();
        _program.Validate();
        if (_program.Language != ProgramLanguage.CSharp) throw new ArgumentException("The C# compiler requires CSharp program language.");
        var references = new List<MetadataReference>();
        var identities = new List<string>
        {
            "csharp-snapshot-compiler-v1", _options.ConfigurationHash, "CSharp12", "library-release-deterministic-safe",
            typeof(CSharpCompilation).Assembly.ManifestModule.ModuleVersionId.ToString("D"),
            typeof(Compilation).Assembly.ManifestModule.ModuleVersionId.ToString("D"),
            typeof(CSharpPatchCompiler).Assembly.ManifestModule.ModuleVersionId.ToString("D"),
            typeof(ProgramGenome).Assembly.ManifestModule.ModuleVersionId.ToString("D"),
            typeof(EvolutionResourceLedger).Assembly.ManifestModule.ModuleVersionId.ToString("D"),
            _program.MaxProgramChars.ToString(CultureInfo.InvariantCulture),
            _program.EnforceEvolveBlocks ? "enforce" : "free",
            _program.ResolveEvolveBlockMarkers().Start, _program.ResolveEvolveBlockMarkers().End
        };
        var hashes = new HashSet<string>(StringComparer.Ordinal);
        foreach (string path in _options.ReferencePaths)
        {
            byte[] bytes = ReadReference(path, MaximumReferenceBytes - ReferenceBytes);
            ReferenceBytes += bytes.Length;
            string hash = Hash(bytes);
            if (!hashes.Add(hash)) throw new ArgumentException("Duplicate reference assembly content.");
            identities.Add(hash);
            var reference = MetadataReference.CreateFromImage(ImmutableArray.CreateRange(bytes));
            // Roslyn references load metadata lazily. Force validation during charged setup, before model dispatch.
            using (Metadata metadata = reference.GetMetadata())
            {
                if (metadata is not AssemblyMetadata assembly || assembly.GetModules().Length != 1 ||
                    !assembly.GetModules()[0].GetMetadataReader().IsAssembly)
                    throw new BadImageFormatException("A trusted reference must be a single managed assembly image.");
            }
            references.Add(reference);
        }
        _references = references.ToArray();
        ReferenceHashes = Array.AsReadOnly(hashes.OrderBy(hash => hash, StringComparer.Ordinal).ToArray());
        VersionHash = EvolutionHash.Combine(identities);
    }

    internal int ReferenceBytes { get; }
    internal IReadOnlyList<string> ReferenceHashes { get; private set; } = Array.Empty<string>();
    internal string VersionHash { get; }

    internal CSharpPatchPreparation Prepare(ProgramGenome parent, CancellationToken cancellationToken)
    {
        if (parent.Language != ProgramLanguage.CSharp || parent.Source.Length > Math.Min(_options.MaxSourceChars, _program.MaxProgramChars))
            throw new ArgumentException("The parent is not a bounded C# source snapshot.");
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(_options.CompilationTimeoutSeconds));
        SyntaxTree tree = CSharpSyntaxTree.ParseText(parent.Source, _parse, cancellationToken: timeout.Token);
        SyntaxNode root = tree.GetRoot(timeout.Token);
        if (tree.GetDiagnostics(timeout.Token).Any(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error))
            throw new ArgumentException("The parent must have valid C# syntax before syntax-guided evolution.");
        List<(int Start, int End)> regions = EditableRegions(parent.Source);
        var targets = new List<CSharpEditTarget>();
        foreach (SyntaxNode node in root.DescendantNodes())
        {
            timeout.Token.ThrowIfCancellationRequested();
            if (node is not StatementSyntax && node is not ExpressionSyntax) continue;
            if (node.Span.Length == 0 || !WithinMethodBody(node)) continue;
            if (!regions.Any(region => node.Span.Start >= region.Start && node.Span.End <= region.End)) continue;
            string original = parent.Source.Substring(node.Span.Start, node.Span.Length);
            targets.Add(new(node.Span.Start, node.Span.Length, node.Kind().ToString(), node is StatementSyntax, Hash(original), original.Substring(0, Math.Min(160, original.Length))));
            if (targets.Count == _options.MaxCatalogNodes) break;
        }
        if (targets.Count == 0) throw new ArgumentException("No editable method-body statements or expressions fit the declared boundaries.");
        return new CSharpPatchPreparation(parent, tree, targets.AsReadOnly());
    }

    internal CSharpPatchAttempt Apply(CSharpPatchPreparation preparation, string response, CancellationToken cancellationToken)
    {
        bool compilerInvoked = false;
        string hypothesis = string.Empty;
        ProgramGenome? proposed = null;
        var timer = Stopwatch.StartNew();
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(_options.CompilationTimeoutSeconds));
        CSharpPatchAttempt Reject(string feedback) => new(null, proposed, hypothesis, feedback, compilerInvoked, string.Empty, timer.Elapsed);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (response.Length > _options.MaxResponseChars) return Reject("The JSON proposal exceeds the response bound.");
            using JsonDocument document = JsonDocument.Parse(response, new JsonDocumentOptions { MaxDepth = 8 });
            JsonElement plan = document.RootElement;
            RequireProperties(plan, "schemaVersion", "parentId", "hypothesis", "edits");
            if (plan.GetProperty("schemaVersion").GetInt32() != 1 || plan.GetProperty("parentId").GetString() != preparation.Parent.Id)
                return Reject("The schema or exact parent identity does not match this snapshot.");
            hypothesis = plan.GetProperty("hypothesis").GetString() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(hypothesis) || hypothesis.Length > 1024) return Reject("Supply a testable hypothesis of at most 1024 characters.");
            StrictUtf8.GetByteCount(hypothesis);
            JsonElement edits = plan.GetProperty("edits");
            if (edits.ValueKind != JsonValueKind.Array || edits.GetArrayLength() < 1 || edits.GetArrayLength() > _options.MaxEdits)
                return Reject("Supply a bounded, nonempty edit array.");
            var changes = new List<TextChange>();
            foreach (JsonElement edit in edits.EnumerateArray())
            {
                RequireProperties(edit, "start", "length", "kind", "expectedSha256", "replacement");
                int start = edit.GetProperty("start").GetInt32(), length = edit.GetProperty("length").GetInt32();
                string? kind = edit.GetProperty("kind").GetString(), expected = edit.GetProperty("expectedSha256").GetString();
                CSharpEditTarget? target = preparation.Targets.FirstOrDefault(item => item.Start == start && item.Length == length && item.Kind == kind && item.ExpectedSha256 == expected);
                if (target is null) return Reject("An edit does not identify a node in the supplied original-snapshot catalog.");
                string replacement = edit.GetProperty("replacement").GetString() ?? string.Empty;
                if (replacement.Length == 0 || replacement.Length > _options.MaxSourceChars) return Reject("A replacement is empty or exceeds its bound.");
                StrictUtf8.GetByteCount(replacement);
                SyntaxNode parsed = target.IsStatement
                    ? SyntaxFactory.ParseStatement(replacement, options: _parse, consumeFullText: true)
                    : SyntaxFactory.ParseExpression(replacement, options: _parse, consumeFullText: true);
                if (parsed.ContainsDiagnostics || parsed.ContainsSkippedText ||
                    parsed.ContainsDirectives || parsed.FullSpan.Length != replacement.Length)
                    return Reject("A replacement must be one complete statement or expression, with no directives or skipped syntax.");
                changes.Add(new TextChange(new TextSpan(start, length), replacement));
            }
            changes.Sort((left, right) => left.Span.Start.CompareTo(right.Span.Start));
            for (int index = 1; index < changes.Count; index++)
                if (changes[index - 1].Span.End > changes[index].Span.Start) return Reject("Edits overlap; submit disjoint original-snapshot nodes.");
            long newLength = preparation.Parent.Source.Length + changes.Sum(change => (long)(change.NewText?.Length ?? 0) - change.Span.Length);
            if (newLength > Math.Min(_options.MaxSourceChars, _program.MaxProgramChars)) return Reject("The complete candidate exceeds the source bound.");
            string source = SourceText.From(preparation.Parent.Source, StrictUtf8).WithChanges(changes).ToString();
            if (_program.EnforceEvolveBlocks && !ProgramEditBoundary.PreservesProtectedText(preparation.Parent.Source, source, _program.ResolveEvolveBlockMarkers()))
                return Reject("The completed patch changed protected text or marker structure.");
            if (source == preparation.Parent.Source) return Reject("The patch did not change the exact source.");
            proposed = new ProgramGenome(source, ProgramLanguage.CSharp, hypothesis);
            SyntaxTree candidateTree = CSharpSyntaxTree.ParseText(source, _parse, cancellationToken: timeout.Token);
            var compilation = CSharpCompilation.Create("EvolutionCandidate", new[] { candidateTree }, _references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, optimizationLevel: OptimizationLevel.Release,
                    allowUnsafe: false, deterministic: true, concurrentBuild: false));
            using var image = new BoundedImageStream();
            compilerInvoked = true;
            var emitted = compilation.Emit(image, cancellationToken: timeout.Token);
            if (!emitted.Success) return Reject(CompilerFeedback(emitted.Diagnostics));
            return new(proposed, proposed, hypothesis, string.Empty, true, Hash(image.ToArray()), timer.Elapsed);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return Reject("The bounded compiler operation timed out; simplify the patch.");
        }
        catch (Exception exception) when (exception is JsonException or ArgumentException or InvalidOperationException or FormatException or OverflowException)
        {
            return Reject("The proposal is not a valid bounded patch for the supplied schema and snapshot.");
        }
        catch (ImageLimitException)
        {
            return Reject("The emitted image exceeded the bounded artifact size.");
        }
    }

    private List<(int Start, int End)> EditableRegions(string source)
    {
        if (!_program.EnforceEvolveBlocks) return new() { (0, source.Length) };
        var extracted = EvolveBlock.Extract(source, _program.ResolveEvolveBlockMarkers());
        if (!extracted.IsWellFormed || !extracted.HasRegions) throw new ArgumentException("Valid evolve blocks are required.");
        var starts = ProgramText.LineStarts(source);
        return extracted.Regions.Select(region => (starts[region.StartLineIndex + 1], starts[region.EndLineIndex])).ToList();
    }

    private static bool WithinMethodBody(SyntaxNode node)
    {
        BaseMethodDeclarationSyntax? method = node.Ancestors().OfType<BaseMethodDeclarationSyntax>().FirstOrDefault();
        return method is not null && ((method.Body is { } body && body.Span.Contains(node.Span)) ||
            (method.ExpressionBody is { } expression && expression.Expression.Span.Contains(node.Span)));
    }

    private static void RequireProperties(JsonElement value, params string[] required)
    {
        if (value.ValueKind != JsonValueKind.Object) throw new JsonException();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonProperty property in value.EnumerateObject())
            if (!required.Contains(property.Name, StringComparer.Ordinal) || !seen.Add(property.Name)) throw new JsonException();
        if (seen.Count != required.Length) throw new JsonException();
    }

    private static string CompilerFeedback(IEnumerable<Diagnostic> diagnostics)
    {
        // Diagnostic messages and mapped file names may contain source text or #line payloads. Report only compiler
        // IDs and physical UTF-16 spans; the source and bounded patch remain in the request's untrusted data section.
        string[] errors = diagnostics.Where(item => item.Severity == DiagnosticSeverity.Error).Take(8)
            .Select(item => item.Id + " at " + item.Location.SourceSpan.Start.ToString(CultureInfo.InvariantCulture) +
                "+" + item.Location.SourceSpan.Length.ToString(CultureInfo.InvariantCulture)).ToArray();
        return "Compilation failed: " + string.Join("; ", errors) + ". Repair against the same original parent and catalog.";
    }

    private static byte[] ReadReference(string path, int remaining)
    {
        using var stream = new FileStream(Path.GetFullPath(path), FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length is <= 0 or > 32 * 1024 * 1024 || stream.Length > remaining) throw new ArgumentException("The trusted reference bundle exceeds its byte bound.");
        int length = checked((int)stream.Length);
        byte[] bytes = new byte[length];
        int read = 0;
        while (read < length)
        {
            int count = stream.Read(bytes, read, length - read);
            if (count == 0) throw new IOException("A reference image ended before its declared length.");
            read += count;
        }
        if (stream.ReadByte() != -1) throw new IOException("A reference image changed while being read.");
        return bytes;
    }

    internal static string Hash(string value) => Hash(StrictUtf8.GetBytes(value));
    private static string Hash(byte[] value) => Convert.ToHexString(SHA256.HashData(value)).ToLowerInvariant();

    private sealed class ImageLimitException : IOException { }
    internal sealed class BoundedImageStream : MemoryStream
    {
        public override void Write(byte[] buffer, int offset, int count)
        {
            Check(count);
            base.Write(buffer, offset, count);
        }
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            Check(buffer.Length);
            base.Write(buffer);
        }
        public override void WriteByte(byte value)
        {
            Check(1);
            base.WriteByte(value);
        }
        public override void SetLength(long value)
        {
            if (value > MaximumImageBytes) throw new ImageLimitException();
            base.SetLength(value);
        }
        private void Check(int count)
        {
            if (Position > MaximumImageBytes - count) throw new ImageLimitException();
        }
    }
}

internal sealed record CSharpEditTarget(int Start, int Length, string Kind, bool IsStatement, string ExpectedSha256, string Preview);
internal sealed record CSharpPatchPreparation(ProgramGenome Parent, SyntaxTree Tree, IReadOnlyList<CSharpEditTarget> Targets);
internal sealed record CSharpPatchAttempt(ProgramGenome? Candidate, ProgramGenome? Proposed, string Hypothesis, string Feedback,
    bool CompilerInvoked, string EmittedSha256, TimeSpan Elapsed);
