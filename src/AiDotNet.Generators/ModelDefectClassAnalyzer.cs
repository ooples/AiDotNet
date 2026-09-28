using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Compile-time gates for three model defect classes the OCR audit found, each with a checked-in baseline
/// (<see cref="ModelDefectBaselines"/>) that may only shrink.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>ADNDEF001: the same model implemented twice: two concrete types with the same simple name and
/// the same primary <c>[ResearchPaper]</c>. Nine such pairs existed in OCR alone, each with one copy
/// silently untested.</item>
/// <item>ADNDEF002: a LayerHelper factory that yields a bare <c>MultiHeadAttentionLayer</c>. In a flat
/// chain nothing adds the input back, so the "transformer" has no residual connections.</item>
/// <item>ADNDEF003: a model built from a LayerHelper factory that models of OTHER papers also use,
/// without <c>[ArchitectureFromPaper]</c> naming whose architecture it reuses. This is how generic
/// templates stood in for paper architectures.</item>
/// <item>ADNDEF004: a baseline entry that no longer violates, so a fix cannot land without removing
/// its line.</item>
/// </list>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class ModelDefectClassAnalyzer : DiagnosticAnalyzer
{
    internal static readonly DiagnosticDescriptor DuplicatePaper = new(
        "ADNDEF001", "Two models implement the same paper",
        "'{0}' and {1} other model(s) ({2}) implement the same paper '{3}'; keep the faithful copy and delete the rest",
        "AiDotNet.ModelDefects", DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor ResidualFreeAttention = new(
        "ADNDEF002", "Transformer factory yields attention without a residual connection",
        "LayerHelper.{0} yields a bare MultiHeadAttentionLayer, so the chain computes LN(Attn(x)) with no skip; use TransformerEncoderLayer, TransformerEncoderBlock or PreLNTransformerBlock",
        "AiDotNet.ModelDefects", DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor UndeclaredSharedFactory = new(
        "ADNDEF003", "Model reuses another paper's layer factory without declaring it",
        "'{0}' builds its layers from LayerHelper.{1}, which models of {2} other paper(s) also use; build this paper's own architecture, or add [ArchitectureFromPaper] naming whose architecture it reuses",
        "AiDotNet.ModelDefects", DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor StaleBaseline = new(
        "ADNDEF004", "Model defect baseline entry is stale",
        "'{0}' is listed in ModelDefectBaselines.{1} but no longer violates; remove its line so the baseline stays an exact inventory",
        "AiDotNet.ModelDefects", DiagnosticSeverity.Error, isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(DuplicatePaper, ResidualFreeAttention, UndeclaredSharedFactory, StaleBaseline);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            if (start.Compilation.AssemblyName != "AiDotNet") return;
            var papers = new ConcurrentBag<(string Type, string Paper, string Title, Location Location, string SimpleName)>();
            var residualFree = new ConcurrentDictionary<string, Location>(StringComparer.Ordinal);
            var factoryUses = new ConcurrentBag<(string Factory, string Type)>();
            var declared = new ConcurrentDictionary<string, Location>(StringComparer.Ordinal);

            start.RegisterSymbolAction(symbolContext =>
            {
                var type = (INamedTypeSymbol)symbolContext.Symbol;
                if (type.TypeKind != TypeKind.Class || type.IsAbstract || type.Locations.Length == 0) return;
                string name = type.ToDisplayString();
                var attributes = type.GetAttributes();
                var primary = attributes.FirstOrDefault(a => a.AttributeClass?.Name == "ResearchPaperAttribute");
                if (primary is { ConstructorArguments.Length: >= 2 }
                    && primary.ConstructorArguments[1].Value is string url)
                {
                    string title = primary.ConstructorArguments[0].Value as string ?? url;
                    papers.Add((name, NormalizePaper(url), title, type.Locations[0], type.Name));
                }
                if (attributes.Any(a => a.AttributeClass?.Name == "ArchitectureFromPaperAttribute"))
                    declared[name] = type.Locations[0];
            }, SymbolKind.NamedType);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var invocation = (InvocationExpressionSyntax)nodeContext.Node;
                if (nodeContext.SemanticModel.GetSymbolInfo(invocation).Symbol is not IMethodSymbol method
                    || method.ContainingType?.Name != "LayerHelper" || !method.Name.StartsWith("CreateDefault", StringComparison.Ordinal))
                    return;
                var owner = nodeContext.ContainingSymbol?.ContainingType;
                if (owner is null || owner.Name == "LayerHelper") return;
                factoryUses.Add((method.Name, owner.ToDisplayString()));
            }, SyntaxKind.InvocationExpression);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var yield = (YieldStatementSyntax)nodeContext.Node;
                if (yield.Expression is null) return;
                var method = nodeContext.ContainingSymbol as IMethodSymbol;
                if (method?.ContainingType?.Name != "LayerHelper") return;
                var yielded = nodeContext.SemanticModel.GetTypeInfo(yield.Expression).Type;
                if (yielded?.Name == "MultiHeadAttentionLayer")
                    residualFree.TryAdd(method.Name, method.Locations.FirstOrDefault() ?? Location.None);
            }, SyntaxKind.YieldReturnStatement);

            start.RegisterCompilationEndAction(end =>
            {
                // ADNDEF001: the same model implemented twice, i.e. the same simple class name AND the same
                // primary paper. Paper alone is not enough: a model and its loss, or task heads and size
                // variants of one architecture, legitimately cite one paper (that matched 700+ types).
                var seenDuplicates = new HashSet<string>(StringComparer.Ordinal);
                foreach (var group in papers.GroupBy(p => p.Paper + "|" + p.SimpleName).Where(g => g.Select(p => p.Type).Distinct().Count() > 1))
                {
                    var members = group.GroupBy(p => p.Type).Select(g => g.First()).OrderBy(p => p.Type, StringComparer.Ordinal).ToList();
                    foreach (var member in members)
                    {
                        seenDuplicates.Add(member.Type);
                        if (ModelDefectBaselines.DuplicatePapers.Contains(member.Type)) continue;
                        var others = members.Where(m => m.Type != member.Type).Select(m => m.Type).ToList();
                        end.ReportDiagnostic(Diagnostic.Create(DuplicatePaper, member.Location,
                            member.Type, others.Count, string.Join(", ", others), member.Title));
                    }
                }

                // ADNDEF002: residual-free attention in a factory.
                foreach (var entry in residualFree)
                    if (!ModelDefectBaselines.ResidualFreeFactories.Contains(entry.Key))
                        end.ReportDiagnostic(Diagnostic.Create(ResidualFreeAttention, entry.Value, entry.Key));

                // ADNDEF003: a factory shared across papers, used without a declaration.
                var paperOf = papers.GroupBy(p => p.Type).ToDictionary(g => g.Key, g => g.First().Paper, StringComparer.Ordinal);
                var locationOf = papers.GroupBy(p => p.Type).ToDictionary(g => g.Key, g => g.First().Location, StringComparer.Ordinal);
                var seenShared = new HashSet<string>(StringComparer.Ordinal);
                foreach (var factory in factoryUses.GroupBy(u => u.Factory))
                {
                    var users = factory.Select(u => u.Type).Distinct().Where(paperOf.ContainsKey).ToList();
                    int distinctPapers = users.Select(u => paperOf[u]).Distinct().Count();
                    if (distinctPapers < 2) continue;
                    foreach (var user in users)
                    {
                        if (declared.ContainsKey(user)) continue;
                        seenShared.Add(user);
                        if (ModelDefectBaselines.UndeclaredSharedFactoryModels.Contains(user)) continue;
                        end.ReportDiagnostic(Diagnostic.Create(UndeclaredSharedFactory, locationOf[user],
                            user, factory.Key, distinctPapers - 1));
                    }
                }

                // ADNDEF004: the baselines only shrink.
                ReportStale(end, ModelDefectBaselines.DuplicatePapers, seenDuplicates, nameof(ModelDefectBaselines.DuplicatePapers));
                ReportStale(end, ModelDefectBaselines.ResidualFreeFactories, new HashSet<string>(residualFree.Keys), nameof(ModelDefectBaselines.ResidualFreeFactories));
                ReportStale(end, ModelDefectBaselines.UndeclaredSharedFactoryModels, seenShared, nameof(ModelDefectBaselines.UndeclaredSharedFactoryModels));
            });
        });
    }

    private static void ReportStale(CompilationAnalysisContext end, HashSet<string> baseline, HashSet<string> current, string name)
    {
        foreach (var entry in baseline)
            if (!current.Contains(entry))
                end.ReportDiagnostic(Diagnostic.Create(StaleBaseline, Location.None, entry, name));
    }

    // One key per paper whatever link form a model cites: arXiv ids (with or without the 10.48550 DOI
    // prefix and version suffix), then DOIs, then the bare URL.
    internal static string NormalizePaper(string url)
    {
        string u = url.Trim().ToLowerInvariant().TrimEnd('/');
        var arxiv = Regex.Match(u, @"(?:arxiv\.org/(?:abs|pdf)/|arxiv\.)(\d{4}\.\d{4,5})");
        if (arxiv.Success) return "arxiv:" + arxiv.Groups[1].Value;
        var doi = Regex.Match(u, @"doi\.org/(.+)$");
        if (doi.Success) return "doi:" + doi.Groups[1].Value;
        return Regex.Replace(u, @"^https?://(www\.)?", string.Empty);
    }
}