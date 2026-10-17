using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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
/// <item>ADNDEF001: the same model implemented twice: two concrete types with the same name (ignoring a
/// Model, Network, Net or TTS suffix) and the same primary <c>[ResearchPaper]</c>. Nine such pairs existed in OCR
/// alone, each with one copy silently untested. A pair that runs one network for two tasks declares
/// <c>[TaskVariantOf]</c>; a pair the paper defines as different networks declares
/// <c>[DistinctImplementationOf]</c>.</item>
/// <item>ADNDEF002: a LayerHelper factory that yields a bare <c>MultiHeadAttentionLayer</c>. In a flat
/// chain nothing adds the input back, so the "transformer" has no residual connections.</item>
/// <item>ADNDEF003: a model built from a LayerHelper factory that models of OTHER papers also use,
/// without <c>[ArchitectureFromPaper]</c> naming whose architecture it reuses. This is how generic
/// templates stood in for paper architectures.</item>
/// <item>ADNDEF004: a baseline entry that no longer violates, so a fix cannot land without removing
/// its line.</item>
/// <item>ADNDEF005: a <c>[TaskVariantOf]</c> or <c>[DistinctImplementationOf]</c> declaration the code contradicts: it
/// names a model that is not a same-paper namesake, a task variant shares no network with its canonical model, or
/// a "distinct" pair shares one.</item>
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

    internal static readonly DiagnosticDescriptor UnverifiedVariant = new(
        "ADNDEF005", "Duplicate-model declaration contradicted by the code",
        "'{0}' declares [{1}] of '{2}', but {3}",
        "AiDotNet.ModelDefects", DiagnosticSeverity.Error, isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(DuplicatePaper, ResidualFreeAttention, UndeclaredSharedFactory, StaleBaseline, UnverifiedVariant);

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
            // The normalized paper each [ArchitectureFromPaper] names; a blank declaration is recorded as null so it
            // suppresses nothing.
            var declared = new ConcurrentDictionary<string, string?>(StringComparer.Ordinal);
            // What each type constructs (AiDotNet classes) or calls (LayerHelper factories), keyed by its outermost type.
            var uses = new ConcurrentBag<(string Owner, string Used)>();
            // [TaskVariantOf] (IsVariant) and [DistinctImplementationOf] declarations.
            var pairings = new ConcurrentBag<(string Type, string Target, bool IsVariant, Location Location)>();

            start.RegisterSymbolAction(symbolContext =>
            {
                var type = (INamedTypeSymbol)symbolContext.Symbol;
                if (type.TypeKind != TypeKind.Class || type.IsAbstract || type.Locations.Length == 0) return;
                string name = type.ToDisplayString();
                // A subclass runs its base's network (Bark is a facade over BarkModel).
                for (var baseType = type.BaseType; baseType is not null && baseType.ContainingAssembly?.Name == "AiDotNet"; baseType = baseType.BaseType)
                    uses.Add((Outermost(type), Outermost(baseType)));
                var attributes = type.GetAttributes();
                var primary = attributes.FirstOrDefault(a => a.AttributeClass?.Name == "ResearchPaperAttribute");
                if (primary is { ConstructorArguments.Length: >= 2 }
                    && primary.ConstructorArguments[1].Value is string url)
                {
                    string title = primary.ConstructorArguments[0].Value as string ?? url;
                    papers.Add((name, NormalizePaper(url), title, type.Locations[0], type.Name));
                }
                foreach (var pairing in attributes.Where(a => a.AttributeClass?.Name is "TaskVariantOfAttribute" or "DistinctImplementationOfAttribute"))
                {
                    if (pairing.ConstructorArguments.Length < 2 || pairing.ConstructorArguments[0].Value is not INamedTypeSymbol target
                        || pairing.ConstructorArguments[1].Value is not string text || string.IsNullOrWhiteSpace(text))
                        continue;
                    pairings.Add((name, target.OriginalDefinition.ToDisplayString(), pairing.AttributeClass!.Name == "TaskVariantOfAttribute",
                        pairing.ApplicationSyntaxReference?.GetSyntax(symbolContext.CancellationToken).GetLocation() ?? type.Locations[0]));
                }
                var declaration = attributes.FirstOrDefault(a => a.AttributeClass?.Name == "ArchitectureFromPaperAttribute");
                if (declaration is not null)
                {
                    string? declaredUrl = declaration.ConstructorArguments.Length >= 2
                        ? declaration.ConstructorArguments[0].Value as string : null;
                    string? reason = declaration.ConstructorArguments.Length >= 2
                        ? declaration.ConstructorArguments[1].Value as string : null;
                    declared[name] = declaredUrl is string paperUrl && !string.IsNullOrWhiteSpace(paperUrl)
                        && reason is string reasonText && !string.IsNullOrWhiteSpace(reasonText)
                        ? NormalizePaper(paperUrl)
                        : null;
                }
            }, SymbolKind.NamedType);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var invocation = (InvocationExpressionSyntax)nodeContext.Node;
                if (nodeContext.SemanticModel.GetSymbolInfo(invocation).Symbol is not IMethodSymbol method)
                    return;
                var owner = nodeContext.ContainingSymbol?.ContainingType;
                if (owner is null) return;
                // A static call into another type's code uses it (CLAP builds CLAPModel's audio tower).
                if (method.IsStatic && method.ContainingType is { } callee && callee.Name != "LayerHelper"
                    && callee.ContainingAssembly?.Name == "AiDotNet" && callee.TypeKind == TypeKind.Class && !IsSettings(callee))
                    uses.Add((Outermost(owner), Outermost(callee)));
                if (method.ContainingType?.Name != "LayerHelper" || owner.Name == "LayerHelper") return;
                uses.Add((Outermost(owner), "LayerHelper." + FactoryKey(method)));
                if (!method.Name.StartsWith("CreateDefault", StringComparison.Ordinal)) return;
                factoryUses.Add((FactoryKey(method), owner.ToDisplayString()));
            }, SyntaxKind.InvocationExpression);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                if (nodeContext.SemanticModel.GetTypeInfo(nodeContext.Node, nodeContext.CancellationToken).Type is not INamedTypeSymbol created
                    || created.TypeKind != TypeKind.Class || created.ContainingAssembly?.Name != "AiDotNet" || IsSettings(created))
                    return;
                var symbol = nodeContext.ContainingSymbol;
                var owner = symbol as INamedTypeSymbol ?? symbol?.ContainingType;
                if (owner is null) return;
                // The created type itself: building a network's nested Output record does not run the network.
                uses.Add((Outermost(owner), created.OriginalDefinition.ToDisplayString()));
            }, SyntaxKind.ObjectCreationExpression, SyntaxKind.ImplicitObjectCreationExpression);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var yield = (YieldStatementSyntax)nodeContext.Node;
                if (yield.Expression is null) return;
                var method = nodeContext.ContainingSymbol as IMethodSymbol;
                if (method?.ContainingType?.Name != "LayerHelper") return;
                if (YieldsAttention(nodeContext.SemanticModel, yield.Expression, nodeContext.CancellationToken))
                    residualFree.TryAdd(FactoryKey(method), method.Locations.FirstOrDefault() ?? Location.None);
            }, SyntaxKind.YieldReturnStatement);

            start.RegisterCompilationEndAction(end =>
            {
                // ADNDEF001: the same model implemented twice, i.e. the same name (ignoring a Model, Network, Net or TTS
                // suffix) AND the same primary paper. Paper alone is not enough: a model and its loss, or task heads and
                // size variants of one architecture, legitimately cite one paper (that matched 700+ types). Members joined
                // by a verified [TaskVariantOf] or [DistinctImplementationOf] are accounted for.
                var seenDuplicates = new HashSet<string>(StringComparer.Ordinal);
                var paperTypes = papers.GroupBy(p => p.Type).ToDictionary(g => g.Key, g => g.First().Paper, StringComparer.Ordinal);
                var graph = new UsageGraph(uses);
                var groupOf = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
                var duplicateGroups = papers.GroupBy(p => p.Paper + "|" + NormalizeModelName(p.SimpleName))
                    .Where(g => g.Select(p => p.Type).Distinct().Count() > 1).ToList();
                foreach (var group in duplicateGroups)
                {
                    var set = new HashSet<string>(group.Select(p => p.Type), StringComparer.Ordinal);
                    foreach (var type in set) groupOf[type] = set;
                }
                var joined = new Dictionary<string, string>(StringComparer.Ordinal);
                string Root(string t) { while (joined.TryGetValue(t, out var parent) && parent != t) t = parent; return t; }
                foreach (var pairing in pairings.OrderBy(p => p.Type, StringComparer.Ordinal).ThenBy(p => p.Target, StringComparer.Ordinal))
                {
                    string attribute = pairing.IsVariant ? "TaskVariantOf" : "DistinctImplementationOf";
                    if (!groupOf.TryGetValue(pairing.Type, out var group) || !group.Contains(pairing.Target) || pairing.Target == pairing.Type)
                    {
                        end.ReportDiagnostic(Diagnostic.Create(UnverifiedVariant, pairing.Location, pairing.Type, attribute, pairing.Target,
                            "the two are not models of the same paper under the same name; remove the declaration"));
                        continue;
                    }
                    string? network = graph.SharedExclusive(pairing.Type, pairing.Target, group, paperTypes[pairing.Type], paperTypes);
                    if (pairing.IsVariant && network is null)
                    {
                        end.ReportDiagnostic(Diagnostic.Create(UnverifiedVariant, pairing.Location, pairing.Type, attribute, pairing.Target,
                            "the two share no network that models of other papers do not also use; build both on one network"));
                        continue;
                    }
                    if (!pairing.IsVariant && network is not null)
                    {
                        end.ReportDiagnostic(Diagnostic.Create(UnverifiedVariant, pairing.Location, pairing.Type, attribute, pairing.Target,
                            "both run '" + network + "', which no model of another paper uses; they are one network, so declare [TaskVariantOf]"));
                        continue;
                    }
                    joined[Root(pairing.Type)] = Root(pairing.Target);
                }
                foreach (var group in duplicateGroups)
                {
                    var members = group.GroupBy(p => p.Type).Select(g => g.First()).OrderBy(p => p.Type, StringComparer.Ordinal).ToList();
                    if (members.Select(m => Root(m.Type)).Distinct(StringComparer.Ordinal).Count() == 1) continue;
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
                foreach (var entry in residualFree.Where(e => !ModelDefectBaselines.ResidualFreeFactories.Contains(e.Key)))
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
                        // A declaration suppresses the report only when it names the paper of ANOTHER user of this
                        // factory: that is the claim it makes. A blank one, or one naming an unrelated paper, does not.
                        if (declared.TryGetValue(user, out var declaredPaper) && declaredPaper is not null
                            && users.Any(other => other != user && paperOf[other] == declaredPaper))
                            continue;
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

    /// <summary>A model's simple name without the suffixes namesakes differ by (ResNetNetwork and ResNet, VITSTTS and VITS).</summary>
    internal static string NormalizeModelName(string name)
    {
        string[] suffixes = { "Network", "Model", "Net", "TTS" };
        bool stripped = true;
        while (stripped)
        {
            stripped = false;
            foreach (var suffix in suffixes)
            {
                if (name.Length > suffix.Length && name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                {
                    name = name.Substring(0, name.Length - suffix.Length);
                    stripped = true;
                    break;
                }
            }
        }
        return name.ToLowerInvariant();
    }

    // Options and configuration objects say what to build, not what is built: two models sharing one prove nothing.
    private static bool IsSettings(INamedTypeSymbol type)
    {
        for (var t = type; t is not null; t = t.BaseType)
            if (t.Name.EndsWith("Options", StringComparison.Ordinal) || t.Name.EndsWith("Configuration", StringComparison.Ordinal)
                || t.Name.EndsWith("Config", StringComparison.Ordinal) || t.Name == "Exception" || t.Name == "Attribute")
                return true;
        return false;
    }

    private static string Outermost(INamedTypeSymbol type)
    {
        var definition = type.OriginalDefinition;
        while (definition.ContainingType is not null) definition = definition.ContainingType.OriginalDefinition;
        return definition.ToDisplayString();
    }

    /// <summary>Which types construct which, and which call which LayerHelper factories.</summary>
    private sealed class UsageGraph
    {
        private readonly Dictionary<string, HashSet<string>> _forward = new(StringComparer.Ordinal);
        private readonly Dictionary<string, HashSet<string>> _reverse = new(StringComparer.Ordinal);

        public UsageGraph(IEnumerable<(string Owner, string Used)> uses)
        {
            foreach (var (owner, used) in uses)
            {
                if (owner == used) continue;
                Add(_forward, owner, used);
                Add(_reverse, used, owner);
            }
        }

        private static void Add(Dictionary<string, HashSet<string>> map, string key, string value)
        {
            if (!map.TryGetValue(key, out var set)) map[key] = set = new HashSet<string>(StringComparer.Ordinal);
            set.Add(value);
        }

        /// <summary>The type itself and everything it constructs or calls, transitively.</summary>
        private HashSet<string> Reach(string type)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal) { type };
            var queue = new Queue<string>();
            queue.Enqueue(type);
            while (queue.Count > 0)
                if (_forward.TryGetValue(queue.Dequeue(), out var next))
                    foreach (var n in next)
                        if (seen.Add(n)) queue.Enqueue(n);
            return seen;
        }

        /// <summary>Whether every model that reaches <paramref name="piece"/> is in <paramref name="group"/> or cites
        /// <paramref name="paper"/> (a language identifier on the wav2vec 2.0 network is a third task, not a copy).</summary>
        private bool UsedOnlyBy(string piece, HashSet<string> group, string paper, Dictionary<string, string> models)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal) { piece };
            var queue = new Queue<string>();
            queue.Enqueue(piece);
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                models.TryGetValue(current, out var cited);
                if (cited is not null && !group.Contains(current) && cited != paper) return false;
                // A model that reaches the piece through a group member (a detector built on the ResNet backbone) uses
                // that member; it does not implement the network again.
                if (current != piece && (group.Contains(current) || cited == paper)) continue;
                if (_reverse.TryGetValue(current, out var users))
                    foreach (var u in users)
                        if (seen.Add(u)) queue.Enqueue(u);
            }
            return true;
        }

        /// <summary>A piece both models reach that no model outside the group reaches, or null.</summary>
        public string? SharedExclusive(string a, string b, HashSet<string> group, string paper, Dictionary<string, string> models)
        {
            var reachB = Reach(b);
            return Reach(a).Where(reachB.Contains).OrderBy(p => p, StringComparer.Ordinal)
                .FirstOrDefault(piece => UsedOnlyBy(piece, group, paper, models));
        }
    }

    private static void ReportStale(CompilationAnalysisContext end, HashSet<string> baseline, HashSet<string> current, string name)
    {
        foreach (var entry in baseline.Where(e => !current.Contains(e)))
            end.ReportDiagnostic(Diagnostic.Create(StaleBaseline, Location.None, entry, name));
    }

    // One key per paper whatever link form a model cites: arXiv ids (with or without the 10.48550 DOI
    // prefix and version suffix), then DOIs, then the bare URL.
    //
    // Plain string scanning, not Regex: the patterns were linear, but a Regex match timeout is a wall-clock budget,
    // and on a saturated build machine the 1 s budget expired and failed the whole compile with AD0001
    // (RegexMatchTimeoutException) - a nondeterministic build failure on a deterministic input.
    internal static string NormalizePaper(string url)
    {
        string u = url.Trim().ToLowerInvariant().TrimEnd('/');
        // (?:arxiv\.org/(?:abs|pdf)/|arxiv\.)(\d{4}\.\d{4,5}), leftmost match.
        for (int i = 0; i < u.Length; i++)
        {
            string? id = null;
            if (string.CompareOrdinal(u, i, "arxiv.org/abs/", 0, 14) == 0 || string.CompareOrdinal(u, i, "arxiv.org/pdf/", 0, 14) == 0)
                id = ArxivId(u, i + 14);
            if (id is null && string.CompareOrdinal(u, i, "arxiv.", 0, 6) == 0)
                id = ArxivId(u, i + 6);
            if (id is not null) return "arxiv:" + id;
        }
        // doi\.org/(.+)$
        int doi = u.IndexOf("doi.org/", StringComparison.Ordinal);
        if (doi >= 0 && doi + 8 < u.Length) return "doi:" + u.Substring(doi + 8);
        // ^https?://(www\.)?
        int scheme = u.StartsWith("https://", StringComparison.Ordinal) ? 8 : u.StartsWith("http://", StringComparison.Ordinal) ? 7 : 0;
        if (scheme == 0) return u;
        string bare = u.Substring(scheme);
        return bare.StartsWith("www.", StringComparison.Ordinal) ? bare.Substring(4) : bare;
    }

    // \d{4}\.\d{4,5} at position start (greedy), or null.
    private static string? ArxivId(string u, int start)
    {
        int i = start;
        for (int k = 0; k < 4; k++, i++)
            if (i >= u.Length || u[i] < '0' || u[i] > '9') return null;
        if (i >= u.Length || u[i] != '.') return null;
        i++;
        int digits = 0;
        while (digits < 5 && i < u.Length && u[i] >= '0' && u[i] <= '9')
        {
            i++;
            digits++;
        }
        return digits >= 4 ? u.Substring(start, i - start) : null;
    }

    /// <summary>
    /// The identity of a LayerHelper factory: its name when LayerHelper declares only one method of that name (so the
    /// baselines keep their plain entries), otherwise the name with its parameter types, so two overloads serving
    /// different papers are never merged into one "shared" factory.
    /// </summary>
    internal static string FactoryKey(IMethodSymbol method)
    {
        var definition = method.OriginalDefinition;
        bool overloaded = definition.ContainingType.GetMembers(definition.Name).OfType<IMethodSymbol>().Skip(1).Any();
        if (!overloaded) return definition.Name;
        return definition.Name + "("
            + string.Join(", ", definition.Parameters.Select(p => p.Type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)))
            + ")";
    }

    /// <summary>
    /// Whether a yield returns a bare MultiHeadAttentionLayer, looking through conversions (<c>(ILayer&lt;T&gt;)new ...</c>)
    /// and through a yielded local to every value assigned to it in the method.
    /// </summary>
    private static bool YieldsAttention(SemanticModel model, ExpressionSyntax expression, System.Threading.CancellationToken cancellationToken)
    {
        var operation = model.GetOperation(expression, cancellationToken);
        return IsAttentionValue(model, operation, cancellationToken, depth: 0);
    }

    private static bool IsAttentionValue(SemanticModel model, Microsoft.CodeAnalysis.IOperation? operation,
        System.Threading.CancellationToken cancellationToken, int depth)
    {
        while (operation is Microsoft.CodeAnalysis.Operations.IConversionOperation conversion)
            operation = conversion.Operand;
        if (operation is null || depth > 4) return false;
        if (operation.Type?.Name == "MultiHeadAttentionLayer") return true;
        if (operation is not Microsoft.CodeAnalysis.Operations.ILocalReferenceOperation localReference) return false;

        // Every value the local can hold: its initializer and each assignment in the enclosing method body.
        var local = localReference.Local;
        var body = expressionBody(operation.Syntax);
        if (body is null) return false;
        foreach (var declarator in body.DescendantNodes().OfType<VariableDeclaratorSyntax>())
        {
            if (declarator.Initializer is null
                || !SymbolEqualityComparer.Default.Equals(model.GetDeclaredSymbol(declarator, cancellationToken), local)) continue;
            if (IsAttentionValue(model, model.GetOperation(declarator.Initializer.Value, cancellationToken), cancellationToken, depth + 1))
                return true;
        }
        foreach (var assignment in body.DescendantNodes().OfType<AssignmentExpressionSyntax>()
                     .Where(a => SymbolEqualityComparer.Default.Equals(model.GetSymbolInfo(a.Left, cancellationToken).Symbol, local)))
        {
            if (IsAttentionValue(model, model.GetOperation(assignment.Right, cancellationToken), cancellationToken, depth + 1))
                return true;
        }
        return false;

        static SyntaxNode? expressionBody(SyntaxNode node) =>
            node.AncestorsAndSelf().FirstOrDefault(n => n is BaseMethodDeclarationSyntax or LocalFunctionStatementSyntax or AccessorDeclarationSyntax);
    }
}