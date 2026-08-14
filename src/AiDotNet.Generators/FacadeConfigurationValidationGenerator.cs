using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental generator that enforces, at BUILD time, that every fluent
/// <c>Configure*</c> method on the facade actually does something with what it was given.
/// </summary>
/// <remarks>
/// <para>
/// The bug class this exists to kill: a <c>Configure*</c> method accepts a value, stores it in a
/// private field, returns <c>this</c>, and nothing ever reads that field. The caller gets no
/// exception and no warning -- just a model built as if they had never called the method. Silent
/// wrong behaviour is worse than a loud failure, and it is invisible in a diff because each half
/// (the assignment, the missing read) looks fine on its own.
/// </para>
/// <para>
/// This is mechanically decidable, so it belongs in an analyzer rather than a review checklist,
/// for the reasons set out on <see cref="GoldenPatternValidationGenerator"/>: the analyzer sees
/// 100% of the code on every build and cannot be merged around.
/// </para>
/// <para>
/// <b>Why this is cheap.</b> The fields in question are <c>private</c>, so C# accessibility means a
/// read can only appear inside the declaring type's own (partial) declarations. The analyzer
/// therefore only walks syntax trees that declare the facade type, not the whole compilation.
/// </para>
/// <para>
/// <b>Severity policy.</b> Ships as Warning, matching AIDN070-076. Ratchet to Error once the
/// backlog is zero.
/// </para>
/// <para>
/// <b>Rule-id range.</b> Starts at AIDN090 deliberately: 001-076 are in use here, and the
/// 080-089 block is claimed by analyzers added on the model-family branch (AIDN082/084/085/087),
/// so 090+ avoids an id collision when those branches meet.
/// </para>
/// </remarks>
[Generator]
public class FacadeConfigurationValidationGenerator : IIncrementalGenerator
{
    private const string Category = "AiDotNet.FacadeConfiguration";

    /// <summary>The facade types whose Configure* surface is validated.</summary>
    private static readonly string[] FacadeTypeNames = { "AiModelBuilder" };

    internal static readonly DiagnosticDescriptor ConfiguredValueNeverRead = new(
        id: "AIDN090",
        title: "Configure* method stores a value nothing ever reads",
        messageFormat: "'{0}' assigns '{1}', but nothing ever reads it -- the configuration is silently dropped",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A fluent Configure* method must have an observable effect. Assigning a private field that " +
                     "no code reads means the caller's configuration is accepted and discarded with no error: the " +
                     "model is built as if the method had never been called. Either consume the field where the " +
                     "model is built, route the value into the pipeline, or delete the field if the method already " +
                     "takes effect by another route.");

    internal static readonly DiagnosticDescriptor ConfiguredValueOnlyExposed = new(
        id: "AIDN091",
        title: "Configured value is only reachable through an accessor nobody calls",
        messageFormat: "'{0}' is read only by '{1}', which has no callers -- the configuration is still effectively unused",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Exposing a configured field through a property does not by itself make the configuration take " +
                     "effect. If the accessor has no callers, the value is stored, exposed, and still never acted on. " +
                     "Consume it, or remove the configuration surface that promises it is honoured.");

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Only the syntax trees that declare a facade partial can contain reads of its private
        // fields, so the analysis is scoped to those trees rather than the whole compilation.
        var facadeTrees = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds
                    && FacadeTypeNames.Contains(cds.Identifier.ValueText),
                transform: static (ctx, _) => (ClassDeclarationSyntax)ctx.Node)
            .Collect();

        context.RegisterSourceOutput(facadeTrees, static (spc, declarations) =>
        {
            if (declarations.IsDefaultOrEmpty) return;

            var fields = new Dictionary<string, FieldInfo>(System.StringComparer.Ordinal);
            var reads = new Dictionary<string, int>(System.StringComparer.Ordinal);
            var accessorReads = new Dictionary<string, string>(System.StringComparer.Ordinal);
            var accessorNames = new HashSet<string>(System.StringComparer.Ordinal);
            var accessorCalls = new Dictionary<string, int>(System.StringComparer.Ordinal);
            var assigningMethod = new Dictionary<string, string>(System.StringComparer.Ordinal);

            foreach (var decl in declarations)
            {
                CollectFields(decl, fields);
                CollectAccessors(decl, accessorReads, accessorNames);
            }

            foreach (var decl in declarations)
            {
                CollectUsages(decl, fields, reads, accessorNames, accessorCalls, assigningMethod);
            }

            foreach (var kvp in fields)
            {
                string name = kvp.Key;
                var info = kvp.Value;

                // Only fields a Configure* method actually assigns are in scope. Plain internal
                // state that happens to be unused is a different (and much noisier) problem.
                if (!assigningMethod.TryGetValue(name, out var method)) continue;

                reads.TryGetValue(name, out int readCount);
                if (readCount > 0) continue;

                if (accessorReads.TryGetValue(name, out var accessor))
                {
                    accessorCalls.TryGetValue(accessor, out int callCount);
                    if (callCount == 0)
                    {
                        spc.ReportDiagnostic(Diagnostic.Create(
                            ConfiguredValueOnlyExposed, info.Location, name, accessor));
                    }
                    continue;
                }

                spc.ReportDiagnostic(Diagnostic.Create(
                    ConfiguredValueNeverRead, info.Location, method, name));
            }
        });
    }

    private readonly struct FieldInfo
    {
        internal FieldInfo(Location location) => Location = location;
        internal Location Location { get; }
    }

    private static void CollectFields(ClassDeclarationSyntax decl, Dictionary<string, FieldInfo> fields)
    {
        foreach (var member in decl.Members.OfType<FieldDeclarationSyntax>())
        {
            if (!member.Modifiers.Any(SyntaxKind.PrivateKeyword)) continue;
            foreach (var v in member.Declaration.Variables)
            {
                string name = v.Identifier.ValueText;
                if (!fields.ContainsKey(name))
                    fields[name] = new FieldInfo(v.Identifier.GetLocation());
            }
        }
    }

    /// <summary>
    /// Records expression-bodied accessors of the shape <c>X => _field;</c>. These are the reason a
    /// naive "field is never read" check produces false negatives: the accessor IS a read, so the
    /// field looks live even when nothing calls the accessor.
    /// </summary>
    private static void CollectAccessors(
        ClassDeclarationSyntax decl,
        Dictionary<string, string> accessorReads,
        HashSet<string> accessorNames)
    {
        foreach (var prop in decl.Members.OfType<PropertyDeclarationSyntax>())
        {
            accessorNames.Add(prop.Identifier.ValueText);
            if (prop.ExpressionBody?.Expression is IdentifierNameSyntax id
                && id.Identifier.ValueText.StartsWith("_", System.StringComparison.Ordinal))
            {
                accessorReads[id.Identifier.ValueText] = prop.Identifier.ValueText;
            }
        }
    }

    private static void CollectUsages(
        ClassDeclarationSyntax decl,
        Dictionary<string, FieldInfo> fields,
        Dictionary<string, int> reads,
        HashSet<string> accessorNames,
        Dictionary<string, int> accessorCalls,
        Dictionary<string, string> assigningMethod)
    {
        foreach (var id in decl.DescendantNodes().OfType<IdentifierNameSyntax>())
        {
            string name = id.Identifier.ValueText;

            if (accessorNames.Contains(name) && !IsOwnDeclaration(id))
            {
                accessorCalls.TryGetValue(name, out int c);
                accessorCalls[name] = c + 1;
            }

            if (!fields.ContainsKey(name)) continue;

            if (IsAssignmentTarget(id))
            {
                var method = id.FirstAncestorOrSelf<MethodDeclarationSyntax>();
                string? methodName = method?.Identifier.ValueText;
                if (methodName is not null
                    && methodName.StartsWith("Configure", System.StringComparison.Ordinal)
                    && !assigningMethod.ContainsKey(name))
                {
                    assigningMethod[name] = methodName;
                }
                continue;
            }

            // An expression-bodied accessor read is tracked separately, not as a real read.
            if (id.Parent is ArrowExpressionClauseSyntax { Parent: PropertyDeclarationSyntax }) continue;

            reads.TryGetValue(name, out int r);
            reads[name] = r + 1;
        }
    }

    private static bool IsOwnDeclaration(IdentifierNameSyntax id) =>
        id.FirstAncestorOrSelf<PropertyDeclarationSyntax>() is { } prop
        && prop.Identifier.ValueText == id.Identifier.ValueText;

    /// <summary>
    /// True when this identifier is the target of a plain assignment (<c>_x = v</c>). Compound and
    /// null-coalescing assignments (<c>_x += v</c>, <c>_x ??= v</c>) read the field first, so they
    /// are deliberately NOT treated as pure writes.
    /// </summary>
    private static bool IsAssignmentTarget(IdentifierNameSyntax id) =>
        id.Parent is AssignmentExpressionSyntax assign
        && assign.Left == id
        && assign.IsKind(SyntaxKind.SimpleAssignmentExpression);
}
