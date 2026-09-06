using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps <c>[PaperOptimizer]</c> declarations honest, and reports models that still train at the
/// optimizer's generic defaults instead of their paper's settings.
/// </summary>
/// <remarks>
/// <para>
/// Issue #1928: 685 optimizer constructions across 592 files pass no options at all, so those
/// models silently inherit the optimizer class's own defaults — for AdamW that is
/// <c>WeightDecay = 0.01</c> applied to every parameter on every step, which commit
/// <c>1972a510a</c> showed to be actively wrong for span-based NER.
/// </para>
/// <para>
/// The hard part of that issue is not mechanism, it is data: each value has to be read out of a
/// paper. This analyzer exists to make the remaining backlog countable rather than invisible, and
/// to stop the two ways a declaration can be worse than no declaration at all — an uncited value,
/// and an ambiguous one.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class PaperOptimizerAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A model citing a paper but not declaring the optimizer settings that paper states.</summary>
    /// <remarks>
    /// Info, not Warning, deliberately. The backlog starts in the hundreds, and several hundred
    /// warnings on day one is the kind of noise that gets suppressed wholesale rather than worked
    /// down — which would defeat the point of counting it. This follows the ladder AIDN087 already
    /// documents: report while the backlog is large, promote as it shrinks.
    /// </remarks>
    private static readonly DiagnosticDescriptor MissingPaperOptimizer = new(
        "AIDN101",
        "Model cites a paper but does not declare the optimizer settings that paper specifies",
        "'{0}' has [ResearchPaper] but no [PaperOptimizer], so it trains at the optimizer class's "
            + "generic defaults rather than its paper's. Add [PaperOptimizer(...)] with a Source "
            + "naming the section the values come from, or leave it undeclared if the paper does "
            + "not state them",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Models that declare no paper hyperparameters silently inherit the optimizer "
            + "class defaults, which rarely match the published training recipe.");

    /// <summary>A declared hyperparameter with no citation.</summary>
    /// <remarks>
    /// Error rather than Warning because an uncited value is worse than an absent one. An absent
    /// value falls back to a documented library default; an invented one looks authoritative,
    /// propagates into results, and nobody re-derives it.
    /// </remarks>
    private static readonly DiagnosticDescriptor MissingSource = new(
        "AIDN102",
        "Declared paper hyperparameters must cite where they come from",
        "'{0}' declares paper hyperparameters without a Source. Name the section or table they "
            + "come from (for example Source = \"Sec. 4.1, Table 8\"), or remove the values -- an "
            + "uncited number reads as authoritative and will not be re-checked",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A hyperparameter that cannot be pointed at a section of the paper should not "
            + "be declared; the library default is the honest answer.");

    /// <summary>Two declarations competing for the same optimizer and variant.</summary>
    private static readonly DiagnosticDescriptor DuplicateDeclaration = new(
        "AIDN103",
        "Duplicate [PaperOptimizer] for the same optimizer and variant",
        "'{0}' declares [PaperOptimizer] more than once for {1} and variant '{2}'. Resolution picks "
            + "one of them, so the other is silently dead -- give each declaration a distinct "
            + "Variant, or keep a single entry",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Ambiguous declarations make the applied hyperparameters depend on attribute "
            + "ordering rather than on what the paper says.");


    /// <summary>A declared recipe that nothing routes through the factory, so it never applies.</summary>
    /// <remarks>
    /// The declaration is only half the work: the model must also build its optimizer through
    /// <c>PaperOptimizerFactory.CreateFor</c>, or the recipe sits in source looking authoritative
    /// while the model keeps training on its hardcoded default. That failure is invisible at
    /// runtime -- nothing throws, the numbers are simply not the paper's -- which is exactly the
    /// class of defect #1928 is about, so it is worth a build error rather than a note.
    /// </remarks>
    private static readonly DiagnosticDescriptor DeclarationNotWired = new(
        "AIDN104",
        "Declared paper recipe is never used, because the optimizer is still hardcoded",
        "'{0}' declares [PaperOptimizer] but constructs its optimizer directly, so the recipe is "
            + "inert. Route the construction through PaperOptimizerFactory.CreateFor, keeping the "
            + "existing constructor as the fallback: optimizer ?? PaperOptimizerFactory.CreateFor(this) "
            + "?? new SomeOptimizer(this)",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A recipe that is declared but not wired reads as if the model trains at its "
            + "paper's settings when it does not.");

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(MissingPaperOptimizer, MissingSource, DuplicateDeclaration, DeclarationNotWired);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeClass, SyntaxKind.ClassDeclaration);
    }

    private static void AnalyzeClass(SyntaxNodeAnalysisContext context)
    {
        var declaration = (ClassDeclarationSyntax)context.Node;
        if (declaration.Modifiers.Any(SyntaxKind.AbstractKeyword)) return;

        if (context.SemanticModel.GetDeclaredSymbol(declaration) is not INamedTypeSymbol type) return;

        var paperOptimizers = new List<AttributeData>();
        bool citesPaper = false;

        foreach (var attribute in type.GetAttributes())
        {
            string? name = attribute.AttributeClass?.Name;
            if (name is "ResearchPaperAttribute" or "ResearchPaper") citesPaper = true;
            else if (name is "PaperOptimizerAttribute" or "PaperOptimizer") paperOptimizers.Add(attribute);
        }

        if (paperOptimizers.Count == 0)
        {
            // Two conditions, both required. A type with no [ResearchPaper] has no published
            // recipe to be missing. And a paper-citing type that never constructs an optimizer --
            // an analytic or classical model -- has no training defaults to get wrong, so
            // reporting it would be noise. Flagging on [ResearchPaper] alone reports 2098 types;
            // requiring an actual no-options optimizer construction reports only those that
            // silently inherit the optimizer class's defaults, which is what #1928 is about.
            if (citesPaper && ConstructsOptimizerWithoutOptions(declaration))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    MissingPaperOptimizer, declaration.Identifier.GetLocation(), type.Name));
            }

            return;
        }

        var seen = new HashSet<string>();
        foreach (var attribute in paperOptimizers)
        {
            Location location = attribute.ApplicationSyntaxReference is { } reference
                ? Location.Create(reference.SyntaxTree, reference.Span)
                : declaration.Identifier.GetLocation();

            string variant = GetStringArgument(attribute, "Variant") ?? string.Empty;
            string source = GetStringArgument(attribute, "Source") ?? string.Empty;
            string optimizer = attribute.ConstructorArguments.Length > 0
                ? attribute.ConstructorArguments[0].Value?.ToString() ?? "?"
                : "?";

            if (DeclaresAnyHyperparameter(attribute) && string.IsNullOrWhiteSpace(source))
            {
                context.ReportDiagnostic(Diagnostic.Create(MissingSource, location, type.Name));
            }

            if (!seen.Add(optimizer + "|" + variant))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    DuplicateDeclaration, location, type.Name, optimizer,
                    variant.Length == 0 ? "(default)" : variant));
            }
        }

        if (ConstructsOptimizerWithoutOptions(declaration) && !RoutesThroughPaperOptimizerFactory(declaration))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                DeclarationNotWired, declaration.Identifier.GetLocation(), type.Name));
        }
    }

    /// <summary>True when the class builds its optimizer through the paper-recipe factory.</summary>
    private static bool RoutesThroughPaperOptimizerFactory(ClassDeclarationSyntax declaration)
        => declaration.DescendantNodes()
            .OfType<MemberAccessExpressionSyntax>()
            .Any(access => access.Name.Identifier.Text == "CreateFor"
                && access.Expression is IdentifierNameSyntax { Identifier.Text: "PaperOptimizerFactory" });

    /// <summary>
    /// True when the class contains a <c>new SomethingOptimizer&lt;...&gt;(this)</c> with no options
    /// argument -- the exact shape that silently inherits the optimizer class's own defaults.
    /// </summary>
    /// <remarks>
    /// 670 of the 685 sites #1928 counts are literally <c>optimizer ?? new X(this)</c>, so matching
    /// a single-argument construction is both precise and sufficient. A construction that already
    /// passes an options object is not reported, because that model has made a deliberate choice --
    /// whether or not it matches its paper is a different question from this rule's.
    /// </remarks>
    private static bool ConstructsOptimizerWithoutOptions(ClassDeclarationSyntax declaration)
    {
        foreach (var creation in declaration.DescendantNodes().OfType<ObjectCreationExpressionSyntax>())
        {
            string name = creation.Type switch
            {
                GenericNameSyntax generic => generic.Identifier.Text,
                IdentifierNameSyntax identifier => identifier.Identifier.Text,
                QualifiedNameSyntax qualified => qualified.Right.Identifier.Text,
                _ => string.Empty,
            };

            if (!name.EndsWith("Optimizer", System.StringComparison.Ordinal)) continue;
            if (creation.ArgumentList is null) continue;
            if (creation.ArgumentList.Arguments.Count != 1) continue;

            return true;
        }

        return false;
    }

    /// <summary>
    /// True when the declaration sets at least one hyperparameter.
    /// </summary>
    /// <remarks>
    /// The attribute uses <c>NaN</c> as its "unset" marker, because it is the one double that can
    /// never be a legitimate hyperparameter. Testing against 0 instead would misread a deliberate
    /// <c>WeightDecay = 0</c> — the very case that matters most, since it is how a model says "this
    /// paper specifies plain Adam, do not apply AdamW's decoupled decay".
    /// </remarks>
    private static bool DeclaresAnyHyperparameter(AttributeData attribute)
    {
        foreach (var named in attribute.NamedArguments)
        {
            if (named.Key is not ("LearningRate" or "WeightDecay" or "Beta1" or "Beta2" or "Epsilon"))
                continue;

            if (named.Value.Value is double value && !double.IsNaN(value)) return true;
        }

        return false;
    }

    private static string? GetStringArgument(AttributeData attribute, string name)
    {
        foreach (var named in attribute.NamedArguments)
        {
            if (named.Key == name) return named.Value.Value as string;
        }

        return null;
    }
}
