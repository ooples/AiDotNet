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
/// Keeps <c>[PaperOptimizer]</c> declarations honest, and reports models that still train at the
/// optimizer's generic defaults instead of their paper's settings.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class PaperOptimizerAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor MissingPaperOptimizer = new(
        "AIDN101",
        "Model cites a paper but does not declare the optimizer settings that paper specifies",
        "'{0}' has [ResearchPaper] but no [PaperOptimizer], so it trains at the optimizer class's "
            + "generic defaults rather than its paper's. Add [PaperOptimizer(...)] with a Source "
            + "naming the section the values come from, or leave it undeclared if the paper does "
            + "not state them.",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Models that declare no paper hyperparameters silently inherit the optimizer "
            + "class defaults, which rarely match the published training recipe.");

    private static readonly DiagnosticDescriptor MissingSource = new(
        "AIDN102",
        "Every declared paper optimizer must cite its source",
        "'{0}' declares [PaperOptimizer] without a Source. Name the section or table the optimizer "
            + "recipe comes from (for example Source = \"Sec. 4.1, Table 8\"), or remove the "
            + "declaration -- an uncited recipe reads as authoritative and will not be re-checked.",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Every optimizer recipe, including an optimizer-only declaration, is a claim "
            + "about a paper and must identify where that claim can be verified.");

    private static readonly DiagnosticDescriptor DuplicateDeclaration = new(
        "AIDN103",
        "Duplicate [PaperOptimizer] for the same variant",
        "'{0}' declares [PaperOptimizer] more than once for variant '{1}'. Resolution selects by "
            + "variant before optimizer kind, so one recipe is silently dead -- give each "
            + "declaration a distinct Variant, or keep a single entry.",
        "AiDotNet.PaperFidelity",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A model variant identifies one complete training recipe. Multiple optimizer "
            + "kinds for the same variant make resolution depend on attribute ordering.");

    private static readonly DiagnosticDescriptor DeclarationNotWired = new(
        "AIDN104",
        "Declared paper recipe is never used, because the optimizer is still hardcoded",
        "'{0}' declares [PaperOptimizer] but constructs its optimizer directly, so the recipe is "
            + "inert. Route the construction through PaperOptimizerFactory.CreateFor, keeping the "
            + "existing constructor as the fallback: optimizer ?? PaperOptimizerFactory.CreateFor(this) "
            + "?? new SomeOptimizer(this).",
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
        context.RegisterCompilationStartAction(compilationContext =>
            compilationContext.RegisterSymbolStartAction(StartTypeAnalysis, SymbolKind.NamedType));
    }

    private static void StartTypeAnalysis(SymbolStartAnalysisContext context)
    {
        var type = (INamedTypeSymbol)context.Symbol;
        if (type.TypeKind != TypeKind.Class || type.DeclaringSyntaxReferences.Length == 0) return;

        bool citesPaper = type.GetAttributes().Any(IsResearchPaper);
        bool ownsRecipe = type.GetAttributes().Any(IsPaperOptimizer);
        bool hasEffectiveRecipe = EnumerateEffectiveRecipes(type).Any(IsSpecifiedRecipe);
        if (!citesPaper && !ownsRecipe && !hasEffectiveRecipe) return;

        // Symbol-start/end provides a SemanticModel for every partial declaration while retaining
        // one final reporting point for the complete type. This avoids both duplicate partial-type
        // diagnostics and Compilation.GetSemanticModel inside an analyzer.
        var constructions = new ConcurrentBag<bool>();
        if (!type.IsAbstract)
        {
            context.RegisterSyntaxNodeAction(syntaxContext =>
            {
                var creation = (ObjectCreationExpressionSyntax)syntaxContext.Node;
                if (creation.ArgumentList?.Arguments.Count != 1) return;
                if (syntaxContext.SemanticModel.GetTypeInfo(creation).Type is not INamedTypeSymbol createdType)
                    return;
                if (!ImplementsGradientOptimizer(createdType)) return;

                TypeDeclarationSyntax? ownerDeclaration = creation.Ancestors()
                    .OfType<TypeDeclarationSyntax>()
                    .FirstOrDefault();
                if (ownerDeclaration is null) return;
                if (!SymbolEqualityComparer.Default.Equals(
                        syntaxContext.SemanticModel.GetDeclaredSymbol(ownerDeclaration), type))
                    return;

                constructions.Add(SelectionUsesFactory(creation, syntaxContext.SemanticModel));
            }, SyntaxKind.ObjectCreationExpression);
        }

        context.RegisterSymbolEndAction(endContext => AnalyzeType(
            endContext, type, constructions, citesPaper, hasEffectiveRecipe));
    }

    private static void AnalyzeType(
        SymbolAnalysisContext context,
        INamedTypeSymbol type,
        ConcurrentBag<bool> constructions,
        bool citesPaper,
        bool hasEffectiveRecipe)
    {

        var declarations = type.DeclaringSyntaxReferences
            .Select(reference => reference.GetSyntax(context.CancellationToken))
            .OfType<ClassDeclarationSyntax>()
            .ToArray();
        if (declarations.Length == 0) return;

        var declaredRecipes = type.GetAttributes().Where(IsPaperOptimizer).ToArray();
        ValidateOwnedRecipes(context, type, declarations[0], declaredRecipes);

        // Abstract bases own and are diagnosed for their declarations, but wiring is a concrete
        // model responsibility. Derived types consume inherited recipes without repeating their
        // base's AIDN102/AIDN103 diagnostics.
        if (type.IsAbstract) return;

        if (constructions.IsEmpty) return;

        if (!hasEffectiveRecipe)
        {
            if (citesPaper)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    MissingPaperOptimizer, declarations[0].Identifier.GetLocation(), type.Name));
            }

            return;
        }

        if (constructions.Any(usesFactory => !usesFactory))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                DeclarationNotWired, declarations[0].Identifier.GetLocation(), type.Name));
        }
    }

    private static void ValidateOwnedRecipes(
        SymbolAnalysisContext context,
        INamedTypeSymbol type,
        ClassDeclarationSyntax fallbackDeclaration,
        IReadOnlyList<AttributeData> declaredRecipes)
    {
        // Seed with inherited variants so a derived declaration cannot shadow a base declaration.
        // Only the newly introduced (derived) declaration is reported, so an invalid base recipe
        // still produces exactly one diagnostic at its owning declaration.
        var seenVariants = new HashSet<string>(StringComparer.Ordinal);
        for (INamedTypeSymbol? current = type.BaseType; current is not null; current = current.BaseType)
        {
            foreach (var inherited in current.GetAttributes().Where(IsPaperOptimizer))
            {
                seenVariants.Add(RecipeKey(inherited));
            }
        }

        foreach (var attribute in declaredRecipes)
        {
            Location location = attribute.ApplicationSyntaxReference is { } reference
                ? Location.Create(reference.SyntaxTree, reference.Span)
                : fallbackDeclaration.Identifier.GetLocation();
            string source = GetStringArgument(attribute, "Source") ?? string.Empty;
            string variant = GetStringArgument(attribute, "Variant") ?? string.Empty;
            string key = RecipeKey(attribute);

            if (string.IsNullOrWhiteSpace(source))
            {
                context.ReportDiagnostic(Diagnostic.Create(MissingSource, location, type.Name));
            }

            if (!seenVariants.Add(key))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    DuplicateDeclaration,
                    location,
                    type.Name,
                    variant.Length == 0 ? "(default)" : variant));
            }
        }
    }

    private static bool ImplementsGradientOptimizer(INamedTypeSymbol type)
        => type.AllInterfaces.Any(candidate => candidate.Name == "IGradientBasedOptimizer"
            && candidate.Arity == 3
            && candidate.ContainingNamespace.ToDisplayString() == "AiDotNet.Interfaces");

    /// <summary>
    /// Checks the expression that selects the directly-created fallback. A factory call elsewhere
    /// in the class is not evidence that its result controls this optimizer assignment.
    /// </summary>
    private static bool SelectionUsesFactory(
        ObjectCreationExpressionSyntax creation,
        SemanticModel semanticModel)
    {
        SyntaxNode selection = creation;
        while (selection.Parent is ParenthesizedExpressionSyntax
            or CastExpressionSyntax
            or BinaryExpressionSyntax
            or ConditionalExpressionSyntax)
        {
            selection = selection.Parent;
        }

        if (selection.Parent is EqualsValueClauseSyntax equalsValue)
            selection = equalsValue.Value;
        else if (selection.Parent is AssignmentExpressionSyntax assignment
            && ReferenceEquals(assignment.Right, selection))
            selection = assignment.Right;
        else if (selection.Parent is ArrowExpressionClauseSyntax arrow)
            selection = arrow.Expression;
        else if (selection.Parent is ReturnStatementSyntax returnStatement
            && returnStatement.Expression is not null)
            selection = returnStatement.Expression;

        return selection.DescendantNodesAndSelf()
            .OfType<InvocationExpressionSyntax>()
            .Any(invocation => IsPaperOptimizerFactoryCall(semanticModel, invocation));
    }

    private static bool IsPaperOptimizerFactoryCall(
        SemanticModel semanticModel,
        InvocationExpressionSyntax invocation)
    {
        if (semanticModel.GetSymbolInfo(invocation).Symbol is not IMethodSymbol method) return false;

        // VerifyHandBuilt counts as reaching the factory as much as CreateFor does. Models that
        // already build their paper correctly -- a dimension-aware Noam schedule, options that
        // deliberately disable clipping the paper does not use -- keep their own optimizer and have
        // the declaration verify it. Recognising only CreateFor would report those as unwired and
        // push them towards being replaced by something less faithful.
        return method.Name is "CreateFor" or "VerifyHandBuilt"
            && method.ContainingType.Name == "PaperOptimizerFactory"
            && method.ContainingType.ContainingNamespace.ToDisplayString() == "AiDotNet.Optimizers";
    }

    /// <summary>The key a declaration is unique on: its variant AND its component.</summary>
    /// <remarks>
    /// Component is part of the identity, not a detail of it. A composite model legitimately
    /// declares one row per part -- Stable Audio Open gives separate rates for its autoencoder,
    /// its discriminators and its DiT -- and keying on variant alone would report every one of
    /// those as a duplicate of the others.
    /// </remarks>
    private static string RecipeKey(AttributeData attribute)
        => (GetStringArgument(attribute, "Variant") ?? string.Empty)
            + "|" + (GetStringArgument(attribute, "Component") ?? string.Empty);

    private static IEnumerable<AttributeData> EnumerateEffectiveRecipes(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.BaseType)
        {
            foreach (var attribute in current.GetAttributes().Where(IsPaperOptimizer))
            {
                yield return attribute;
            }
        }
    }

    private static bool IsSpecifiedRecipe(AttributeData attribute)
    {
        if (attribute.ConstructorArguments.Length == 0) return false;
        object? value = attribute.ConstructorArguments[0].Value;
        return value is not null && Convert.ToInt64(value) != 0;
    }

    private static bool IsPaperOptimizer(AttributeData attribute)
        => attribute.AttributeClass?.Name is "PaperOptimizerAttribute" or "PaperOptimizer";

    private static bool IsResearchPaper(AttributeData attribute)
        => attribute.AttributeClass?.Name is "ResearchPaperAttribute" or "ResearchPaper";

    private static string? GetStringArgument(AttributeData attribute, string name)
    {
        foreach (var named in attribute.NamedArguments.Where(named => named.Key == name))
        {
            return named.Value.Value as string;
        }

        return null;
    }

}
