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
/// Compile-time gates that keep a model's options class honest: every option is consumed by
/// something, and the values a paper states are declared where a test can check them.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>ADNOPT001: a public settable/init property of a model's options class that nothing in the
/// library reads. Epic #2090 found 104 of 117 neural-network models storing an options object and
/// never reading it; Dreamer, World Models and Decision Transformer all passed every generated test
/// because those tests only proved the models RUN.</item>
/// <item>ADNOPT002: an options class configuring a model that cites a <c>[ResearchPaper]</c>, with no
/// <c>[PaperDefault]</c> on any of its properties.</item>
/// <item>ADNOPT003: a <c>[PaperDefault]</c> with no source, or whose value cannot be assigned to the
/// property it annotates.</item>
/// <item>ADNOPT004: an <c>[OptionBehaviourExempt]</c> with a blank reason.</item>
/// <item>ADNOPT005: a baseline entry in <see cref="OptionsFidelityBaselines"/> that no longer violates,
/// so a fix cannot land without removing its line.</item>
/// </list>
/// <para>
/// <b>What counts as a read (ADNOPT001).</b> Reads outside the options hierarchy count directly.
/// Reads inside value-producing options helpers and computed getters count only when a consumer
/// reaches them, including calls through other options helpers. Uncalled helpers, options constructors,
/// Clone, nameof, plain writes and validation-only void methods do not count. Reads may be implicit
/// identifiers as well as member accesses. The analysis is assembly-wide; generated runtime option-
/// sensitivity tests then check each model individually. Generated consumers count too, but diagnostics
/// are reported only in user code.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class OptionsFidelityAnalyzer : DiagnosticAnalyzer
{
    internal const string ModelOptionsMetadataName = "AiDotNet.Models.Options.ModelOptions";
    internal const string PaperDefaultAttributeName = "AiDotNet.Attributes.PaperDefaultAttribute";
    internal const string OptionBehaviourExemptAttributeName = "AiDotNet.Attributes.OptionBehaviourExemptAttribute";

    internal static readonly DiagnosticDescriptor UnreadOption = new(
        "ADNOPT001", "Options property is never read by the model",
        "'{0}' is never read outside its options class, its copy constructor and Clone: no model, base class or helper consumes it, so setting it changes nothing. Read it where the model is built or trained, or delete it.",
        "AiDotNet.OptionsFidelity", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "An option that nothing reads advertises configurability that does not exist. Existing violations are listed in OptionsFidelityBaselines.UnreadOptions, which may only shrink.",
        customTags: WellKnownDiagnosticTags.CompilationEnd);

    internal static readonly DiagnosticDescriptor MissingPaperDefaults = new(
        "ADNOPT002", "Options class of a paper model declares no paper defaults",
        "'{0}' configures {1}, which cites a research paper, but declares no [PaperDefault]; annotate the properties whose values the paper states, with the section or table each comes from",
        "AiDotNet.OptionsFidelity", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "Without declared paper values nothing checks that the shipped defaults are the paper's. Existing violations are listed in OptionsFidelityBaselines.OptionsWithoutPaperDefaults, which may only shrink.",
        customTags: WellKnownDiagnosticTags.CompilationEnd);

    internal static readonly DiagnosticDescriptor InvalidPaperDefault = new(
        "ADNOPT003", "Invalid [PaperDefault] declaration",
        "[PaperDefault] on '{0}' {1}",
        "AiDotNet.OptionsFidelity", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "A paper value must cite where it is stated and must be assignable to the property it declares.");

    internal static readonly DiagnosticDescriptor BlankExemptionReason = new(
        "ADNOPT004", "Option-sensitivity exemption has no reason",
        "[OptionBehaviourExempt] on '{0}' has a blank reason; state why perturbing this option cannot change observable behaviour at test scale",
        "AiDotNet.OptionsFidelity", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "An exemption nobody had to explain is indistinguishable from a hidden defect.");

    internal static readonly DiagnosticDescriptor StaleBaseline = new(
        "ADNOPT005", "Options fidelity baseline entry is stale",
        "'{0}' is listed in OptionsFidelityBaselines.{1} but no longer violates; remove its line so the baseline stays an exact inventory",
        "AiDotNet.OptionsFidelity", DiagnosticSeverity.Error, isEnabledByDefault: true,
        customTags: WellKnownDiagnosticTags.CompilationEnd);

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(UnreadOption, MissingPaperDefaults, InvalidPaperDefault, BlankExemptionReason, StaleBaseline);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            if (start.Compilation.AssemblyName != "AiDotNet") return;
            var modelOptions = start.Compilation.GetTypeByMetadataName(ModelOptionsMetadataName);
            if (modelOptions is null) return;

            // Only member accesses whose name is some options property's name can be a read that
            // matters, so the semantic lookup is skipped for every other member access in the assembly.
            var candidateNames = new HashSet<string>(StringComparer.Ordinal);
            var candidateMethods = new HashSet<string>(StringComparer.Ordinal);
            var sourceOptions = SourceTypes(start.Compilation.Assembly.GlobalNamespace).Where(t => DerivesFrom(t, modelOptions)).ToList();
            foreach (var type in sourceOptions)
            {
                foreach (var property in type.GetMembers().OfType<IPropertySymbol>())
                    candidateNames.Add(property.Name);
                foreach (var method in type.GetMembers().OfType<IMethodSymbol>().Where(m => !m.ReturnsVoid))
                    candidateMethods.Add(method.Name);
            }
            // OptionReadGraph drops every Clone target, so binding the assembly's Clone calls (tensor clones
            // among them) found nothing and cost most of the invocation pass.
            candidateMethods.Remove("Clone");

            // A simple name binds to an options property only through member scope (code inside an options
            // type or a type nested in one) or a using static import; property patterns have their own action.
            // Without this check every identifier sharing a name with some option (LearningRate, SampleRate,
            // ModelPath in model code) was bound: the compat build's net471 compile spent 684 s here.
            var usingStaticTrees = new HashSet<SyntaxTree>(start.Compilation.SyntaxTrees.Where(tree =>
                tree.GetRoot(start.CancellationToken)
                    .DescendantNodes(n => n is CompilationUnitSyntax || n is BaseNamespaceDeclarationSyntax)
                    .OfType<UsingDirectiveSyntax>()
                    .Any(u => u.StaticKeyword.IsKind(SyntaxKind.StaticKeyword))));
            bool globalUsingStatic = usingStaticTrees.Any(tree =>
                tree.GetRoot(start.CancellationToken).ChildNodes().OfType<UsingDirectiveSyntax>()
                    .Any(u => u.StaticKeyword.IsKind(SyntaxKind.StaticKeyword) && u.GlobalKeyword.IsKind(SyntaxKind.GlobalKeyword)));

            var read = new OptionReadGraph(modelOptions, sourceOptions);
            var models = new ConcurrentBag<(INamedTypeSymbol Model, INamedTypeSymbol Options, bool CitesPaper)>();

            start.RegisterSyntaxNodeAction(
                nodeContext => RecordRead(nodeContext, ((MemberAccessExpressionSyntax)nodeContext.Node).Name, modelOptions, candidateNames, read),
                SyntaxKind.SimpleMemberAccessExpression);
            start.RegisterSyntaxNodeAction(
                nodeContext => RecordRead(nodeContext, ((MemberBindingExpressionSyntax)nodeContext.Node).Name, modelOptions, candidateNames, read),
                SyntaxKind.MemberBindingExpression);
            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                // `options is { UseBias: true }` reads UseBias without a member access.
                var subpattern = (SubpatternSyntax)nodeContext.Node;
                if (subpattern.ExpressionColon?.Expression is IdentifierNameSyntax name)
                    RecordRead(nodeContext, name, modelOptions, candidateNames, read);
            }, SyntaxKind.Subpattern);

            start.RegisterSymbolAction(symbolContext =>
            {
                var type = (INamedTypeSymbol)symbolContext.Symbol;
                if (type.TypeKind != TypeKind.Class || type.IsAbstract || type.Locations.Length == 0) return;
                if (!type.AllInterfaces.Any(IsFullModelInterface)) return;
                var options = FindDeclaredOptionsType(type, modelOptions);
                if (options is null) return;
                bool citesPaper = type.GetAttributes().Any(a => a.AttributeClass?.Name == "ResearchPaperAttribute");
                models.Add((type, options, citesPaper));
            }, SymbolKind.NamedType);

            start.RegisterSymbolAction(symbolContext =>
                ValidatePropertyAttributes(symbolContext, (IPropertySymbol)symbolContext.Symbol), SymbolKind.Property);

            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var name = (IdentifierNameSyntax)nodeContext.Node;
                if ((name.Parent is MemberAccessExpressionSyntax access && access.Name == name)
                    || name.Parent is MemberBindingExpressionSyntax) return;
                if (!candidateNames.Contains(name.Identifier.ValueText)) return;
                if (!globalUsingStatic && !usingStaticTrees.Contains(name.SyntaxTree)
                    && !InOptionsMemberScope(nodeContext.ContainingSymbol, modelOptions)) return;
                RecordRead(nodeContext, name, modelOptions, candidateNames, read);
            }, SyntaxKind.IdentifierName);
            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                var expression = ((InvocationExpressionSyntax)nodeContext.Node).Expression;
                string? methodName = expression switch
                {
                    MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
                    MemberBindingExpressionSyntax binding => binding.Name.Identifier.ValueText,
                    SimpleNameSyntax name => name.Identifier.ValueText,
                    _ => null
                };
                if (methodName is null || !candidateMethods.Contains(methodName)) return;
                if (nodeContext.SemanticModel.GetSymbolInfo(nodeContext.Node, nodeContext.CancellationToken).Symbol is IMethodSymbol method)
                    read.Call(nodeContext.ContainingSymbol, method);
            }, SyntaxKind.InvocationExpression);
            start.RegisterSyntaxNodeAction(nodeContext =>
            {
                if (nodeContext.ContainingSymbol is not IMethodSymbol
                    { MethodKind: MethodKind.PropertySet, AssociatedSymbol: IPropertySymbol alias } setter
                    || !DerivesFrom(alias.ContainingType, modelOptions)) return;
                var assignment = (AssignmentExpressionSyntax)nodeContext.Node;
                if (!assignment.Right.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>().Any(name =>
                    name.Identifier.ValueText == "value" && !IsInsideNameof(name) &&
                    nodeContext.SemanticModel.GetSymbolInfo(name, nodeContext.CancellationToken).Symbol is IParameterSymbol parameter &&
                    SymbolEqualityComparer.Default.Equals(parameter.ContainingSymbol, setter))) return;
                if (nodeContext.SemanticModel.GetSymbolInfo(assignment.Left, nodeContext.CancellationToken).Symbol is IPropertySymbol target)
                    read.Forward(alias, target);
            }, SyntaxKind.SimpleAssignmentExpression);


            start.RegisterCompilationEndAction(end =>
            {
                read.Resolve();
                var optionsTypes = models.Select(m => m.Options.OriginalDefinition)
                    .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                    .ToList();

                // ADNOPT001: every option of every model's options type, declared in source, with nothing reading it.
                var unread = new HashSet<string>(StringComparer.Ordinal);
                var reportedProperties = new HashSet<IPropertySymbol>(SymbolEqualityComparer.Default);
                foreach (var options in optionsTypes)
                {
                    // reportedProperties.Add keeps a property shared by several options types to one report.
                    foreach (var property in ConfigurableProperties(options, modelOptions)
                                 .Where(p => !read.ContainsKey(p))
                                 .Where(reportedProperties.Add))
                    {
                        string key = PropertyKey(property);
                        unread.Add(key);
                        if (OptionsFidelityBaselines.UnreadOptions.Contains(key)) continue;
                        end.ReportDiagnostic(Diagnostic.Create(UnreadOption, property.Locations.FirstOrDefault() ?? Location.None, key));
                    }
                }

                // ADNOPT002: options of a paper-citing model that declare no paper value anywhere in their chain.
                var undeclared = new HashSet<string>(StringComparer.Ordinal);
                foreach (var group in models.Where(m => m.CitesPaper)
                             .GroupBy<(INamedTypeSymbol Model, INamedTypeSymbol Options, bool CitesPaper), INamedTypeSymbol>(
                                 m => m.Options.OriginalDefinition, SymbolEqualityComparer.Default))
                {
                    var options = group.Key;
                    if (DeclaresAnyPaperDefault(options)) continue;
                    string key = options.ToDisplayString();
                    undeclared.Add(key);
                    if (OptionsFidelityBaselines.OptionsWithoutPaperDefaults.Contains(key)) continue;
                    var consumers = group.Select(m => m.Model.ToDisplayString()).Distinct().OrderBy(n => n, StringComparer.Ordinal).ToList();
                    end.ReportDiagnostic(Diagnostic.Create(MissingPaperDefaults, options.Locations.FirstOrDefault() ?? Location.None,
                        key, string.Join(", ", consumers)));
                }

                // ADNOPT005: the baselines only shrink.
                ReportStale(end, OptionsFidelityBaselines.UnreadOptions, unread, nameof(OptionsFidelityBaselines.UnreadOptions));
                ReportStale(end, OptionsFidelityBaselines.OptionsWithoutPaperDefaults, undeclared, nameof(OptionsFidelityBaselines.OptionsWithoutPaperDefaults));
            });
        });
    }

    /// <summary>The stable key of an options property: its declaring type's display name, a dot, and its name.</summary>
    internal static string PropertyKey(IPropertySymbol property)
        => property.OriginalDefinition.ContainingType.ToDisplayString() + "." + property.Name;

    /// <summary>
    /// The options type that configures a model: the first public constructor parameter, scanning
    /// constructors from fewest parameters, whose type derives from <c>ModelOptions</c>.
    /// </summary>
    /// <remarks>
    /// The same rule the test scaffold generator uses to find the options it perturbs, so the static
    /// and runtime checks always agree on which class "the model's options" is.
    /// </remarks>
    internal static INamedTypeSymbol? FindDeclaredOptionsType(INamedTypeSymbol model, INamedTypeSymbol modelOptions)
        => FindDeclaredOptionsType(model, candidate => DerivesFrom(candidate, modelOptions));

    /// <summary>
    /// The same rule, matching <c>ModelOptions</c> by its metadata name, for callers that see AiDotNet as a
    /// referenced assembly rather than as source (the test scaffold generator).
    /// </summary>
    internal static INamedTypeSymbol? FindDeclaredOptionsType(INamedTypeSymbol model)
        => FindDeclaredOptionsType(model, candidate =>
        {
            for (INamedTypeSymbol? walk = candidate; walk is not null; walk = walk.BaseType)
            {
                if (walk.ToDisplayString() == ModelOptionsMetadataName) return true;
            }

            return false;
        });

    private static INamedTypeSymbol? FindDeclaredOptionsType(INamedTypeSymbol model, Func<INamedTypeSymbol, bool> isOptions)
    {
        foreach (var constructor in model.InstanceConstructors
                     .Where(c => c.DeclaredAccessibility == Accessibility.Public)
                     .OrderBy(c => c.Parameters.Length))
        {
            foreach (var parameter in constructor.Parameters)
            {
                if (parameter.Type is INamedTypeSymbol candidate && isOptions(candidate))
                    return candidate;
            }
        }

        return null;
    }

    /// <summary>
    /// Public instance properties with a public setter or init accessor, declared in source on an
    /// options type or any of its bases up to and including <c>ModelOptions</c>.
    /// </summary>
    internal static IEnumerable<IPropertySymbol> ConfigurableProperties(INamedTypeSymbol options, INamedTypeSymbol modelOptions)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (INamedTypeSymbol? walk = options.OriginalDefinition; walk is not null; walk = walk.BaseType?.OriginalDefinition)
        {
            // seen.Add keeps the most-derived declaration of a name: the walk starts at the options type itself.
            foreach (var property in walk.GetMembers().OfType<IPropertySymbol>().Where(IsConfigurable).Where(p => seen.Add(p.Name)))
                yield return property;

            if (SymbolEqualityComparer.Default.Equals(walk, modelOptions)) yield break;
        }
    }

    private static bool IsConfigurable(IPropertySymbol property)
        => !property.IsStatic && !property.IsIndexer && !property.IsOverride
            && property.DeclaredAccessibility == Accessibility.Public
            && property.SetMethod is { DeclaredAccessibility: Accessibility.Public }
            && property.Locations.Any(l => l.IsInSource);

    private static void RecordRead(
        SyntaxNodeAnalysisContext context,
        SimpleNameSyntax name,
        INamedTypeSymbol modelOptions,
        HashSet<string> candidateNames,
        OptionReadGraph read)
    {
        if (!candidateNames.Contains(name.Identifier.ValueText)) return;
        // XML cref links describe an option; they never consume its runtime value.
        // Binding these links depends on whether XML documentation output is enabled.
        if (name.Ancestors().OfType<DocumentationCommentTriviaSyntax>().Any()) return;
        if (context.SemanticModel.GetSymbolInfo(name, context.CancellationToken).Symbol is not IPropertySymbol property) return;
        var declaring = property.OriginalDefinition.ContainingType;
        if (declaring is null || IsWriteOnly(context.Node) || IsInsideNameof(context.Node)) return;
        if (DerivesFrom(declaring, modelOptions))
            read.Record(context.ContainingSymbol, property);
        else
            read.RecordInterface(context.ContainingSymbol, property);
    }

    /// <summary>True when the access is only the target of a plain assignment.</summary>
    /// <remarks>
    /// <c>x.P = v</c> writes; <c>x.P += v</c>, <c>x.P++</c> and every other use read. A conditional
    /// access <c>x?.P = v</c> is unwrapped to its enclosing conditional-access expression first.
    /// </remarks>
    private static bool IsWriteOnly(SyntaxNode access)
    {
        SyntaxNode target = access;
        if (access is MemberBindingExpressionSyntax
            && access.Parent is ConditionalAccessExpressionSyntax conditional
            && conditional.WhenNotNull == access)
        {
            target = conditional;
        }

        return target.Parent is AssignmentExpressionSyntax assignment
            && assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)
            && assignment.Left == target;
    }

    /// <summary>True when code in <paramref name="containing"/> can reach an options property by its simple
    /// name: an enclosing type (the type itself or one it is nested in) derives from the options base.</summary>
    private static bool InOptionsMemberScope(ISymbol? containing, INamedTypeSymbol modelOptions)
    {
        for (var type = containing as INamedTypeSymbol ?? containing?.ContainingType; type is not null; type = type.ContainingType)
            if (DerivesFrom(type, modelOptions)) return true;
        return false;
    }

    private static bool IsInsideNameof(SyntaxNode node)
        => node.Ancestors().OfType<InvocationExpressionSyntax>()
            .Any(invocation => invocation.Expression is IdentifierNameSyntax { Identifier.ValueText: "nameof" });

    /// <summary>
    /// Whether a read sits where it does not show the option is consumed: inside the options hierarchy
    /// that declares it, inside a copy constructor, or inside a <c>Clone</c> method.
    /// </summary>
    private static bool IsExcludedLocation(ISymbol? containing, INamedTypeSymbol declaring)
    {
        // Lambdas and local functions belong to the member they are written in.
        while (containing is IMethodSymbol { MethodKind: MethodKind.AnonymousFunction or MethodKind.LocalFunction })
            containing = containing.ContainingSymbol;
        if (containing is null) return false;

        var owner = containing as INamedTypeSymbol ?? containing.ContainingType;
        for (var walk = owner; walk is not null; walk = walk.ContainingType)
        {
            if (DerivesFrom(walk, declaring)) return true;
        }

        if (containing is IMethodSymbol method)
        {
            if (method.Name == "Clone") return true;
            if (method.MethodKind == MethodKind.Constructor
                && method.Parameters.Length == 1
                && SymbolEqualityComparer.Default.Equals(method.Parameters[0].Type.OriginalDefinition, method.ContainingType.OriginalDefinition))
            {
                return true;
            }
        }

        return false;
    }

    private static void ValidatePropertyAttributes(SymbolAnalysisContext context, IPropertySymbol property)
    {
        foreach (var attribute in property.GetAttributes())
        {
            string? name = attribute.AttributeClass?.ToDisplayString();
            Location location = attribute.ApplicationSyntaxReference is { } reference
                ? Location.Create(reference.SyntaxTree, reference.Span)
                : property.Locations.FirstOrDefault() ?? Location.None;
            string display = property.ContainingType.ToDisplayString() + "." + property.Name;

            if (name == OptionBehaviourExemptAttributeName)
            {
                string? reason = attribute.ConstructorArguments.Length == 1 ? attribute.ConstructorArguments[0].Value as string : null;
                if (string.IsNullOrWhiteSpace(reason))
                    context.ReportDiagnostic(Diagnostic.Create(BlankExemptionReason, location, display));
                continue;
            }

            if (name != PaperDefaultAttributeName) continue;

            string? source = attribute.NamedArguments.Where(a => a.Key == "Source").Select(a => a.Value.Value as string).FirstOrDefault();
            if (string.IsNullOrWhiteSpace(source))
            {
                context.ReportDiagnostic(Diagnostic.Create(InvalidPaperDefault, location, display,
                    "has no Source; name the section or table of the paper that states this value"));
            }

            if (attribute.ConstructorArguments.Length != 1) continue;
            string? problem = DescribeIncompatibility(attribute.ConstructorArguments[0], property.Type);
            if (problem is not null)
                context.ReportDiagnostic(Diagnostic.Create(InvalidPaperDefault, location, display, problem));
        }
    }

    /// <summary>Why a declared constant cannot be the value of a property of <paramref name="propertyType"/>, or null when it can.</summary>
    internal static string? DescribeIncompatibility(TypedConstant value, ITypeSymbol propertyType)
    {
        if (value.IsNull) return "declares null; a paper value is a concrete number, flag, member or list";

        var target = UnwrapNullable(propertyType);
        var element = ListElementType(target);
        if (element is not null)
        {
            if (value.Kind != TypedConstantKind.Array)
                return $"declares a single value for the list-valued property type '{propertyType.ToDisplayString()}'; declare an array";
            foreach (var item in value.Values)
            {
                string? itemProblem = DescribeScalarIncompatibility(item, UnwrapNullable(element));
                if (itemProblem is not null) return itemProblem;
            }

            return null;
        }

        if (value.Kind == TypedConstantKind.Array)
            return $"declares an array for the scalar property type '{propertyType.ToDisplayString()}'";
        return DescribeScalarIncompatibility(value, target);
    }

    private static string? DescribeScalarIncompatibility(TypedConstant value, ITypeSymbol target)
    {
        if (value.IsNull) return "declares a null element";
        var valueType = value.Type;
        if (valueType is null) return "declares a value of unknown type";

        // A generic numeric property (T LearningRate) is closed over double or float at run time.
        if (target.TypeKind == TypeKind.TypeParameter)
            return IsNumeric(valueType) ? null : $"declares a {valueType.ToDisplayString()} for a generic numeric property; declare a number";

        if (target.TypeKind == TypeKind.Enum)
            return SymbolEqualityComparer.Default.Equals(valueType, target)
                ? null
                : $"declares a {valueType.ToDisplayString()} for the enum property type '{target.ToDisplayString()}'; declare a member of that enum";

        if (IsNumeric(target))
            return IsNumeric(valueType) ? null : $"declares a {valueType.ToDisplayString()} for the numeric property type '{target.ToDisplayString()}'";

        if (target.SpecialType is SpecialType.System_Boolean or SpecialType.System_String or SpecialType.System_Char)
            return valueType.SpecialType == target.SpecialType ? null
                : $"declares a {valueType.ToDisplayString()} for the property type '{target.ToDisplayString()}'";

        return $"is on a property of type '{target.ToDisplayString()}', which cannot hold an attribute constant; declare paper values only on numeric, bool, enum, string or list-of-those properties";
    }

    private static bool IsNumeric(ITypeSymbol type) => type.SpecialType is
        SpecialType.System_Byte or SpecialType.System_SByte or SpecialType.System_Int16 or SpecialType.System_UInt16 or
        SpecialType.System_Int32 or SpecialType.System_UInt32 or SpecialType.System_Int64 or SpecialType.System_UInt64 or
        SpecialType.System_Single or SpecialType.System_Double or SpecialType.System_Decimal;

    private static ITypeSymbol UnwrapNullable(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } nullable
            ? nullable.TypeArguments[0]
            : type;

    /// <summary>The element type of an array or generic list-like property, or null for a scalar.</summary>
    internal static ITypeSymbol? ListElementType(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol { Rank: 1 } array) return array.ElementType;
        if (type is INamedTypeSymbol { IsGenericType: true, TypeArguments.Length: 1 } named
            && named.SpecialType != SpecialType.System_Nullable_T
            && (named.Name is "List" or "IList" or "IReadOnlyList" or "ICollection" or "IReadOnlyCollection" or "IEnumerable"))
        {
            return named.TypeArguments[0];
        }

        return null;
    }

    private static bool DeclaresAnyPaperDefault(INamedTypeSymbol options)
    {
        for (INamedTypeSymbol? walk = options.OriginalDefinition; walk is not null; walk = walk.BaseType?.OriginalDefinition)
        {
            foreach (var property in walk.GetMembers().OfType<IPropertySymbol>())
            {
                if (property.GetAttributes().Any(a => a.AttributeClass?.ToDisplayString() == PaperDefaultAttributeName))
                    return true;
            }
        }

        return false;
    }

    private static bool IsFullModelInterface(INamedTypeSymbol candidate)
        => candidate.Name == "IFullModel" && candidate.Arity == 3
            && candidate.ContainingNamespace.ToDisplayString() == "AiDotNet.Interfaces";

    internal static bool DerivesFrom(INamedTypeSymbol type, INamedTypeSymbol expected)
    {
        var target = expected.OriginalDefinition;
        for (INamedTypeSymbol? walk = type.OriginalDefinition; walk is not null; walk = walk.BaseType?.OriginalDefinition)
        {
            if (SymbolEqualityComparer.Default.Equals(walk, target)) return true;
        }

        return false;
    }

    private static IEnumerable<INamedTypeSymbol> SourceTypes(INamespaceSymbol root)
    {
        var pending = new Stack<INamespaceOrTypeSymbol>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var current = pending.Pop();
            foreach (var member in current.GetMembers())
            {
                if (member is INamespaceSymbol ns)
                {
                    pending.Push(ns);
                }
                else if (member is INamedTypeSymbol type)
                {
                    if (type.Locations.Any(l => l.IsInSource)) yield return type;
                    pending.Push(type);
                }
            }
        }
    }

    private static void ReportStale(CompilationAnalysisContext end, HashSet<string> baseline, HashSet<string> current, string name)
    {
        foreach (var entry in baseline.Where(e => !current.Contains(e)).OrderBy(e => e, StringComparer.Ordinal))
            end.ReportDiagnostic(Diagnostic.Create(StaleBaseline, Location.None, entry, name));
    }
    // Follow value-producing options helpers only when a consumer reaches them. Uncalled helpers,
    // validation-only void methods, copy constructors and Clone do not make an option consumed.
    private sealed class OptionReadGraph
    {
        private readonly INamedTypeSymbol _modelOptions;
        private readonly ConcurrentDictionary<IPropertySymbol, byte> _properties = new(SymbolEqualityComparer.Default);
        private readonly ConcurrentDictionary<IMethodSymbol, byte> _roots = new(SymbolEqualityComparer.Default);
        private readonly ConcurrentDictionary<IMethodSymbol, ConcurrentDictionary<IPropertySymbol, byte>> _reads = new(SymbolEqualityComparer.Default);
        private readonly ConcurrentDictionary<IMethodSymbol, ConcurrentDictionary<IMethodSymbol, byte>> _calls = new(SymbolEqualityComparer.Default);

        private readonly Dictionary<IPropertySymbol, List<IPropertySymbol>> _interfaces = new(SymbolEqualityComparer.Default);
        private readonly ConcurrentDictionary<IPropertySymbol, ConcurrentDictionary<IPropertySymbol, byte>> _forwarded = new(SymbolEqualityComparer.Default);

        internal OptionReadGraph(INamedTypeSymbol modelOptions, IEnumerable<INamedTypeSymbol> options)
        {
            _modelOptions = modelOptions;
            foreach (var type in options)
            foreach (var contract in type.AllInterfaces)
            foreach (var member in contract.GetMembers().OfType<IPropertySymbol>())
            {
                if (type.FindImplementationForInterfaceMember(member) is not IPropertySymbol implementation) continue;
                var key = member.OriginalDefinition;
                if (!_interfaces.TryGetValue(key, out var implementations))
                    _interfaces[key] = implementations = new List<IPropertySymbol>();
                implementations.Add(implementation.OriginalDefinition);
            }
        }

        internal void RecordInterface(ISymbol? containing, IPropertySymbol property)
        {
            if (_interfaces.TryGetValue(property.OriginalDefinition, out var implementations))
                foreach (var implementation in implementations) Record(containing, implementation);
        }

        internal void Forward(IPropertySymbol alias, IPropertySymbol target) =>
            _forwarded.GetOrAdd(alias.OriginalDefinition, _ => new(SymbolEqualityComparer.Default))
                .TryAdd(target.OriginalDefinition, 0);
        internal bool ContainsKey(IPropertySymbol property) => _properties.ContainsKey(property);

        private static IMethodSymbol? Method(ISymbol? symbol)
        {
            while (symbol is IMethodSymbol { MethodKind: MethodKind.AnonymousFunction or MethodKind.LocalFunction })
                symbol = symbol.ContainingSymbol;
            return symbol as IMethodSymbol;
        }

        private bool Excluded(IMethodSymbol? method) => method is not null &&
            (method.Name == "Clone" || (method.MethodKind == MethodKind.Constructor && DerivesFrom(method.ContainingType, _modelOptions)));
        internal void Record(ISymbol? containing, IPropertySymbol property)
        {
            var caller = Method(containing);
            if (Excluded(caller)) return;
            bool insideOptions = IsExcludedLocation(containing, property.OriginalDefinition.ContainingType);
            if (insideOptions && (caller is null || caller.ReturnsVoid)) return;
            for (IPropertySymbol? walk = property.OriginalDefinition; walk is not null; walk = walk.OverriddenProperty?.OriginalDefinition)
            {
                if (insideOptions)
                    _reads.GetOrAdd(caller!.OriginalDefinition, _ => new(SymbolEqualityComparer.Default)).TryAdd(walk, 0);
                else
                    _properties.TryAdd(walk, 0);
                if (walk.GetMethod is { } getter) Call(containing, getter);
            }
        }

        internal void Call(ISymbol? containing, IMethodSymbol target)
        {
            if (target.ReturnsVoid || Excluded(target) || !DerivesFrom(target.ContainingType, _modelOptions)) return;
            var caller = Method(containing);
            if (Excluded(caller)) return;
            target = target.OriginalDefinition;
            if (caller is not null && DerivesFrom(caller.ContainingType, _modelOptions))
            {
                if (!caller.ReturnsVoid)
                    _calls.GetOrAdd(caller.OriginalDefinition, _ => new(SymbolEqualityComparer.Default)).TryAdd(target, 0);
            }
            else if (containing?.ContainingType is not { } owner || !DerivesFrom(owner, _modelOptions))
                _roots.TryAdd(target, 0);
        }

        internal void Resolve()
        {
            var visited = new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);
            var pending = new Stack<IMethodSymbol>(_roots.Keys);
            while (pending.Count > 0)
            {
                var method = pending.Pop();
                if (!visited.Add(method)) continue;
                if (_reads.TryGetValue(method, out var properties))
                    foreach (var property in properties.Keys) _properties.TryAdd(property, 0);
                if (_calls.TryGetValue(method, out var calls))
                    foreach (var call in calls.Keys) pending.Push(call);
            }
            bool changed;
            do
            {
                changed = false;
                foreach (var pair in _forwarded)
                    if (pair.Value.Keys.Any(_properties.ContainsKey))
                        changed |= _properties.TryAdd(pair.Key, 0);
            } while (changed);
        }
    }
}
