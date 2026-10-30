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
/// <b>What counts as a read (ADNOPT001).</b> Deliberately conservative, so a report is always a real
/// unread option: any member access of the property, anywhere in the AiDotNet assembly, that is not
/// the target of a plain assignment and is not inside <c>nameof</c>, the options class hierarchy that
/// declares the property, a copy constructor, or a <c>Clone</c> method. Being assembly-wide rather
/// than per-model means a property read by a base class, a helper the options are handed to, or a
/// sibling model counts as read; the generated runtime option-sensitivity test is what then checks
/// each individual model. Generated code is analyzed for reads too (a generated consumer is still a
/// consumer), but no diagnostic is reported inside it. The generated clone plan and YAML binder do not
/// count against this: the plan reaches members by reflection and the binder only assigns.
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
            foreach (var type in SourceTypes(start.Compilation.Assembly.GlobalNamespace))
            {
                if (!DerivesFrom(type, modelOptions)) continue;
                foreach (var property in type.GetMembers().OfType<IPropertySymbol>())
                    candidateNames.Add(property.Name);
            }

            var read = new ConcurrentDictionary<IPropertySymbol, byte>(SymbolEqualityComparer.Default);
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

            start.RegisterCompilationEndAction(end =>
            {
                var optionsTypes = models.Select(m => m.Options.OriginalDefinition)
                    .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                    .ToList();

                // ADNOPT001: every option of every model's options type, declared in source, with nothing reading it.
                var unread = new HashSet<string>(StringComparer.Ordinal);
                var reportedProperties = new HashSet<IPropertySymbol>(SymbolEqualityComparer.Default);
                foreach (var options in optionsTypes)
                {
                    foreach (var property in ConfigurableProperties(options, modelOptions))
                    {
                        if (!reportedProperties.Add(property)) continue;
                        if (read.ContainsKey(property)) continue;
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
    {
        foreach (var constructor in model.InstanceConstructors
                     .Where(c => c.DeclaredAccessibility == Accessibility.Public)
                     .OrderBy(c => c.Parameters.Length))
        {
            foreach (var parameter in constructor.Parameters)
            {
                if (parameter.Type is INamedTypeSymbol candidate && DerivesFrom(candidate, modelOptions))
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
            foreach (var property in walk.GetMembers().OfType<IPropertySymbol>())
            {
                if (property.IsStatic || property.IsIndexer || property.IsOverride) continue;
                if (property.DeclaredAccessibility != Accessibility.Public) continue;
                if (property.SetMethod is not { DeclaredAccessibility: Accessibility.Public }) continue;
                if (!property.Locations.Any(l => l.IsInSource)) continue;
                if (!seen.Add(property.Name)) continue;
                yield return property;
            }

            if (SymbolEqualityComparer.Default.Equals(walk, modelOptions)) yield break;
        }
    }

    private static void RecordRead(
        SyntaxNodeAnalysisContext context,
        SimpleNameSyntax name,
        INamedTypeSymbol modelOptions,
        HashSet<string> candidateNames,
        ConcurrentDictionary<IPropertySymbol, byte> read)
    {
        if (!candidateNames.Contains(name.Identifier.ValueText)) return;
        if (context.SemanticModel.GetSymbolInfo(name, context.CancellationToken).Symbol is not IPropertySymbol property) return;
        var declaring = property.OriginalDefinition.ContainingType;
        if (declaring is null || !DerivesFrom(declaring, modelOptions)) return;
        if (IsWriteOnly(context.Node) || IsInsideNameof(context.Node)) return;
        if (IsExcludedLocation(context.ContainingSymbol, declaring)) return;

        // A read through a base declaration reaches every override, and a read of an override reaches
        // the declaration it overrides; both count, since either is the option being consumed.
        for (IPropertySymbol? walk = property.OriginalDefinition; walk is not null; walk = walk.OverriddenProperty?.OriginalDefinition)
            read.TryAdd(walk, 0);
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
}
