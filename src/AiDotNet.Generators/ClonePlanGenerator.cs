using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Emits a compile-time clone plan for every cloneable type, so a clone carries the right members
/// without anyone hand-writing a copy constructor.
/// </summary>
/// <remarks>
/// <para>
/// The problem this removes: 464 of 594 options classes have no copy constructor, and 1802
/// hand-written clone paths exist across the library. Every one is a place a property can be
/// dropped silently, which is the defect behind the Tacotron2 and TimeBridge clone bugs and behind
/// the 71 copy constructors that omitted the inherited <c>ModelOptions.Seed</c>. Generating 464
/// more constructors would multiply that surface rather than remove it.
/// </para>
/// <para>
/// A <b>plan</b> is emitted rather than copy code because a Roslyn generator can only add members
/// to a <c>partial</c> type and none of the options classes are partial. Registering a plan leaves
/// every existing class untouched, and means a class written by a consumer works without them
/// declaring anything.
/// </para>
/// <para>
/// <b>What counts as configuration.</b> Everything settable, unless provably otherwise. Read-only
/// and computed properties are skipped because the compiler proves they are derived from what is
/// carried, so re-deriving them keeps a clone consistent rather than merely equal. Delegates and
/// interfaces are deliberately <i>kept</i>: activation functions, kernels and schedules arrive that
/// way and are genuine configuration, so excluding them by type shape would produce a clone that
/// behaves differently while looking correct.
/// </para>
/// <para>
/// <b>What is not configuration.</b> Learned parameters travel through
/// <c>GetParameters()</c>/<c>UpdateParameters(Vector&lt;T&gt;)</c>, which every layer implements and
/// which training exercises on every step; optimizer state travels through the optimizer's own
/// <c>Serialize</c>/<c>Deserialize</c>. Neither is inferred here, because inferring learned state
/// from a property's type would misread a <c>Tensor&lt;T&gt;</c> that is genuinely configuration --
/// a fixed prior or a mask.
/// </para>
/// </remarks>
[Generator]
public class ClonePlanGenerator : IIncrementalGenerator
{
    private const string NotConfiguration = "NotConfigurationAttribute";
    private const string ExternalResource = "ExternalResourceAttribute";

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is ClassDeclarationSyntax { BaseList: not null } c
                    && !c.Modifiers.Any(m => m.ValueText == "abstract")
                    && !c.Modifiers.Any(m => m.ValueText == "static"),
                static (ctx, _) => (INamedTypeSymbol?)ctx.SemanticModel.GetDeclaredSymbol(ctx.Node))
            .Where(static s => s is not null && IsCloneable(s!));

        var collected = candidates.Collect();
        context.RegisterSourceOutput(collected, static (spc, types) => Execute(spc, types!));
    }

    /// <summary>
    /// Determines whether a type participates in cloning.
    /// </summary>
    /// <param name="symbol">The candidate type.</param>
    /// <returns><see langword="true"/> when a plan should be emitted.</returns>
    /// <remarks>
    /// Membership is decided by the base chain rather than by a naming convention so that a
    /// consumer's own subclass is included automatically -- which is the point of the feature. A
    /// name-based rule would silently exclude anyone who named their class differently.
    /// </remarks>
    private static bool IsCloneable(INamedTypeSymbol symbol)
    {
        if (!IsNameableFromGeneratedCode(symbol)) return false;

        for (var b = symbol.BaseType; b is not null; b = b.BaseType)
        {
            switch (b.Name)
            {
                case "ModelOptions":
                case "NeuralNetworkOptions":
                case "RegressionOptions":
                case "TimeSeriesRegressionOptions":
                case "RiskModelOptions":
                case "LayerBase":
                case "NeuralNetworkBase":
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Determines whether generated code in the same assembly can name this type.
    /// </summary>
    /// <param name="symbol">The candidate type.</param>
    /// <returns><see langword="true"/> when <c>typeof(...)</c> would compile against it.</returns>
    /// <remarks>
    /// <para>
    /// A type nested inside another as <c>private</c> or <c>protected</c> is invisible outside its
    /// declaring type, so emitting <c>typeof(Outer.Inner)</c> produces CS0122 no matter how
    /// cloneable the type is. <c>STCConnectorLayer&lt;T&gt;.RegStageBlock</c> is exactly that: a
    /// nested helper deriving from <c>LayerBase</c>, matched by the base-chain rule and then
    /// unnameable.
    /// </para>
    /// <para>
    /// Every containing type is checked, not just the type itself: an accessible class nested in an
    /// inaccessible one is still unreachable. Internal is fine, since generated code lands in the
    /// same assembly.
    /// </para>
    /// <para>
    /// Such a type is not left without a clone — it falls back to the reflected plan at runtime,
    /// which reflection can reach precisely because it does not have to name the type in source.
    /// </para>
    /// </remarks>
    private static bool IsNameableFromGeneratedCode(INamedTypeSymbol symbol)
    {
        for (var current = symbol; current is not null; current = current.ContainingType)
        {
            if (current.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Internal))
            {
                return false;
            }
        }

        return true;
    }

    private static void Execute(SourceProductionContext context, ImmutableArray<INamedTypeSymbol?> types)
    {
        // Deduplicate: a partial class surfaces once per syntax tree, and the same symbol reached
        // through different trees must not register two plans.
        var distinct = types
            .Where(t => t is not null)
            .Select(t => t!)
            .GroupBy(t => t.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat))
            .Select(g => g.First())
            .OrderBy(t => t.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat), System.StringComparer.Ordinal)
            .ToList();

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using System.Reflection;");
        sb.AppendLine("using AiDotNet.Models;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Generated;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Compile-time clone plans. Registered once, on first use of the clone registry.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class CloneRegistrations");
        sb.AppendLine("{");
        sb.AppendLine("    private static bool _done;");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Registers every generated plan. Idempotent.</summary>");
        sb.AppendLine("    internal static void RegisterAll()");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_done) return;");
        sb.AppendLine("        _done = true;");
        sb.AppendLine();

        foreach (var type in distinct)
        {
            EmitRegistration(sb, type);
        }

        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Binds one configuration property, skipping it if the shape changed since generation.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    /// <remarks>");
        sb.AppendLine("    /// A null result means the generated plan and the runtime type disagree, which the");
        sb.AppendLine("    /// analyzer is there to prevent. Skipping rather than throwing keeps a stale plan from");
        sb.AppendLine("    /// taking down an application that is otherwise working.");
        sb.AppendLine("    /// </remarks>");
        sb.AppendLine("    private static void Add(List<ClonePlanEntry> entries, Type owner, string name, CloneCopyKind kind)");
        sb.AppendLine("    {");
        sb.AppendLine("        var p = owner.GetProperty(name, BindingFlags.Public | BindingFlags.Instance);");
        sb.AppendLine("        if (p is not null && p.CanRead && p.CanWrite) entries.Add(new ClonePlanEntry(p, kind));");
        sb.AppendLine("    }");
        sb.AppendLine("}");

        context.AddSource("CloneRegistrations.g.cs", sb.ToString());
    }

    private static void EmitRegistration(StringBuilder sb, INamedTypeSymbol type)
    {
        var entries = CollectConfiguration(type);
        if (entries.Count == 0) return;

        // An open generic cannot be reified here; typeof(Foo<>) is the runtime handle the registry
        // keys on, and a closed instantiation resolves through it.
        var display = type.IsGenericType
            ? type.ConstructUnboundGenericType().ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            : type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

        sb.AppendLine("        {");
        sb.AppendLine($"            var t = typeof({display});");
        sb.AppendLine("            var e = new List<ClonePlanEntry>();");

        foreach (var (name, kind) in entries)
        {
            sb.AppendLine($"            Add(e, t, \"{name}\", CloneCopyKind.{kind});");
        }

        sb.AppendLine("            CloneRegistry.Register(new ClonePlan(t, e));");
        sb.AppendLine("        }");
        sb.AppendLine();
    }

    /// <summary>
    /// Collects the configuration surface, walking the full inheritance chain.
    /// </summary>
    /// <param name="type">The type to inspect.</param>
    /// <returns>Property names paired with how each is carried, base first.</returns>
    /// <remarks>
    /// The chain is walked explicitly rather than read off the derived type alone. A
    /// declaration-only view is precisely what omitted <c>ModelOptions.Seed</c> from 71 hand-written
    /// copy constructors: the property is real and settable, but it is declared somewhere else.
    /// </remarks>
    private static List<(string Name, string Kind)> CollectConfiguration(INamedTypeSymbol type)
    {
        var result = new List<(string, string)>();
        var seen = new HashSet<string>(System.StringComparer.Ordinal);
        var chain = new List<INamedTypeSymbol>();

        for (var current = type; current is not null && current.SpecialType != SpecialType.System_Object; current = current.BaseType)
        {
            chain.Add(current);
        }

        chain.Reverse();

        foreach (var level in chain)
        {
            var properties = level.GetMembers()
                .OfType<IPropertySymbol>()
                .Where(p => p.DeclaredAccessibility == Accessibility.Public)
                .Where(p => !p.IsStatic && !p.IsIndexer)
                .Where(p => p.GetMethod is not null && p.SetMethod is not null)
                .Where(p => !IsExcluded(p))
                .OrderBy(p => p.Name, System.StringComparer.Ordinal);

            foreach (var property in properties)
            {
                if (seen.Add(property.Name))
                {
                    result.Add((property.Name, CopyKindFor(property.Type)));
                }
            }
        }

        return result;
    }

    private static bool IsExcluded(IPropertySymbol property)
        => property.GetAttributes().Any(a =>
            a.AttributeClass?.Name is NotConfiguration or ExternalResource);

    /// <summary>
    /// Chooses how a value is carried: duplicated, or shared.
    /// </summary>
    /// <param name="type">The property type.</param>
    /// <returns>The <c>CloneCopyKind</c> member name.</returns>
    /// <remarks>
    /// Mutable containers are duplicated, because a bare assignment leaves the clone and the
    /// original writing through one buffer -- a difference invisible to a property-by-property
    /// equality check, which is why the generated tests also assert that mutating a clone cannot
    /// affect its original. Strings are shared: reference types, but immutable, so copying is waste.
    /// </remarks>
    private static string CopyKindFor(ITypeSymbol type)
    {
        if (type.SpecialType == SpecialType.System_String) return "ByReference";
        if (type.TypeKind == TypeKind.Array) return "Deep";

        if (type is INamedTypeSymbol { IsGenericType: true } named)
        {
            switch (named.ConstructedFrom.Name)
            {
                case "List":
                case "Dictionary":
                case "HashSet":
                case "IList":
                case "ICollection":
                    return "Deep";
            }
        }

        return "ByReference";
    }
}
