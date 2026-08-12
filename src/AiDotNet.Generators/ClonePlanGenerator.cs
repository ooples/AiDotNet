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

        // Anything the library treats as a model. The base-name list below predates this and covers
        // options classes, whose root is not an interface; models are reached by interface instead so
        // that a family added later -- or a model written in a consumer's own assembly -- is included
        // without anyone editing this list. Every model family's root already declares IFullModel,
        // which is what makes it the membership test rather than a convention about class names.
        if (symbol.AllInterfaces.Any(i => i.Name == "IFullModel")) return true;

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
        var candidates = CollectConstructorCandidates(
            type, type.AllInterfaces.Any(i => i.Name == "IFullModel"));
        var constructor = candidates is null || candidates.Count == 0 ? null : candidates[0];

        // A type with no settable configuration is still worth a plan when its constructor was
        // recorded. That is the normal shape of a model: the arguments it was built from live in
        // private fields, so the property scan finds nothing, and skipping it here is what left
        // every model without a plan and forced a hand-written CreateNewInstance.
        if (entries.Count == 0 && constructor is null) return;

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
        if (constructor is null)
        {
            sb.AppendLine("            CloneRegistry.Register(new ClonePlan(t, e));");
        }
        else
        {
            var names = string.Join(", ", constructor.Select(n => $"\"{n}\""));
            var all = string.Join(", ", candidates!.Select(c =>
                "new string[] { " + string.Join(", ", c.Select(n => $"\"{n}\"")) + " }"));
            sb.AppendLine(
                $"            CloneRegistry.Register(new ClonePlan(t, e, new[] {{ {names} }}, "
                + $"new IReadOnlyList<string>[] {{ {all} }}));");
        }
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

    /// <summary>
    /// Records the constructor a clone should call, when calling one is the only way to rebuild.
    /// </summary>
    /// <param name="type">The type being planned.</param>
    /// <returns>
    /// The constructor's parameters, in order, named by the member that supplies each; or
    /// <see langword="null"/> when the type is rebuilt by allocating and assigning instead.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>Only when there is no parameterless constructor.</b> A type that can be allocated bare was
    /// already cloning correctly through assignment, and recording a constructor for it would change
    /// working behaviour for no gain. Models are the types this exists for: a diffusion model takes
    /// its scheduler and its noise predictor as arguments and offers no bare constructor at all, so
    /// before this the only way to rebuild one was to hand-write a <c>CreateNewInstance</c> override
    /// -- which is the 1147 overrides this removes.
    /// </para>
    /// <para>
    /// <b>Every parameter must map, or none are recorded.</b> A partially satisfiable constructor is
    /// worse than no constructor: it would compile, run, and quietly leave the unmapped arguments at
    /// their defaults, producing a clone that differs from its original in a way no property
    /// comparison detects. When the match fails the type keeps the assignment path and, if it has no
    /// bare constructor either, <c>CloneEngine</c> says so by name at runtime rather than guessing.
    /// </para>
    /// <para>
    /// <b>Candidates are ordered widest first, and chosen at run time.</b> A constructor derives
    /// things from its arguments -- buffers sized from a layer count, sub-models built from a depth
    /// setting -- so re-deriving from more state is better. But which constructor applies depends on
    /// the instance: a model built natively has no ONNX path stored, and rebuilding it through the
    /// wider ONNX constructor passes null and throws. <c>CloneEngine</c> makes that choice.
    /// </para>
    /// </remarks>
    internal static List<string>? CollectConstructorParameters(INamedTypeSymbol type, bool isModel)
    {
        var candidates = CollectConstructorCandidates(type, isModel);
        return candidates is null || candidates.Count == 0 ? null : candidates[0];
    }

    /// <summary>
    /// Records every constructor a clone could call, widest first.
    /// </summary>
    /// <param name="type">The type being planned.</param>
    /// <param name="isModel">Whether the library treats this type as a model.</param>
    /// <returns>One entry per satisfiable constructor, or <see langword="null"/> when none is.</returns>
    internal static List<List<string>>? CollectConstructorCandidates(INamedTypeSymbol type, bool isModel)
    {
        var constructors = type.InstanceConstructors
            .Where(c => c.DeclaredAccessibility is Accessibility.Public or Accessibility.Internal)
            .Where(c => !c.IsStatic)
            .Where(c => c.Parameters.Length > 0)
            .ToList();

        if (constructors.Count == 0) return null;

        // A type that can be allocated bare kept working through assignment, so leave it alone --
        // unless it is a model, whose configuration lives in fields that assignment cannot reach.
        if (!isModel && type.InstanceConstructors.Any(c => c.Parameters.Length == 0)) return null;

        // EVERY satisfiable constructor is recorded, widest first -- not just the widest.
        //
        // Recording only the widest was wrong, and the sweep proved it: 51 models failed to clone
        // with "onnxModelPath cannot be null". Those models take a model path in one constructor and
        // an optimizer in another, the ONNX one is wider, and a natively-built instance has no path
        // stored -- so rebuilding it through the widest constructor passed null and threw. Which
        // constructor applies is a property of the INSTANCE, and nothing known here can decide it.
        //
        // Width still orders the candidates, because a narrower overload usually forwards to the
        // wider one with defaults filled in and re-deriving from more state is better. CloneEngine
        // walks them in this order and takes the first whose required arguments the instance holds.
        var candidates = new List<List<string>>();

        foreach (var constructor in constructors.OrderByDescending(c => c.Parameters.Length))
        {
            var mapped = new List<string>(constructor.Parameters.Length);
            var satisfied = true;

            foreach (var parameter in constructor.Parameters)
            {
                // ref/out cannot be reproduced from reading a stored value.
                if (parameter.RefKind != RefKind.None) { satisfied = false; break; }

                var member = FindSource(type, parameter);

                if (member is null) { satisfied = false; break; }

                mapped.Add(member);
            }

            if (satisfied) candidates.Add(mapped);
        }

        return candidates.Count == 0 ? null : candidates;
    }

    /// <summary>
    /// Finds the member that holds what was passed for a constructor parameter.
    /// </summary>
    /// <param name="type">The type being planned.</param>
    /// <param name="parameter">The constructor parameter to source.</param>
    /// <returns>The member's name, or <see langword="null"/> when nothing holds the value.</returns>
    /// <remarks>
    /// <para>
    /// Fields are searched, not only properties, because that is where a model keeps what it was
    /// built from. <c>DDPMModel</c> takes a scheduler and a U-Net and stores them in <c>_unet</c>
    /// and a base-class property; a property-only scan finds one of the two and gives up, which is
    /// why models had no plan at all before this.
    /// </para>
    /// <para>
    /// The naming rule is the one <c>LayerStateGenerator</c> already proved on the layers: the
    /// parameter name itself, an underscore prefix, or the PascalCase form. It is deliberately not a
    /// search for "a field of the right type" -- two constructor parameters of the same type would
    /// then bind in whichever order the members happened to be declared, and the clone would silently
    /// swap them.
    /// </para>
    /// <para>
    /// The type must match, which is what makes a name coincidence harmless: a field that happens to
    /// share a parameter's name but not its type is rejected and the constructor goes unrecorded.
    /// </para>
    /// </remarks>
    internal static string? FindSource(INamedTypeSymbol type, IParameterSymbol parameter)
    {
        var candidates = new[]
        {
            parameter.Name,
            "_" + parameter.Name,
            char.ToUpperInvariant(parameter.Name[0]) + parameter.Name.Substring(1),
        };

        for (var current = type; current is not null; current = current.BaseType)
        {
            foreach (var candidate in candidates)
            {
                foreach (var member in current.GetMembers(candidate))
                {
                    switch (member)
                    {
                        case IPropertySymbol { IsStatic: false, IsIndexer: false } property
                            when property.GetMethod is not null
                                 && IsCarriedAs(property.Type, parameter.Type):
                            return property.Name;

                        case IFieldSymbol { IsStatic: false, IsConst: false } field
                            when IsCarriedAs(field.Type, parameter.Type):
                            return field.Name;
                    }
                }
            }
        }

        return null;
    }

    /// </remarks>
    private static bool IsCarriedAs(ITypeSymbol property, ITypeSymbol parameter)
    {
        var from = property.WithNullableAnnotation(NullableAnnotation.None);
        var to = parameter.WithNullableAnnotation(NullableAnnotation.None);

        if (SymbolEqualityComparer.Default.Equals(from, to)) return true;

        for (var b = (from as INamedTypeSymbol)?.BaseType; b is not null; b = b.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(b.WithNullableAnnotation(NullableAnnotation.None), to))
            {
                return true;
            }
        }

        return from.AllInterfaces.Any(i =>
            SymbolEqualityComparer.Default.Equals(i.WithNullableAnnotation(NullableAnnotation.None), to));
    }

}
