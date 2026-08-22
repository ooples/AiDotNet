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
    /// <summary>
    /// Stands in a recorded constructor for "pass this parameter's declared default".
    /// </summary>
    /// <remarks>
    /// Not a member name -- no C# member can be called this -- so it cannot collide with one. The
    /// same literal is spelled out in <c>CloneEngine</c>, which is in a different assembly and cannot
    /// reference this one; changing it here requires changing it there.
    /// </remarks>
    internal const string UseDefault = "=default";

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
            .Where(static symbol => IsCloneable(symbol));

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
    private static bool IsCloneable(INamedTypeSymbol? symbol)
    {
        // Roslyn normally supplies a symbol for a class declaration, but incomplete/error
        // compilations are valid generator inputs. Keep that nullable boundary explicit instead
        // of hiding it with null-forgiving syntax before this callback.
        if (symbol is null) return false;
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
        sb.AppendLine("[global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ClonePlanGenerator\", \"1.0.0\")]");
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

        var registrationMethods = new StringBuilder();
        int registrationIndex = 0;
        foreach (var type in distinct)
        {
            if (EmitRegistration(registrationMethods, type, registrationIndex))
            {
                sb.AppendLine($"        Register_{registrationIndex:D6}();");
                registrationIndex++;
            }
        }

        sb.AppendLine("    }");
        sb.AppendLine();
        sb.Append(registrationMethods);
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

    private static bool EmitRegistration(StringBuilder sb, INamedTypeSymbol type, int registrationIndex)
    {
        var entries = CollectConfiguration(type);
        var candidates = CollectConstructorCandidates(
            type, type.AllInterfaces.Any(i => i.Name == "IFullModel"));
        var constructor = candidates is null || candidates.Count == 0 ? null : candidates[0];

        // A type with no settable configuration is still worth a plan when its constructor was
        // recorded. That is the normal shape of a model: the arguments it was built from live in
        // private fields, so the property scan finds nothing, and skipping it here is what left
        // every model without a plan and forced a hand-written CreateNewInstance.
        if (entries.Count == 0 && constructor is null) return false;

        // An open generic cannot be reified here; typeof(Foo<>) is the runtime handle the registry
        // keys on, and a closed instantiation resolves through it.
        var display = type.IsGenericType
            ? type.ConstructUnboundGenericType().ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
            : type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

        // Keep each plan in its own method. A single RegisterAll body containing every plan grew
        // beyond 1.1 MB of IL in the main assembly. Coverage and static-analysis tools construct a
        // control-flow graph before applying generated-code exclusions, so that monolith consumed
        // an entire CI shard budget even when the shard never cloned a model. Small generated
        // methods keep total work linear while RegisterAll remains the one idempotent entry point.
        sb.AppendLine($"    private static void Register_{registrationIndex:D6}()");
        sb.AppendLine("    {");
        sb.AppendLine($"        var t = typeof({display});");
        sb.AppendLine("        var e = new List<ClonePlanEntry>();");

        foreach (var (name, kind) in entries)
        {
            sb.AppendLine($"        Add(e, t, \"{name}\", CloneCopyKind.{kind});");
        }
        if (constructor is null)
        {
            sb.AppendLine("        CloneRegistry.Register(new ClonePlan(t, e));");
        }
        else
        {
            var names = string.Join(", ", constructor.Select(n => $"\"{n}\""));
            var all = string.Join(", ", candidates!.Select(c =>
                "new string[] { " + string.Join(", ", c.Select(n => $"\"{n}\"")) + " }"));
            sb.AppendLine(
                $"        CloneRegistry.Register(new ClonePlan(t, e, new[] {{ {names} }}, "
                + $"new IReadOnlyList<string>[] {{ {all} }}));");
        }
        sb.AppendLine("    }");
        sb.AppendLine();
        return true;
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
            var mapped = new string?[constructor.Parameters.Length];
            var satisfied = true;

            // RESOLVE BY ELIMINATION, IN TWO PASSES. A parameter named after a member takes it
            // first; only then does the type fallback run, and it ignores anything already spoken
            // for. Resolving each parameter in isolation made a constructor with two arguments of
            // one type unresolvable even when only one was ambiguous: a self-supervised method
            // takes a studentProjector and a teacherProjector, the teacher matches the base's
            // TeacherProjector by name, and the student -- which is the only projector left -- was
            // still refused as "not unique" because the claimed member was counted against it.
            for (int i = 0; i < constructor.Parameters.Length; i++)
            {
                // ref/out cannot be reproduced from reading a stored value.
                if (constructor.Parameters[i].RefKind != RefKind.None) { satisfied = false; break; }

                mapped[i] = FindDirectConstructorAssignment(type, constructor, constructor.Parameters[i])
                    ?? FindSource(type, constructor.Parameters[i]);
            }

            var claimed = new HashSet<string>(System.StringComparer.Ordinal);
            foreach (var already in mapped)
            {
                if (already is not null) claimed.Add(already);
            }

            for (int i = 0; satisfied && i < constructor.Parameters.Length; i++)
            {
                if (mapped[i] is not null) continue;

                var parameter = constructor.Parameters[i];

                // Ignoring claimed members is a PREFERENCE, not a rule. Two parameters may legitimately
                // read the same member, and banning it outright took the GAN family's resolutions away:
                // a parameter that used to source a member by type now found it spoken for and refused
                // the whole constructor. Falling back to the unrestricted search makes this strictly
                // additive -- it can only resolve parameters that were unresolvable before.
                var member = FindUniqueByType(type, parameter, claimed)
                    ?? FindUniqueByType(type, parameter, NothingClaimed);

                if (member is null)
                {
                    // An OPTIONAL parameter nothing stores gets its declared default. That is not a
                    // concession -- it is exactly what the hand-written override did: `new Foo(_options)`
                    // left every unstored argument at its default too. 240 models are blocked on a
                    // `seed` and 67 on a `maxGradNorm` that is passed to an initializer and never kept,
                    // and refusing them bought nothing, because there is no value to preserve. A
                    // REQUIRED parameter still refuses: onnxModelPath is required, which is what keeps
                    // an ONNX model from being rebuilt as a native one.
                    if (!parameter.IsOptional) { satisfied = false; break; }

                    mapped[i] = UseDefault;
                    continue;
                }

                mapped[i] = member;
                claimed.Add(member);
            }

            if (!satisfied) continue;

            var resolved = new List<string>(mapped.Length);
            foreach (var member in mapped)
            {
                // Only reachable with every slot decided: pass one leaves a name or null, and pass
                // two replaces every remaining null with a member or the default sentinel, or the
                // constructor was abandoned above.
                resolved.Add(member ?? UseDefault);
            }

            candidates.Add(resolved);
        }

        return candidates.Count == 0 ? null : candidates;
    }

    /// <summary>
    /// Finds a member that the selected constructor directly assigns from a parameter, even when
    /// their names intentionally differ (for example <c>ImageSize = imageWidth</c>).
    /// </summary>
    /// <remarks>
    /// This is stronger evidence than a naming heuristic: it reads the constructor's actual storage
    /// operation. Only a direct parameter RHS is accepted. Derived expressions remain unresolved so
    /// the generator cannot mistake a computed runtime value for the original argument.
    /// </remarks>
    private static string? FindDirectConstructorAssignment(
        INamedTypeSymbol type,
        IMethodSymbol constructor,
        IParameterSymbol parameter)
    {
        string? found = null;
        foreach (var syntaxReference in constructor.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax() is not ConstructorDeclarationSyntax declaration)
                continue;

            foreach (var assignment in declaration.DescendantNodes().OfType<AssignmentExpressionSyntax>())
            {
                if (assignment.Right is not IdentifierNameSyntax right
                    || !string.Equals(right.Identifier.ValueText, parameter.Name,
                        System.StringComparison.Ordinal))
                    continue;

                string? memberName = assignment.Left switch
                {
                    IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                    MemberAccessExpressionSyntax
                    {
                        Expression: ThisExpressionSyntax,
                        Name: SimpleNameSyntax name
                    } => name.Identifier.ValueText,
                    _ => null,
                };
                if (memberName is null) continue;

                bool isReadableMember = false;
                for (var current = type; current is not null && !isReadableMember; current = current.BaseType)
                {
                    isReadableMember = current.GetMembers(memberName).Any(member => member switch
                    {
                        IPropertySymbol { IsStatic: false, IsIndexer: false } property
                            when property.GetMethod is not null
                                 && IsCarriedAs(property.Type, parameter.Type) => true,
                        IFieldSymbol { IsStatic: false, IsConst: false } field
                            when IsCarriedAs(field.Type, parameter.Type) => true,
                        _ => false,
                    });
                }
                if (!isReadableMember) continue;

                if (found is not null
                    && !string.Equals(found, memberName, System.StringComparison.Ordinal))
                    return null;
                found = memberName;
            }
        }

        return found;
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

        // The type fallback is NOT tried here. It runs in a second pass over the whole constructor,
        // once every name match is known, so it can ignore members another parameter already claimed.
        return FindByNameSuffix(type, parameter);
    }

    /// <summary>
    /// Finds the member whose name ends with the parameter's, when the exact name did not match.
    /// </summary>
    /// <param name="type">The type being planned.</param>
    /// <param name="parameter">The constructor parameter to source.</param>
    /// <returns>That member's name, or <see langword="null"/> when there is not exactly one.</returns>
    /// <remarks>
    /// <para>
    /// A qualifying prefix is the common way this library disambiguates a stored argument:
    /// <c>AttentiveNAS</c> keeps its <c>searchSpace</c> in <c>_nasSearchSpace</c>, and
    /// <c>BayTransProtoAlgorithm</c> keeps its <c>options</c> in <c>_algoOptions</c>. Both hold
    /// exactly what the constructor was given; only the name is decorated.
    /// </para>
    /// <para>
    /// Tried BEFORE the type search and preferred over it, because a name that ends with the
    /// parameter's is evidence about THIS parameter, where a unique type is only evidence that
    /// nothing else could be meant. Where two members qualify, neither is chosen.
    /// </para>
    /// </remarks>
    private static string? FindByNameSuffix(INamedTypeSymbol type, IParameterSymbol parameter)
    {
        var suffix = char.ToUpperInvariant(parameter.Name[0]) + parameter.Name.Substring(1);
        string? found = null;

        for (var current = type; current is not null; current = current.BaseType)
        {
            var fields = new List<string>();
            var properties = new List<string>();

            foreach (var member in current.GetMembers())
            {
                var name = member switch
                {
                    IPropertySymbol { IsStatic: false, IsIndexer: false } p
                        when p.GetMethod is not null && IsCarriedAs(p.Type, parameter.Type) => p.Name,
                    IFieldSymbol { IsStatic: false, IsConst: false } f
                        when IsCarriedAs(f.Type, parameter.Type) => f.Name,
                    _ => null,
                };

                if (name is null) continue;

                // A STORED ARGUMENT IS DECORATED AT EITHER END. A qualifying prefix is the common
                // case (_nasSearchSpace, _bayesOptions); a qualifying suffix is the other one --
                // StackingClassifier takes a `Func<IClassifier<T>> finalEstimator` and keeps it in
                // _finalEstimatorFactory, which holds exactly what the constructor was given.
                //
                // The type check above is what makes this safe rather than loose: the same class
                // also declares _finalEstimator, and that one is refused on TYPE (a classifier, not
                // the factory) before its name is ever considered.
                var bare = name.TrimStart('_');
                var decorated = name.EndsWith(suffix, System.StringComparison.Ordinal)
                    || (bare.Length > parameter.Name.Length
                        && bare.StartsWith(parameter.Name, System.StringComparison.OrdinalIgnoreCase));
                if (!decorated) continue;

                if (member is IFieldSymbol) fields.Add(name); else properties.Add(name);
            }

            // A PROPERTY AND ITS OWN BACKING FIELD ARE ONE VALUE, NOT TWO CANDIDATES. Counting them
            // separately is what made the pair ambiguous, and the ambiguity rule then refused BOTH.
            // Every NAS model is shaped this way -- AttentiveNAS declares `_nasSearchSpace` and
            // exposes `NasSearchSpace => _nasSearchSpace` beside it -- so all eight were reported
            // unrebuildable over a parameter they do store, by the very lookup written to find it.
            // The field is preferred because it is the slot the constructor assigned.
            properties.RemoveAll(p => fields.Any(
                f => string.Equals(f.TrimStart('_'), p, System.StringComparison.OrdinalIgnoreCase)));

            // Anything still standing alongside another is genuinely ambiguous, and neither is chosen.
            var matches = fields.Count + properties.Count;
            if (matches > 1) return null;
            if (matches == 1) found = fields.Count == 1 ? fields[0] : properties[0];

            // Most-derived wins. AttentiveNAS keeps its searchSpace in _nasSearchSpace while a base
            // also exposes SearchSpace; both hold it, and the one the constructor assigned is the one
            // declared alongside that constructor. Refusing the pair left the model unrebuildable
            // over a naming decision that changes nothing about its state.
            if (found is not null) return found;
        }

        return null;
    }

    /// <summary>
    /// Finds the single member of a parameter's exact type, when the name did not match.
    /// </summary>
    /// <param name="type">The type being planned.</param>
    /// <param name="parameter">The constructor parameter to source.</param>
    /// <returns>That member's name, or <see langword="null"/> when there is not exactly one.</returns>
    /// <remarks>
    /// <para>
    /// The name rule alone missed 132 models, all the same way: the constructor takes
    /// <c>options</c> and the field is <c>_algoOptions</c>. The value IS stored -- just not under a
    /// name the rule guesses -- so refusing produced a model that needed a hand-written clone for a
    /// naming choice rather than for anything about its state.
    /// </para>
    /// <para>
    /// EXACTLY ONE, and by exact type. Two members of the same type would bind in whichever order
    /// they happen to be declared, so a clone could silently swap a generator for a discriminator.
    /// Uniqueness is what makes this unambiguous, and it is checked across the whole inheritance
    /// chain rather than one level, because the member usually lives on a base.
    /// </para>
    /// <para>
    /// Base types are excluded from the exact-type search only for very common primitives, where a
    /// unique match is a coincidence rather than a correspondence.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Name-then-type sourcing for a parameter considered on its own, with nothing claimed.
    /// </summary>
    /// <remarks>
    /// For the analyzer, which reports which parameters block a model and must answer that per
    /// parameter. The plan itself resolves a constructor as a whole, so it uses the two passes.
    /// </remarks>
    internal static string? FindAnySource(INamedTypeSymbol type, IParameterSymbol parameter)
        => FindSource(type, parameter) ?? FindUniqueByType(type, parameter, NothingClaimed);

    private static readonly HashSet<string> NothingClaimed = new(System.StringComparer.Ordinal);

    private static string? FindUniqueByType(
        INamedTypeSymbol type,
        IParameterSymbol parameter,
        HashSet<string> claimed)
    {
        // A lone int or string field matching a lone int or string parameter says nothing: those
        // types recur, and the match would be luck. Richer types are genuinely identifying.
        if (parameter.Type.SpecialType is not SpecialType.None) return null;
        if (parameter.Type.TypeKind == TypeKind.Enum) return null;

        var fields = new List<string>();
        var properties = new List<string>();

        for (var current = type; current is not null; current = current.BaseType)
        {
            foreach (var member in current.GetMembers())
            {
                string? name = member switch
                {
                    IPropertySymbol { IsStatic: false, IsIndexer: false } p
                        when p.GetMethod is not null && IsSameType(p.Type, parameter.Type) => p.Name,
                    IFieldSymbol { IsStatic: false, IsConst: false } f
                        when IsSameType(f.Type, parameter.Type) => f.Name,
                    _ => null,
                };

                if (name is null) continue;

                // Spoken for by a parameter that matched it by name, so it is not evidence about
                // this one -- that is what lets the last unclaimed member of a repeated type resolve.
                if (claimed.Contains(name)) continue;

                if (member is IFieldSymbol) fields.Add(name); else properties.Add(name);
            }
        }

        // A property and its own backing field are one value here too, for the same reason they are
        // in FindByNameSuffix: counting them separately made a type that occurs exactly once look
        // like it occurred twice, and "not unique" then refused it. The projector a self-supervised
        // method is built with is stored as _projector and read back through Projector, so the pair
        // alone was enough to lose it.
        properties.RemoveAll(p => fields.Any(
            f => string.Equals(f.TrimStart('_'), p, System.StringComparison.OrdinalIgnoreCase)));

        if (fields.Count + properties.Count != 1) return null;

        return fields.Count == 1 ? fields[0] : properties[0];
    }

    /// <summary>
    /// Compares two types ignoring nullable annotation.
    /// </summary>
    /// <param name="a">The first type.</param>
    /// <param name="b">The second type.</param>
    /// <returns><see langword="true"/> when they are the same type.</returns>
    private static bool IsSameType(ITypeSymbol a, ITypeSymbol b)
        => SymbolEqualityComparer.Default.Equals(
            a.WithNullableAnnotation(NullableAnnotation.None),
            b.WithNullableAnnotation(NullableAnnotation.None));

    /// <summary>
    /// Determines whether a member's value can be passed for a parameter without a conversion.
    /// </summary>
    /// <param name="property">The member's type.</param>
    /// <param name="parameter">The parameter type.</param>
    /// <returns><see langword="true"/> when the value is passable as-is.</returns>
    /// <remarks>
    /// Reference conversions only -- a base class or an implemented interface. Numeric and
    /// user-defined conversions are deliberately refused: the value is passed through
    /// <c>ConstructorInfo.Invoke</c>, which performs no user-defined conversion, so accepting one
    /// here would produce a plan that compiles and then throws at the point of cloning.
    /// </remarks>
    private static bool IsCarriedAs(ITypeSymbol property, ITypeSymbol parameter)
    {
        var from = property.WithNullableAnnotation(NullableAnnotation.None);
        var to = parameter.WithNullableAnnotation(NullableAnnotation.None);

        if (SymbolEqualityComparer.Default.Equals(from, to)) return true;

        // A resolved optional value is commonly stored in a non-nullable field: constructors spell
        // `int? outputChannels = null` and then persist `_outputChannels = outputChannels ?? input`.
        // Passing that stored int back to ConstructorInfo for Nullable<int> is the exact CLR boxing
        // representation of a nullable with HasValue=true. Refusing it pinned outputChannels (and
        // similar shape-bearing options) to null, rebuilding custom predictors with default widths.
        if (to is INamedTypeSymbol
            {
                OriginalDefinition.SpecialType: SpecialType.System_Nullable_T,
                TypeArguments.Length: 1
            } nullable
            && SymbolEqualityComparer.Default.Equals(
                from, nullable.TypeArguments[0].WithNullableAnnotation(NullableAnnotation.None)))
        {
            return true;
        }

        for (var b = (from as INamedTypeSymbol)?.BaseType; b is not null; b = b.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(b.WithNullableAnnotation(NullableAnnotation.None), to))
            {
                return true;
            }
        }

        if (from.AllInterfaces.Any(i =>
            SymbolEqualityComparer.Default.Equals(i.WithNullableAnnotation(NullableAnnotation.None), to)))
        {
            return true;
        }

        // THE MEMBER MAY HOLD THE ARGUMENT MORE GENERALLY THAN THE CONSTRUCTOR TAKES IT. A time
        // series model passes its ARModelOptions to the base and reads it back off the base's
        // Options property, whose type is the general options base -- the model's own clone did
        // `new ARModel<T>((ARModelOptions<T>)Options)`, downcasting exactly this way. Refusing the
        // pair left three models unrebuildable over a value they never stopped holding.
        //
        // Safe because it is the RUNTIME value that settles it: CloneEngine now skips a candidate
        // constructor whose argument is not an instance of the parameter type, so a member that
        // happens to hold something else moves on to the next candidate instead of throwing.
        for (var b = (to as INamedTypeSymbol)?.BaseType; b is not null; b = b.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(b.WithNullableAnnotation(NullableAnnotation.None), from))
            {
                return true;
            }
        }

        return to.AllInterfaces.Any(i =>
            SymbolEqualityComparer.Default.Equals(i.WithNullableAnnotation(NullableAnnotation.None), from));
    }

}
