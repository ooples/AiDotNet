using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Emits each model's state declarations into the model, so a model author writes none.
/// </summary>
/// <remarks>
/// <para>
/// A hand-written <c>RegisterState</c> that lists every field is the same boilerplate as a
/// hand-written <c>Serialize</c> wearing a different hat: it is common behaviour living in the model,
/// and it is one more place to forget the field somebody adds next year. The generator already knows
/// the type's members and already classifies them, so it can write both halves and the author can
/// write nothing at all.
/// </para>
/// <para>
/// Reuses <see cref="ParameterMemberSemanticModel"/> rather than inventing a second opinion about
/// what a member is. Its vocabulary already answers the question this generator asks:
/// <c>Trainable</c> is in the parameter vector and must NOT be written twice; <c>Fitted</c>,
/// <c>Frozen</c> and <c>Buffer</c> are learned state that the vector does not carry; <c>Scratch</c>
/// is recomputable; <c>Alias</c> is a view of something else; <c>External</c> belongs to another
/// runtime. Unclassified numeric state is already an error under AIDN088, which is what makes
/// "persist everything I can place" safe -- there is nothing it cannot place and silently skip.
/// </para>
/// <para>
/// Emits into a <c>partial</c> declaration, the same way the parameter generator does, because that
/// is the only way generated code can reach a private field. AIDN085 already establishes that
/// convention for weights and 1099 types in this library are already partial.
/// </para>
/// </remarks>
[Generator]
public class ModelStateGenerator : IIncrementalGenerator
{
    /// <summary>A model owns state the generator can persist, but is not partial.</summary>
    private static readonly DiagnosticDescriptor MustBePartial = new(
        "ADN0061",
        "Model must be partial for its state to be persisted automatically",
        "'{0}' owns state that is not in its parameter vector ({1}) but is not declared 'partial', so "
            + "the state generator cannot reach it and nothing persists it. Add 'partial' and the "
            + "declarations are written for you",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>State was annotated, and the registry has no way to carry its type.</summary>
    private static readonly DiagnosticDescriptor UnsupportedStateShape = new(
        "ADN0062",
        "Declared state has a shape the registry cannot carry",
        "'{0}.{1}' is annotated as state but its type '{2}' has no ModelStateRegistry declaration, so "
            + "nothing would persist it. Add an overload for that shape, or hold the value in one the "
            + "registry already carries",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax { BaseList: not null },
                transform: static (ctx, _) =>
                    ctx.SemanticModel.GetDeclaredSymbol((ClassDeclarationSyntax)ctx.Node) as INamedTypeSymbol)
            .Where(static symbol => symbol is not null);

        context.RegisterSourceOutput(candidates, static (spc, symbol) => Emit(spc, symbol));
    }

    private static void Emit(SourceProductionContext spc, INamedTypeSymbol? type)
    {
        // A semantic model can legitimately return no declared symbol while the user's
        // compilation is incomplete. Treat that as no candidate; never assert it away with !.
        if (type is null) return;

        // ABSTRACT BASES ARE INCLUDED. Skipping them meant state declared on a shared base was never
        // generated for anyone: every decision-tree model keeps its structure in
        // DecisionTreeRegressionBase.Root, and no concrete model declares it, so nothing persisted it
        // and each model wrote its own Serialize to walk the tree by hand. An abstract class can carry
        // an override perfectly well, and RegisterGeneratedState chains through base calls, so putting
        // the declaration where the member actually lives is both correct and the only place it can go.
        if (type.IsStatic) return;

        // The numeric type comes from the base hook's own signature rather than from a guess about
        // which type parameter is "the" numeric one.
        var hook = FindHook(type);
        if (hook is null) return;
        if (hook.Parameters.Length != 1) return;
        if (hook.Parameters[0].Type is not INamedTypeSymbol { TypeArguments.Length: 1 } registry) return;

        var numeric = registry.TypeArguments[0].ToDisplayString();

        var members = new List<(string Name, string Call)>();

        foreach (var member in type.GetMembers())
        {
            if (member.IsStatic || member.IsImplicitlyDeclared) continue;

            var memberType = member switch
            {
                IFieldSymbol f when !f.IsConst => f.Type,
                IPropertySymbol { IsIndexer: false } p when p.GetMethod is not null && p.SetMethod is not null => p.Type,
                _ => null,
            };
            if (memberType is null) continue;

            // Readonly storage cannot be reassigned on restore, so declaring it would produce a
            // payload nothing could apply.
            if (member is IFieldSymbol { IsReadOnly: true }) continue;

            var classification = ParameterMemberSemanticModel.Classify(member);

            // OPT-OUT, NOT OPT-IN, and this is the whole reason 330 hand-written Serialize/Deserialize
            // pairs exist. Requiring [Fitted], [Frozen] or [Buffer] before a member is persisted means
            // an author who adds a field and annotates nothing gets a model that serialises
            // incompletely and no error -- so the only way to be sure was to write the pair by hand,
            // which is two places to forget the same field instead of one. Storage is now persisted by
            // DEFAULT and a member has to say why it should not be.
            //
            // The four exclusions are the ones that would be wrong to persist, not the ones nobody
            // annotated:
            //   Trainable  already in the parameter vector; writing it again would restore it twice
            //   Scratch    a work buffer whose value between calls means nothing
            //   Alias      another name for a member already carried
            //   External   not this model's to save
            // Conflicting is excluded too, because a member carrying contradictory annotations is a
            // question for AIDN089 to answer rather than something to guess at here.
            if (classification.Kind is ParameterMemberSemanticModel.Kind.Trainable
                or ParameterMemberSemanticModel.Kind.Scratch
                or ParameterMemberSemanticModel.Kind.Alias
                or ParameterMemberSemanticModel.Kind.External
                or ParameterMemberSemanticModel.Kind.Conflicting)
            {
                continue;
            }

            // Whether somebody ASKED for this member to be state, as opposed to it being swept in by
            // the default. It decides how loudly an unsupported shape is reported, below.
            var annotated = classification.Kind is ParameterMemberSemanticModel.Kind.Fitted
                or ParameterMemberSemanticModel.Kind.Frozen
                or ParameterMemberSemanticModel.Kind.Buffer;

            // Keyed by DECLARING TYPE and member, not by member alone. A name is unique within one
            // class and nothing more: VectorAutoRegressionModel and VARMAModel each keep a private
            // Matrix<T> _residuals, which is ordinary C# and means the derived model's generated
            // registration met the base's under the same key and threw "State '_residuals' is
            // already declared". Every model with a field that shares a name with one further up its
            // own hierarchy had the same fault waiting in it.
            var call = DeclareCall(member.Name, $"{type.Name}.{member.Name}", memberType, numeric, annotated,
                nullableTarget: memberType.NullableAnnotation == NullableAnnotation.Annotated
                    || memberType.IsValueType);

            if (call is null)
            {
                // A member the default swept in whose type the registry cannot carry is passed over in
                // silence, because that is exactly what happened to it before persistence became the
                // default -- reporting it would turn "no change" into hundreds of new build errors
                // about members nobody claimed were state. An ANNOTATED member is the opposite case and
                // is still reported below.
                if (!annotated) continue;

                // LOUD, NOT SILENT. This member was CLASSIFIED as state -- somebody annotated it --
                // and the generator cannot express its type. Skipping quietly would drop annotated
                // state from the payload and produce a model that restores almost everything, which
                // is the exact failure this work exists to remove. GeneralizedAdditiveModel proved
                // it: its List<Vector<T>> knots were annotated, silently skipped, and Predict then
                // refused with "loaded without its fitted knot vectors".
                spc.ReportDiagnostic(Diagnostic.Create(
                    UnsupportedStateShape,
                    member.Locations.FirstOrDefault() ?? type.Locations.FirstOrDefault(),
                    type.Name,
                    member.Name,
                    memberType.ToDisplayString()));
                continue;
            }

            members.Add((member.Name, call));
        }

        if (members.Count == 0) return;

        // The type AND everything containing it. A nested partial can only be reopened inside partial
        // outers, so reporting only the inner one would name a fix that does not compile on its own.
        for (var scope = type; scope is not null; scope = scope.ContainingType)
        {
            if (IsPartial(scope)) continue;

            spc.ReportDiagnostic(Diagnostic.Create(
                MustBePartial,
                scope.Locations.FirstOrDefault(),
                scope.Name,
                string.Join(", ", members.Select(m => m.Name))));
            return;
        }

        spc.AddSource($"{type.ToDisplayString().Replace('<', '_').Replace('>', '_').Replace(',', '_')}.State.g.cs",
            Render(type, numeric, members));
    }

    /// <summary>True when the member is itself something that can serialize its own state.</summary>
    private static bool IsSerializableModel(ITypeSymbol type)
    {
        if (type.AllInterfaces.Any(i => i.Name is "IModelSerializer" or "IFullModel" or "INeuralNetwork")
            || type.Name is "IModelSerializer" or "IFullModel" or "INeuralNetwork")
        {
            return true;
        }

        return false;
    }

    /// <summary>Finds the inherited hook this generator overrides, and with it the numeric type.</summary>
    private static IMethodSymbol? FindHook(INamedTypeSymbol type)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            var hook = current.GetMembers("RegisterGeneratedState").OfType<IMethodSymbol>().FirstOrDefault();
            if (hook is not null) return hook;
        }

        return null;
    }

    /// <summary>Maps a member's type onto the registry call that persists it.</summary>
    /// <remarks>
    /// Returns null for a shape the registry cannot express. That is deliberately silent HERE and
    /// loud elsewhere: AIDN088 already refuses to let numeric state go unclassified, so a member that
    /// reaches this point and has no mapping is a container the registry has not learned yet, and the
    /// model keeps its own declaration until it does.
    /// </remarks>
    private static string? DeclareCall(
        string name, string id, ITypeSymbol memberType, string numeric, bool annotated, bool nullableTarget)
    {
        // A nullable value type has a state the registry cannot express: "not set" is not a number,
        // and the getter cannot hand an int? to something expecting an int. MOMENT proved it -- the
        // display string had its '?' trimmed before matching, so an int? matched the int case and the
        // generated lambda would not compile. Declining is the honest answer; inventing a zero for it
        // would silently turn "never configured" into "configured to zero" on every round trip.
        if (memberType is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T })
        {
            return null;
        }

        // Namespaces stripped before matching. The display string is fully qualified, so
        // List<AiDotNet.Tensors.LinearAlgebra.Vector<T>> does not end with "List<Vector<T>>" and a
        // naive suffix test silently declined it -- which is how the knots got dropped.
        var display = memberType.ToDisplayString().TrimEnd('?');
        var key = System.Text.RegularExpressions.Regex
            .Replace(display, @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
            .Replace($"<{numeric}>", "<T>");

        // A non-nullable field cannot be handed a null, and the null-forgiving operator is not an
        // option here -- AIDN071 rejects it precisely because it suppresses the question rather than
        // answering it. So a null in the payload leaves the constructed value in place, which is the
        // honest reading: the saving model had nothing there to restore.
        var setter = nullableTarget
            ? $"v => {name} = v"
            : $"v => {{ if (v is not null) {name} = v; }}";
        var getter = $"() => {name}";

        return key switch
        {
            var k when k.EndsWith(".Vector<T>") || k == "Vector<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            var k when k.EndsWith(".Matrix<T>") || k == "Matrix<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            var k when k.EndsWith(".Tensor<T>") || k == "Tensor<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "List<Vector<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "List<Matrix<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Matrix<T>[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Vector<int>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Dictionary<int, Vector<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Vector<T>[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "int[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "double[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "T[]" => $"state.DeclareArray(\"{id}\", {getter}, {setter});",
            "int" => $"state.DeclareInt32(\"{id}\", {getter}, {setter});",
            "double" => $"state.DeclareDouble(\"{id}\", {getter}, {setter});",
            "bool" => $"state.DeclareBoolean(\"{id}\", {getter}, {setter});",
            "string" => $"state.DeclareString(\"{id}\", {getter}, {setter});",
            "T" => $"state.DeclareScalar(\"{id}\", {getter}, {setter});",

            // A nested model carries its own state through its own Serialize, so the parent only has
            // to say that it is there. Restored IN PLACE, because the parent builds it and what
            // travels is its state rather than its identity.
            // A parameter source keeps its state in a vector rather than a payload.
            // A RECURSIVE NODE GRAPH -- a decision tree. The registry has carried these all along
            // through DeclareGraph; nothing could reach it, because describing a node meant writing the
            // description by hand, which is the boilerplate this work removes rather than relocates.
            // The node type says everything needed: its own properties give the fields, the ones typed
            // as itself give the children, and its parameterless constructor gives Create. Eight
            // tree and ensemble model families hand-wrote a Serialize to walk this structure, and
            // deleting those pairs without this failed 26 tests -- exactly the silent state loss the
            // deleter's own design warns about.
            _ when IsRecursiveNode(memberType) is { } node => GraphCall(id, name, node, numeric),

            // THE CHILD PATHS STAY OPT-IN even though storage is now opt-out, and the difference is
            // real rather than cautious. Storage is state by its nature -- a Matrix<T> a model holds
            // between calls is something it learned. An object is not: a model also holds its
            // optimizer, its scheduler, its loss function, and none of those are state to restore.
            // Sweeping them in on the default is how three networks came to persist a _trainOptimizer
            // that is never assigned, which the compiler reported as CS0649 and which would have
            // travelled in every payload as a null. So a nested model is carried when somebody says it
            // is state, and otherwise left alone.
            _ when !IsInfrastructure(memberType) && memberType.AllInterfaces.Any(i => i.Name == "IParameterSource")
                   && !IsSerializableModel(memberType) =>
                $"state.DeclareParameterSource(\"{id}\", {getter});",

            _ when !IsInfrastructure(memberType) && IsSerializableModel(memberType) =>
                $"state.DeclareChild<{memberType.ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});",

            // A list of nested models -- an agent's per-actor target networks, a mixer's per-agent
            // heads. Same rule as a single child: each carries its own state, restored in place.
            // A LIST of models is carried by default, unlike a single one, and the split is not
            // arbitrary. The reason single children stay opt-in is the optimizer a model holds, and an
            // optimizer is held in a field of its own -- nobody keeps a list of them. A list of models
            // is an ensemble's members: a random forest IS its trees, and dropping them leaves a
            // forest that restores with nothing to predict from. RandomForest, DART and
            // ExtremelyRandomizedTrees all failed their round trip on exactly that.
            _ when memberType is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } list
                   && IsSerializableModel(list.TypeArguments[0]) =>
                $"state.DeclareChildList<{list.TypeArguments[0].ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});",

            _ => null,
        };
    }

    private static string Render(INamedTypeSymbol type, string numeric, List<(string Name, string Call)> members)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();

        var ns = type.ContainingNamespace.IsGlobalNamespace ? null : type.ContainingNamespace.ToDisplayString();
        if (ns is not null)
        {
            sb.AppendLine($"namespace {ns};");
            sb.AppendLine();
        }

        // A NESTED type has to be reopened through the types that contain it. Emitting `partial class
        // Inner` at namespace level declares a DIFFERENT type -- one with no base -- and the override
        // then has nothing to override, which is what CS0115 was reporting for five model classes
        // declared inside their test fixtures. Outermost first, so the chain reads the way it is
        // written in the source.
        var chain = new List<INamedTypeSymbol>();
        for (var outer = type.ContainingType; outer is not null; outer = outer.ContainingType)
        {
            chain.Insert(0, outer);
        }

        var indent = string.Empty;

        foreach (var outer in chain)
        {
            sb.AppendLine($"{indent}partial class {outer.Name}{TypeParametersOf(outer)}");
            sb.AppendLine($"{indent}{{");
            indent += "    ";
        }

        sb.AppendLine($"{indent}partial class {type.Name}{TypeParametersOf(type)}");
        sb.AppendLine($"{indent}{{");
        sb.AppendLine($"{indent}    /// <summary>Auto-generated state declarations for this model's own members.</summary>");
        sb.AppendLine($"{indent}    protected override void RegisterGeneratedState(global::AiDotNet.Models.ModelStateRegistry<{numeric}> state)");
        sb.AppendLine($"{indent}    {{");
        sb.AppendLine($"{indent}        base.RegisterGeneratedState(state);");

        // Ordered by name so the payload does not depend on declaration order, which a refactor can
        // change without anybody meaning to.
        foreach (var member in members.OrderBy(m => m.Name, System.StringComparer.Ordinal))
        {
            sb.AppendLine($"{indent}        {member.Call}");
        }

        sb.AppendLine($"{indent}    }}");
        sb.AppendLine($"{indent}}}");

        for (var i = chain.Count - 1; i >= 0; i--)
        {
            indent = indent.Substring(0, indent.Length - 4);
            sb.AppendLine($"{indent}}}");
        }

        return sb.ToString();
    }

    /// <summary>The node type, when a member holds the root of a graph whose nodes point at their own kind.</summary>
    /// <remarks>
    /// Recognised by shape rather than by name, so a consumer's own tree is carried on the same terms
    /// as ours. Three things have to hold: it is a class, it can be built with no arguments -- the
    /// registry has to make one per node on restore -- and at least one settable property is typed as
    /// the node itself, which is what makes it a graph rather than a plain object.
    /// </remarks>
    private static INamedTypeSymbol? IsRecursiveNode(ITypeSymbol memberType)
    {
        if (memberType is not INamedTypeSymbol { TypeKind: TypeKind.Class } named) return null;
        if (named.IsAbstract) return null;

        var self = named.ToDisplayString().TrimEnd('?');

        if (!named.InstanceConstructors.Any(c => c.Parameters.Length == 0 && c.DeclaredAccessibility == Accessibility.Public))
        {
            return null;
        }

        var recursive = named.GetMembers().OfType<IPropertySymbol>().Any(p =>
            !p.IsStatic && p.SetMethod is not null && p.GetMethod is not null
            && p.Type.ToDisplayString().TrimEnd('?') == self);

        return recursive ? named : null;
    }

    /// <summary>Builds the DeclareGraph call that carries a node graph.</summary>
    /// <remarks>
    /// Only the property shapes NodeShape can carry are described. What is left out is left out on
    /// purpose: a decision node also holds the training samples that produced it and, in a model-tree,
    /// a fitted sub-model, and neither is needed to reproduce a prediction. Carrying the samples would
    /// put the training set inside every saved model.
    /// </remarks>
    private static string GraphCall(string id, string name, INamedTypeSymbol node, string numeric)
    {
        var self = node.ToDisplayString().TrimEnd('?');
        var qualified = "global::" + self;
        var shape = new StringBuilder();

        shape.Append($".Create(() => new {qualified}())");

        foreach (var property in node.GetMembers().OfType<IPropertySymbol>()
            .Where(p => !p.IsStatic && p.GetMethod is not null && p.SetMethod is not null)
            .OrderBy(p => p.Name, System.StringComparer.Ordinal))
        {
            var propertyType = property.Type.ToDisplayString().TrimEnd('?');
            var bare = System.Text.RegularExpressions.Regex
                .Replace(propertyType, @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
                .Replace($"<{numeric}>", "<T>");

            var call = propertyType == self ? "Child"
                : bare switch
                {
                    "int" => "Int32",
                    "bool" => "Boolean",
                    "T" => "Scalar",
                    "Vector<T>" => "Vector",
                    _ => null,
                };

            if (call is null) continue;

            shape.Append($".{call}(n => n.{property.Name}, (n, v) => n.{property.Name} = v)");
        }

        return $"state.DeclareGraph<{qualified}>(\"{id}\", () => {name}, v => {name} = v, n => n{shape});";
    }

    /// <summary>True for a member that is training machinery rather than state to restore.</summary>
    /// <remarks>
    /// THE RIGHT DISCRIMINATOR, replacing "single children are opt-in". That gate was written to keep a
    /// model's optimizer out of the payload, and it did -- along with every legitimate sub-model held in
    /// a field of its own. SiameseNetwork keeps its twin in <c>_subnetwork</c> and its head in
    /// <c>_outputLayer</c>, both single, and lost both: clone output moved 0.585 -> 0.502. Its optimizer
    /// sits in the very next field, which is what makes the distinction clear -- it is not arity that
    /// separates them, it is WHAT THEY ARE.
    /// <para>
    /// Matched by interface name so a consumer's own optimizer is excluded on the same terms as ours,
    /// and kept short on purpose: everything not on this list is state, which is the direction the
    /// default should fail in.
    /// </para>
    /// </remarks>
    private static bool IsInfrastructure(ITypeSymbol type)
    {
        static bool Machinery(string name)
            => name is "IOptimizer" or "IGradientBasedOptimizer" or "ILossFunction"
                or "ILearningRateScheduler" or "IRegularization" or "IActivationFunction";

        return Machinery(type.Name) || type.AllInterfaces.Any(i => Machinery(i.Name));
    }

    private static bool IsPartial(INamedTypeSymbol type)
        => type.DeclaringSyntaxReferences
            .Select(r => r.GetSyntax())
            .OfType<ClassDeclarationSyntax>()
            .Any(d => d.Modifiers.Any(m => m.ValueText == "partial"));

    private static string TypeParametersOf(INamedTypeSymbol type)
    {
        return type.TypeParameters.Length > 0
            ? "<" + string.Join(", ", type.TypeParameters.Select(p => p.Name)) + ">"
            : string.Empty;
    }
}
