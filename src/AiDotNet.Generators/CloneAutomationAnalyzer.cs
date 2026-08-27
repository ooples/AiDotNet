using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps cloning automated: reports a hand-written clone the base already reproduces, and reports a
/// model the clone plan cannot rebuild.
/// </summary>
/// <remarks>
/// <para>
/// Without this the overrides grow back. <c>CreateNewInstance</c>, <c>DeepCopy</c> and <c>Clone</c>
/// were abstract on eleven base classes, so every concrete model was compelled to write one, and
/// 1465 of them did. Making the bases concrete removes the compulsion but not the habit: the next
/// model added to the library will still be written with a copy of its neighbour's override, and
/// nothing would say otherwise.
/// </para>
/// <para>
/// The two rules are deliberately different in severity, because they describe different situations.
/// A redundant override is a mistake with a mechanical fix -- delete it -- and there are currently
/// none, so it is an error and stays at zero. A model the plan cannot rebuild is a backlog item with
/// a real fix (store the constructor argument in a field so it can be read back), and there are
/// hundreds, so it is informational and names the parameter that blocks each one.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class CloneAutomationAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A hand-written clone the base class already reproduces.</summary>
    private static readonly DiagnosticDescriptor RedundantOverride = new(
        "ADN0058",
        "Clone override duplicates what the base class already does",
        "'{0}.{1}' only reconstructs the type, and the clone plan already records the constructor to "
            + "do that. Delete the override: the base reproduces it, and a hand-written copy is a "
            + "place a future constructor argument can be dropped without anything failing",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>A model whose constructor cannot be replayed from what the instance still holds.</summary>
    private static readonly DiagnosticDescriptor Unreproducible = new(
        "ADN0059",
        "Model cannot be rebuilt from its own state",
        "'{0}' cannot be rebuilt by the clone plan, so it still needs a hand-written clone. {1} "
            + "Store each one in a field named after it ('_name') and the generator will replay the "
            + "constructor instead",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    /// <summary>State a layer persists by hand instead of declaring it.</summary>
    /// <remarks>
    /// <para>
    /// Every hand-written <c>Serialize</c> in this library was inspected, and not one writes state
    /// that lacks a declaration mechanism. They write constructor parameters (<c>[LayerState]</c>
    /// already generates both halves), tensors (<c>RegisterTrainableParameter</c> and
    /// <c>RegisterBuffer</c> already own those), or the resolved shape (the base payload now carries
    /// it). They exist because the state was never declared, so nothing generated it.
    /// </para>
    /// <para>
    /// ERROR, DELIBERATELY, AND RED UNTIL THE BACKLOG IS ZERO. This rule exists to verify FULL
    /// compliance, and a warning is the thing everyone learns to scroll past -- 368 hand-written
    /// halves accumulated under exactly that kind of silence. The build stays red until every one of
    /// them declares its state, which is the point: the count is the work, and it is not allowed to
    /// be invisible.
    /// </para>
    /// </remarks>
    private static readonly DiagnosticDescriptor HandWrittenSerialization = new(
        "ADN0060",
        "Serialization is hand-written instead of declared",
        "'{0}.{1}' persists state by hand. Declare the state instead and the generator writes and "
            + "reads it: mark constructor parameters [LayerState], register tensors with "
            + "RegisterTrainableParameter or RegisterBuffer, and let the base carry the resolved "
            + "shape. A hand-written pair is two places to forget the same field",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>A concrete model has taken ownership of framework lifecycle plumbing.</summary>
    /// <remarks>
    /// A method can be non-redundant only because the shared generator/base path still has a gap.
    /// Keeping the method makes that gap permanent and lets the next model copy it. Concrete models
    /// therefore declare state and construction inputs; only abstract family bases may implement a
    /// lifecycle policy. The rule is intentionally independent of body shape and plan availability.
    /// </remarks>
    private static readonly DiagnosticDescriptor ConcreteLifecycleOverride = new(
        "ADN0063",
        "Concrete model lifecycle must be generated",
        "'{0}.{1}' is model-owned lifecycle plumbing. Delete the override and express its construction, "
            + "clone, serialization, or parameter ownership through generated declarations and shared "
            + "base infrastructure",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    /// <inheritdoc/>
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(
            RedundantOverride,
            Unreproducible,
            HandWrittenSerialization,
            ConcreteLifecycleOverride);

    /// <inheritdoc/>
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
        context.RegisterSyntaxNodeAction(AnalyzeType, SyntaxKind.ClassDeclaration);
    }

    /// <summary>
    /// Reports an override whose whole body is a reconstruction the plan already performs.
    /// </summary>
    /// <param name="context">The analysis context.</param>
    /// <remarks>
    /// Only a body that is exactly one <c>return new ...;</c> is reported. An override that also
    /// resolves a shape, branches on a mode or copies a field is doing something the base does not,
    /// and telling someone to delete it would be wrong -- that work has to move deliberately.
    /// </remarks>
    private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
    {
        var method = (MethodDeclarationSyntax)context.Node;

        if (!method.Modifiers.Any(m => m.ValueText == "override")) return;

        var name = method.Identifier.ValueText;

        // The acceptance boundary is architectural, not syntactic. A complicated override is not
        // evidence that model-owned lifecycle code is necessary; it is evidence that the common
        // path still needs to learn that state shape. Abstract family bases remain the right home
        // for genuinely shared policy, while concrete models only declare their unique state.
        bool isParameterOwnershipHook = name is "GetExtraTrainableLayers" or "GetExtraTrainableTensors";
        if ((name is "Clone" or "DeepCopy" or "CreateNewInstance"
                or "Serialize" or "Deserialize"
                or "SerializeNetworkSpecificData" or "DeserializeNetworkSpecificData"
                || isParameterOwnershipHook)
            && context.ContainingSymbol is IMethodSymbol { ContainingType: { IsAbstract: false } lifecycleOwner }
            && (IsModel(lifecycleOwner) || IsLayer(lifecycleOwner) || isParameterOwnershipHook))
        {
            // Explicit compatibility formats may own only the public byte encoding. Construction,
            // cloning, parameter ownership, and network-specific state remain generated/base-owned.
            if (name is "Serialize" or "Deserialize"
                && HasCustomSerializationFormat(lifecycleOwner))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                ConcreteLifecycleOverride,
                method.Identifier.GetLocation(),
                lifecycleOwner.Name,
                name));
            return;
        }
        // Serialization is reported wherever it is hand-written, without asking whether the base
        // "already reproduces" it. That question is the wrong one: the state a layer persists by
        // hand is state nothing declared, so the base COULD not reproduce it, and treating that as
        // justification is what let 297 hand-written halves accumulate. The remedy is to declare the
        // state, not to prove the override earns its place.
        if (name is "Serialize" or "Deserialize")
        {
            // An override that hands the work to base is a DECORATOR, not a second copy of the
            // persistence. Counting calls, logging, timing or taking a lock around
            // `base.Serialize()` adds no field that anyone can forget, because the base still
            // writes every declared member. Reporting those said "declare your state instead" to
            // code that already does exactly that.
            //
            // Deliberately narrow: this exempts a body only when it CALLS base. A body that calls
            // base and then appends its own hand-written payload is still reported, which is the
            // shape that actually risks drift.
            if (DelegatesToBase(method, name)) return;

            if (context.ContainingSymbol is IMethodSymbol { OverriddenMethod: not null and not { IsAbstract: true } }
                && context.ContainingSymbol.ContainingType is INamedTypeSymbol owner)
            {
                // A small number of public checkpoint formats deliberately encode state differently
                // from its in-memory representation (for example, Adam8Bit's versioned quantized
                // payload and legacy compatibility checks). Require an explicit type-level marker
                // for those cases; every unmarked hand-written serializer remains an error.
                if (HasCustomSerializationFormat(owner)) return;

                context.ReportDiagnostic(Diagnostic.Create(
                    HandWrittenSerialization, method.Identifier.GetLocation(), owner.Name, name));
            }

            return;
        }

        // CreateInstanceForCopy belongs here for exactly the same reason as the other three: it is a
        // factory hook whose whole body is "build one of me", which is what the recorded constructor
        // already does. Leaving it out of the list was an omission rather than a decision -- 14 sites
        // sat in the same shape as the 607 that went, and nothing was naming them.
        if (name is not ("CreateNewInstance" or "DeepCopy" or "Clone" or "CreateInstanceForCopy")) return;
        if (method.ParameterList.Parameters.Count != 0) return;

        if (!IsSingleReturnOfNewObject(method) && !IsPureForwarder(method, name)) return;

        if (context.ContainingSymbol is not IMethodSymbol symbol) return;

        // An override that satisfies an abstract member is not optional, whatever its body looks
        // like. A test file declares its own MockModelBase with `public abstract Clone()`, and
        // telling three mocks to delete the only implementation of it produced CS0534 instead of a
        // cleaner tree. Redundancy is a property of the base being CONCRETE, not of the body alone.
        if (symbol.OverriddenMethod is null || symbol.OverriddenMethod.IsAbstract) return;

        if (symbol.ContainingType is not INamedTypeSymbol type) return;
        if (ClonePlanGenerator.CollectConstructorParameters(type, IsModel(type)) is null) return;

        context.ReportDiagnostic(Diagnostic.Create(
            RedundantOverride, method.Identifier.GetLocation(), type.Name, name));
    }

    /// <summary>
    /// Reports a model the plan cannot rebuild, naming the constructor parameters that block it.
    /// </summary>
    /// <param name="context">The analysis context.</param>
    private static void AnalyzeType(SyntaxNodeAnalysisContext context)
    {
        var declaration = (ClassDeclarationSyntax)context.Node;

        if (declaration.Modifiers.Any(m => m.ValueText is "abstract" or "static")) return;
        // GetDeclaredSymbol, not ContainingSymbol: for a class declaration the containing symbol is
        // the namespace, so the cast below silently never matched and this rule reported nothing.
        if (context.SemanticModel.GetDeclaredSymbol(declaration) is not INamedTypeSymbol type) return;
        if (!IsModel(type)) return;
        if (ClonePlanGenerator.CollectConstructorParameters(type, isModel: true) is not null) return;

        var constructors = type.InstanceConstructors
            .Where(c => c.DeclaredAccessibility is Accessibility.Public or Accessibility.Internal)
            .Where(c => !c.IsStatic && c.Parameters.Length > 0)
            .ToList();

        if (constructors.Count == 0) return;

        var widest = constructors.Max(c => c.Parameters.Length);
        var candidates = constructors.Where(c => c.Parameters.Length == widest).ToList();

        // An ambiguous overload set is a different situation from a missing field, and saying
        // "add a backing field" would send someone to fix the wrong thing.
        var reason = candidates.Count > 1
            ? $"It declares {candidates.Count} constructors taking {widest} arguments, so nothing "
              + "records which one this instance was built with."
            : "These constructor parameters have no member holding their value: "
              + string.Join(", ", candidates[0].Parameters
                  .Where(p => p.RefKind != RefKind.None || ClonePlanGenerator.FindAnySource(type, p) is null)
                  .Select(p => $"'{p.Name}'"))
              + ".";

        context.ReportDiagnostic(Diagnostic.Create(
            Unreproducible, declaration.Identifier.GetLocation(), type.Name, reason));
    }

    /// <summary>
    /// Determines whether the body is exactly one object creation returned.
    /// </summary>
    /// <param name="method">The override to inspect.</param>
    /// <returns><see langword="true"/> when the body reconstructs and does nothing else.</returns>
    /// <summary>
    /// Determines whether a call only moves parameters onto a freshly built copy.
    /// </summary>
    /// <param name="call">The invocation to classify.</param>
    /// <returns><see langword="true"/> when it is one of the base's own parameter-transfer methods.</returns>
    /// <remarks>
    /// By simple name, because the receiver varies -- <c>clone.SetParameters(...)</c>,
    /// <c>clone._projectionLayer.SetParameters(...)</c>, a bare <c>TryShareParametersFrom(this)</c> --
    /// and the receiver is not what makes the call redundant. What makes it redundant is that
    /// Serialize already carries every declared parameter, so restating the transfer by hand restates
    /// the payload. The list is closed on purpose: a fourth name is a decision to make deliberately,
    /// not something to add because a body happened to contain it.
    /// </remarks>
    private static bool IsParameterTransfer(InvocationExpressionSyntax call)
    {
        var name = call.Expression switch
        {
            MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
            IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
            _ => null,
        };

        return name is "SetParameters" or "SetParameterChunks" or "TryShareParametersFrom";
    }

    private static bool IsSingleReturnOfNewObject(MethodDeclarationSyntax method)
    {
        var expression = method.ExpressionBody?.Expression;

        if (expression is not null)
        {
            return expression is ObjectCreationExpressionSyntax { Initializer: null };
        }

        if (method.Body is null || method.Body.Statements.Count == 0) return false;

        // EVERY return must hand back a plain construction, and nothing else may happen. A body that
        // only chooses BETWEEN constructors is still pure reconstruction, and that is by far the
        // commonest shape here: 584 of these pick an ONNX constructor when a model path is present
        // and a native one when it is not.
        //
        //     if (!_useNativeMode && _options.ModelPath is { } mp) return new AudioMAE<T>(Architecture, mp, _options);
        //     return new AudioMAE<T>(Architecture, _options);
        //
        // The plan already decides that at runtime -- it records every satisfiable constructor and
        // picks the one the INSTANCE can supply, which is exactly what taking the widest one
        // unconditionally got wrong when it passed null for onnxModelPath and made 51 models throw.
        // So the base reproduces this body, and reporting only the one-liner left 584 of them
        // invisible to the deletion loop.
        var returns = 0;

        // Which locals hold a plain construction, so `return clone;` can be told apart from
        // `return _cachedThing;`. Only the former is still pure reconstruction.
        var constructed = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);

        foreach (var local in method.Body.DescendantNodes().OfType<LocalDeclarationStatementSyntax>())
        {
            foreach (var declarator in local.Declaration.Variables)
            {
                if (declarator.Initializer?.Value is ObjectCreationExpressionSyntax { Initializer: null })
                {
                    constructed.Add(declarator.Identifier.ValueText);
                }
            }
        }

        foreach (var statement in method.Body.DescendantNodes().OfType<StatementSyntax>())
        {
            switch (statement)
            {
                case ReturnStatementSyntax { Expression: ObjectCreationExpressionSyntax { Initializer: null } }:
                    returns++;
                    break;

                // `return clone;` where clone was built above and nothing else was done to it.
                case ReturnStatementSyntax { Expression: IdentifierNameSyntax id }
                    when constructed.Contains(id.Identifier.ValueText):
                    returns++;
                    break;

                // A local holding a constructor argument, e.g. `var options = new ASTOptions(_options);`
                // or `var unetClone = (UNetNoisePredictor<T>)_unet.Clone();`.
                case LocalDeclarationStatementSyntax:
                // The branch itself; its own statements are visited separately.
                case IfStatementSyntax:
                case BlockSyntax:
                    break;

                // MOVING PARAMETERS ACROSS IS NOT EXTRA WORK -- it is the copy the base already makes.
                // The diffusion family's Clone overrides construct, then transfer, then return:
                //
                //     var clone = new OSDSModel<T>(...);
                //     if (!clone.TryShareParametersFrom(this)) clone.SetParameterChunks(GetParameterChunks());
                //     return clone;
                //
                // DeepCopy routes through Serialize and Deserialize, which carry every parameter the
                // model declared, so those two lines restate what the payload does. They were written
                // because the engine handed sub-modules across by reference and the copy shared its
                // U-Net with the original -- and the engine now clones them. The allowlist is
                // deliberately these three names and nothing else: anything further is real work the
                // base cannot know about, and still disqualifies the body.
                case ExpressionStatementSyntax { Expression: InvocationExpressionSyntax call }
                    when IsParameterTransfer(call):
                    break;

                // COPYING A CONFIGURATION VALUE ONTO THE NEW INSTANCE IS NOT EXTRA WORK EITHER. The
                // clone plan copies every configuration member, so `clone.ContextLength = ContextLength;`
                // only restates what CopyConfiguration already did.
                //
                // This is the shape that hid the real damage. Requiring the body to be pure
                // reconstruction meant a single field copy bought silence, and the override that
                // looked the most deliberate was the most dangerous: AnimateDiffModel rebuilt itself
                // with `options: null, scheduler: null`, patched three fields back, and dropped its
                // options and scheduler on every clone -- while the analyzer that exists to find
                // exactly that stayed quiet because line four was an assignment.
                //
                // Restricted to members of a local this body CONSTRUCTED: assigning to anything else
                // reaches outside the new instance, which the base cannot stand in for.
                case ExpressionStatementSyntax
                {
                    Expression: AssignmentExpressionSyntax
                    {
                        Left: MemberAccessExpressionSyntax { Expression: IdentifierNameSyntax target }
                    }
                }
                    when constructed.Contains(target.Identifier.ValueText):
                    break;

                // A loop, or any other call -- work the constructor did not do -- means the body is
                // not pure reconstruction and the base cannot stand in for it. Invocations on the new
                // instance stay disqualifying even when they look like setters: this rule is an
                // ERROR, and a method can do work no plan reproduces.
                default:
                    return false;
            }
        }

        return returns > 0;
    }

    /// <summary>
    /// True when the override only calls its own sibling and adds nothing.
    /// </summary>
    /// <param name="method">The override being analysed.</param>
    /// <param name="name">The override's name.</param>
    /// <returns><see langword="true"/> for a body that is exactly <c>SomeSibling()</c>.</returns>
    /// <remarks>
    /// <para>
    /// This class is not merely redundant, it is FATAL. The bases define
    /// <c>Clone() =&gt; DeepCopy()</c>, so a type that also defines <c>DeepCopy() =&gt; Clone()</c>
    /// closes a two-frame cycle as soon as its own real <c>Clone</c> is removed. 227 types carried
    /// that forwarder and 85 of them were already cyclic; <c>SuperNet</c> crashed the test host with
    /// a stack overflow after 12015 repetitions.
    /// </para>
    /// <para>
    /// It is also the deletion hazard the rest of this analyzer does not model. Proving the BASE
    /// reproduces an override says nothing about whether a SIBLING in the same type delegates to
    /// what is being removed, so removing <c>Clone</c> is correct in isolation and fatal next to a
    /// forwarder. Reporting the forwarder means the deletion loop removes BOTH, and the pair cannot
    /// regrow into a cycle.
    /// </para>
    /// </remarks>
    /// <summary>
    /// True when the body calls <c>base.&lt;name&gt;(...)</c> and writes nothing itself.
    /// </summary>
    /// <param name="method">The override being analysed.</param>
    /// <param name="name">Its name, so the call must be to the SAME member on the base.</param>
    /// <remarks>
    /// The second condition matters. A body may call base and then append its own payload -
    /// <c>ModelWrapperBase</c> does exactly that with the declared-state trailer - and that body IS
    /// a place a field can be forgotten, so it stays reported. Constructing a <c>BinaryWriter</c> or
    /// <c>BinaryReader</c> is the tell, and it is the tell every hand-written pair in this codebase
    /// exhibits.
    /// </remarks>
    private static bool DelegatesToBase(MethodDeclarationSyntax method, string name)
    {
        SyntaxNode? body = method.Body ?? (SyntaxNode?)method.ExpressionBody?.Expression;
        if (body is null) return false;

        var callsBase = body.DescendantNodesAndSelf()
            .OfType<InvocationExpressionSyntax>()
            .Any(call => call.Expression is MemberAccessExpressionSyntax
            {
                Expression: BaseExpressionSyntax,
                Name.Identifier.ValueText: { } called
            } && called == name);

        if (!callsBase) return false;

        var writesItself = body.DescendantNodesAndSelf()
            .OfType<ObjectCreationExpressionSyntax>()
            .Any(created => created.Type.ToString() is var t
                && (t.EndsWith("BinaryWriter", System.StringComparison.Ordinal)
                    || t.EndsWith("BinaryReader", System.StringComparison.Ordinal)));

        return !writesItself;
    }

    private static bool IsPureForwarder(MethodDeclarationSyntax method, string name)
    {
        var expression = method.ExpressionBody?.Expression;

        if (expression is null)
        {
            if (method.Body is null || method.Body.Statements.Count != 1) return false;
            if (method.Body.Statements[0] is not ReturnStatementSyntax { Expression: { } returned })
            {
                return false;
            }

            expression = returned;
        }

        // Only an unqualified or this-qualified call, and never to itself -- `Clone() => Clone()`
        // would be its own infinite recursion rather than a forwarder to a sibling.
        var invoked = expression switch
        {
            InvocationExpressionSyntax { ArgumentList.Arguments.Count: 0 } call => call.Expression switch
            {
                IdentifierNameSyntax id => id.Identifier.ValueText,
                MemberAccessExpressionSyntax { Expression: ThisExpressionSyntax } member
                    => member.Name.Identifier.ValueText,
                _ => null,
            },
            _ => null,
        };

        return invoked is "Clone" or "DeepCopy" or "CreateNewInstance" && invoked != name;
    }

    /// <summary>
    /// Determines whether the library treats this type as a model.
    /// </summary>
    /// <param name="type">The type to classify.</param>
    /// <returns><see langword="true"/> when it declares a model persistence surface.</returns>
    private static bool IsModel(INamedTypeSymbol type)
    {
        // Optimizers also expose serializer + shape contracts, but their lifecycle belongs to the
        // optimizer hierarchy. Treating that structural overlap as a model made ADN0063 demand
        // deletion of distributed optimizer checkpoint payloads that this generator does not own.
        if (type.AllInterfaces.Any(i => i.Name == "IOptimizer")) return false;

        bool isFullModel = type.AllInterfaces.Any(i => i.Name == "IFullModel");
        bool isSerializableShapedModel = type.AllInterfaces.Any(i => i.Name == "IModelSerializer")
            && type.AllInterfaces.Any(i => i.Name == "IModelShape");
        return isFullModel || isSerializableShapedModel;
    }

    /// <summary>Determines whether a type belongs to the generated layer lifecycle.</summary>
    private static bool IsLayer(INamedTypeSymbol type)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.Name == "LayerBase") return true;
        }

        return false;
    }

    /// <summary>Returns whether a type explicitly owns a stable custom checkpoint wire format.</summary>
    private static bool HasCustomSerializationFormat(INamedTypeSymbol type)
        => type.GetAttributes().Any(attribute
            => attribute.AttributeClass?.Name == "CustomSerializationFormatAttribute");
}
