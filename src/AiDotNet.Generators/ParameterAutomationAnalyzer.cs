using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Drives the migration to automatic parameter and buffer registration, and keeps it from
/// regressing once complete.
/// </summary>
/// <remarks>
/// <para>
/// A layer's parameter surface should be DERIVED, never written. <c>LayerBase</c> folds
/// <c>ParameterCount</c>, <c>GetParameters</c> and <c>SetParameters</c> out of one ordered
/// enumeration — own tensors, registered buffers, then sub-layers — so a hand-written copy can only
/// restate that fold or silently disagree with it. Disagreement is not cosmetic:
/// <c>SetParameters</c> pairs by length, so a checkpoint restores into the wrong tensors and the
/// model keeps its initial weights with nothing failing.
/// </para>
/// <para>
/// The measured cost of leaving this to authors: RWKVLayer registered 8 weight matrices and omitted
/// 10 more learned tensors — both LayerNorm affine pairs, the time- and channel-mixing coefficients
/// RWKV is named for, and the first-token bonus. The optimizer never updated them, and nothing
/// reported it, because the layer's three hand-written surfaces agreed with each other on the wrong
/// set.
/// </para>
/// <para>
/// Implemented as an <c>IIncrementalGenerator</c> that reports diagnostics rather than a
/// <c>DiagnosticAnalyzer</c>: this project wires generators through
/// <c>OutputItemType="Analyzer"</c> and standalone analyzer types are not loaded, which is why the
/// first version compiled, shipped the right IDs, and silently reported nothing. Every other
/// diagnostic in this repo (AIDN050-052 and friends) uses this shape.
/// </para>
/// <para>
/// Warning severity during migration so the build stays green while layers are converted. These
/// become errors once the count reaches zero, at which point forgetting is impossible rather than
/// merely detectable.
/// </para>
/// </remarks>
[Generator]
public class ParameterAutomationAnalyzer : IIncrementalGenerator
{
    private const string Category = "AiDotNet.ParameterAutomation";

    private static readonly DiagnosticDescriptor MissingAutoParameters = new(
        id: "AIDN070",
        title: "Layer does not use automatic parameter discovery",
        messageFormat: "Layer '{0}' is not marked [AutoParameters]; its tensor fields must be registered by hand, which is how parameters get silently omitted from training",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Add [AutoParameters] so every non-nullable tensor field is registered automatically. " +
                     "Mark exceptions with [Buffer] (persistent, never trained) or [Scratch] (transient).");

    private static readonly DiagnosticDescriptor RedundantParameterSurface = new(
        id: "AIDN071",
        title: "Parameter surface is derived and should not be overridden",
        messageFormat: "'{0}' overrides {1}; LayerBase derives it from the same registry, so this can only restate the fold or drift from it",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Delete the override and let LayerBase derive the value. ParameterCount, GetParameters and " +
                     "SetParameters fold one enumeration in one order, so they cannot disagree.");

    private static readonly DiagnosticDescriptor RedundantModelSurface = new(
        id: "AIDN072",
        title: "Model parameter surface is derived and should not be overridden",
        messageFormat: "'{0}' overrides {1}; the model base derives it from the registered components, so this can only restate the fold or drift from it",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Declare the model's components in RegisterComponents and delete the override. " +
                     "A hand-written model surface is how a count and a vector come to disagree: 44 " +
                     "diffusion models mixed a COUNT from one child with a VECTOR LENGTH from another " +
                     "in one expression, and VideoUNetPredictor's estimate was nine times out.");

    private static readonly DiagnosticDescriptor UndiscoverableWeight = new(
        id: "AIDN073",
        title: "Field holds weights the parameter generator cannot see",
        messageFormat: "'{0}.{1}' is {2}, so automatic discovery skips it; it contributes to no ParameterCount, no checkpoint and no optimizer",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Make it a non-readonly Tensor<T> if it is trained, or mark it [Buffer] (persistent " +
                     "state that is never optimized) or [Scratch] (rebuilt each forward). Say which — " +
                     "silence here is indistinguishable from an oversight, and the parameter-count " +
                     "contract test cannot catch it because the count and the vector omit the SAME field " +
                     "and therefore agree on a wrong answer. InformerModel reported 1,688 parameters " +
                     "against a real 167,640 this way: every Q/K/V/O projection, FFN weight and LayerNorm " +
                     "gain it owned was readonly, so nothing counted, saved or trained them.");

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var classes = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax,
                transform: static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol)
            .Where(static s => s is not null);

        context.RegisterSourceOutput(classes.Collect(), static (spc, symbols) =>
        {
            var seen = new System.Collections.Generic.HashSet<string>();
            foreach (var type in symbols)
            {
                if (type is null || type.IsAbstract || type.TypeKind != TypeKind.Class) continue;

                // Models whose base already derives the surface from a component registry.
                if (ExtendsAny(type, "AiDotNet.Diffusion.DiffusionModelBase<",
                                     "AiDotNet.Diffusion.VAE.VAEModelBase<"))
                {
                    var modelLoc = type.Locations.FirstOrDefault(l => l.IsInSource);
                    if (modelLoc is not null && seen.Add(type.ToDisplayString()))
                    {
                        foreach (var member in type.GetMembers())
                        {
                            if (!member.IsOverride) continue;
                            string? ms = member switch
                            {
                                IPropertySymbol p2 when p2.Name == "ParameterCount" => "ParameterCount",
                                IMethodSymbol m2 when m2.Name == "GetParameters" && m2.Parameters.Length == 0 => "GetParameters",
                                IMethodSymbol m2 when m2.Name == "SetParameters" && m2.Parameters.Length == 1 => "SetParameters",
                                _ => null,
                            };
                            if (ms is null) continue;
                            var mloc = member.Locations.FirstOrDefault(l => l.IsInSource);
                            if (mloc is not null)
                                spc.ReportDiagnostic(Diagnostic.Create(RedundantModelSurface, mloc, type.Name, ms));
                        }
                    }
                    continue;
                }

                if (!ExtendsLayerBase(type)) continue;
                if (!seen.Add(type.ToDisplayString())) continue;   // partial declarations

                var location = type.Locations.FirstOrDefault(l => l.IsInSource);
                if (location is null) continue;

                bool auto = type.GetAttributes()
                    .Any(a => a.AttributeClass?.Name is "AutoParametersAttribute" or "AutoParameters");

                // A layer holding no tensors of its own has nothing to discover; nagging it is noise.
                bool ownsTensors = type.GetMembers().OfType<IFieldSymbol>()
                    .Any(f => !f.IsStatic && !f.IsImplicitlyDeclared && f.AssociatedSymbol is null
                              && IsTensorType(f.Type));

                if (!auto && ownsTensors)
                    spc.ReportDiagnostic(Diagnostic.Create(MissingAutoParameters, location, type.Name));

                // AIDN073: fields the generator's discovery predicate will silently skip. Only checked
                // under [AutoParameters], where discovery is the ONLY way a weight reaches the surface;
                // without the attribute the author is registering by hand and knows what they own.
                if (auto)
                {
                    foreach (var f in type.GetMembers().OfType<IFieldSymbol>())
                    {
                        if (f.IsStatic || f.IsImplicitlyDeclared || f.AssociatedSymbol is not null) continue;
                        if (HasAnyAttribute(f, "BufferAttribute", "Buffer", "ScratchAttribute", "Scratch")) continue;

                        // Nullable is the sanctioned way to say "optional / not always present", and the
                        // gradient and cache fields that use it are legion. Flagging them would bury the
                        // real finding, so only the two skips that silently ate real weights are reported.
                        bool nullable = f.NullableAnnotation == NullableAnnotation.Annotated
                                        || f.Type.NullableAnnotation == NullableAnnotation.Annotated;
                        if (nullable) continue;

                        // An ARRAY of matrices is skipped for the same reason a single one is, and
                        // LoHaAdapter holds its Hadamard factors that way -- report the element type
                        // so the message names something the author can actually find in the file.
                        var probe = f.Type is IArrayTypeSymbol arr ? arr.ElementType : f.Type;

                        string? why = null;
                        if (IsTensorType(probe) && f.IsReadOnly)
                            why = "readonly";
                        else if (IsMatrixOrVectorType(probe))
                            why = probe.OriginalDefinition.ToDisplayString().Contains(".Matrix<")
                                ? (f.Type is IArrayTypeSymbol ? "an array of Matrix<T>" : "Matrix<T>") + " rather than Tensor<T>"
                                : (f.Type is IArrayTypeSymbol ? "an array of Vector<T>" : "Vector<T>") + " rather than Tensor<T>";

                        if (why is null) continue;
                        var fl = f.Locations.FirstOrDefault(l => l.IsInSource);
                        if (fl is null) continue;
                        spc.ReportDiagnostic(Diagnostic.Create(UndiscoverableWeight, fl, type.Name, f.Name, why));
                    }
                }

                foreach (var member in type.GetMembers())
                {
                    if (!member.IsOverride) continue;
                    string? surface = member switch
                    {
                        IPropertySymbol p when p.Name == "ParameterCount" => "ParameterCount",
                        IMethodSymbol m when m.Name == "GetParameters" && m.Parameters.Length == 0 => "GetParameters",
                        IMethodSymbol m when m.Name == "SetParameters" && m.Parameters.Length == 1 => "SetParameters",
                        _ => null,
                    };
                    if (surface is null) continue;
                    var ml = member.Locations.FirstOrDefault(l => l.IsInSource);
                    if (ml is null) continue;
                    spc.ReportDiagnostic(Diagnostic.Create(RedundantParameterSurface, ml, type.Name, surface));
                }
            }
        });
    }

    /// <summary>True when <paramref name="type"/> derives from any of the given base metadata prefixes.</summary>
    private static bool ExtendsAny(INamedTypeSymbol type, params string[] prefixes)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            var name = b.OriginalDefinition.ToDisplayString();
            foreach (var prefix in prefixes)
            {
                if (name.StartsWith(prefix, System.StringComparison.Ordinal)) return true;
            }
        }
        return false;
    }

    private static bool ExtendsLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.NeuralNetworks.Layers.LayerBase<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    private static bool IsTensorType(ITypeSymbol type)
    {
        for (var c = type; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.Tensors.LinearAlgebra.Tensor<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Matrix&lt;T&gt; and Vector&lt;T&gt;: numeric containers the discovery predicate does not accept,
    /// so weights held in them reach no parameter surface. LoRALayer held its A and B this way and
    /// derived a ParameterCount of zero.
    /// </summary>
    private static bool IsMatrixOrVectorType(ITypeSymbol type)
    {
        var name = type.OriginalDefinition.ToDisplayString();
        return name.StartsWith("AiDotNet.Tensors.LinearAlgebra.Matrix<", System.StringComparison.Ordinal)
            || name.StartsWith("AiDotNet.Tensors.LinearAlgebra.Vector<", System.StringComparison.Ordinal);
    }

    private static bool HasAnyAttribute(ISymbol symbol, params string[] names)
    {
        foreach (var a in symbol.GetAttributes())
        {
            var n = a.AttributeClass?.Name;
            if (n is null) continue;
            foreach (var candidate in names)
            {
                if (string.Equals(n, candidate, System.StringComparison.Ordinal)) return true;
            }
        }
        return false;
    }
}
