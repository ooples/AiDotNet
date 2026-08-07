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
}
