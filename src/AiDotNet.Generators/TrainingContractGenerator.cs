using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Emits each neural-network model's training contract from the members it already declares, so the base class runs
/// every model's training step and no model overrides <c>Train</c> to supply it:
/// <list type="bullet">
/// <item><c>TrainingOptimizer =&gt; _optimizer</c> when the model declares a gradient-based <c>_optimizer</c> field
/// (the optimizer its constructor built, usually the paper recipe);</item>
/// <item><c>TrainingUnavailableReason</c> from a <c>bool _useNativeMode</c> field (an ONNX-backed instance cannot
/// train), or for an <c>IDetectionBackbone</c>, which trains only inside its parent detector;</item>
/// <item><c>OnTrainingStepStarting</c> calling the model's <c>EnsureNativeInitialized()</c>, so its native layers
/// exist before a training step reads them;</item>
/// <item><c>PrepareInputForTraining =&gt; PreprocessImage(input)</c> (or <c>PreprocessAudio</c>) when the model's own
/// native <c>PredictCore</c> preprocesses its input, so training sees the input inference sees. A model whose
/// prediction skips it, or preprocesses only for its ONNX graph, keeps the raw input.</item>
/// </list>
/// A model that declares either member itself keeps its own.
/// </summary>
[Generator]
public class TrainingContractGenerator : IIncrementalGenerator
{
    private const string NeuralNetworkBasePrefix = "AiDotNet.NeuralNetworks.NeuralNetworkBase<";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var models = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax { BaseList: not null },
                transform: static (ctx, ct) => Describe(ctx, ct))
            .Where(static model => model is not null);

        context.RegisterSourceOutput(models, static (spc, model) =>
        {
            if (model is not null) spc.AddSource(model.HintName, model.Source);
        });
    }

    private sealed class Contract
    {
        public Contract(string hintName, string source)
        {
            HintName = hintName;
            Source = source;
        }

        public string HintName { get; }
        public string Source { get; }

        public override bool Equals(object? obj) => obj is Contract other && other.HintName == HintName && other.Source == Source;
        public override int GetHashCode() => HintName.GetHashCode() ^ Source.GetHashCode();
    }

    private static Contract? Describe(GeneratorSyntaxContext ctx, System.Threading.CancellationToken ct)
    {
        var declaration = (ClassDeclarationSyntax)ctx.Node;
        if (ctx.SemanticModel.GetDeclaredSymbol(declaration, ct) is not INamedTypeSymbol type || type.IsStatic) return null;

        // One emission per type: the first declaration with a base list owns it.
        foreach (var reference in type.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax(ct) is ClassDeclarationSyntax { BaseList: not null } owner)
            {
                if (owner != declaration) return null;
                break;
            }
        }

        if (!IsPartial(type, ct)) return null;
        var numeric = NeuralNetworkNumericType(type);
        if (numeric is null) return null;

        bool emitOptimizer = !Declares(type, "TrainingOptimizer")
            && DeclaresGradientOptimizerField(type, numeric, ctx.SemanticModel.Compilation);
        bool backbone = !Declares(type, "TrainingUnavailableReason") && FirstImplementsDetectionBackbone(type);
        bool emitReason = !Declares(type, "TrainingUnavailableReason") && (backbone || DeclaresNativeModeField(type));
        // The front end PredictCore applies to its public input, which training must apply too: the image one, or the
        // audio one (mel, MFCC, STFT) where no training-pair transform of the type already applies it.
        string? preprocess = Declares(type, "PrepareInputForTraining") ? null
            : PredictCoreCalls(type, "PreprocessImage", ct) ? "PreprocessImage"
            : !Declares(type, "PrepareTrainingPair") && PredictCoreCalls(type, "PreprocessAudio", ct) ? "PreprocessAudio"
            : null;
        bool emitPreprocess = preprocess is not null;
        bool emitDisposedGuard = !Declares(type, "ThrowIfCannotTrain") && DeclaresThrowIfDisposed(type);
        bool emitNativeInit = !Declares(type, "OnTrainingStepStarting") && DeclaresParameterlessVoid(type, "EnsureNativeInitialized");
        if (!emitOptimizer && !emitReason && !emitPreprocess && !emitDisposedGuard && !emitNativeInit) return null;

        string t = numeric.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        string tensor = $"global::AiDotNet.Tensors.LinearAlgebra.Tensor<{t}>";
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        if (!type.ContainingNamespace.IsGlobalNamespace)
        {
            sb.AppendLine($"namespace {type.ContainingNamespace.ToDisplayString()};");
            sb.AppendLine();
        }

        var chain = new List<INamedTypeSymbol>();
        for (var outer = type.ContainingType; outer is not null; outer = outer.ContainingType) chain.Insert(0, outer);
        for (int depth = 0; depth < chain.Count; depth++)
        {
            var outerIndent = new string(' ', 4 * depth);
            sb.AppendLine($"{outerIndent}partial class {chain[depth].Name}{TypeParametersOf(chain[depth])}");
            sb.AppendLine($"{outerIndent}{{");
        }

        var indent = new string(' ', 4 * chain.Count);

        sb.AppendLine($"{indent}partial class {type.Name}{TypeParametersOf(type)}");
        sb.AppendLine($"{indent}{{");
        const string Generated = "[global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainingContractGenerator\", \"1.0.0\")]";
        if (emitOptimizer)
        {
            sb.AppendLine($"{indent}    /// <summary>The optimizer this model's constructor built; the base training step uses it.</summary>");
            sb.AppendLine($"{indent}    {Generated}");
            sb.AppendLine($"{indent}    protected internal override global::AiDotNet.Interfaces.IGradientBasedOptimizer<{t}, {tensor}, {tensor}>? TrainingOptimizer => _optimizer;");
        }

        if (emitReason)
        {
            if (backbone)
            {
                sb.AppendLine($"{indent}    /// <summary>A detection backbone trains as part of its parent detector, which runs the joint step.</summary>");
                sb.AppendLine($"{indent}    {Generated}");
                sb.AppendLine($"{indent}    protected override string? TrainingUnavailableReason =>");
                sb.AppendLine($"{indent}        $\"{{GetType().Name}}: detection backbones train as part of a parent detector (FasterRCNN, YOLO, DETR, ...), which runs the joint forward and backward pass.\";");
            }
            else
            {
                sb.AppendLine($"{indent}    /// <summary>An instance running an ONNX session instead of native layers cannot train.</summary>");
                sb.AppendLine($"{indent}    {Generated}");
                sb.AppendLine($"{indent}    protected override string? TrainingUnavailableReason =>");
                sb.AppendLine($"{indent}        _useNativeMode ? base.TrainingUnavailableReason : OnnxTrainingUnavailableReason;");
            }
        }

        if (emitDisposedGuard)
        {
            sb.AppendLine($"{indent}    /// <summary>A disposed instance cannot train.</summary>");
            sb.AppendLine($"{indent}    {Generated}");
            sb.AppendLine($"{indent}    protected override void ThrowIfCannotTrain()");
            sb.AppendLine($"{indent}    {{");
            sb.AppendLine($"{indent}        ThrowIfDisposed();");
            sb.AppendLine($"{indent}        base.ThrowIfCannotTrain();");
            sb.AppendLine($"{indent}    }}");
        }

        if (emitNativeInit)
        {
            sb.AppendLine($"{indent}    /// <summary>The native layers are built before the first training step reads them.</summary>");
            sb.AppendLine($"{indent}    {Generated}");
            sb.AppendLine($"{indent}    protected override void OnTrainingStepStarting({tensor} input, {tensor} target)");
            sb.AppendLine($"{indent}    {{");
            sb.AppendLine($"{indent}        base.OnTrainingStepStarting(input, target);");
            sb.AppendLine($"{indent}        EnsureNativeInitialized();");
            sb.AppendLine($"{indent}    }}");
        }

        if (emitPreprocess)
        {
            sb.AppendLine($"{indent}    /// <summary>Training sees the input <c>PredictCore</c> sees: the same preprocessing.</summary>");
            sb.AppendLine($"{indent}    {Generated}");
            sb.AppendLine($"{indent}    protected override {tensor} PrepareInputForTraining({tensor} input) => {preprocess}(input);");
        }

        sb.AppendLine($"{indent}}}");
        for (int depth = chain.Count - 1; depth >= 0; depth--)
            sb.AppendLine($"{new string(' ', 4 * depth)}}}");

        string hint = type.ToDisplayString().Replace('<', '_').Replace('>', '_').Replace(',', '_').Replace(' ', '_');
        return new Contract($"{hint}.TrainingContract.g.cs", sb.ToString());
    }

    private static bool IsPartial(INamedTypeSymbol type, System.Threading.CancellationToken ct)
    {
        for (var current = type; current is not null; current = current.ContainingType)
        {
            bool partial = current.DeclaringSyntaxReferences
                .Select(r => r.GetSyntax(ct))
                .OfType<TypeDeclarationSyntax>()
                .Any(d => d.Modifiers.Any(SyntaxKind.PartialKeyword));
            if (!partial) return false;
        }

        return true;
    }

    private static ITypeSymbol? NeuralNetworkNumericType(INamedTypeSymbol type)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (current.OriginalDefinition.ToDisplayString().StartsWith(NeuralNetworkBasePrefix, System.StringComparison.Ordinal)
                && current.TypeArguments.Length == 1)
            {
                return current.TypeArguments[0];
            }
        }

        return null;
    }

    private static bool Declares(INamedTypeSymbol type, string name) => type.GetMembers(name).Length > 0;

    private static bool DeclaresGradientOptimizerField(INamedTypeSymbol type, ITypeSymbol numeric, Compilation compilation)
    {
        var field = type.GetMembers("_optimizer").OfType<IFieldSymbol>().FirstOrDefault(f => !f.IsStatic);
        if (field is null) return false;
        var gradientOptimizer = compilation.GetTypeByMetadataName("AiDotNet.Interfaces.IGradientBasedOptimizer`3");
        var tensorType = compilation.GetTypeByMetadataName("AiDotNet.Tensors.LinearAlgebra.Tensor`1");
        if (gradientOptimizer is null || tensorType is null) return false;
        var tensor = tensorType.Construct(numeric);
        var expected = gradientOptimizer.Construct(numeric, tensor, tensor);
        return SymbolEqualityComparer.Default.Equals(field.Type, expected)
            || field.Type.AllInterfaces.Any(i => SymbolEqualityComparer.Default.Equals(i, expected));
    }

    /// <summary>
    /// Whether the type's own PredictCore calls <paramref name="name"/> on its native way to the layers. A call only the
    /// ONNX branch makes does not count: training runs the native forward (DeepFilterNet preprocesses for its ONNX graph
    /// alone, and its native forward starts from the waveform).
    /// </summary>
    private static bool PredictCoreCalls(INamedTypeSymbol type, string name, System.Threading.CancellationToken ct)
    {
        if (type.GetMembers(name).Length == 0 && !InheritsMember(type, name)) return false;
        foreach (var predict in type.GetMembers("PredictCore").OfType<IMethodSymbol>())
        {
            foreach (var reference in predict.DeclaringSyntaxReferences)
            {
                if (reference.GetSyntax(ct) is not MethodDeclarationSyntax method) continue;
                SyntaxNode? body = (SyntaxNode?)method.Body ?? method.ExpressionBody;
                if (body is null) continue;
                if (body.DescendantNodes().OfType<InvocationExpressionSyntax>().Any(call => call.Expression switch
                    {
                        IdentifierNameSyntax id => id.Identifier.ValueText == name,
                        MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText == name,
                        _ => false,
                    } && !InOnnxOnlyBranch(call, body)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Whether <paramref name="node"/> runs only in ONNX mode: under <c>if (IsOnnxMode ...)</c> or
    /// <c>if (!_useNativeMode)</c>, in the else of <c>if (_useNativeMode)</c>, or in the matching arm of a conditional
    /// expression.
    /// </summary>
    private static bool InOnnxOnlyBranch(SyntaxNode node, SyntaxNode body)
    {
        for (SyntaxNode? child = node, parent = node.Parent; parent is not null && child != body; child = parent, parent = parent.Parent)
        {
            ExpressionSyntax? condition;
            bool inTrueArm;
            switch (parent)
            {
                case IfStatementSyntax ifStatement when child == ifStatement.Statement || child == ifStatement.Else:
                    condition = ifStatement.Condition;
                    inTrueArm = child == ifStatement.Statement;
                    break;
                case ConditionalExpressionSyntax conditional when child == conditional.WhenTrue || child == conditional.WhenFalse:
                    condition = conditional.Condition;
                    inTrueArm = child == conditional.WhenTrue;
                    break;
                default:
                    continue;
            }

            string text = condition.ToString();
            bool onnxWhenTrue = text.Contains("Onnx") || text.Contains("!_useNativeMode");
            bool nativeWhenTrue = !onnxWhenTrue && text.Contains("_useNativeMode");
            if (onnxWhenTrue && inTrueArm || nativeWhenTrue && !inTrueArm) return true;
        }

        return false;
    }

    private static bool InheritsMember(INamedTypeSymbol type, string name)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
            if (current.GetMembers(name).Length > 0) return true;
        return false;
    }

    /// <summary>Whether this type, and not a base class, is the first in its chain to implement IDetectionBackbone.</summary>
    private static bool FirstImplementsDetectionBackbone(INamedTypeSymbol type)
    {
        static bool Implements(INamedTypeSymbol? t) => t is not null && t.AllInterfaces.Any(i =>
            i.OriginalDefinition.ToDisplayString() == "AiDotNet.Interfaces.IDetectionBackbone<T>");
        return Implements(type) && !Implements(type.BaseType);
    }

    private static bool DeclaresThrowIfDisposed(INamedTypeSymbol type) => DeclaresParameterlessVoid(type, "ThrowIfDisposed");

    private static bool DeclaresParameterlessVoid(INamedTypeSymbol type, string name)
        => type.GetMembers(name).OfType<IMethodSymbol>()
            .Any(m => !m.IsStatic && m.Parameters.Length == 0 && m.ReturnsVoid);

    private static bool DeclaresNativeModeField(INamedTypeSymbol type)
        => type.GetMembers("_useNativeMode").OfType<IFieldSymbol>()
            .Any(f => !f.IsStatic && f.Type.SpecialType == SpecialType.System_Boolean);

    private static string TypeParametersOf(INamedTypeSymbol type)
        => type.TypeParameters.Length > 0 ? "<" + string.Join(", ", type.TypeParameters.Select(p => p.Name)) + ">" : string.Empty;
}