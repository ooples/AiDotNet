using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Source generator that discovers [TrainableParameter] fields on LayerBase subclasses
/// and emits GetTrainableParameters, SetTrainableParameters, and ZeroGrad overrides.
/// Also discovers ILayer fields and emits InitializeSubLayers() for recursive parameter collection.
/// </summary>
/// <remarks>
/// <para>This is the production equivalent of PyTorch's nn.Parameter auto-registration.
/// Developers mark fields with [TrainableParameter] and the generator handles all training
/// infrastructure automatically — zero manual boilerplate, zero runtime overhead.</para>
///
/// <para><b>Convention-based gradient discovery:</b> For a parameter field named _foo,
/// the generator looks for _fooGradient (Tensor&lt;T&gt;?). If found, ZeroGrad will
/// null it. For non-nullable gradient fields, it calls Fill(NumOps.Zero).</para>
///
/// <para><b>Sub-layer discovery:</b> Fields typed as ILayer&lt;T&gt; or LayerBase&lt;T&gt;
/// subclasses are emitted as RegisterSubLayer calls in a generated InitializeSubLayers method.</para>
///
/// <para><b>Parameter roles:</b> [TrainableParameter(Role = "weight")] attributes generate
/// GetParameterRoles() for per-role optimizer configuration (e.g., weight decay exemption for biases).</para>
/// </remarks>
[Generator]
public class TrainableParameterGenerator : IIncrementalGenerator
{
    private const string TrainableParameterAttributeName = "AiDotNet.Attributes.TrainableParameterAttribute";
    private const string LayerBaseTypeName = "AiDotNet.NeuralNetworks.Layers.LayerBase";
    private const string TensorTypeName = "AiDotNet.Tensors.LinearAlgebra.Tensor";
    private const string ILayerTypeName = "AiDotNet.Interfaces.ILayer";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Find all class declarations that might have [TrainableParameter] fields
        var classDeclarations = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds &&
                    cds.Modifiers.Any(m => m.Text == "partial"),
                transform: static (ctx, _) => (ClassDeclarationSyntax)ctx.Node)
            .Where(static c => c is not null);

        var compilationAndClasses = context.CompilationProvider.Combine(classDeclarations.Collect());

        context.RegisterSourceOutput(compilationAndClasses, static (spc, source) => Execute(source.Left, source.Right, spc));
    }

    private static void Execute(Compilation compilation, ImmutableArray<ClassDeclarationSyntax> classes, SourceProductionContext context)
    {
        if (classes.IsDefaultOrEmpty) return;

        var attributeSymbol = compilation.GetTypeByMetadataName(TrainableParameterAttributeName);
        if (attributeSymbol is null) return;

        // Group by containing class (multiple partial declarations possible)
        var processedClasses = new HashSet<string>();

        foreach (var classDecl in classes)
        {
            var model = compilation.GetSemanticModel(classDecl.SyntaxTree);
            var classSymbol = model.GetDeclaredSymbol(classDecl) as INamedTypeSymbol;
            if (classSymbol is null) continue;

            // Check if class extends LayerBase<T>
            if (!ExtendsLayerBase(classSymbol)) continue;

            // Skip if already processed (multiple partial files)
            var fullName = classSymbol.ToDisplayString();
            if (!processedClasses.Add(fullName)) continue;

            // Collect [TrainableParameter] fields
            var paramFields = new List<ParameterFieldInfo>();
            var gradientFields = new Dictionary<string, GradientFieldInfo>();
            var subLayerFields = new List<SubLayerFieldInfo>();

            foreach (var member in classSymbol.GetMembers())
            {
                if (member is not IFieldSymbol field) continue;

                // Check for [TrainableParameter]
                var attr = field.GetAttributes()
                    .FirstOrDefault(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, attributeSymbol));

                if (attr is not null)
                {
                    var role = "PersistentTensorRole.Weights";
                    var order = 0;

                    foreach (var namedArg in attr.NamedArguments)
                    {
                        if (namedArg.Key == "Role" && namedArg.Value.Value is int roleVal)
                            role = $"PersistentTensorRole.{(PersistentTensorRoleEnum)roleVal}";
                        else if (namedArg.Key == "Order" && namedArg.Value.Value is int orderVal)
                            order = orderVal;
                    }

                    paramFields.Add(new ParameterFieldInfo(field.Name, role, order));
                }

                // Check for gradient fields (convention: {name}Gradient)
                if (field.Name.EndsWith("Gradient") && IsTensorType(field.Type))
                {
                    var isNullable = field.NullableAnnotation == NullableAnnotation.Annotated ||
                                     field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    gradientFields[field.Name] = new GradientFieldInfo(field.Name, isNullable);
                }

                // Check for sub-layer fields
                if (IsLayerType(field.Type) && !field.IsStatic)
                {
                    var isNullable = field.NullableAnnotation == NullableAnnotation.Annotated ||
                                     field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    subLayerFields.Add(new SubLayerFieldInfo(field.Name, isNullable));
                }
            }

            if (paramFields.Count == 0 && subLayerFields.Count == 0) continue;

            // Stable sort by Order, preserving declaration order for equal Order values.
            // List.Sort is not stable, so we use a secondary key (original index).
            for (int idx = 0; idx < paramFields.Count; idx++)
                paramFields[idx] = paramFields[idx] with { DeclIndex = idx };
            paramFields.Sort((a, b) =>
            {
                int cmp = a.Order.CompareTo(b.Order);
                return cmp != 0 ? cmp : a.DeclIndex.CompareTo(b.DeclIndex);
            });

            // Generate the partial class source
            var source = GenerateSource(classSymbol, paramFields, gradientFields, subLayerFields);
            // Use fully qualified name to avoid collisions across namespaces
            var qualifiedName = classSymbol.ToDisplayString().Replace('.', '_').Replace('<', '_').Replace('>', '_');
            var hintName = $"{qualifiedName}.TrainableParameters.g.cs";
            context.AddSource(hintName, source);
        }
    }

    private static string GenerateSource(
        INamedTypeSymbol classSymbol,
        List<ParameterFieldInfo> paramFields,
        Dictionary<string, GradientFieldInfo> gradientFields,
        List<SubLayerFieldInfo> subLayerFields)
    {
        var ns = classSymbol.ContainingNamespace.ToDisplayString();
        var className = classSymbol.Name;
        var typeParams = classSymbol.TypeParameters.Length > 0
            ? "<" + string.Join(", ", classSymbol.TypeParameters.Select(tp => tp.Name)) + ">"
            : "";

        // Collect containing type chain for nested classes
        var containingTypes = new List<INamedTypeSymbol>();
        var outer = classSymbol.ContainingType;
        while (outer is not null)
        {
            containingTypes.Insert(0, outer);
            outer = outer.ContainingType;
        }

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated />");
        sb.AppendLine("// Generated by TrainableParameterGenerator");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using AiDotNet.Tensors.Engines;");
        sb.AppendLine("using AiDotNet.Tensors.LinearAlgebra;");
        sb.AppendLine();
        sb.AppendLine($"namespace {ns};");
        sb.AppendLine();

        // Emit containing type wrappers for nested classes
        foreach (var ct in containingTypes)
        {
            var ctTypeParams = ct.TypeParameters.Length > 0
                ? "<" + string.Join(", ", ct.TypeParameters.Select(tp => tp.Name)) + ">"
                : "";
            sb.AppendLine($"partial class {ct.Name}{ctTypeParams}");
            sb.AppendLine("{");
        }

        sb.AppendLine($"partial class {className}{typeParams}");
        sb.AppendLine("{");

        // GetTrainableParameters
        if (paramFields.Count > 0)
        {
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Returns all trainable parameter tensors marked with [TrainableParameter].");
            sb.AppendLine("    /// Auto-generated — do not modify. Edit the [TrainableParameter] attributes instead.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> GetTrainableParameters()");
            sb.AppendLine($"        => new Tensor<{GetTypeParamName(classSymbol)}>[] {{ {string.Join(", ", paramFields.Select(f => f.Name))} }};");
            sb.AppendLine();

            // SetTrainableParameters
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Replaces trainable parameter tensors (e.g., with ParameterBuffer views).");
            sb.AppendLine("    /// Auto-generated — updates both the field and the registered tensor list.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine($"    public override void SetTrainableParameters(System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> parameters)");
            sb.AppendLine("    {");
            sb.AppendLine($"        if (parameters.Count != {paramFields.Count})");
            sb.AppendLine($"            throw new System.ArgumentException($\"Expected {paramFields.Count} parameters, got {{parameters.Count}}.\");");
            for (int i = 0; i < paramFields.Count; i++)
            {
                sb.AppendLine($"        {paramFields[i].Name} = parameters[{i}] ?? throw new System.ArgumentNullException(nameof(parameters), \"Parameter at index {i} is null.\");");
            }
            sb.AppendLine("        base.SetTrainableParameters(parameters);");
            sb.AppendLine("    }");
            sb.AppendLine();

            // ZeroGrad
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Clears all gradient fields discovered by convention ({paramName}Gradient).");
            sb.AppendLine("    /// Auto-generated from [TrainableParameter] field naming conventions.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    public override void ZeroGrad()");
            sb.AppendLine("    {");
            sb.AppendLine("        base.ZeroGrad();");
            foreach (var param in paramFields)
            {
                var gradName = param.Name + "Gradient";
                if (gradientFields.TryGetValue(gradName, out var grad))
                {
                    if (grad.IsNullable)
                        sb.AppendLine($"        {grad.Name} = null;");
                    else
                        sb.AppendLine($"        {grad.Name}.Fill(NumOps.Zero);");
                }
            }
            sb.AppendLine("    }");
        }

        // GetParameterRoles — maps parameter names to their roles for per-role learning rates / weight decay
        // Role always has a value (defaults to PersistentTensorRole.Weights), so emit for all param fields
        if (paramFields.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases).");
            sb.AppendLine("    /// Auto-generated from [TrainableParameter(Role = \"...\")] attributes.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine($"    public virtual System.Collections.Generic.Dictionary<string, string> GetParameterRoles()");
            sb.AppendLine("    {");
            sb.AppendLine($"        return new System.Collections.Generic.Dictionary<string, string>");
            sb.AppendLine("        {");
            foreach (var param in paramFields)
            {
                sb.AppendLine($"            {{ \"{param.Name}\", \"{param.Role}\" }},");
            }
            sb.AppendLine("        };");
            sb.AppendLine("    }");
        }

        // Sub-layer registration via EnsureInitialized override (called before every forward pass)
        if (subLayerFields.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("    private bool _subLayersRegistered;");
            sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Registers discovered sub-layer fields for recursive parameter collection.");
            sb.AppendLine("    /// Auto-generated from fields typed as ILayer or LayerBase subclasses.");
            sb.AppendLine("    /// Called automatically via EnsureInitialized before the first forward pass.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    protected override void EnsureInitialized()");
            sb.AppendLine("    {");
            sb.AppendLine("        base.EnsureInitialized();");
            sb.AppendLine("        if (_subLayersRegistered) return;");
            sb.AppendLine("        _subLayersRegistered = true;");
            foreach (var sl in subLayerFields)
            {
                if (sl.IsNullable)
                    sb.AppendLine($"        if ({sl.Name} is not null) RegisterSubLayer({sl.Name});");
                else
                    sb.AppendLine($"        RegisterSubLayer({sl.Name});");
            }
            sb.AppendLine("    }");
        }

        sb.AppendLine("}");

        // Close containing type wrappers for nested classes
        foreach (var _ in containingTypes)
        {
            sb.AppendLine("}");
        }

        return sb.ToString();
    }

    private static string GetTypeParamName(INamedTypeSymbol classSymbol)
    {
        return classSymbol.TypeParameters.Length > 0 ? classSymbol.TypeParameters[0].Name : "T";
    }

    private static bool ExtendsLayerBase(INamedTypeSymbol type)
    {
        var current = type.BaseType;
        while (current is not null)
        {
            var display = current.OriginalDefinition.ToDisplayString();
            if (display.StartsWith(LayerBaseTypeName + "<") || display == LayerBaseTypeName)
                return true;
            current = current.BaseType;
        }
        return false;
    }

    private static bool IsTensorType(ITypeSymbol type)
    {
        var original = type is INamedTypeSymbol named ? named.OriginalDefinition : type;
        var display = original.ToDisplayString();
        return display.StartsWith(TensorTypeName + "<") || display == TensorTypeName;
    }

    private static bool IsLayerType(ITypeSymbol type)
    {
        // Check if type implements ILayer<T>
        if (type is INamedTypeSymbol named)
        {
            foreach (var iface in named.AllInterfaces)
            {
                var display = iface.OriginalDefinition.ToDisplayString();
                if (display.StartsWith(ILayerTypeName + "<") || display == ILayerTypeName)
                    return true;
            }
            // Also check the type itself
            var typeDisplay = named.OriginalDefinition.ToDisplayString();
            if (typeDisplay.StartsWith(LayerBaseTypeName + "<") || typeDisplay == LayerBaseTypeName)
                return true;
        }
        return false;
    }

    // Simple enum matching for PersistentTensorRole values in attribute
    private enum PersistentTensorRoleEnum
    {
        Weights = 0,
        Biases = 1,
        NormalizationParams = 2,
        Embeddings = 3,
        AttentionCache = 4,
        OptimizerState = 5,
        Constant = 6,
        Other = 7
    }

    private record struct ParameterFieldInfo(string Name, string Role, int Order, int DeclIndex = 0);
    private record struct GradientFieldInfo(string Name, bool IsNullable);
    private record struct SubLayerFieldInfo(string Name, bool IsNullable);
}
