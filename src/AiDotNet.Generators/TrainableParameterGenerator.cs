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

        // Opt-in inverted discovery. A class marked [AutoParameters] treats every non-nullable
        // tensor field as a trainable parameter unless it is explicitly excluded, which is how
        // PyTorch behaves by construction (nn.Parameter is a distinct type, so a weight cannot be
        // stored without announcing itself). Per-class so the inversion can be verified one layer
        // at a time instead of flipped library-wide in one step.
        var autoParamsSymbol = compilation.GetTypeByMetadataName("AiDotNet.Attributes.AutoParametersAttribute");
        var scratchSymbol = compilation.GetTypeByMetadataName("AiDotNet.Attributes.ScratchAttribute");
        var bufferSymbol = compilation.GetTypeByMetadataName("AiDotNet.Attributes.BufferAttribute");
        // Bail only if NO discovery route exists. This used to return whenever
        // TrainableParameterAttribute was missing, which also disabled register-call discovery,
        // sub-layer registration, buffers and [AutoParameters] -- every mechanism, gated on one
        // attribute none of them need.
        if (attributeSymbol is null && autoParamsSymbol is null && bufferSymbol is null) return;

        // Group by containing class (multiple partial declarations possible)
        var processedClasses = new HashSet<string>();

        foreach (var classDecl in classes)
        {
            var model = compilation.GetSemanticModel(classDecl.SyntaxTree);
            var classSymbol = model.GetDeclaredSymbol(classDecl) as INamedTypeSymbol;
            if (classSymbol is null) continue;

            // Check if class extends LayerBase<T>
            if (!ExtendsLayerBase(classSymbol)) continue;

            // A layer that hand-writes its parameter accessors manages its own plumbing; generating
            // partial copies would be a duplicate-member error.
            //
            // Only those two members gate the whole class. Including EnsureInitialized here as well
            // was too coarse and had a serious consequence: DenseLayer hand-writes EnsureInitialized
            // but NOT the accessors, so it silently lost its generated SetTrainableParameters -- the
            // one that assigns _weights/_biases -- and fell back to LayerBase's, which rebinds only
            // the registered-tensor list. The layer's own fields kept their old tensors, so
            // GetTrainableParameters reported the new values while Forward still used the old ones.
            // Copy-on-write cloning relies on exactly this setter, so every COW clone of a model
            // containing a DenseLayer came back computing with stale weights.
            // Suppress only the members the author actually declared, not the whole class. The
            // all-or-nothing form meant 13 layers that hand-write ONE accessor silently lost every
            // other generated member -- sub-layer registration, ZeroGrad, buffer registration --
            // none of which can collide with what they wrote.
            bool declaresGetter = DeclaresAny(classSymbol, "GetTrainableParameters");
            bool declaresSetter = DeclaresAny(classSymbol, "SetTrainableParameters");
            if (declaresGetter && declaresSetter)
                continue;

            // Skip if already processed (multiple partial files)
            var fullName = classSymbol.ToDisplayString();
            if (!processedClasses.Add(fullName)) continue;

            // Collect [TrainableParameter] fields
            var paramFields = new List<ParameterFieldInfo>();
            var gradientFields = new Dictionary<string, GradientFieldInfo>();
            var subLayerFields = new List<SubLayerFieldInfo>();

            bool autoParameters = autoParamsSymbol is not null && classSymbol.GetAttributes()
                .Any(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, autoParamsSymbol));

            var bufferFields = new List<(string Field, string Name, string Role)>();

            // A field handed to RegisterBuffer IS a buffer, whether or not it also carries
            // [Buffer]. Without this, inverted discovery promoted BatchNormalization's _runningMean
            // and _runningVariance to TRAINABLE -- counted once as parameters and again through the
            // buffer registry (144 against a saved 96), and, far worse, handed to the optimizer.
            // Running statistics are estimates of the data, not weights; a gradient step on them is
            // silent corruption of every subsequent inference.
            var imperativeBuffers = new HashSet<string>();
            foreach (var (bufName, _) in DiscoverFromRegisterCalls(classDecl, model, "RegisterBuffer"))
                imperativeBuffers.Add(bufName);

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
                    var optional = false;
                    string? shape = null;

                    foreach (var namedArg in attr.NamedArguments)
                    {
                        if (namedArg.Key == "Role" && namedArg.Value.Value is int roleVal)
                            role = $"PersistentTensorRole.{(PersistentTensorRoleEnum)roleVal}";
                        else if (namedArg.Key == "Order" && namedArg.Value.Value is int orderVal)
                            order = orderVal;
                        else if (namedArg.Key == "Optional" && namedArg.Value.Value is bool optVal)
                            optional = optVal;
                        else if (namedArg.Key == "Shape" && namedArg.Value.Value is string shapeVal)
                            shape = shapeVal;
                    }

                    // A nullable field carrying an explicit [TrainableParameter] is Optional by
                    // construction: the author declared it a parameter AND declared it may be
                    // absent. BatchEnsembleLayer._bias is the shape -- allocated only when useBias
                    // is set, guarded by "if (_bias != null)" on every use. Requiring the author to
                    // also write Optional = true would make the nullable annotation a trap, and
                    // until now the omission was masked: the register-call replacement dropped
                    // every nullable field, so _bias was declared trainable and then silently left
                    // out of the surface the optimizer walks.
                    var explicitNullable = field.NullableAnnotation == NullableAnnotation.Annotated
                                           || field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    paramFields.Add(new ParameterFieldInfo(
                        field.Name, role, order, DeclIndex: 0,
                        TypeName: field.Type.ToDisplayString(),
                        Optional: optional || explicitNullable, Nullable: explicitNullable,
                        Shape: shape));
                }

                // Inverted default: an unmarked, non-nullable, non-readonly tensor field IS a
                // parameter. readonly is excluded because the generated SetTrainableParameters
                // REBINDS the field (tape-buffer views are swapped in wholesale), and assigning a
                // readonly field outside a constructor is CS0191. A readonly tensor whose CONTENTS
                // are trainable can still opt in explicitly with [TrainableParameter].
                // Order matters -- this runs only when [TrainableParameter] was absent, so an
                // explicit role, Order or Optional always wins over the inferred one.
                else if (autoParameters && !field.IsStatic && !field.IsReadOnly
                         // Auto-property backing fields (<Input>k__BackingField) are compiler-
                         // generated: their name is not valid C# to emit, and the properties they
                         // back are caches (FeedForwardLayer's Input/Output hold the last forward
                         // pass). An author cannot put [Scratch] on a field that does not exist in
                         // source, so a property-backed weight must use a real field to opt in.
                         && !field.IsImplicitlyDeclared && field.AssociatedSymbol is null
                         // Arrays of tensors are NOT a single parameter. IsTensorType is a prefix
                         // test on the display string, and "Tensor<T>[]" starts with "Tensor<", so
                         // an array slipped through and the generated code tried to assign the whole
                         // array into one Tensor<T> slot (ContinuumMemorySystemLayer._storedInputs).
                         && field.Type is not IArrayTypeSymbol
                         && IsTensorOfLayerElement(field.Type, classSymbol)
                         && !field.Name.EndsWith("Gradient", System.StringComparison.Ordinal)
                         && field.NullableAnnotation != NullableAnnotation.Annotated
                         && field.Type.NullableAnnotation != NullableAnnotation.Annotated
                         && !HasAttr(field, scratchSymbol)
                         && !HasAttr(field, bufferSymbol)
                         && !imperativeBuffers.Contains(field.Name))
                {
                    // Biases infer their role from the name so per-role optimizer configuration
                    // (weight-decay exemption) keeps working without an attribute.
                    var inferredRole = field.Name.IndexOf("bias", System.StringComparison.OrdinalIgnoreCase) >= 0
                        ? "PersistentTensorRole.Biases"
                        : "PersistentTensorRole.Weights";
                    paramFields.Add(new ParameterFieldInfo(
                        field.Name, inferredRole, 0, DeclIndex: 0,
                        TypeName: field.Type.ToDisplayString(), Optional: false));
                }

                // Collect [Buffer] fields: persistent state that is serialized but never trained.
                // Marking alone is not enough -- without emitting RegisterBuffer the tensors leave
                // the trainable set and join nothing, disappearing from ParameterCount and the flat
                // vector entirely. ReservoirLayer proved it: "Expected 320 parameters, got 0".
                if (!field.IsStatic && IsTensorType(field.Type) && HasAttr(field, bufferSymbol))
                {
                    var bufRole = "PersistentTensorRole.Constant";
                    var bufName = field.Name.TrimStart('_');
                    var bAttr = field.GetAttributes().FirstOrDefault(a =>
                        SymbolEqualityComparer.Default.Equals(a.AttributeClass, bufferSymbol));
                    if (bAttr is not null)
                    {
                        foreach (var na in bAttr.NamedArguments)
                        {
                            if (na.Key == "Role" && na.Value.Value is int br)
                                bufRole = $"PersistentTensorRole.{(PersistentTensorRoleEnum)br}";
                            else if (na.Key == "Name" && na.Value.Value is string bn && bn.Length > 0)
                                bufName = bn;
                        }
                    }
                    bufferFields.Add((field.Name, bufName, bufRole));
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
                    subLayerFields.Add(new SubLayerFieldInfo(field.Name, isNullable, IsCollection: false));
                }
                // ...and sub-layers held in a COLLECTION. A composite that keeps its children in a
                // List<> got no registration at all, so GetSubLayers() returned nothing for them and
                // the recursive parameter walk never reached their weights: they were built, they ran
                // in Forward, and they silently never trained. CitrinetBlockLayer reported 0 children
                // while holding 9. This is what PyTorch's nn.ModuleList exists to prevent -- a plain
                // Python list of modules is likewise invisible to .parameters().
                else if (!field.IsStatic && IsLayerCollectionType(field.Type))
                {
                    var isNullable = field.NullableAnnotation == NullableAnnotation.Annotated ||
                                     field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    subLayerFields.Add(new SubLayerFieldInfo(field.Name, isNullable, IsCollection: true));
                }
            }

            // Discover trainable parameters from RegisterTrainableParameter() calls.
            // Registration order is the canonical order — it matches _registeredTensors
            // in LayerBase, so base.SetTrainableParameters positional assignment works.
            // If RegisterTrainableParameter calls exist, they REPLACE attribute-discovered
            // params to ensure correct ordering (attributes may be in declaration order
            // which differs from registration order).
            {
                var registeredFields = DiscoverFromRegisterCalls(classDecl, model, "RegisterTrainableParameter");


                // Under [AutoParameters] the discovered set is AUTHORITATIVE and must not be
                // replaced by the register-call list. Replacing would drop every field the
                // inversion found: RWKVLayer registers 8 weight matrices imperatively while
                // holding 10 more learned tensors (both LayerNorm affine pairs, the time- and
                // channel-mixing coefficients, the first-token bonus), so the replace path would
                // silently restore exactly the bug the inversion exists to fix. Registration order
                // still governs the layers that have not opted in.
                // ...unless the layer registers tensors the generator cannot SEE. Weights held
                // in a Dictionary<string, Tensor<T>> or a List<Tensor<T>> and registered in a loop
                // are not fields, so field discovery finds none of them; emitting a surface from
                // the fields alone would OVERRIDE the runtime registry and drop every one
                // (HeterogeneousGraphLayer's per-edge-type weights, biases and basis coefficients
                // all vanished, and its Parameters_CountShouldMatchVector went to zero). Such a
                // collection cannot be promoted by default either -- the same shape is far more
                // often a cache (_lastInputs, _gpuCachedHiddenStates), and silently training a
                // cache is worse than the bug. So imperative registration stays authoritative for
                // exactly these layers, which is how they already worked.
                if (autoParameters && !HasUnmappableRegistration(classDecl, model, classSymbol))
                    registeredFields = new List<(string, string)>();

                if (registeredFields.Count > 0)
                {
                    // Build attribute-discovered roles map for enrichment
                    var attrRoles = new Dictionary<string, string>();
                    var attrOptional = new Dictionary<string, bool>();
                    foreach (var pf in paramFields)
                    {
                        attrRoles[pf.Name] = pf.Role;
                        attrOptional[pf.Name] = pf.Optional;
                    }

                    // Replace paramFields with registration-ordered list
                    paramFields.Clear();
                    var seen = new HashSet<string>();
                    foreach (var (fieldName, role) in registeredFields)
                    {
                        if (!seen.Add(fieldName)) continue;

                        // Verify the field exists, is a Tensor<T>, and is non-nullable
                        var matchingField = classSymbol.GetMembers()
                            .OfType<IFieldSymbol>()
                            .FirstOrDefault(f => f.Name == fieldName && IsTensorType(f.Type)
                                && f.NullableAnnotation != NullableAnnotation.Annotated
                                && f.Type.NullableAnnotation != NullableAnnotation.Annotated);
                        if (matchingField is not null)
                        {
                            // Prefer attribute role if available (more specific)
                            var finalRole = attrRoles.TryGetValue(fieldName, out var attrRole)
                                ? attrRole : role;
                            var finalOptional = attrOptional.TryGetValue(fieldName, out var optFlag)
                                && optFlag;
                            paramFields.Add(new ParameterFieldInfo(
                                matchingField.Name, finalRole, paramFields.Count, DeclIndex: 0,
                                TypeName: matchingField.Type.ToDisplayString(), Optional: finalOptional));
                        }
                    }
                }
            }

            if (paramFields.Count == 0 && subLayerFields.Count == 0 && bufferFields.Count == 0) continue;

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
            var source = GenerateSource(classSymbol, paramFields, gradientFields, subLayerFields, bufferFields);
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
        List<SubLayerFieldInfo> subLayerFields,
        List<(string Field, string Name, string Role)> bufferFields)
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

        // Buffer registration. Persistent, never trained: LayerBase folds these into
        // ParameterCount / GetParameters / SetParameters but deliberately keeps them out of
        // GetTrainableParameters, so the optimizer and the tape cannot touch them. This mirrors the
        // PyTorch parameters()-versus-state_dict() split, with the difference that both surfaces
        // here are covered by one flat vector and one checked count.
        if (bufferFields.Count > 0)
        {
            sb.AppendLine("    private bool _buffersRegistered;");
            sb.AppendLine();
            sb.AppendLine("    /// <summary>Auto-generated: registers [Buffer] fields as persistent non-trainable state.</summary>");
            sb.AppendLine("    private void EnsureBuffersRegistered()");
            sb.AppendLine("    {");
            sb.AppendLine("        if (_buffersRegistered) return;");
            sb.AppendLine("        _buffersRegistered = true;");
            foreach (var bf in bufferFields)
            {
                sb.AppendLine($"        if ({bf.Field} is not null) RegisterBuffer({bf.Field}, \"{bf.Name}\", {bf.Role});");
            }
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine("    /// <inheritdoc />");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<(string Name, Tensor<{GetTypeParamName(classSymbol)}> Tensor)> GetRegisteredBuffers()");
            sb.AppendLine("    {");
            sb.AppendLine("        EnsureBuffersRegistered();");
            sb.AppendLine("        return base.GetRegisteredBuffers();");
            sb.AppendLine("    }");
            sb.AppendLine();
        }


        // DeclaredParameterShapes — emitted from [TrainableParameter(Shape = "...")].
        //
        // This is the whole point of the Shape argument: LayerBase.TryAdoptRestoredParameters can see
        // THAT a tensor was supplied before the first forward but not whether its shape is right, and
        // only the layer knows that its weights are [inputSize, outputSize]. Declaring it on the field
        // lets the generator supply that fact, so no layer hand-writes the override.
        var shapedFields = paramFields.Where(p => !string.IsNullOrWhiteSpace(p.Shape)).ToList();
        if (shapedFields.Count > 0)
        {
            string tp = GetTypeParamName(classSymbol);
            string tupleType = $"(Tensor<{tp}>? Tensor, AiDotNet.Tensors.LinearAlgebra.TensorShape Expected, PersistentTensorRole Role)";
            string arrayType = $"(Tensor<{tp}>?, AiDotNet.Tensors.LinearAlgebra.TensorShape, PersistentTensorRole)";

            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// The shape each [TrainableParameter] must have once this layer's shapes are resolved.");
            sb.AppendLine("    /// Auto-generated — do not modify. Edit the [TrainableParameter(Shape = \"...\")] arguments instead.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    /// <remarks>");
            sb.AppendLine("    /// Returns empty while any declared axis is still the -1 lazy sentinel, which is the base's");
            sb.AppendLine("    /// signal that this layer cannot answer yet. An axis written as * becomes -2, meaning the layer");
            sb.AppendLine("    /// adapts that axis and a mismatch there is normal rather than a broken restore.");
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{tupleType}> DeclaredParameterShapes()");
            sb.AppendLine("    {");
            sb.AppendLine($"        var __declared = new {arrayType}[]");
            sb.AppendLine("        {");
            foreach (var pf in shapedFields)
            {
                var axes = pf.Shape!.Split(',')
                    .Select(a => a.Trim())
                    .Where(a => a.Length > 0)
                    .Select(a => a == "*" ? "-2" : a);
                sb.AppendLine($"            ({pf.Name}, ShapeOf({string.Join(", ", axes)}), {pf.Role}),");
            }
            sb.AppendLine("        };");
            sb.AppendLine();
            sb.AppendLine("        for (int __i = 0; __i < __declared.Length; __i++)");
            sb.AppendLine("        {");
            sb.AppendLine("            var __s = __declared[__i].Item2;");
            sb.AppendLine("            for (int __d = 0; __d < __s.Length; __d++)");
            sb.AppendLine("            {");
            sb.AppendLine("                if (__s[__d] < 0 && __s[__d] != -2)");
            sb.AppendLine($"                    return System.Array.Empty<{arrayType}>();");
            sb.AppendLine("            }");
            sb.AppendLine("        }");
            sb.AppendLine();
            sb.AppendLine("        return __declared;");
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        // GetTrainableParameters
        bool hasOptional = paramFields.Any(p => p.Optional);
        if (paramFields.Count > 0)
        {
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Returns all trainable parameter tensors marked with [TrainableParameter].");
            sb.AppendLine("    /// Auto-generated — do not modify. Edit the [TrainableParameter] attributes instead.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    /// <remarks>");
            sb.AppendLine("    /// Always discovers nested sub-layers via EnsureSubLayersRegistered()");
            sb.AppendLine("    /// so optimizers and exporters see the full parameter graph even for");
            sb.AppendLine("    /// lazy parents that have not received a Forward() call yet. Only");
            sb.AppendLine("    /// the *weight materialization* step is gated on IsShapeResolved:");
            sb.AppendLine("    /// for lazy layers that haven't yet received a Forward() call,");
            sb.AppendLine("    /// InputShape/OutputShape still hold the -1 sentinel and");
            sb.AppendLine("    /// EnsureInitialized would overflow on TensorAllocator.Rent. In that");
            sb.AppendLine("    /// case we return the (still-empty) placeholder tensors — those");
            sb.AppendLine("    /// layers will materialize their real weights on their first Forward()");
            sb.AppendLine("    /// and a subsequent CollectTrainableParameters pass will pick them up.");
            if (hasOptional)
            {
                sb.AppendLine("    /// Optional parameters (lazily-materialized, conditionally-used fields)");
                sb.AppendLine("    /// are omitted while they remain empty [0,0] placeholders so they are");
                sb.AppendLine("    /// not exposed as trainable params that can never receive a gradient");
                sb.AppendLine("    /// update; they re-appear once materialized. SetTrainableParameters is");
                sb.AppendLine("    /// emitted symmetrically (consumes a slot only for currently-present fields).");
            }
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> GetTrainableParameters()");
            sb.AppendLine("    {");
            if (subLayerFields.Count > 0)
            {
                sb.AppendLine("        EnsureSubLayersRegistered();");
            }
            sb.AppendLine("        if (IsShapeResolved) EnsureInitialized();");
            if (hasOptional)
            {
                sb.AppendLine($"        var __params = new System.Collections.Generic.List<Tensor<{GetTypeParamName(classSymbol)}>>({paramFields.Count});");
                foreach (var f in paramFields)
                {
                    if (f.Optional)
                        sb.AppendLine($"        if ({PresenceExpr(f)}) __params.Add({f.Name});");
                    else
                        sb.AppendLine($"        __params.Add({f.Name});");
                }
                sb.AppendLine("        return __params;");
            }
            else
            {
                sb.AppendLine($"        return new Tensor<{GetTypeParamName(classSymbol)}>[] {{ {string.Join(", ", paramFields.Select(f => f.Name))} }};");
            }
            sb.AppendLine("    }");
            sb.AppendLine();

            // Counting view: the same fields in the same order, minus the EnsureInitialized
            // trampoline. ParameterCount is read by ComputeTopologyFingerprint and by Dispose, and
            // materializing weights just to size them threw OutOfMemoryException tearing down a
            // 774M-parameter model. Sub-layer registration is still performed -- it allocates
            // nothing and the count would otherwise miss children.
            sb.AppendLine("    /// <summary>Field list for ParameterCount: no lazy materialization.</summary>");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> GetTrainableParametersUnmaterialized()");
            sb.AppendLine("    {");
            if (subLayerFields.Count > 0)
            {
                sb.AppendLine("        EnsureSubLayersRegistered();");
            }
            if (hasOptional)
            {
                sb.AppendLine($"        var __counting = new System.Collections.Generic.List<Tensor<{GetTypeParamName(classSymbol)}>>({paramFields.Count});");
                foreach (var f in paramFields)
                {
                    if (f.Optional)
                        sb.AppendLine($"        if ({PresenceExpr(f)}) __counting.Add({f.Name});");
                    else
                        sb.AppendLine($"        __counting.Add({f.Name});");
                }
                sb.AppendLine("        return __counting;");
            }
            else
            {
                sb.AppendLine($"        return new Tensor<{GetTypeParamName(classSymbol)}>[] {{ {string.Join(", ", paramFields.Select(f => f.Name))} }};");
            }
            sb.AppendLine("    }");
            sb.AppendLine();

            // SetTrainableParameters
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Replaces trainable parameter tensors (e.g., with ParameterBuffer views).");
            sb.AppendLine("    /// Auto-generated — updates both the field and the registered tensor list.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine($"    public override void SetTrainableParameters(System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> parameters)");
            sb.AppendLine("    {");
            // Local helper: emit the assignment of a field from a parameters[idx]
            // slot, with the sparse-leaf downcast when the field's concrete type
            // is a Tensor<T> subclass. indexExpr may be a literal index (fixed path)
            // or a post-increment cursor (optional path).
            void EmitFieldAssign(ParameterFieldInfo pf, string indexExpr, string idxLabel)
            {
                bool needsCast = pf.TypeName is not null
                    && !(pf.TypeName.StartsWith(TensorTypeName + "<") || pf.TypeName == TensorTypeName);
                if (needsCast)
                {
                    sb.AppendLine($"        {pf.Name} = (parameters[{indexExpr}] ?? throw new System.ArgumentNullException(nameof(parameters), \"Parameter at index {idxLabel} is null.\")) as global::{pf.TypeName}");
                    sb.AppendLine($"            ?? throw new System.ArgumentException(\"Parameter at index {idxLabel} is not a {pf.TypeName}. Tape-buffer must preserve sparse leaf types.\", nameof(parameters));");
                }
                else
                {
                    sb.AppendLine($"        {pf.Name} = parameters[{indexExpr}] ?? throw new System.ArgumentNullException(nameof(parameters), \"Parameter at index {idxLabel} is null.\");");
                }
            }

            // Re-sync _registeredTensors with the newly assigned field values.
            // We cannot call base.SetTrainableParameters because _registeredTensors
            // may have a different count than paramFields when multiple parameters
            // share the same PersistentTensorRole (the replace-by-role logic in
            // RegisterTrainableParameter collapses them). Instead, clear and
            // re-register each field so the runtime list matches the generator's
            // field count exactly. The role is read from the [TrainableParameter]
            // attribute at compile time — no hardcoded mapping needed.
            if (hasOptional)
            {
                // Count-aware: an optional field consumes a parameter slot (and is
                // re-registered) only while it is currently a materialized tensor
                // (Length > 0) — mirroring GetTrainableParameters exactly, so the
                // get/set round-trip stays consistent. The predicate is evaluated on
                // the *current* field state, which matches the state
                // GetTrainableParameters saw when the caller built the list.
                //
                // Validate the count up front — BEFORE any state mutation — so a
                // rejected call leaves the layer untouched rather than partially
                // updated (a short list would otherwise throw mid-assignment from
                // parameters[__i], a long list only after every field + registration
                // had already been mutated). The expected count is the number of
                // currently-present trainable tensors, computed with the same
                // predicates the assignment loop uses (field tensors are not mutated
                // between here and the loop, so the values are stable).
                sb.AppendLine("        if (parameters is null)");
                sb.AppendLine("            throw new System.ArgumentNullException(nameof(parameters));");
                sb.AppendLine("        int __expected = 0;");
                foreach (var pf in paramFields)
                {
                    if (pf.Optional)
                        sb.AppendLine($"        if ({PresenceExpr(pf)}) __expected++;");
                    else
                        sb.AppendLine("        __expected++;");
                }
                // A RESTORE may legitimately supply MORE than are currently present: an optional
                // field is absent only because nothing has materialized it yet, and a checkpoint
                // that carries a value for it is precisely what should bring it into being.
                // EmbeddingLayer's input projection is the case -- it exists only for continuous
                // input, so a fresh clone has one tensor and the saved model has two, and refusing
                // the longer list left 576 values with nowhere to go. Accept the all-optionals-
                // present count as well; anything between the two remains ambiguous and is
                // rejected as before.
                sb.AppendLine($"        const int __withAllOptional = {paramFields.Count};");
                sb.AppendLine("        bool __materializeOptional = parameters.Count == __withAllOptional && __expected != __withAllOptional;");
                sb.AppendLine("        if (parameters.Count != __expected && !__materializeOptional)");
                sb.AppendLine("            throw new System.ArgumentException($\"Expected {__expected} parameters (currently-present trainable tensors) or {__withAllOptional} (all optional present), got {parameters.Count}.\", nameof(parameters));");
                sb.AppendLine("        int __i = 0;");
                sb.AppendLine("        ClearRegisteredParameters();");
                foreach (var pf in paramFields)
                {
                    if (pf.Optional)
                    {
                        sb.AppendLine($"        if (__materializeOptional || ({PresenceExpr(pf)}))");
                        sb.AppendLine("        {");
                        EmitFieldAssign(pf, "__i", "__i");
                        sb.AppendLine($"            AppendTrainableParameter({pf.Name}, {pf.Role});");
                        sb.AppendLine("            __i++;");
                        sb.AppendLine("        }");
                    }
                    else
                    {
                        EmitFieldAssign(pf, "__i", "__i");
                        sb.AppendLine($"        AppendTrainableParameter({pf.Name}, {pf.Role});");
                        sb.AppendLine("        __i++;");
                    }
                }
            }
            else
            {
                sb.AppendLine($"        if (parameters.Count != {paramFields.Count})");
                sb.AppendLine($"            throw new System.ArgumentException($\"Expected {paramFields.Count} parameters, got {{parameters.Count}}.\");");
                for (int i = 0; i < paramFields.Count; i++)
                {
                    EmitFieldAssign(paramFields[i], i.ToString(), i.ToString());
                }
                // Rebind in place when the registration already has the right shape. The
                // clear-and-re-append path below unregisters every tensor from the engine and
                // registers it again, and the engine's persistent pool is not order-stable across
                // that cycle -- which changes gradient reduction order and makes training
                // run-to-run nondeterministic. Training calls this setter (ParameterBuffer views),
                // so the churn happened on every step: two identical runs of BiaffineNER's
                // LossStrictlyDecreases / OptimizerStep probes gave different results.
                //
                // When the base registration count is unchanged there is nothing to re-register:
                // the fields are already assigned above, and the base setter swaps the registry
                // entries positionally without touching the engine. Do not use the generated
                // GetTrainableParameters count here: lazy fields are visible through that override
                // before the base registry is populated. Only a changed count needs the full
                // rebuild, and AppendTrainableParameter is used there (not
                // RegisterTrainableParameter) to avoid role-based dedup -- layers like
                // MultiHeadAttentionLayer carry several parameters with the same role
                // (e.g. 4 x Weights) that replace-by-role logic would collapse to one.
                sb.AppendLine($"        if (RegisteredTrainableParameterCount == {paramFields.Count})");
                sb.AppendLine("        {");
                sb.AppendLine("            base.SetTrainableParameters(parameters);");
                sb.AppendLine("            return;");
                sb.AppendLine("        }");
                sb.AppendLine();
                sb.AppendLine("        ClearRegisteredParameters();");
                for (int i = 0; i < paramFields.Count; i++)
                {
                    sb.AppendLine($"        AppendTrainableParameter({paramFields[i].Name}, {paramFields[i].Role});");
                }
            }
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

            // ReturnPooledParameters — issue #1136 plan part 3 hook.
            // Returns rented parameter tensors back to the TensorAllocator
            // pool so sequential Diffusion / NN tests on 16 GB CI runners
            // don't accumulate pool-orphaned buffers in the gen-2 LOH and
            // OOM after a few hundred tests. Emitted as a separate hook
            // (instead of a full Dispose(bool) override) so layers with
            // their own Dispose(bool) override (DenseLayer, ConvolutionalLayer,
            // SpiralConvLayer, SynapticPlasticityLayer) don't get a
            // duplicate-member error — LayerBase.Dispose(bool) calls this
            // hook on every Dispose path, and hand-written overrides
            // continue to work via the base.Dispose(disposing) call they
            // already make.
            sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Returns rented parameter tensors to the TensorAllocator pool.");
            sb.AppendLine("    /// Auto-generated from [TrainableParameter] fields per issue #1136 plan part 3.");
            sb.AppendLine("    /// Called from <see cref=\"LayerBase{T}.Dispose(bool)\"/>; do not call directly.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    protected override void ReturnPooledParameters()");
            sb.AppendLine("    {");
            sb.AppendLine("        // Lazy-init layers that never received a Forward have zero-length");
            sb.AppendLine("        // placeholder tensors that were never Rented — skip them.");
            sb.AppendLine("        if (!IsShapeResolved) return;");
            foreach (var param in paramFields)
            {
                sb.AppendLine($"        if ({param.Name} != null && {param.Name}.Length > 0)");
                sb.AppendLine("        {");
                sb.AppendLine($"            AiDotNet.Tensors.Helpers.TensorAllocator.Return({param.Name});");
                sb.AppendLine("        }");
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
            // `virtual` is illegal on a member of a sealed type (CS0549), and three sealed
            // layers -- ColumnParallelLinear, RowParallelLinear, Stage3ShardedLinear -- hit
            // exactly that once they became partial. A sealed class cannot be derived from, so
            // the modifier carries no meaning there anyway.
            sb.AppendLine($"    public {(classSymbol.IsSealed ? "" : "virtual ")}System.Collections.Generic.Dictionary<string, string> GetParameterRoles()");
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

        // Sub-layer registration. Hoisted into EnsureSubLayersRegistered() so it runs
        // independently of weight materialization — GetTrainableParameters() must see the
        // sub-layer graph even on a pre-Forward() collection pass, which would otherwise
        // skip EnsureInitialized() (and its TensorAllocator.Rent on -1 placeholder shapes).
        if (subLayerFields.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("    private bool _subLayersRegistered;");
            sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Registers discovered sub-layer fields. Cheap (no weight allocation), so");
            sb.AppendLine("    /// safe to call before the first Forward() — keeps optimizer/export discovery");
            sb.AppendLine("    /// working for lazy parents that haven't yet resolved their own input shape.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    /// <remarks>");
            sb.AppendLine("    /// A non-nullable child field is still null until the layer initializes: a lazy");
            sb.AppendLine("    /// composite builds its children in EnsureInitialized, and GetParameters now folds");
            sb.AppendLine("    /// in GetSubLayers(), so this runs first and RegisterSubLayer threw on the null.");
            sb.AppendLine("    /// Register what exists and latch only when nothing was missing — registration is");
            sb.AppendLine("    /// identity-based and idempotent, so the retry after initialization is free.");
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine("    private void EnsureSubLayersRegistered()");
            sb.AppendLine("    {");
            sb.AppendLine("        if (_subLayersRegistered) return;");
            sb.AppendLine("        bool __complete = true;");
            foreach (var sl in subLayerFields)
            {
                if (sl.IsCollection)
                {
                    // Null-guarded regardless of annotation: a collection field can legitimately be
                    // left unassigned on a branch the constructor did not take, and RegisterSubLayer
                    // is identity-based and idempotent, so re-walking a list is harmless.
                    sb.AppendLine($"        if ({sl.Name} is null) __complete = false;");
                    sb.AppendLine("        else");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            foreach (var __sub in {sl.Name})");
                    sb.AppendLine("            {");
                    sb.AppendLine("                if (__sub is not null) RegisterSubLayer(__sub);");
                    sb.AppendLine("            }");
                    sb.AppendLine("        }");
                }
                else if (sl.IsNullable)
                    sb.AppendLine($"        if ({sl.Name} is not null) RegisterSubLayer({sl.Name});");
                else
                    sb.AppendLine($"        if ({sl.Name} is not null) RegisterSubLayer({sl.Name}); else __complete = false;");
            }
            sb.AppendLine("        _subLayersRegistered = __complete;");
            sb.AppendLine("    }");
            sb.AppendLine();
            // Emitted only when the layer does not write its own; a hand-written override is
            // respected rather than duplicated (which is what the class-level skip used to do,
            // at the cost of the accessors above).
            if (!DeclaresAny(classSymbol, "EnsureInitialized"))
            {
                sb.AppendLine("    /// <summary>");
                sb.AppendLine("    /// Auto-generated EnsureInitialized: registers sub-layers (cheap), then");
                sb.AppendLine("    /// delegates to base for weight allocation.");
                sb.AppendLine("    /// </summary>");
                sb.AppendLine("    protected override void EnsureInitialized()");
                sb.AppendLine("    {");
                sb.AppendLine("        EnsureSubLayersRegistered();");
                sb.AppendLine("        base.EnsureInitialized();");
                sb.AppendLine("    }");
            }
            if (!DeclaresAny(classSymbol, "GetSubLayers"))
            {
                // GetSubLayers is itself a public structural query and must be complete before
                // Forward. Initialization is too late for Deserialize/Clone/export/topology walks:
                // a fresh composite has not executed Forward yet, so those paths otherwise see no
                // children and can silently restore the entire flat vector into the parent's
                // fallback Parameters slot. This applies even when the generator also emits
                // EnsureInitialized and even when the parent owns trainable tensors of its own.
                //
                // Registering from GetSubLayers keeps the timing lazy. It must NOT move into a
                // constructor: that places children in front of the pre-step buffer-view walk
                // beside the parent that already handles them, which silently breaks training.
                sb.AppendLine("    /// <inheritdoc />");
                sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<ILayer<{GetTypeParamName(classSymbol)}>> GetSubLayers()");
                sb.AppendLine("    {");
                sb.AppendLine("        EnsureSubLayersRegistered();");
                sb.AppendLine("        return base.GetSubLayers();");
                sb.AppendLine("    }");
            }
        }

        sb.AppendLine("}");

        // Close containing type wrappers for nested classes
        foreach (var _ in containingTypes)
        {
            sb.AppendLine("}");
        }

        return sb.ToString();
    }

    /// <summary>
    /// The element type to write into the generated signatures.
    /// </summary>
    /// <remarks>
    /// Usually the layer's own type parameter, but a layer may FIX its element type
    /// (<c>QuantizedDenseLayer : LayerBase&lt;float&gt;</c>). Falling back to the literal "T" for
    /// those emitted <c>Tensor&lt;T&gt;</c> against a class with no T, so the only way they built
    /// was for the generator to skip them entirely -- which meant a whole category of layer could
    /// never be automatic. Read the element off the base instead.
    /// </remarks>
    private static string GetTypeParamName(INamedTypeSymbol classSymbol)
    {
        if (classSymbol.TypeParameters.Length > 0) return classSymbol.TypeParameters[0].Name;
        for (var b = classSymbol.BaseType; b is not null; b = b.BaseType)
        {
            if (b.OriginalDefinition.ToDisplayString()
                    .StartsWith("AiDotNet.NeuralNetworks.Layers.LayerBase<", System.StringComparison.Ordinal)
                && b.TypeArguments.Length == 1)
                return b.TypeArguments[0].ToDisplayString();
        }
        return "T";
    }

    /// <summary>True when the class itself declares any of the named members.</summary>
    /// <summary>True when the class already declares one of the methods this generator emits.</summary>
    /// <remarks>
    /// MATCHED ON SIGNATURE, NOT NAME. Any member sharing the name suppressed the whole class's
    /// generation -- an unrelated overload such as `GetTrainableParameters(bool includeFrozen)` was
    /// enough. That silently reintroduced stale parameter rebinding, or dropped sub-layer
    /// registration, for a class whose author had not overridden anything the generator writes.
    /// </remarks>
    private static bool DeclaresAny(INamedTypeSymbol type, params string[] names)
    {
        foreach (var name in names)
        {
            foreach (var member in type.GetMembers(name))
            {
                if (member is not IMethodSymbol m) continue;

                // The generated shapes, exactly:
                //   EnsureInitialized()
                //   GetTrainableParameters()
                //   SetTrainableParameters(IReadOnlyList<Tensor<T>>)
                bool matches = name switch
                {
                    "SetTrainableParameters" => m.Parameters.Length == 1,
                    _ => m.Parameters.Length == 0,
                };
                if (matches) return true;
            }
        }
        return false;
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

    private static bool HasAttr(IFieldSymbol field, INamedTypeSymbol? attributeSymbol)
        => attributeSymbol is not null && field.GetAttributes()
            .Any(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, attributeSymbol));

    /// <summary>
    /// A tensor over the LAYER's own element type -- the only thing that fits the
    /// <c>Tensor&lt;T&gt;</c> parameter surface.
    /// </summary>
    /// <remarks>
    /// <see cref="IsTensorType"/> matches any element type, so <c>Tensor&lt;Complex&lt;T&gt;&gt;</c>
    /// (QuantumLayer's state amplitudes) passed and the generated code tried to hand it to
    /// <c>IReadOnlyList&lt;Tensor&lt;T&gt;&gt;</c>. Such a field is real state, but it cannot be a
    /// parameter until the surface is generic over the element type; it needs [Buffer]/[Scratch] or
    /// a hand-written flattening.
    /// </remarks>
    private static bool IsTensorOfLayerElement(ITypeSymbol type, INamedTypeSymbol classSymbol)
    {
        var elem = GetTypeParamName(classSymbol);
        for (var c = type; c is not null; c = c.BaseType)
        {
            if (c is not INamedTypeSymbol named) continue;
            if (!named.OriginalDefinition.ToDisplayString()
                    .StartsWith("AiDotNet.Tensors.LinearAlgebra.Tensor<", System.StringComparison.Ordinal))
                continue;
            return named.TypeArguments.Length == 1
                   && named.TypeArguments[0].ToDisplayString() == elem;
        }
        return false;
    }

    private static bool IsTensorType(ITypeSymbol type)
    {
        // Walk the inheritance chain so SparseTensor<T> (and any future
        // Tensor<T> subclass like JaggedTensor / RaggedTensor) is treated
        // as a trainable-parameter-eligible type. The generator previously
        // only matched the literal Tensor<T> spelling, which excluded
        // SparseLinearLayer's _weights field from auto-registration even
        // though it was meant to be tape-trainable.
        var current = type;
        while (current is not null)
        {
            var original = current is INamedTypeSymbol named ? named.OriginalDefinition : current;
            var display = original.ToDisplayString();
            if (display.StartsWith(TensorTypeName + "<") || display == TensorTypeName)
                return true;
            current = current.BaseType;
        }
        return false;
    }

    /// <summary>
    /// True for a field holding MANY sub-layers: TLayer[], List&lt;TLayer&gt;, IReadOnlyList&lt;TLayer&gt;
    /// and friends, where TLayer satisfies <see cref="IsLayerType"/>.
    /// </summary>
    private static bool IsLayerCollectionType(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol array)
            return IsLayerType(array.ElementType);

        if (type is INamedTypeSymbol named && named.IsGenericType && named.TypeArguments.Length == 1)
        {
            // Only walk types that are actually enumerable, so a Func<TLayer> or similar
            // single-argument generic is not mistaken for a collection of layers.
            var enumerable = named.AllInterfaces.Any(i =>
                i.OriginalDefinition.ToDisplayString() == "System.Collections.Generic.IEnumerable<T>");
            if (enumerable && IsLayerType(named.TypeArguments[0]))
                return true;
        }

        return false;
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

    // Mirror of PersistentTensorRole enum for mapping attribute integer values
    // to fully-qualified enum names in generated code. MUST be kept in sync with
    // AiDotNet.Tensors.Engines.PersistentTensorRole — if you add a new role there, add it here.
    private enum PersistentTensorRoleEnum
    {
        Weights = 0,
        Biases = 1,
        NormalizationParams = 2,
        Embeddings = 3,
        AttentionCache = 4,
        OptimizerState = 5,
        Constant = 6,
        Other = 7,
        ScaleParameters = 8,
    }

    /// <summary>
    /// Scans all syntax trees of a class for invocations of RegisterTrainableParameter(_field, role)
    /// and extracts the field name and role. This enables auto-discovery without [TrainableParameter]
    /// attributes — matching the pattern used for RegisterSubLayer discovery.
    /// </summary>
    /// <summary>
    /// True when some <c>RegisterTrainableParameter</c> argument is not a field of this class --
    /// a loop variable over a collection, an indexer, a local. Field discovery cannot account for
    /// those, so the generated surface would be incomplete rather than merely redundant.
    /// </summary>
    private static bool HasUnmappableRegistration(
        ClassDeclarationSyntax classDecl, SemanticModel model, INamedTypeSymbol classSymbol)
    {
        foreach (var invocation in classDecl.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            string invokedName = invocation.Expression switch
            {
                IdentifierNameSyntax id => id.Identifier.Text,
                MemberAccessExpressionSyntax ma => ma.Name.Identifier.Text,
                _ => string.Empty,
            };
            if (invokedName != "RegisterTrainableParameter") continue;
            if (invocation.ArgumentList.Arguments.Count == 0) continue;

            var arg = invocation.ArgumentList.Arguments[0].Expression;
            if (model.SyntaxTree != arg.SyntaxTree) continue;
            var symbol = model.GetSymbolInfo(arg).Symbol;
            if (symbol is IFieldSymbol f
                && SymbolEqualityComparer.Default.Equals(f.ContainingType, classSymbol))
                continue;
            return true;
        }
        return false;
    }

    private static List<(string FieldName, string Role)> DiscoverFromRegisterCalls(
        ClassDeclarationSyntax classDecl, SemanticModel model, string methodName)
    {
        var results = new List<(string, string)>();
        var seen = new HashSet<string>();

        // Walk all invocation expressions in the class body
        foreach (var invocation in classDecl.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            // Match method name: RegisterTrainableParameter(...)
            string invokedName;
            if (invocation.Expression is IdentifierNameSyntax id)
                invokedName = id.Identifier.Text;
            else if (invocation.Expression is MemberAccessExpressionSyntax ma)
                invokedName = ma.Name.Identifier.Text;
            else
                continue;

            if (invokedName != methodName) continue;

            var args = invocation.ArgumentList.Arguments;
            if (args.Count < 2) continue;

            // First arg: the field reference (e.g., _weights)
            var fieldArg = args[0].Expression;
            string fieldName;
            if (fieldArg is IdentifierNameSyntax fieldId)
                fieldName = fieldId.Identifier.Text;
            else if (fieldArg is MemberAccessExpressionSyntax fieldMa && fieldMa.Expression is ThisExpressionSyntax)
                fieldName = fieldMa.Name.Identifier.Text;
            else
                continue;

            // Second arg: the role enum (e.g., PersistentTensorRole.Weights)
            var roleArg = args[1].Expression.ToString();
            // Normalize to full enum reference
            if (!roleArg.Contains("PersistentTensorRole"))
                roleArg = $"PersistentTensorRole.{roleArg}";

            // Deduplicate (same field may be registered in multiple constructors)
            if (seen.Add(fieldName))
            {
                results.Add((fieldName, roleArg));
            }
        }

        return results;
    }

    /// <summary>
    /// Captured info per [TrainableParameter] field. <see cref="TypeName"/>
    /// is the field's declared type as a fully-qualified display string;
    /// when it differs from <c>AiDotNet.Tensors.LinearAlgebra.Tensor&lt;T&gt;</c>
    /// (e.g., <c>SparseTensor&lt;T&gt;</c>) the generator emits a downcast
    /// in SetTrainableParameters so the field assignment compiles.
    /// </summary>
    /// <summary>
    /// "This parameter is currently present." Absent means either a genuinely null field (a
    /// nullable <c>[TrainableParameter]</c> whose feature is switched off) or a non-null empty
    /// placeholder awaiting materialization -- both must be skipped, and dereferencing the first
    /// to test the second throws.
    /// </summary>
    private static string PresenceExpr(ParameterFieldInfo pf)
        => pf.Nullable ? $"{pf.Name} is not null && {pf.Name}.Length > 0" : $"{pf.Name}.Length > 0";

    private record struct ParameterFieldInfo(string Name, string Role, int Order, int DeclIndex = 0, string? TypeName = null, bool Optional = false, bool Nullable = false, string? Shape = null);
    private record struct GradientFieldInfo(string Name, bool IsNullable);
    private record struct SubLayerFieldInfo(string Name, bool IsNullable, bool IsCollection);
}
