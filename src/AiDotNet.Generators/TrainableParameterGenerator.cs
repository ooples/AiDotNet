using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
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

    private static readonly DiagnosticDescriptor DuplicateBufferIdentity = new(
        "ADNBUF001",
        "Generated buffer identity is ambiguous",
        "'{0}' declares persistent fields '{1}' with the same buffer identity '{2}'. Give each distinct state tensor a unique [Buffer(Name = ...)] identity.",
        "AiDotNet.ParameterAutomation",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A stable buffer identity maps to exactly one tensor and one semantic role within a layer.");

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

        // [AutoParameters] remains a migration marker, but it never assigns semantics. PyTorch can
        // infer from nn.Parameter because that is a distinct type; Tensor<T> is also used for
        // activations, caches, datasets and buffers, so treating its CLR type or nullability as a
        // role silently corrupts the parameter graph.
        var autoParamsSymbol = compilation.GetTypeByMetadataName("AiDotNet.Attributes.AutoParametersAttribute");
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

            bool hasAutoParameters = autoParamsSymbol is not null
                && classSymbol.GetAttributes().Any(attribute =>
                    SymbolEqualityComparer.Default.Equals(attribute.AttributeClass, autoParamsSymbol));

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

            // A loop/local registration (for example, foreach (var tensor in dictionary.Values))
            // is an explicit parameter declaration, but it cannot be reconstructed from fields at
            // compile time. In that case the runtime registry is the complete source of truth. The
            // detector for this case existed for a long time but was never consumed, so a single
            // annotated field caused the generator to emit a field-only override that HID every
            // dynamically registered tensor. HeterogeneousGraphLayer consequently registered all
            // of its per-type weights and then reported an empty parameter surface.
            bool useRuntimeParameterRegistry = HasAnyUnmappableRegistration(compilation, classSymbol);

            // Skip if already processed (multiple partial files)
            var fullName = classSymbol.ToDisplayString();
            if (!processedClasses.Add(fullName)) continue;

            // Collect [TrainableParameter] fields
            var paramFields = new List<ParameterFieldInfo>();
            var gradientFields = new Dictionary<string, GradientFieldInfo>();
            var subLayerFields = new List<SubLayerFieldInfo>();

            var bufferFields = new List<(string Field, string Name, string Role, string StateRole)>();

            foreach (var member in classSymbol.GetMembers())
            {
                if (member is not IFieldSymbol field) continue;

                var classification = ParameterMemberSemanticModel.Classify(field);

                // Check for [TrainableParameter]
                var attr = field.GetAttributes()
                    .FirstOrDefault(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, attributeSymbol));

                if (classification.Kind == ParameterMemberSemanticModel.Kind.Trainable && attr is not null)
                {
                    var role = "PersistentTensorRole.Weights";
                    var order = 0;
                    var optional = false;
                    string? shape = null;
                    string? condition = null;

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
                        else if (namedArg.Key == "Condition" && namedArg.Value.Value is string conditionVal)
                            condition = conditionVal;
                    }

                    var explicitNullable = field.NullableAnnotation == NullableAnnotation.Annotated
                                           || field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    if (IsTensorOfLayerElement(field.Type, classSymbol))
                    {
                        paramFields.Add(new ParameterFieldInfo(
                            field.Name, role, order, DeclIndex: 0,
                            TypeName: field.Type.ToDisplayString(),
                            // Nullable storage is emitted through the safe conditional mechanics while
                            // AIDN090 requires the author to declare WHY it is nullable. This is a
                            // migration bridge, not semantic inference: the diagnostic remains until
                            // Optional/Availability is explicit, and the classifier never promotes an
                            // unannotated nullable tensor into the graph.
                            Optional: optional || explicitNullable, Nullable: explicitNullable,
                            Shape: shape, Condition: condition));
                    }
                    else if (TryGetTensorCollection(field.Type, classSymbol, out var collectionKind))
                    {
                        // A collection is one semantic declaration but contributes one optimizer tensor
                        // per element. Its stable order is positional for arrays/lists and canonical-key
                        // order for dictionaries, matching ModelParameterGenerator's collection sources.
                        // The collection object may be readonly; generated restore mutates its elements,
                        // never rebinds the field itself.
                        paramFields.Add(new ParameterFieldInfo(
                            field.Name, role, order, DeclIndex: 0,
                            TypeName: field.Type.ToDisplayString(),
                            Optional: optional || explicitNullable, Nullable: explicitNullable,
                            Shape: shape, CollectionKind: collectionKind, Condition: condition));
                    }
                }

                // Collect every declared persistent non-optimizer role through the same base
                // buffer mechanism. The manifest retains whether it is fitted, frozen, or a true
                // auxiliary buffer; the optimizer view excludes all three.
                // Marking alone is not enough -- without emitting RegisterBuffer the tensors leave
                // the trainable set and join nothing, disappearing from ParameterCount and the flat
                // vector entirely. ReservoirLayer proved it: "Expected 320 parameters, got 0".
                if (!field.IsStatic && IsTensorType(field.Type)
                    && classification.Kind is ParameterMemberSemanticModel.Kind.Fitted
                        or ParameterMemberSemanticModel.Kind.Frozen
                        or ParameterMemberSemanticModel.Kind.Buffer)
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
                    string stateRole = classification.Kind switch
                    {
                        ParameterMemberSemanticModel.Kind.Fitted =>
                            "global::AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState",
                        ParameterMemberSemanticModel.Kind.Frozen =>
                            "global::AiDotNet.Models.Parameters.ParameterSlotRole.Frozen",
                        _ => "global::AiDotNet.Models.Parameters.ParameterSlotRole.Buffer"
                    };
                    bufferFields.Add((field.Name, bufName, bufRole, stateRole));
                }

                // Check for gradient fields (convention: {name}Gradient)
                if (field.Name.EndsWith("Gradient") &&
                    (IsTensorType(field.Type) || TryGetTensorCollection(field.Type, classSymbol, out _)))
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
                    var sliShape = field.GetAttributes()
                        .FirstOrDefault(a => a.AttributeClass?.Name == "SubLayerInputAttribute")
                        ?.ConstructorArguments.FirstOrDefault().Value as string;
                    subLayerFields.Add(new SubLayerFieldInfo(field.Name, isNullable, IsCollection: false, InputShape: sliShape));
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

            foreach (var duplicate in bufferFields
                         .GroupBy(item => item.Name, System.StringComparer.Ordinal)
                         .Where(group => group.Count() > 1))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    DuplicateBufferIdentity,
                    classSymbol.Locations.FirstOrDefault(),
                    classSymbol.Name,
                    string.Join(", ", duplicate.Select(item => item.Field)),
                    duplicate.Key));
            }

            // Merge trainable parameters declared by attribute with the compatibility
            // RegisterTrainableParameter route. Both are explicit semantic declarations; neither
            // is tensor-type inference. Dropping either route makes a partially migrated layer's
            // generated graph incomplete, so count/read/write can agree with each other while an
            // actual weight silently disappears.
            {
                var registeredFields = DiscoverFromRegisterCalls(classSymbol, "RegisterTrainableParameter");

                // Weights held
                // in a Dictionary<string, Tensor<T>> or a List<Tensor<T>> and registered in a loop
                // are not fields, so field discovery finds none of them; emitting a surface from
                // the fields alone would OVERRIDE the runtime registry and drop every one
                // (HeterogeneousGraphLayer's per-edge-type weights, biases and basis coefficients
                // all vanished, and its Parameters_CountShouldMatchVector went to zero). Such a
                // collection cannot be promoted by default either -- the same shape is far more
                // often a cache (_lastInputs, _gpuCachedHiddenStates), and silently training a
                // cache is worse than the bug. So imperative registration stays authoritative for
                // exactly these layers, which is how they already worked.
                if (registeredFields.Count > 0)
                {
                    var seen = new HashSet<string>(paramFields.Select(parameter => parameter.Name));
                    int nextOrder = paramFields.Count == 0
                        ? 0
                        : paramFields.Max(parameter => parameter.Order) + 1;
                    foreach (var (fieldName, role) in registeredFields)
                    {
                        if (!seen.Add(fieldName)) continue;

                        // A nullable registered field remains explicit trainable state. Preserve
                        // its conditional presence in the generated manifest; AIDN090 separately
                        // requires the author to declare the lifecycle that explains the null.
                        var matchingField = classSymbol.GetMembers()
                            .OfType<IFieldSymbol>()
                            .FirstOrDefault(f => f.Name == fieldName && IsTensorType(f.Type));
                        if (matchingField is not null)
                        {
                            bool nullable = matchingField.NullableAnnotation == NullableAnnotation.Annotated
                                || matchingField.Type.NullableAnnotation == NullableAnnotation.Annotated;
                            paramFields.Add(new ParameterFieldInfo(
                                matchingField.Name, role, nextOrder++, DeclIndex: 0,
                                TypeName: matchingField.Type.ToDisplayString(),
                                Optional: nullable, Nullable: nullable));
                        }
                    }
                }
            }

            bool hasImperativePersistentRegistration =
                ParameterMemberSemanticModel.GetRegistrationClassifications(classSymbol).Count > 0
                || HasPersistentRegistrationInvocation(classSymbol);
            bool hasInheritedPersistentContract = HasInheritedPersistentContract(
                classSymbol, attributeSymbol, bufferSymbol);
            bool hasUnmodeledLayerContainer = classSymbol.GetMembers()
                .OfType<IFieldSymbol>()
                .Any(field => !field.IsStatic && IsPotentialLayerContainer(field.Type));
            bool emitParameterFreeContract = hasAutoParameters
                && !useRuntimeParameterRegistry
                && !hasImperativePersistentRegistration
                && !hasInheritedPersistentContract
                && !hasUnmodeledLayerContainer
                && paramFields.Count == 0
                && subLayerFields.Count == 0
                && bufferFields.Count == 0;

            if (paramFields.Count == 0 && subLayerFields.Count == 0 && bufferFields.Count == 0
                && !emitParameterFreeContract) continue;

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
            var source = GenerateSource(
                classSymbol, paramFields, gradientFields, subLayerFields, bufferFields,
                useRuntimeParameterRegistry, emitParameterFreeContract);
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
        List<(string Field, string Name, string Role, string StateRole)> bufferFields,
        bool useRuntimeParameterRegistry,
        bool emitParameterFreeContract)
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

        if (emitParameterFreeContract)
        {
            sb.AppendLine("    /// <summary>Auto-generated: this migrated layer declares no persistent parameter state.</summary>");
            sb.AppendLine("    protected override bool IsDeclaredParameterFree => true;");
            sb.AppendLine();
        }

        // One inheritance-aware component manifest drives the public flat parameter surfaces.
        // Base declarations are appended first, then this class's fields in source declaration
        // order. This is what keeps a derived adapter's own tensors after the factors declared by
        // its base class without teaching the generator any model or adapter names.
        EmitOrderedParameterManifest(
            sb, classSymbol, paramFields, subLayerFields, bufferFields);

        // A complete local shape declaration can recover one deferred input axis from a flat
        // checkpoint length exactly. Emit the algebra from the author's Shape expressions so a
        // lazy layer does not need a model-specific guess (or a hand-written SetParameters).
        EmitDeferredInputShapeInference(sb, classSymbol, paramFields, subLayerFields, bufferFields);

        // Buffer registration. Persistent, never trained: LayerBase folds these into
        // ParameterCount / GetParameters / SetParameters but deliberately keeps them out of
        // GetTrainableParameters, so the optimizer and the tape cannot touch them. This mirrors the
        // PyTorch parameters()-versus-state_dict() split, with the difference that both surfaces
        // here are covered by one flat vector and one checked count.
        if (bufferFields.Count > 0)
        {
            sb.AppendLine("    /// <summary>Auto-generated: registers [Buffer] fields as persistent non-trainable state.</summary>");
            sb.AppendLine("    private void EnsureBuffersRegistered()");
            sb.AppendLine("    {");
            foreach (var bf in bufferFields)
            {
                sb.AppendLine($"        if ({bf.Field} is not null) RegisterBuffer({bf.Field}, \"{bf.Name}\", {bf.Role}, {bf.StateRole});");
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


        // DeclaredSubLayerShapes — emitted from [SubLayerInput("...")] on the sub-layer fields.
        //
        // A composite's children do not all receive the composite's own input, and only the
        // composite knows which gets what. Declaring it on the field lets the generator supply that
        // fact to LayerBase.BringUpDeclaredSubLayers, so no composite implements the method.
        var shapedSubLayers = subLayerFields
            .Where(sl => !sl.IsCollection && !string.IsNullOrWhiteSpace(sl.InputShape))
            .ToList();
        if (shapedSubLayers.Count > 0)
        {
            string tpS = GetTypeParamName(classSymbol);
            string subTuple = $"(LayerBase<{tpS}>? Child, AiDotNet.Tensors.LinearAlgebra.TensorShape InputShape)";
            string subArray = $"(LayerBase<{tpS}>?, AiDotNet.Tensors.LinearAlgebra.TensorShape)";

            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// The input shape each sub-layer receives from this composite.");
            sb.AppendLine("    /// Auto-generated — do not modify. Edit the [SubLayerInput(\"...\")] arguments instead.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    /// <remarks>");
            sb.AppendLine("    /// Empty while any declared child is still null or any axis is still negative: a composite");
            sb.AppendLine("    /// builds its children inside its initializer, so both are ordinary states before that runs.");
            sb.AppendLine("    /// Cached, because the initializer deliberately re-enters.");
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{subTuple}> DeclaredSubLayerShapes()");
            sb.AppendLine("    {");
            sb.AppendLine("        if (__declaredSubLayerShapes is not null) return __declaredSubLayerShapes;");
            foreach (var sl in shapedSubLayers)
            {
                sb.AppendLine($"        if ({sl.Name} is null) return System.Array.Empty<{subArray}>();");
            }
            sb.AppendLine($"        var __sub = new {subArray}[]");
            sb.AppendLine("        {");
            foreach (var sl in shapedSubLayers)
            {
                var axes = string.Join(", ", sl.InputShape!.Split(',').Select(a => a.Trim()).Where(a => a.Length > 0));
                sb.AppendLine($"            ({sl.Name}, ShapeOf({axes})),");
            }
            sb.AppendLine("        };");
            sb.AppendLine("        for (int __i = 0; __i < __sub.Length; __i++)");
            sb.AppendLine("        {");
            sb.AppendLine("            var __s = __sub[__i].Item2;");
            sb.AppendLine("            for (int __d = 0; __d < __s.Length; __d++)");
            sb.AppendLine($"                if (__s[__d] < 0) return System.Array.Empty<{subArray}>();");
            sb.AppendLine("        }");
            sb.AppendLine("        __declaredSubLayerShapes = __sub;");
            sb.AppendLine("        return __declaredSubLayerShapes;");
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine($"    private {subArray}[]? __declaredSubLayerShapes;");
            sb.AppendLine();
        }

        // DeclaredParameterShapes — emitted from [TrainableParameter(Shape = "...")].
        //
        // This is the whole point of the Shape argument: LayerBase.TryAdoptRestoredParameters can see
        // THAT a tensor was supplied before the first forward but not whether its shape is right, and
        // only the layer knows that its weights are [inputSize, outputSize]. Declaring it on the field
        // lets the generator supply that fact, so no layer hand-writes the override.
        var shapedFields = useRuntimeParameterRegistry
            ? new List<ParameterFieldInfo>()
            : paramFields.Where(p => p.CollectionKind == ParameterCollectionKind.Direct &&
                                     !string.IsNullOrWhiteSpace(p.Shape)).ToList();
        if (shapedFields.Count > 0)
        {
            string tp = GetTypeParamName(classSymbol);
            string tupleType = $"(Tensor<{tp}>? Tensor, AiDotNet.Tensors.LinearAlgebra.TensorShape Expected, PersistentTensorRole Role)";
            string arrayType = $"(Tensor<{tp}>?, AiDotNet.Tensors.LinearAlgebra.TensorShape, PersistentTensorRole)";

            string activeShapeDeclarations = shapedFields.Any(field => field.Condition is null)
                ? "true"
                : string.Join(" || ", shapedFields.Select(field => $"({field.Condition})"));
            sb.AppendLine("    /// <summary>Whether an active parameter declaration is waiting for its shape.</summary>");
            sb.AppendLine($"    protected override bool HasActiveDeclaredParameterShapes => {activeShapeDeclarations};");
            sb.AppendLine();

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
            sb.AppendLine($"        var __declared = new System.Collections.Generic.List<{tupleType}>({shapedFields.Count});");
            foreach (var pf in shapedFields)
            {
                var axes = pf.Shape!.Split(',')
                    .Select(a => a.Trim())
                    .Where(a => a.Length > 0)
                    .Select(ToValidationShapeAxis);
                if (pf.Condition is not null)
                    sb.AppendLine($"        if ({pf.Condition})");
                sb.AppendLine($"            __declared.Add(({pf.Name}, ShapeOf({string.Join(", ", axes)}), {pf.Role}));");
            }
            sb.AppendLine();
            sb.AppendLine("        for (int __i = 0; __i < __declared.Count; __i++)");
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

        // A bound adaptive axis, written *(<expression>), keeps wildcard restore semantics but
        // supplies the manifest with its current allocation-free size. Emit an aligned shape list
        // only when one is present; ordinary declarations keep the zero-overhead default.
        bool hasBoundAdaptiveAxes = shapedFields.Any(field =>
            field.Shape!.Split(',').Any(axis => TryGetAdaptiveAxisBinding(axis.Trim(), out _)));
        if (hasBoundAdaptiveAxes)
        {
            const string countShapeType = "AiDotNet.Tensors.LinearAlgebra.TensorShape";
            sb.AppendLine("    /// <summary>Concrete sizing view for bound adaptive parameter axes.</summary>");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{countShapeType}> DeclaredParameterCountShapes()");
            sb.AppendLine("    {");
            sb.AppendLine($"        var __declared = new System.Collections.Generic.List<{countShapeType}>({shapedFields.Count});");
            foreach (var pf in shapedFields)
            {
                var countAxes = pf.Shape!.Split(',')
                    .Select(a => a.Trim())
                    .Where(a => a.Length > 0)
                    .Select(ToCountingShapeAxis);
                if (pf.Condition is not null)
                    sb.AppendLine($"        if ({pf.Condition})");
                sb.AppendLine($"            __declared.Add(ShapeOf({string.Join(", ", countAxes)}));");
            }
            sb.AppendLine("        for (int __i = 0; __i < __declared.Count; __i++)");
            sb.AppendLine("            for (int __d = 0; __d < __declared[__i].Length; __d++)");
            sb.AppendLine("                if (__declared[__i][__d] <= 0) return System.Array.Empty<AiDotNet.Tensors.LinearAlgebra.TensorShape>();");
            sb.AppendLine("        return __declared;");
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        // GetTrainableParameters
        bool hasCollections = paramFields.Any(p => p.CollectionKind != ParameterCollectionKind.Direct);
        bool hasOptional = paramFields.Any(p =>
            (p.CollectionKind == ParameterCollectionKind.Direct && p.Optional) || p.Condition is not null);
        if (paramFields.Count > 0 && !useRuntimeParameterRegistry)
        {
            bool hasFixedParameterView = !hasCollections && !hasOptional;
            if (hasFixedParameterView)
            {
                string tensorType = $"Tensor<{GetTypeParamName(classSymbol)}>";
                sb.AppendLine($"    private {tensorType}[]? __aidnTrainableParameterViewStorage;");
                sb.AppendLine($"    private System.Collections.ObjectModel.ReadOnlyCollection<{tensorType}>? __aidnTrainableParameterView;");
                sb.AppendLine();
                sb.AppendLine("    /// <summary>Returns the stable, allocation-free view of fixed generated parameter fields.</summary>");
                sb.AppendLine($"    private System.Collections.Generic.IReadOnlyList<{tensorType}> __aidnGetTrainableParameterView()");
                sb.AppendLine("    {");
                sb.AppendLine("        var __storage = System.Threading.Volatile.Read(ref __aidnTrainableParameterViewStorage);");
                sb.AppendLine("        if (__storage is null)");
                sb.AppendLine("        {");
                sb.AppendLine($"            var __created = new {tensorType}[{paramFields.Count}];");
                sb.AppendLine("            __storage = System.Threading.Interlocked.CompareExchange(");
                sb.AppendLine("                ref __aidnTrainableParameterViewStorage, __created, null) ?? __created;");
                sb.AppendLine("        }");
                for (int i = 0; i < paramFields.Count; i++)
                    sb.AppendLine($"        __storage[{i}] = {paramFields[i].Name};");
                sb.AppendLine("        var __view = System.Threading.Volatile.Read(ref __aidnTrainableParameterView);");
                sb.AppendLine("        if (__view is null)");
                sb.AppendLine("        {");
                sb.AppendLine("            var __createdView = System.Array.AsReadOnly(__storage);");
                sb.AppendLine("            __view = System.Threading.Interlocked.CompareExchange(");
                sb.AppendLine("                ref __aidnTrainableParameterView, __createdView, null) ?? __createdView;");
                sb.AppendLine("        }");
                sb.AppendLine("        return __view;");
                sb.AppendLine("    }");
                sb.AppendLine();
                sb.AppendLine("    private void __aidnRefreshTrainableParameterViewIfCreated()");
                sb.AppendLine("    {");
                sb.AppendLine("        var __storage = System.Threading.Volatile.Read(ref __aidnTrainableParameterViewStorage);");
                sb.AppendLine("        if (__storage is null) return;");
                for (int i = 0; i < paramFields.Count; i++)
                    sb.AppendLine($"        __storage[{i}] = {paramFields[i].Name};");
                sb.AppendLine("    }");
                sb.AppendLine();
            }

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
            if (hasOptional || hasCollections)
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
            sb.AppendLine("        if (IsShapeResolved || ParametersAreConstructionSized) EnsureInitializationSerialized();");
            if (hasOptional || hasCollections)
            {
                sb.AppendLine($"        var __params = new System.Collections.Generic.List<Tensor<{GetTypeParamName(classSymbol)}>>({paramFields.Count});");
                foreach (var f in paramFields)
                {
                    EmitCollectionAdd(sb, f, "__params");
                }
                sb.AppendLine("        return __params;");
            }
            else
            {
                sb.AppendLine("        return __aidnGetTrainableParameterView();");
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
            if (hasOptional || hasCollections)
            {
                sb.AppendLine($"        var __counting = new System.Collections.Generic.List<Tensor<{GetTypeParamName(classSymbol)}>>({paramFields.Count});");
                foreach (var f in paramFields)
                {
                    EmitCollectionAdd(sb, f, "__counting");
                }
                sb.AppendLine("        return __counting;");
            }
            else
            {
                sb.AppendLine("        return __aidnGetTrainableParameterView();");
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
            if (hasOptional || hasCollections)
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
                    if (pf.CollectionKind != ParameterCollectionKind.Direct)
                        EmitCollectionCount(sb, pf, "__expected");
                    else if (pf.Optional || pf.Condition is not null)
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
                sb.AppendLine("        int __withAllOptional = 0;");
                foreach (var pf in paramFields)
                {
                    if (pf.CollectionKind != ParameterCollectionKind.Direct)
                        EmitCollectionCount(sb, pf, "__withAllOptional");
                    else if (pf.Condition is not null)
                        sb.AppendLine($"        if ({pf.Condition}) __withAllOptional++;");
                    else
                        sb.AppendLine("        __withAllOptional++;");
                }
                sb.AppendLine("        bool __materializeOptional = parameters.Count == __withAllOptional && __expected != __withAllOptional;");
                sb.AppendLine("        if (parameters.Count != __expected && !__materializeOptional)");
                sb.AppendLine("            throw new System.ArgumentException($\"Expected {__expected} parameters (currently-present trainable tensors) or {__withAllOptional} (all optional present), got {parameters.Count}.\", nameof(parameters));");
                sb.AppendLine("        int __i = 0;");
                foreach (var pf in paramFields)
                {
                    if (pf.CollectionKind != ParameterCollectionKind.Direct)
                    {
                        EmitCollectionAssign(sb, pf);
                    }
                    else if (pf.Optional || pf.Condition is not null)
                    {
                        string assignCondition = pf.Condition is null
                            ? $"__materializeOptional || ({PresenceExpr(pf)})"
                            : $"({pf.Condition}) && (__materializeOptional || ({PresenceExpr(pf)}))";
                        sb.AppendLine($"        if ({assignCondition})");
                        sb.AppendLine("        {");
                        EmitFieldAssign(pf, "__i", "__i");
                        sb.AppendLine("            __i++;");
                        sb.AppendLine("        }");
                    }
                    else
                    {
                        EmitFieldAssign(pf, "__i", "__i");
                        sb.AppendLine("        __i++;");
                    }
                }
                sb.AppendLine("        if (RegisteredTrainableParameterCount == parameters.Count)");
                sb.AppendLine("        {");
                sb.AppendLine("            base.SetTrainableParameters(parameters);");
                sb.AppendLine("            return;");
                sb.AppendLine("        }");
                sb.AppendLine();
                sb.AppendLine("        ClearRegisteredParameters();");
                foreach (var pf in paramFields)
                    EmitCollectionRegister(sb, pf);
            }
            else
            {
                sb.AppendLine($"        if (parameters.Count != {paramFields.Count})");
                sb.AppendLine($"            throw new System.ArgumentException($\"Expected {paramFields.Count} parameters, got {{parameters.Count}}.\");");
                for (int i = 0; i < paramFields.Count; i++)
                {
                    EmitFieldAssign(paramFields[i], i.ToString(), i.ToString());
                }
                sb.AppendLine("        __aidnRefreshTrainableParameterViewIfCreated();");
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
                if (param.CollectionKind == ParameterCollectionKind.Direct)
                {
                    sb.AppendLine($"        if ({param.Name} != null && {param.Name}.Length > 0)");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            AiDotNet.Tensors.Helpers.TensorAllocator.Return({param.Name});");
                    sb.AppendLine("        }");
                }
                else
                {
                    string values = param.CollectionKind == ParameterCollectionKind.Keyed
                        ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({param.Name})"
                        : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({param.Name})";
                    sb.AppendLine($"        foreach (var __parameter in {values})");
                    sb.AppendLine("        {");
                    sb.AppendLine("            if (__parameter.Length > 0)");
                    sb.AppendLine("                AiDotNet.Tensors.Helpers.TensorAllocator.Return(__parameter);");
                    sb.AppendLine("        }");
                }
            }
            sb.AppendLine("    }");
        }

        // GetParameterRoles — maps parameter names to their roles for per-role learning rates / weight decay
        // Role always has a value (defaults to PersistentTensorRole.Weights), so emit for all param fields
        if (paramFields.Count > 0 && !useRuntimeParameterRegistry)
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
            sb.AppendLine("    /// <summary>Auto-generated: this layer owns child-module structure.</summary>");
            sb.AppendLine("    protected override bool HasDeclaredSubLayerStructure => true;");
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

    private static void EmitOrderedParameterManifest(
        StringBuilder sb,
        INamedTypeSymbol classSymbol,
        List<ParameterFieldInfo> paramFields,
        List<SubLayerFieldInfo> subLayerFields,
        List<(string Field, string Name, string Role, string StateRole)> bufferFields)
    {
        if (paramFields.Count == 0 && subLayerFields.Count == 0 && bufferFields.Count == 0)
            return;

        var parameterNames = new HashSet<string>(paramFields.Select(field => field.Name));
        var subLayersByName = subLayerFields.ToDictionary(field => field.Name, System.StringComparer.Ordinal);
        var buffersByName = bufferFields.ToDictionary(field => field.Field, System.StringComparer.Ordinal);
        var fieldsInSourceOrder = classSymbol.GetMembers()
            .OfType<IFieldSymbol>()
            .Where(field => !field.IsImplicitlyDeclared)
            .OrderBy(field => field.Locations.FirstOrDefault(location => location.IsInSource)
                ?.SourceTree?.FilePath ?? string.Empty, System.StringComparer.Ordinal)
            .ThenBy(field => field.Locations.FirstOrDefault(location => location.IsInSource)
                ?.SourceSpan.Start ?? int.MaxValue)
            .ToList();

        // paramFields already carries the compatibility ordering requested explicitly through
        // [TrainableParameter(Order = ...)]. Map that sequence onto the trainable declaration
        // positions; without an explicit Order, its stable secondary key is source order.
        int parameterCursor = 0;
        var emittedParameters = new HashSet<string>(System.StringComparer.Ordinal);

        sb.AppendLine("    /// <summary>Appends generated parameter components in inheritance and declaration order.</summary>");
        sb.AppendLine("    protected override void AppendDeclaredParameterComponents(");
        sb.AppendLine("        System.Collections.Generic.List<DeclaredParameterComponent> components)");
        sb.AppendLine("    {");
        sb.AppendLine("        base.AppendDeclaredParameterComponents(components);");

        foreach (var field in fieldsInSourceOrder)
        {
            if (parameterNames.Contains(field.Name) && parameterCursor < paramFields.Count)
            {
                var parameter = paramFields[parameterCursor++];
                EmitManifestParameter(sb, parameter);
                emittedParameters.Add(parameter.Name);
                continue;
            }

            if (buffersByName.TryGetValue(field.Name, out var buffer))
            {
                sb.AppendLine($"        DeclareParameterBuffer(components, {buffer.Field}, \"{EscapeStringLiteral(buffer.Name)}\", {buffer.StateRole});");
                continue;
            }

            if (subLayersByName.TryGetValue(field.Name, out var subLayer))
            {
                if (subLayer.IsCollection)
                {
                    sb.AppendLine($"        if ({subLayer.Name} is not null)");
                    sb.AppendLine($"            foreach (var __componentLayer in {subLayer.Name})");
                    sb.AppendLine("                DeclareParameterSubLayer(components, __componentLayer);");
                }
                else
                {
                    sb.AppendLine($"        DeclareParameterSubLayer(components, {subLayer.Name});");
                }
            }
        }

        // Symbols without a source location are rare (typically generated compatibility fields),
        // but silently omitting one would be worse than placing it after source-backed declarations.
        foreach (var parameter in paramFields)
        {
            if (emittedParameters.Add(parameter.Name)) EmitManifestParameter(sb, parameter);
        }

        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static void EmitManifestParameter(StringBuilder sb, ParameterFieldInfo parameter)
    {
        if (parameter.CollectionKind == ParameterCollectionKind.Direct)
        {
            sb.AppendLine($"        DeclareTrainableParameter(components, {parameter.Name});");
            return;
        }

        string values = parameter.CollectionKind == ParameterCollectionKind.Keyed
            ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({parameter.Name})"
            : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({parameter.Name})";
        sb.AppendLine($"        foreach (var __componentTensor in {values})");
        sb.AppendLine("            DeclareTrainableParameter(components, __componentTensor);");
    }

    private static void EmitDeferredInputShapeInference(
        StringBuilder sb,
        INamedTypeSymbol classSymbol,
        List<ParameterFieldInfo> paramFields,
        List<SubLayerFieldInfo> subLayerFields,
        List<(string Field, string Name, string Role, string StateRole)> bufferFields)
    {
        bool completeLocalFormula = paramFields.Count > 0
            && subLayerFields.Count == 0
            && bufferFields.Count == 0
            && paramFields.All(field => field.CollectionKind == ParameterCollectionKind.Direct
                && !field.Optional
                && !string.IsNullOrWhiteSpace(field.Shape)
                && !field.Shape!.Contains("*"));
        if (!completeLocalFormula) return;

        var referencedAxes = new SortedSet<int>();
        foreach (var parameter in paramFields)
        {
            foreach (Match match in Regex.Matches(
                         parameter.Shape!, @"InputShape\s*\[\s*(\d+)\s*\]"))
            {
                if (int.TryParse(match.Groups[1].Value, out int axis)) referencedAxes.Add(axis);
            }
        }
        if (referencedAxes.Count == 0) return;

        sb.AppendLine("    /// <summary>Infers one deferred input axis from complete generated parameter-shape formulas.</summary>");
        sb.AppendLine("    protected override bool TryInferInputShapeFromParameterCount(int parameterCount, out int[] inputShape)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (Parameters.Length == 0)");
        sb.AppendLine("        {");

        foreach (int axis in referencedAxes)
        {
            sb.AppendLine($"            if (InputShape.Length > {axis} && InputShape[{axis}] <= 0)");
            sb.AppendLine("            {");
            sb.AppendLine("                bool __onlyUnknownAxis = true;");
            sb.AppendLine("                for (int __axis = 0; __axis < InputShape.Length; __axis++)");
            sb.AppendLine($"                    if (__axis != {axis} && InputShape[__axis] <= 0) __onlyUnknownAxis = false;");
            sb.AppendLine("                if (__onlyUnknownAxis)");
            sb.AppendLine("                {");
            sb.AppendLine("                    long __atOne = 0;");
            sb.AppendLine("                    long __atTwo = 0;");
            sb.AppendLine("                    checked");
            sb.AppendLine("                    {");
            foreach (var parameter in paramFields)
            {
                sb.AppendLine($"                        __atOne += {ShapeProduct(parameter.Shape!, axis, "1")};");
                sb.AppendLine($"                        __atTwo += {ShapeProduct(parameter.Shape!, axis, "2")};");
            }
            sb.AppendLine("                    }");
            sb.AppendLine("                    long __slope = __atTwo - __atOne;");
            sb.AppendLine("                    long __intercept = __atOne - __slope;");
            sb.AppendLine("                    long __numerator = parameterCount - __intercept;");
            sb.AppendLine("                    if (__slope > 0 && __numerator > 0 && __numerator % __slope == 0");
            sb.AppendLine("                        && __numerator / __slope <= int.MaxValue)");
            sb.AppendLine("                    {");
            sb.AppendLine("                        int __candidate = (int)(__numerator / __slope);");
            sb.AppendLine("                        long __verified = 0;");
            sb.AppendLine("                        checked");
            sb.AppendLine("                        {");
            foreach (var parameter in paramFields)
                sb.AppendLine($"                            __verified += {ShapeProduct(parameter.Shape!, axis, "__candidate")};");
            sb.AppendLine("                        }");
            sb.AppendLine("                        if (__verified == parameterCount)");
            sb.AppendLine("                        {");
            sb.AppendLine("                            inputShape = (int[])InputShape.Clone();");
            sb.AppendLine($"                            inputShape[{axis}] = __candidate;");
            sb.AppendLine("                            return true;");
            sb.AppendLine("                        }");
            sb.AppendLine("                    }");
            sb.AppendLine("                }");
            sb.AppendLine("            }");
        }

        sb.AppendLine("        }");
        sb.AppendLine("        return base.TryInferInputShapeFromParameterCount(parameterCount, out inputShape);");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static string ShapeProduct(string shape, int inputAxis, string replacement)
    {
        var axes = shape.Split(',')
            .Select(axis => axis.Trim())
            .Where(axis => axis.Length > 0)
            .Select(axis => Regex.Replace(
                axis,
                $@"InputShape\s*\[\s*{inputAxis}\s*\]",
                replacement))
            .Select(axis => $"(long)({axis})")
            .ToArray();
        return axes.Length == 0 ? "0L" : string.Join(" * ", axes);
    }

    private static string EscapeStringLiteral(string value)
        => value.Replace("\\", "\\\\").Replace("\"", "\\\"");

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

    private static string ToValidationShapeAxis(string axis)
        => axis == "*" || TryGetAdaptiveAxisBinding(axis, out _) ? "-2" : axis;

    private static string ToCountingShapeAxis(string axis)
        => TryGetAdaptiveAxisBinding(axis, out string binding) ? binding : axis;

    private static bool TryGetAdaptiveAxisBinding(string axis, out string binding)
    {
        binding = string.Empty;
        if (axis.Length < 4
            || !axis.StartsWith("*(", System.StringComparison.Ordinal)
            || axis[axis.Length - 1] != ')')
        {
            return false;
        }

        binding = axis.Substring(2, axis.Length - 3).Trim();
        return binding.Length > 0;
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
    /// Recognizes mutable, deterministically ordered collections of parameter tensors. Enumerable-only
    /// and read-only collection interfaces are deliberately rejected: generated restore must write a
    /// replacement tensor back to the exact slot, and pretending that a read-only surface can do that
    /// would make copy-on-write and optimizer-buffer rebinding silently update stale storage.
    /// </summary>
    private static bool TryGetTensorCollection(
        ITypeSymbol type,
        INamedTypeSymbol classSymbol,
        out ParameterCollectionKind kind)
    {
        kind = ParameterCollectionKind.Direct;

        if (type is IArrayTypeSymbol array && IsTensorOfLayerElement(array.ElementType, classSymbol))
        {
            kind = ParameterCollectionKind.Array;
            return true;
        }

        if (type is not INamedTypeSymbol named || !named.IsGenericType)
            return false;

        if (named.TypeArguments.Length == 1 &&
            IsTensorOfLayerElement(named.TypeArguments[0], classSymbol))
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal))
            {
                kind = ParameterCollectionKind.Indexed;
                return true;
            }
        }

        if (named.TypeArguments.Length == 2 &&
            IsTensorOfLayerElement(named.TypeArguments[1], classSymbol))
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.Dictionary<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IDictionary<", System.StringComparison.Ordinal))
            {
                kind = ParameterCollectionKind.Keyed;
                return true;
            }
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

    /// <summary>
    /// A parameter-free declaration is a closed-world claim. Prove that no base type contributes
    /// trainable fields, buffers, registered state, or child modules before emitting it.
    /// </summary>
    private static bool HasInheritedPersistentContract(
        INamedTypeSymbol classSymbol,
        INamedTypeSymbol? trainableParameterAttribute,
        INamedTypeSymbol? bufferAttribute)
    {
        for (var current = classSymbol.BaseType;
             current is not null && !IsLayerBaseDefinition(current);
             current = current.BaseType)
        {
            // GetRegistrationClassifications maps registrations back to named fields. Runtime
            // registries can also retain tensors supplied through locals (DeepAR's AddProjection
            // helper is the canonical example), so the raw declaration is independently enough
            // to disprove a closed-world parameter-free claim.
            if (ParameterMemberSemanticModel.GetRegistrationClassifications(current).Count > 0
                || HasPersistentRegistrationInvocation(current))
                return true;

            foreach (var field in current.GetMembers().OfType<IFieldSymbol>())
            {
                if (field.IsStatic) continue;
                if (IsLayerType(field.Type)
                    || IsLayerCollectionType(field.Type)
                    || IsPotentialLayerContainer(field.Type))
                {
                    return true;
                }

                foreach (var attribute in field.GetAttributes())
                {
                    if ((trainableParameterAttribute is not null
                         && SymbolEqualityComparer.Default.Equals(
                             attribute.AttributeClass, trainableParameterAttribute))
                        || (bufferAttribute is not null
                            && SymbolEqualityComparer.Default.Equals(
                                attribute.AttributeClass, bufferAttribute)))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private static bool HasPersistentRegistrationInvocation(INamedTypeSymbol type)
    {
        foreach (var syntaxReference in type.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            foreach (var invocation in declaration.DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                string? callName = invocation.Expression switch
                {
                    IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                    MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
                    _ => null
                };
                if (callName is "RegisterTrainableParameter"
                    or "RegisterBuffer"
                    or "RegisterParameterComponent")
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool IsLayerBaseDefinition(INamedTypeSymbol type)
    {
        var display = type.OriginalDefinition.ToDisplayString();
        return display == LayerBaseTypeName || display.StartsWith(LayerBaseTypeName + "<");
    }

    /// <summary>
    /// Conservatively recognizes containers whose value/element type is a layer even when the
    /// collection shape is not yet one the generator can safely enumerate (for example a keyed,
    /// mutable task-adapter dictionary). Such a type cannot be certified parameter-free.
    /// </summary>
    private static bool IsPotentialLayerContainer(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol array) return IsLayerType(array.ElementType);
        if (type is not INamedTypeSymbol named || !named.IsGenericType) return false;
        return named.TypeArguments.Any(IsLayerType);
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
            if (typeDisplay.StartsWith(ILayerTypeName + "<") || typeDisplay == ILayerTypeName
                || typeDisplay.StartsWith(LayerBaseTypeName + "<") || typeDisplay == LayerBaseTypeName)
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

    /// <summary>
    /// Checks every partial declaration for a registration whose tensor is supplied by a loop,
    /// local, indexer, or other expression that cannot be named in generated source.
    /// </summary>
    private static bool HasAnyUnmappableRegistration(
        Compilation compilation,
        INamedTypeSymbol classSymbol)
    {
        foreach (var syntaxReference in classSymbol.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            var semanticModel = compilation.GetSemanticModel(declaration.SyntaxTree);
            if (HasUnmappableRegistration(declaration, semanticModel, classSymbol)) return true;
        }

        return false;
    }

    private static List<(string FieldName, string Role)> DiscoverFromRegisterCalls(
        INamedTypeSymbol classSymbol, string methodName)
    {
        var results = new List<(string, string)>();
        var seen = new HashSet<string>();

        // Walk every partial declaration. Looking only at whichever syntax node reached the
        // incremental pipeline first made registration discovery depend on file ordering.
        foreach (var syntaxReference in classSymbol.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            foreach (var invocation in declaration.DescendantNodes().OfType<InvocationExpressionSyntax>())
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

                // Deduplicate (same field may be registered in multiple constructors).
                if (seen.Add(fieldName))
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
    private static void EmitCollectionAdd(StringBuilder sb, ParameterFieldInfo pf, string destination)
    {
        if (pf.CollectionKind == ParameterCollectionKind.Direct)
        {
            if (pf.Optional || pf.Condition is not null)
                sb.AppendLine($"        if ({PresenceExpr(pf)}) {destination}.Add({pf.Name});");
            else
                sb.AppendLine($"        {destination}.Add({pf.Name});");
            return;
        }

        string values = pf.CollectionKind == ParameterCollectionKind.Keyed
            ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({pf.Name})"
            : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({pf.Name})";
        if (pf.Condition is not null)
        {
            sb.AppendLine($"        if ({pf.Condition})");
            sb.AppendLine("        {");
            sb.AppendLine($"            foreach (var __parameter in {values})");
            sb.AppendLine($"                {destination}.Add(__parameter);");
            sb.AppendLine("        }");
        }
        else
        {
            sb.AppendLine($"        foreach (var __parameter in {values})");
            sb.AppendLine($"            {destination}.Add(__parameter);");
        }
    }

    private static void EmitCollectionCount(StringBuilder sb, ParameterFieldInfo pf, string counter)
    {
        string values = pf.CollectionKind == ParameterCollectionKind.Keyed
            ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({pf.Name})"
            : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({pf.Name})";
        if (pf.Condition is not null)
            sb.AppendLine($"        if ({pf.Condition}) foreach (var __parameter in {values}) {counter}++;");
        else
            sb.AppendLine($"        foreach (var __parameter in {values}) {counter}++;");
    }

    private static void EmitCollectionAssign(StringBuilder sb, ParameterFieldInfo pf)
    {
        if (pf.Condition is not null)
        {
            sb.AppendLine($"        if ({pf.Condition})");
            sb.AppendLine("        {");
        }

        string indent = pf.Condition is null ? "        " : "            ";
        if (pf.CollectionKind is ParameterCollectionKind.Array or ParameterCollectionKind.Indexed)
        {
            string count = pf.CollectionKind == ParameterCollectionKind.Array
                ? $"{pf.Name}.Length"
                : $"{pf.Name}.Count";
            sb.AppendLine($"{indent}if ({pf.Name} is not null)");
            sb.AppendLine($"{indent}{{");
            sb.AppendLine($"{indent}    for (int __slot = 0; __slot < {count}; __slot++)");
            sb.AppendLine($"{indent}        {pf.Name}[__slot] = parameters[__i++] ?? throw new System.ArgumentNullException(nameof(parameters), \"Collection parameter is null.\");");
            sb.AppendLine($"{indent}}}");
        }
        else
        {
            sb.AppendLine($"{indent}if ({pf.Name} is not null)");
            sb.AppendLine($"{indent}{{");
            sb.AppendLine($"{indent}    foreach (var __key in global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedKeys({pf.Name}))");
            sb.AppendLine($"{indent}        {pf.Name}[__key] = parameters[__i++] ?? throw new System.ArgumentNullException(nameof(parameters), \"Collection parameter is null.\");");
            sb.AppendLine($"{indent}}}");
        }

        if (pf.Condition is not null)
            sb.AppendLine("        }");
    }

    private static void EmitCollectionRegister(StringBuilder sb, ParameterFieldInfo pf)
    {
        if (pf.CollectionKind == ParameterCollectionKind.Direct)
        {
            if (pf.Optional || pf.Condition is not null)
                sb.AppendLine($"        if ({PresenceExpr(pf)}) AppendTrainableParameter({pf.Name}, {pf.Role});");
            else
                sb.AppendLine($"        AppendTrainableParameter({pf.Name}, {pf.Role});");
            return;
        }

        string values = pf.CollectionKind == ParameterCollectionKind.Keyed
            ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({pf.Name})"
            : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({pf.Name})";
        if (pf.Condition is not null)
        {
            sb.AppendLine($"        if ({pf.Condition})");
            sb.AppendLine("        {");
            sb.AppendLine($"            foreach (var __parameter in {values})");
            sb.AppendLine($"                AppendTrainableParameter(__parameter, {pf.Role});");
            sb.AppendLine("        }");
        }
        else
        {
            sb.AppendLine($"        foreach (var __parameter in {values})");
            sb.AppendLine($"            AppendTrainableParameter(__parameter, {pf.Role});");
        }
    }

    private static string PresenceExpr(ParameterFieldInfo pf)
    {
        string tensorPresent = pf.Nullable
            ? $"{pf.Name} is not null && {pf.Name}.Length > 0"
            : $"{pf.Name}.Length > 0";
        return pf.Condition is null ? tensorPresent : $"({pf.Condition}) && ({tensorPresent})";
    }

    private enum ParameterCollectionKind
    {
        Direct,
        Array,
        Indexed,
        Keyed,
    }

    private record struct ParameterFieldInfo(
        string Name,
        string Role,
        int Order,
        int DeclIndex = 0,
        string? TypeName = null,
        bool Optional = false,
        bool Nullable = false,
        string? Shape = null,
        ParameterCollectionKind CollectionKind = ParameterCollectionKind.Direct,
        string? Condition = null);
    private record struct GradientFieldInfo(string Name, bool IsNullable);
    private record struct SubLayerFieldInfo(string Name, bool IsNullable, bool IsCollection, string? InputShape = null);
}
