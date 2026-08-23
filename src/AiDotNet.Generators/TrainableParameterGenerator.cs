using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
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

    private static readonly DiagnosticDescriptor DeclaredShapesSuppressed = new(
        "AIDN095",
        "Declared parameter shapes suppressed by a dynamic registration",
        "'{0}' registers a trainable parameter the generator cannot map to a field, so its runtime registry becomes the only source of truth and the [TrainableParameter(Shape = ...)] declarations on {1} are NOT emitted. Restore and copy-on-write clone cannot validate shapes for this layer. Register the field itself, or assign the local to its field in the same method.",
        "AiDotNet.ParameterAutomation",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A layer that declares parameter shapes but also registers an unmappable tensor silently loses DeclaredParameterShapes(). "
            + "Restore then has nothing to validate against and a clone can keep freshly initialized weights instead of trained ones, "
            + "which is invisible at compile time and shows up only as a weight-drift failure much later.");

    private static readonly DiagnosticDescriptor UnguardableDeclaredShapeAxis = new(
        "AIDN098",
        "Declared parameter axis cannot be proven resolved",
        "'{0}' declares the parameter axis '{1}', which the generator cannot trace to a dimension it can guard. A lazy layer carries -1 until its first forward, and arithmetic turns that into a plausible number ('-1 / groups' is 0), so this axis can be declared as a real size before it is known. Write the axis over fields or auto-properties the generator can follow, or resolve the dimension before it is declared.",
        "AiDotNet.ParameterAutomation",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "The -1 lazy sentinel is a value, and values do not survive arithmetic. The declaration's own scan rejects an axis "
            + "that came out negative, which catches a sentinel that was copied and misses one that was divided or multiplied. When the "
            + "generator cannot reach the roots an axis is computed from, it cannot emit the readiness guard either, and the layer can "
            + "publish a shape it has no way to know -- which then rejects a correct checkpoint as non-conforming.");

    private static readonly DiagnosticDescriptor NonPartialTrainableParameter = new(
        "AIDN099",
        "[TrainableParameter] on a non-partial class does nothing",
        "'{0}' declares [TrainableParameter] on {1} but is not partial, so this generator cannot emit into it and the declaration has NO effect. The layer gets no SetTrainableParameters, no DeclaredParameterTensors, and no restore path, so a checkpoint holding its trained weights is silently discarded. Add the partial modifier.",
        "AiDotNet.ParameterAutomation",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Generated parameter automation is emitted as a second partial declaration of the same class, so a non-partial "
            + "class is invisible to the generator's syntax predicate. Nothing fails at compile time -- the attribute is simply inert -- "
            + "and the layer then falls through to fresh initialization on every restore, which surfaces only as a weight-drift or "
            + "round-trip failure far from the declaration. SVTRThinPlateSplineLayer sat in exactly this state.");

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

        // The predicate above admits only partial classes, which is correct for EMISSION and is also
        // why a missing partial is silent: the class that needs the generator most simply never
        // reaches it. This second pass exists to make that state loud. It emits no source.
        var nonPartialDeclarations = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds &&
                    !cds.Modifiers.Any(m => m.Text == "partial") &&
                    cds.Members.OfType<FieldDeclarationSyntax>().Any(static f =>
                        f.AttributeLists.SelectMany(static al => al.Attributes).Any(static a =>
                            IsTrainableParameterAttributeName(a.Name.ToString()))),
                transform: static (ctx, _) => (ClassDeclarationSyntax)ctx.Node)
            .Where(static c => c is not null);

        context.RegisterSourceOutput(nonPartialDeclarations, static (spc, cds) =>
        {
            var offenders = cds.Members.OfType<FieldDeclarationSyntax>()
                .Where(static f => f.AttributeLists.SelectMany(static al => al.Attributes)
                    .Any(static a => IsTrainableParameterAttributeName(a.Name.ToString())))
                .SelectMany(static f => f.Declaration.Variables.Select(static v => v.Identifier.Text))
                .ToList();
            if (offenders.Count == 0) return;

            spc.ReportDiagnostic(Diagnostic.Create(
                NonPartialTrainableParameter,
                cds.Identifier.GetLocation(),
                cds.Identifier.Text,
                string.Join(", ", offenders)));
        });
    }

    /// <summary>
    /// Matches the attribute as written in source, where the <c>Attribute</c> suffix and any
    /// namespace qualification are both optional.
    /// </summary>
    private static bool IsTrainableParameterAttributeName(string name)
    {
        int lastDot = name.LastIndexOf('.');
        if (lastDot >= 0) name = name.Substring(lastDot + 1);
        return name == "TrainableParameter" || name == "TrainableParameterAttribute";
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
            bool useConventionalTensorEnumerator =
                HasUnclassifiedConventionalTensorEnumerator(compilation, classSymbol);
            bool useRuntimeParameterRegistry = HasAnyUnmappableRegistration(compilation, classSymbol)
                || useConventionalTensorEnumerator;

            // Skip if already processed (multiple partial files)
            var fullName = classSymbol.ToDisplayString();
            if (!processedClasses.Add(fullName)) continue;

            // Collect [TrainableParameter] fields
            var paramFields = new List<ParameterFieldInfo>();
            var gradientFields = new Dictionary<string, GradientFieldInfo>();
            var subLayerFields = new List<SubLayerFieldInfo>();

            var bufferFields = new List<(string Field, string Name, string Role, string StateRole, bool InputSized, bool ReadOnly)>();

            foreach (var member in classSymbol.GetMembers())
            {
                if (member is not IFieldSymbol field) continue;

                // COMPILER-GENERATED BACKING FIELDS ARE NOT MEMBERS THE AUTHOR WROTE. An auto-property
                // is backed by a field literally named `<Prop>k__BackingField`, which is not a legal
                // C# identifier, so emitting it produced source that could not compile at all
                // ("Invalid expression term '<'"). The property is the member; its backing store is an
                // implementation detail of the language.
                //
                // This generator already filters them correctly elsewhere -- the guard existed and this
                // loop simply never reached it, which is why the defect stayed invisible until a class
                // holding auto-properties was first made partial.
                if (field.IsImplicitlyDeclared) continue;

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
                    string? lowPrecisionBacking = null;

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
                        else if (namedArg.Key == "LowPrecisionBacking" && namedArg.Value.Value is string backingVal)
                            lowPrecisionBacking = backingVal;
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
                            Shape: shape, Condition: condition,
                            LowPrecisionBacking: lowPrecisionBacking,
                            IsReadOnly: field.IsReadOnly));
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
                            Shape: shape, CollectionKind: collectionKind, Condition: condition,
                            LowPrecisionBacking: lowPrecisionBacking));
                    }
                }

                // Collect every declared persistent non-optimizer role through the same base
                // buffer mechanism. The manifest retains whether it is fitted, frozen, or a true
                // auxiliary buffer; the optimizer view excludes all three.
                // Marking alone is not enough -- without emitting RegisterBuffer the tensors leave
                // the trainable set and join nothing, disappearing from ParameterCount and the flat
                // vector entirely. ReservoirLayer proved it: "Expected 320 parameters, got 0".
                bool hasRegisteredBufferDeclaration = TryGetRegisteredBufferDeclaration(
                    classSymbol, field.Name, out string registeredBufferName, out string registeredBufferRole);
                if (!field.IsStatic && IsTensorType(field.Type)
                    && (classification.Kind is ParameterMemberSemanticModel.Kind.Fitted
                        or ParameterMemberSemanticModel.Kind.Frozen
                        or ParameterMemberSemanticModel.Kind.Buffer
                        || hasRegisteredBufferDeclaration))
                {
                    var bufRole = "PersistentTensorRole.Constant";
                    var bufName = hasRegisteredBufferDeclaration
                        ? registeredBufferName
                        : field.Name.TrimStart('_');
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
                    else if (hasRegisteredBufferDeclaration)
                    {
                        // RegisterBuffer is itself an explicit semantic declaration. Preserve its
                        // role in the generated early registration so a lazy placeholder is not
                        // first published as Constant and later rejected when the layer replaces it
                        // under the author's Weights role.
                        bufRole = registeredBufferRole;
                    }
                    // [FittedParameter(InputSized = true)] separates "persist this" from "count
                    // this". A member whose extent comes from the caller's DATA cannot be part of
                    // the flat vector: its width would change under a forward pass, and every
                    // count-versus-vector contract in the base is written against a width that
                    // only construction can move. It still registers as a buffer below, which is
                    // what serializes and deep-copies it by name.
                    bool inputSized = false;
                    var fittedAttr = field.GetAttributes().FirstOrDefault(a =>
                        a.AttributeClass?.ToDisplayString()
                            == ParameterMemberSemanticModel.FittedAttribute);
                    if (fittedAttr is not null)
                    {
                        foreach (var na in fittedAttr.NamedArguments)
                        {
                            if (na.Key == "InputSized" && na.Value.Value is bool flag)
                                inputSized = flag;
                        }
                    }

                    string stateRole = classification.Kind switch
                    {
                        // InputSized fitted state registers under its own role. The base sweep that
                        // declares every registered buffer keys off exactly this value to leave it
                        // out of the component list, so the role -- not the generated declaration
                        // alone -- is what keeps a caller-sized tensor out of the parameter vector.
                        ParameterMemberSemanticModel.Kind.Fitted when inputSized =>
                            "global::AiDotNet.Models.Parameters.ParameterSlotRole.InputSizedState",
                        ParameterMemberSemanticModel.Kind.Fitted =>
                            "global::AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState",
                        ParameterMemberSemanticModel.Kind.Frozen =>
                            "global::AiDotNet.Models.Parameters.ParameterSlotRole.Frozen",
                        _ => "global::AiDotNet.Models.Parameters.ParameterSlotRole.Buffer"
                    };
                    bufferFields.Add((field.Name, bufName, bufRole, stateRole, inputSized, field.IsReadOnly));
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
                // A layer field is owned by default, but [ParameterAlias] explicitly says that the
                // same child is already owned through another member. Registering both names makes
                // the allocation-free manifest count the child's parameters twice even though the
                // runtime registry correctly deduplicates the shared reference.
                if (IsLayerType(field.Type) && !field.IsStatic
                    && classification.Kind is not (ParameterMemberSemanticModel.Kind.Alias
                        or ParameterMemberSemanticModel.Kind.Scratch
                        or ParameterMemberSemanticModel.Kind.External))
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
                else if (!field.IsStatic && IsLayerCollectionType(field.Type)
                         && !IsAliasLayerCollection(compilation, classSymbol, field)
                         && classification.Kind is not (ParameterMemberSemanticModel.Kind.Alias
                             or ParameterMemberSemanticModel.Kind.Scratch
                             or ParameterMemberSemanticModel.Kind.External))
                {
                    var isNullable = field.NullableAnnotation == NullableAnnotation.Annotated ||
                                     field.Type.NullableAnnotation == NullableAnnotation.Annotated;
                    // The declaration is read here too. It was only read on the single-layer branch,
                    // so a collection carrying [SubLayerInput] recorded no shape and was silently
                    // dropped from DeclaredSubLayerShapes -- the attribute compiled, appeared to
                    // apply, and did nothing.
                    var collectionShape = field.GetAttributes()
                        .FirstOrDefault(a => a.AttributeClass?.Name == "SubLayerInputAttribute")
                        ?.ConstructorArguments.FirstOrDefault().Value as string;
                    subLayerFields.Add(new SubLayerFieldInfo(
                        field.Name, isNullable, IsCollection: true, InputShape: collectionShape));
                }
            }

            // A lazy layer often declares its tensor shapes at the allocation boundary rather
            // than repeating them in attribute strings. Recover the simple, field-only collection
            // expressions used there so the generated restore contract can split a flat checkpoint
            // before the first real input arrives. Only adopt the inferred set when at least one
            // formula binds a canonical input axis; this keeps ordinary construction-sized tensors
            // on their existing zero-overhead path.
            var inferredShapes = new string?[paramFields.Count];
            bool hasInferredInputBinding = false;
            for (int i = 0; i < paramFields.Count; i++)
            {
                if (paramFields[i].CollectionKind != ParameterCollectionKind.Direct
                    || !string.IsNullOrWhiteSpace(paramFields[i].Shape))
                    continue;
                inferredShapes[i] = TryInferAllocationShape(
                    compilation, classSymbol, paramFields[i].Name);
                if (inferredShapes[i]?.IndexOf("InputShape[", System.StringComparison.Ordinal) >= 0)
                    hasInferredInputBinding = true;
            }
            if (hasInferredInputBinding
                && Enumerable.Range(0, paramFields.Count).All(index =>
                    !string.IsNullOrWhiteSpace(paramFields[index].Shape)
                    || !string.IsNullOrWhiteSpace(inferredShapes[index])))
            {
                for (int i = 0; i < paramFields.Count; i++)
                    if (string.IsNullOrWhiteSpace(paramFields[i].Shape))
                        paramFields[i] = paramFields[i] with { Shape = inferredShapes[i] };
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

            // Falling back to the runtime registry is legitimate -- a layer really can register
            // tensors the generator cannot name. Doing it SILENTLY to a layer that also declares
            // shapes is not: those declarations simply vanish from the generated surface, and the
            // first symptom is a clone quietly keeping untrained weights. Say so at compile time.
            var suppressedShapeFields = paramFields
                .Where(p => p.CollectionKind == ParameterCollectionKind.Direct
                            && !string.IsNullOrWhiteSpace(p.Shape))
                .Select(p => p.Name)
                .ToList();
            if (useRuntimeParameterRegistry && suppressedShapeFields.Count > 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    DeclaredShapesSuppressed,
                    classSymbol.Locations.FirstOrDefault(),
                    classSymbol.Name,
                    string.Join(", ", suppressedShapeFields)));
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
                    // Registration order is the live optimizer/tape contract. A partially migrated
                    // layer can annotate most fields and still register all of them imperatively;
                    // appending only the unannotated discoveries after every attributed field made
                    // the generated getter/setter use a different order from the runtime registry.
                    // MambaBlock exposed the consequence: A_log/D were registered before the output
                    // projection but generated after it, so clone adoption paired equal-sized tensors
                    // with the wrong semantic slots while all aggregate counts still agreed.
                    //
                    // Rebuild the local declaration order from the explicit registration sequence,
                    // retaining attribute metadata (shape/optional/backing) for matching fields, then
                    // append genuinely declaration-only parameters. This makes one stable order drive
                    // optimizer collection, flat persistence, copy-on-write adoption and manifests.
                    var declaredByName = paramFields.ToDictionary(
                        parameter => parameter.Name,
                        System.StringComparer.Ordinal);
                    var orderedFields = new List<ParameterFieldInfo>(paramFields.Count);
                    var seen = new HashSet<string>(System.StringComparer.Ordinal);
                    int nextOrder = 0;
                    foreach (var (fieldName, role) in registeredFields)
                    {
                        if (!seen.Add(fieldName)) continue;

                        if (declaredByName.TryGetValue(fieldName, out var declared))
                        {
                            orderedFields.Add(declared with { Order = nextOrder++ });
                            continue;
                        }

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
                            orderedFields.Add(new ParameterFieldInfo(
                                matchingField.Name, role, nextOrder++, DeclIndex: 0,
                                TypeName: matchingField.Type.ToDisplayString(),
                                Optional: nullable, Nullable: nullable));
                        }
                    }

                    foreach (var declared in paramFields)
                    {
                        if (seen.Add(declared.Name))
                            orderedFields.Add(declared with { Order = nextOrder++ });
                    }

                    paramFields = orderedFields;
                }
            }

            // Imperative registration can add fields that were not present during the first
            // allocation-shape pass above. Complete the formulas after that merge so a deferred
            // restore solves against the same full tensor set that Get/SetParameters folds.
            inferredShapes = new string?[paramFields.Count];
            hasInferredInputBinding = false;
            for (int i = 0; i < paramFields.Count; i++)
            {
                if (paramFields[i].CollectionKind != ParameterCollectionKind.Direct
                    || !string.IsNullOrWhiteSpace(paramFields[i].Shape))
                    continue;
                inferredShapes[i] = TryInferAllocationShape(
                    compilation, classSymbol, paramFields[i].Name);
                if (inferredShapes[i]?.IndexOf("InputShape[", System.StringComparison.Ordinal) >= 0)
                    hasInferredInputBinding = true;
            }
            if (hasInferredInputBinding
                && Enumerable.Range(0, paramFields.Count).All(index =>
                    !string.IsNullOrWhiteSpace(paramFields[index].Shape)
                    || !string.IsNullOrWhiteSpace(inferredShapes[index])))
            {
                for (int i = 0; i < paramFields.Count; i++)
                    if (string.IsNullOrWhiteSpace(paramFields[i].Shape))
                        paramFields[i] = paramFields[i] with { Shape = inferredShapes[i] };
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
            var unguardableAxes = new List<string>();
            var source = GenerateSource(
                classSymbol, paramFields, gradientFields, subLayerFields, bufferFields,
                useRuntimeParameterRegistry, useConventionalTensorEnumerator,
                emitParameterFreeContract, unguardableAxes);

            // A declared axis the generator could not trace back to a guardable dimension. Emitting
            // the declaration anyway is what let ConvolutionalLayer publish [8, 0, 3, 3] from an
            // unresolved InputDepth and then reject the correct [8, 1, 3, 3] a checkpoint handed it.
            foreach (var axis in unguardableAxes.Distinct(System.StringComparer.Ordinal))
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    UnguardableDeclaredShapeAxis,
                    classSymbol.Locations.FirstOrDefault(),
                    classSymbol.Name,
                    axis));
            }
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
        List<(string Field, string Name, string Role, string StateRole, bool InputSized, bool ReadOnly)> bufferFields,
        bool useRuntimeParameterRegistry,
        bool useConventionalTensorEnumerator,
        bool emitParameterFreeContract,
        ICollection<string>? unguardableAxes = null)
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine("    protected override bool IsDeclaredParameterFree => true;");
            sb.AppendLine();
        }

        // One inheritance-aware component manifest drives the public flat parameter surfaces.
        // Base declarations are appended first, then this class's fields in source declaration
        // order. This is what keeps a derived adapter's own tensors after the factors declared by
        // its base class without teaching the generator any model or adapter names.
        EmitOrderedParameterManifest(
            sb, classSymbol,
            useRuntimeParameterRegistry ? new List<ParameterFieldInfo>() : paramFields,
            subLayerFields, bufferFields);

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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine("    private void EnsureBuffersRegistered()");
            sb.AppendLine("    {");
            foreach (var bf in bufferFields)
            {
                sb.AppendLine($"        if ({bf.Field} is not null) RegisterBuffer({bf.Field}, \"{bf.Name}\", {bf.Role}, {bf.StateRole});");
            }
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine("    /// <inheritdoc />");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<(string Name, Tensor<{GetTypeParamName(classSymbol)}> Tensor)> GetRegisteredBuffers()");
            sb.AppendLine("    {");
            sb.AppendLine("        EnsureBuffersRegistered();");
            sb.AppendLine("        return base.GetRegisteredBuffers();");
            sb.AppendLine("    }");
            sb.AppendLine();

            sb.AppendLine("    /// <inheritdoc />");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine("    protected override bool CanRestoreBufferField(string name)");
            sb.AppendLine("    {");
            sb.AppendLine("        switch (name)");
            sb.AppendLine("        {");
            foreach (var bf in bufferFields)
            {
                if (bf.ReadOnly) continue;
                sb.AppendLine($"            case \"{EscapeStringLiteral(bf.Name)}\":");
                sb.AppendLine("                return true;");
            }
            sb.AppendLine("            default:");
            sb.AppendLine("                return base.CanRestoreBufferField(name);");
            sb.AppendLine("        }");
            sb.AppendLine("    }");
            sb.AppendLine();

            // Restoring a buffer means writing the FIELD, not just the registry: EnsureBuffersRegistered
            // reads each buffer out of its field, so a registration the field does not back is invisible
            // to the layer's own code. The name-to-field mapping is emitted because only the generator
            // has it; reflecting over it at runtime would turn a rename into a silent no-op.
            sb.AppendLine("    /// <inheritdoc />");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    protected override bool TryRestoreBufferField(string name, Tensor<{GetTypeParamName(classSymbol)}> tensor)");
            sb.AppendLine("    {");
            sb.AppendLine("        switch (name)");
            sb.AppendLine("        {");
            foreach (var bf in bufferFields)
            {
                // A readonly buffer is assigned once, by the constructor, so it is ALWAYS live and
                // the caller's write-through path restores it in place. Emitting a case for it
                // would not compile, and would answer a question the restore never has to ask.
                if (bf.ReadOnly) continue;

                sb.AppendLine($"            case \"{EscapeStringLiteral(bf.Name)}\":");
                sb.AppendLine($"                {bf.Field} = tensor;");
                // Register HERE, not in the caller. A bare RegisterBuffer(tensor, name) takes the
                // default state role, which silently promoted an input-sized slot back into the
                // parameter vector the moment a clone or a restore installed one.
                sb.AppendLine("                EnsureBuffersRegistered();");
                sb.AppendLine("                return true;");
            }
            sb.AppendLine("            default:");
            sb.AppendLine("                return base.TryRestoreBufferField(name, tensor);");
            sb.AppendLine("        }");
            sb.AppendLine("    }");
            sb.AppendLine();
        }


        // DeclaredSubLayerShapes — emitted from [SubLayerInput("...")] on the sub-layer fields.
        //
        // A composite's children do not all receive the composite's own input, and only the
        // composite knows which gets what. Declaring it on the field lets the generator supply that
        // fact to LayerBase.BringUpDeclaredSubLayers, so no composite implements the method.
        // Collections included. A declaration names ONE width, which is exactly right for a bank of
        // siblings that all read the same tensor -- an MoE's experts, for instance. Excluding them
        // left those children to chained sizing, which walks the registration order and hands each
        // expert whatever the PREVIOUS child emitted: MoEFeedForwardLayer registers its router
        // first, so every expert was built against the router's numExperts-wide output instead of
        // the hidden width, and a restore then rejected the saved weights outright.
        var shapedSubLayers = subLayerFields
            .Where(sl => !string.IsNullOrWhiteSpace(sl.InputShape))
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
            sb.AppendLine("    /// Empty while a REQUIRED declared child is still null or any axis is still negative: a");
            sb.AppendLine("    /// composite builds its children inside its initializer, so both are ordinary states before");
            sb.AppendLine("    /// that runs. Cached, because the initializer deliberately re-enters.");
            sb.AppendLine("    /// <para>");
            sb.AppendLine("    /// A NULLABLE declared child is skipped instead, because null is a configuration there rather");
            sb.AppendLine("    /// than a not-built-yet: a transformer block whose dropout rate is zero never constructs its");
            sb.AppendLine("    /// dropout layers. Treating that as \"declaration not ready\" abandoned the whole declaration");
            sb.AppendLine("    /// for the common configuration, so the composite fell back to chained sizing and its counted");
            sb.AppendLine("    /// and materialized surfaces disagreed again.");
            sb.AppendLine("    /// </para>");
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{subTuple}> DeclaredSubLayerShapes()");
            sb.AppendLine("    {");
            sb.AppendLine("        if (__declaredSubLayerShapes is not null) return __declaredSubLayerShapes;");
            foreach (var sl in shapedSubLayers.Where(sl => !sl.IsNullable && !sl.IsCollection))
            {
                sb.AppendLine($"        if ({sl.Name} is null) return System.Array.Empty<{subArray}>();");
            }
            sb.AppendLine($"        var __sub = new System.Collections.Generic.List<{subArray}>({shapedSubLayers.Count});");
            string tp = GetTypeParamName(classSymbol);
            foreach (var sl in shapedSubLayers)
            {
                var axes = string.Join(", ", sl.InputShape!.Split(',').Select(a => a.Trim()).Where(a => a.Length > 0));
                if (sl.IsCollection)
                {
                    // Every element gets the declared width. Elements are filtered by type because a
                    // collection may be declared as ILayer<T>, which carries no shape resolution.
                    sb.AppendLine($"        if ({sl.Name} is not null)");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            foreach (var __child in {sl.Name})");
                    sb.AppendLine($"                if (__child is LayerBase<{tp}> __element)");
                    sb.AppendLine($"                    __sub.Add((__element, ShapeOf({axes})));");
                    sb.AppendLine("        }");
                    continue;
                }

                string entry = $"__sub.Add(({sl.Name}, ShapeOf({axes})));";
                if (sl.IsNullable) sb.AppendLine($"        if ({sl.Name} is not null) {entry}");
                else sb.AppendLine($"        {entry}");
            }
            sb.AppendLine("        for (int __i = 0; __i < __sub.Count; __i++)");
            sb.AppendLine("        {");
            sb.AppendLine("            var __s = __sub[__i].Item2;");
            sb.AppendLine("            for (int __d = 0; __d < __s.Length; __d++)");
            // <= 0, not < 0. Every int field reads ZERO before the constructor assigns it, so a
            // declaration consulted mid-construction produced a zero-width shape that passed a
            // negative-only check and was then CACHED for the life of the layer. A width of zero is
            // never a real one, so treating it as "not ready yet" is correct either way.
            sb.AppendLine($"                if (__s[__d] <= 0) return System.Array.Empty<{subArray}>();");
            sb.AppendLine("        }");
            sb.AppendLine("        __declaredSubLayerShapes = __sub.ToArray();");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{tupleType}> DeclaredParameterShapes()");
            sb.AppendLine("    {");

            // Guard the SOURCES of the shape arithmetic before evaluating it. The scan further down
            // rejects an axis that came out negative, which catches a sentinel that was copied and
            // misses one that was divided: `-1 / Groups` is 0 for any Groups >= 2, and a declared 0
            // reads as a real axis. See CollectDeclaredShapeSentinelRoots for the full case.
            var sentinelRoots = new HashSet<string>(System.StringComparer.Ordinal);
            foreach (var pf in shapedFields)
            {
                foreach (string axis in pf.Shape!.Split(',')
                    .Select(a => a.Trim())
                    .Where(axis => axis.Length > 0 && axis != "*"))
                {
                    string trimmed = TryGetAdaptiveAxisBinding(axis, out string bound) ? bound : axis;

                    // Each axis gets its own walk AND its own `visited` set. The hazard verdict is
                    // per-axis, and `visited` short-circuits on re-entry: sharing it let the first
                    // axis consume an identifier so that every later axis reading the same one
                    // never reached the `member is null` branch and never recorded its unfollowable
                    // read. Two axes computing over the same unfollowable member would then report
                    // only the first, leaving the second unguarded AND unreported -- the precise
                    // failure this diagnostic exists to catch.
                    //
                    // Roots stay shared: any one unresolved root sinks the whole declaration, and
                    // the HashSet keeps the emitted guards unique.
                    var walk = new DeclaredAxisWalk();
                    var sentinelVisited = new HashSet<string>(System.StringComparer.Ordinal);
                    CollectDeclaredShapeSentinelRoots(classSymbol, trimmed, sentinelRoots, sentinelVisited, 0, walk);

                    // Computes something, and reads something we could not follow to a dimension.
                    // Either alone is fine -- a direct read of a -1 is caught by the scan below, and
                    // arithmetic over roots we CAN see is guarded above. Together they are the hole.
                    if (walk.IsHazard) unguardableAxes?.Add(trimmed);
                }
            }

            if (sentinelRoots.Count > 0)
            {
                sb.AppendLine("        // A dimension this declaration reads is still the -1 lazy sentinel, so the shapes below");
                sb.AppendLine("        // cannot be computed yet. Checked on the SOURCES rather than on the result: arithmetic");
                sb.AppendLine("        // launders the sentinel into a plausible non-negative number (-1 / groups == 0), and a");
                sb.AppendLine("        // laundered axis is indistinguishable from a real one by the time it reaches the scan.");
                foreach (string root in sentinelRoots.OrderBy(r => r, System.StringComparer.Ordinal))
                    sb.AppendLine($"        if ({root} < 0) return System.Array.Empty<{arrayType}>();");
                sb.AppendLine();
            }

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

        // DeclaredParameterTensors — the slots and roles alone, with NO shape computed.
        //
        // Gated on the parameter fields themselves and NOT on shapedFields, which is the whole
        // point. A [TrainableParameter] with no Shape = "..." argument still has a tensor and a
        // role; only its DIMENSIONS are unstated. Emitting this alongside the shape declaration
        // would have covered exactly the layers that least need it and skipped the ones that do:
        // SubpixelConvolutionalLayer and SVTRThinPlateSplineLayer declare a role and no shape, so
        // TryAdoptRestoredParameters saw an empty declaration, returned false, and fell through to
        // fresh initialization -- silently discarding a restore that was holding trained weights.
        // That is the "Output[0] differs after serialization roundtrip: original=0" failure, and
        // the 1,984 scalars SVTR loses across a round trip.
        var tensorFields = useRuntimeParameterRegistry
            ? new List<ParameterFieldInfo>()
            : paramFields.Where(p => p.CollectionKind == ParameterCollectionKind.Direct).ToList();
        if (tensorFields.Count > 0)
        {
            string tp2 = GetTypeParamName(classSymbol);
            string tensorTupleType = $"(Tensor<{tp2}>? Tensor, PersistentTensorRole Role)";
            sb.AppendLine("    /// <summary>The declared parameter slots and roles, without their shapes.</summary>");
            sb.AppendLine("    /// <remarks>Auto-generated — computes no axis, so an unresolved layer can still answer.</remarks>");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{tensorTupleType}> DeclaredParameterTensors()");
            sb.AppendLine("    {");
            sb.AppendLine($"        var __declared = new System.Collections.Generic.List<{tensorTupleType}>({tensorFields.Count});");
            foreach (var pf in tensorFields)
            {
                if (pf.Condition is not null)
                    sb.AppendLine($"        if ({pf.Condition})");
                sb.AppendLine($"            __declared.Add(({pf.Name}, {pf.Role}));");
            }
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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

        // A few legacy layers already maintain a deliberate, complete parameter order in a
        // parameterless GetAllTensors() helper. When that helper includes unclassified storage,
        // the annotated fields are necessarily only a subset; emitting the subset would hide
        // real weights from serialization and cloning. Reuse the author's explicit enumeration
        // and let LayerBase perform the identity-based field/container rebind.
        if (useConventionalTensorEnumerator)
        {
            string tensorType = $"Tensor<{GetTypeParamName(classSymbol)}>";
            sb.AppendLine("    /// <summary>Returns the complete convention-enumerated trainable tensor surface.</summary>");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<{tensorType}> GetTrainableParameters()");
            sb.AppendLine("        => GetAllTensors();");
            sb.AppendLine();
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    protected override System.Collections.Generic.IReadOnlyList<{tensorType}> GetTrainableParametersUnmaterialized()");
            sb.AppendLine("        => GetAllTensors();");
            sb.AppendLine();
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    public override void SetTrainableParameters(System.Collections.Generic.IReadOnlyList<{tensorType}> parameters)");
            sb.AppendLine("        => SetConventionEnumeratedTrainableParameters(GetAllTensors(), parameters);");
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
                sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
                sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    public override System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> GetTrainableParameters()");
            sb.AppendLine("    {");
            if (subLayerFields.Count > 0)
            {
                sb.AppendLine("        EnsureSubLayersRegistered();");
            }
            // A declared parameter shape can be complete even when unrelated data axes remain
            // deferred. Channel-pinned convolution factories are the canonical example: their
            // kernels are fully sized, but image/video extents correctly stay dynamic. Use the
            // shared readiness state rather than repeating the older whole-input-shape gate, so
            // the optimizer view and the flat parameter surface materialize at the same boundary.
            sb.AppendLine("        if (OwnParameterReadiness == AiDotNet.Models.Parameters.ParameterReadiness.ShapeResolvedUnmaterialized) EnsureInitializationSerialized();");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
            sb.AppendLine($"    public override void SetTrainableParameters(System.Collections.Generic.IReadOnlyList<Tensor<{GetTypeParamName(classSymbol)}>> parameters)");
            sb.AppendLine("    {");
            // Local helper: emit the assignment of a field from a parameters[idx]
            // slot, with the sparse-leaf downcast when the field's concrete type
            // is a Tensor<T> subclass. indexExpr may be a literal index (fixed path)
            // or a post-increment cursor (optional path).
            void EmitFieldAssign(ParameterFieldInfo pf, string indexExpr, string idxLabel)
            {
                // A READONLY field cannot be REASSIGNED outside its constructor, so the assignment below
                // does not compile for one (CS0191). Skipping such a field instead would be far worse:
                // these are genuine trainable weights, and dropping them from the surface is exactly the
                // silent weight loss this generator exists to prevent.
                //
                // So the VALUES are copied into the tensor the constructor already built. That is not
                // merely a workaround for readonly -- it is the better restore in general, because
                // replacing the tensor breaks the REFERENCE IDENTITY the tape and ParameterBuffer align
                // on, which is the hazard CifAlignmentLayer's own remarks describe. A shape
                // disagreement is a real disagreement and says so rather than silently resizing.
                if (pf.IsReadOnly)
                {
                    sb.AppendLine("        {");
                    sb.AppendLine($"            var __src = parameters[{indexExpr}] ?? throw new System.ArgumentNullException(nameof(parameters), \"Parameter at index {idxLabel} is null.\");");
                    sb.AppendLine($"            if (__src.Length != {pf.Name}.Length)");
                    sb.AppendLine($"                throw new System.ArgumentException($\"Parameter at index {idxLabel} has {{__src.Length}} values but '{pf.Name}' holds {{{pf.Name}.Length}}.\", nameof(parameters));");
                    sb.AppendLine($"            for (int __c = 0; __c < {pf.Name}.Length; __c++) {{ {pf.Name}[__c] = __src[__c]; }}");
                    sb.AppendLine("        }");
                    if (pf.LowPrecisionBacking is not null)
                        sb.AppendLine($"        {pf.LowPrecisionBacking} = null;");
                    return;
                }

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
                if (pf.LowPrecisionBacking is not null)
                    sb.AppendLine($"        {pf.LowPrecisionBacking} = null;");
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
                sb.AppendLine("            MarkTrainableParametersRebound();");
                sb.AppendLine("            return;");
                sb.AppendLine("        }");
                sb.AppendLine();
                sb.AppendLine("        ClearRegisteredParameters();");
                foreach (var pf in paramFields)
                    EmitCollectionRegister(sb, pf);
                sb.AppendLine("        MarkTrainableParametersRebound();");
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
                sb.AppendLine("            MarkTrainableParametersRebound();");
                sb.AppendLine("            return;");
                sb.AppendLine("        }");
                sb.AppendLine();
                sb.AppendLine("        ClearRegisteredParameters();");
                for (int i = 0; i < paramFields.Count; i++)
                {
                    sb.AppendLine($"        AppendTrainableParameter({paramFields[i].Name}, {paramFields[i].Role});");
                }
                sb.AppendLine("        MarkTrainableParametersRebound();");
            }
            sb.AppendLine("    }");
            sb.AppendLine();

            // ZeroGrad
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Clears all gradient fields discovered by convention ({paramName}Gradient).");
            sb.AppendLine("    /// Auto-generated from [TrainableParameter] field naming conventions.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
                sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
                sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
        List<(string Field, string Name, string Role, string StateRole, bool InputSized, bool ReadOnly)> bufferFields)
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
        sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
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
                // An input-sized member is registered (and therefore serialized and deep-copied)
                // but never declared as a component, so it contributes no width to the parameter
                // vector and ParameterCount stays a function of construction alone.
                if (buffer.InputSized)
                {
                    sb.AppendLine($"        // {buffer.Field}: [FittedParameter(InputSized = true)] -- persisted as a buffer, not a parameter.");
                    continue;
                }

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
            string declaration = parameter.LowPrecisionBacking is null
                ? $"DeclareTrainableParameter(components, {parameter.Name});"
                : $"DeclareTrainableParameter(components, {parameter.Name}, {parameter.LowPrecisionBacking});";
            if (parameter.Condition is not null)
                sb.AppendLine($"        if ({parameter.Condition}) {declaration}");
            else
                sb.AppendLine($"        {declaration}");
            return;
        }

        string values = parameter.CollectionKind == ParameterCollectionKind.Keyed
            ? $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({parameter.Name})"
            : $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({parameter.Name})";
        if (parameter.Condition is not null)
        {
            sb.AppendLine($"        if ({parameter.Condition})");
            sb.AppendLine($"            foreach (var __componentTensor in {values})");
            sb.AppendLine("                DeclareTrainableParameter(components, __componentTensor);");
        }
        else
        {
            sb.AppendLine($"        foreach (var __componentTensor in {values})");
            sb.AppendLine("            DeclareTrainableParameter(components, __componentTensor);");
        }
    }

    private static void EmitDeferredInputShapeInference(
        StringBuilder sb,
        INamedTypeSymbol classSymbol,
        List<ParameterFieldInfo> paramFields,
        List<SubLayerFieldInfo> subLayerFields,
        List<(string Field, string Name, string Role, string StateRole, bool InputSized, bool ReadOnly)> bufferFields)
    {
        // Buffers no longer disqualify the formula outright; the emitted method checks at RUNTIME
        // that none is live. Excluding them statically cost DenseLayer -- the most restored layer in
        // the library -- its inference, purely because it declares two optimizer-velocity buffers
        // that are null until training allocates them, and training cannot precede the resolution
        // this infers. A restore into a fresh deferred layer therefore always sees them empty.
        bool completeLocalFormula = paramFields.Count > 0
            && subLayerFields.Count == 0
            && paramFields.All(field => field.CollectionKind == ParameterCollectionKind.Direct
                && !field.Optional
                && field.Condition is null
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
        sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.TrainableParameterGenerator\", \"1.0.0\")]");
        sb.AppendLine("    protected override bool TryInferInputShapeFromParameterCount(int parameterCount, out int[] inputShape)");
        sb.AppendLine("    {");
        var countedBuffers = bufferFields.Where(buffer => !buffer.InputSized).ToList();
        if (countedBuffers.Count > 0)
        {
            sb.AppendLine("        // The formula below counts this layer's PARAMETERS. A live buffer is counted too, so");
            sb.AppendLine("        // inferring while one exists would solve the wrong equation and resolve to a wrong width.");
            sb.AppendLine("        // Input-sized buffers are absent from the vector, so they never perturb the equation");
            sb.AppendLine("        // and must NOT block inference -- a graph layer holds one for its whole life.");
            sb.AppendLine("        bool __noLiveBuffer = true;");
            foreach (var buffer in countedBuffers)
            {
                sb.AppendLine($"        if ({buffer.Field} is not null) __noLiveBuffer = false;");
            }
            sb.AppendLine("        if (__noLiveBuffer && Parameters.Length == 0)");
        }
        else
        {
            sb.AppendLine("        if (Parameters.Length == 0)");
        }
        sb.AppendLine("        {");

        foreach (int axis in referencedAxes)
        {
            sb.AppendLine($"            if (InputShape.Length > {axis} && InputShape[{axis}] <= 0)");
            sb.AppendLine("            {");
            sb.AppendLine("                bool __onlyUnknownAxis = true;");
            foreach (int otherAxis in referencedAxes.Where(candidate => candidate != axis))
                sb.AppendLine($"                if (InputShape.Length <= {otherAxis} || InputShape[{otherAxis}] <= 0) __onlyUnknownAxis = false;");
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

    /// <summary>
    /// Finds the highest-information <c>[axis, ...]</c> expression assigned to one tensor field.
    /// Locals and constructor parameters are rejected because generated members cannot read them;
    /// literals and layer fields/properties remain valid construction formulas.
    /// </summary>
    private static string? TryInferAllocationShape(
        Compilation compilation,
        INamedTypeSymbol classSymbol,
        string fieldName)
    {
        var field = classSymbol.GetMembers(fieldName).OfType<IFieldSymbol>().FirstOrDefault();
        if (field is null) return null;

        string? best = null;
        int bestScore = -1;
        foreach (var reference in classSymbol.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            var semanticModel = compilation.GetSemanticModel(declaration.SyntaxTree);
            foreach (var assignment in declaration.DescendantNodes().OfType<AssignmentExpressionSyntax>())
            {
                if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)
                    || !SymbolEqualityComparer.Default.Equals(
                        semanticModel.GetSymbolInfo(assignment.Left).Symbol, field))
                    continue;

                var collection = assignment.Right.DescendantNodesAndSelf()
                    .OfType<CollectionExpressionSyntax>()
                    .FirstOrDefault();
                if (collection is null || collection.Elements.Count == 0
                    || collection.Elements.Any(element => element is not ExpressionElementSyntax))
                    continue;

                var axes = collection.Elements.Cast<ExpressionElementSyntax>()
                    .Select(element => element.Expression)
                    .ToList();
                bool safe = true;
                foreach (var identifier in axes.SelectMany(axis =>
                             axis.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>()))
                {
                    ISymbol? symbol = semanticModel.GetSymbolInfo(identifier).Symbol;
                    if (symbol is IFieldSymbol fieldSymbol
                        && IsOnTypeHierarchy(fieldSymbol.ContainingType, classSymbol))
                        continue;
                    if (symbol is IPropertySymbol propertySymbol
                        && IsOnTypeHierarchy(propertySymbol.ContainingType, classSymbol))
                        continue;
                    if (symbol is ITypeSymbol) continue;
                    safe = false;
                    break;
                }
                if (!safe) continue;

                string rendered = string.Join(", ", axes.Select(axis => axis.ToString()));
                rendered = Regex.Replace(
                    rendered,
                    @"\b_?inputDepth\b",
                    "InputShape[0]",
                    RegexOptions.IgnoreCase);
                int score = axes.Sum(axis =>
                    axis.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>().Count() * 10
                    + axis.DescendantNodesAndSelf().OfType<LiteralExpressionSyntax>()
                        .Count(literal => literal.Token.ValueText != "0"));
                if (score <= bestScore) continue;
                best = rendered;
                bestScore = score;
            }
        }
        return best;
    }

    private static bool IsOnTypeHierarchy(INamedTypeSymbol? candidate, INamedTypeSymbol type)
    {
        for (var current = type; current is not null; current = current.BaseType)
            if (SymbolEqualityComparer.Default.Equals(current, candidate)) return true;
        return false;
    }

    private static bool TryGetRegisteredBufferDeclaration(
        INamedTypeSymbol owner,
        string fieldName,
        out string name,
        out string role)
    {
        foreach (var reference in owner.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            foreach (var invocation in declaration.DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                string? callName = invocation.Expression switch
                {
                    IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                    MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
                    _ => null
                };
                if (callName != "RegisterBuffer" || invocation.ArgumentList.Arguments.Count < 3)
                    continue;
                if (!invocation.ArgumentList.Arguments[0].Expression.DescendantNodesAndSelf()
                    .OfType<IdentifierNameSyntax>()
                    .Any(identifier => identifier.Identifier.ValueText == fieldName))
                    continue;

                string candidate = invocation.ArgumentList.Arguments[2].Expression.ToString();
                if (candidate.IndexOf("PersistentTensorRole.", System.StringComparison.Ordinal) < 0)
                    continue;
                var nameExpression = invocation.ArgumentList.Arguments[1].Expression;
                name = nameExpression is LiteralExpressionSyntax literal
                    && literal.IsKind(SyntaxKind.StringLiteralExpression)
                        ? literal.Token.ValueText
                        : fieldName.TrimStart('_');
                role = candidate;
                return true;
            }
        }
        name = string.Empty;
        role = string.Empty;
        return false;
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

    /// <summary>
    /// Every member a declared shape axis transitively reads, so the emitted declaration can be
    /// guarded on the sources of its arithmetic rather than only on the number that arithmetic
    /// produced.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The -1 lazy sentinel is a VALUE, and values do not survive arithmetic. The post-hoc scan the
    /// declaration already runs -- reject an axis that came out negative -- catches a sentinel that
    /// was copied, and misses one that was divided. ConvolutionalLayer is the case that proves it:
    /// </para>
    /// <code>
    /// InputDepth = -1;                                   // ctor: "not resolved yet"
    /// private int KernelInChannels => InputDepth / Groups;
    /// [TrainableParameter(Shape = "OutputDepth, KernelInChannels, KernelSize, KernelSize")]
    /// </code>
    /// <para>
    /// For a depthwise convolution Groups is 8, so KernelInChannels is <c>-1 / 8 == 0</c>. That is
    /// not negative, so the scan passes it, and the layer declares <c>[8, 0, 3, 3]</c> -- a shape it
    /// has no way to know and which happens to look concrete. A checkpoint then hands back the
    /// correct <c>[8, 1, 3, 3]</c> and TryAdoptRestoredParameters rejects the RIGHT tensor against a
    /// placeholder the layer should never have emitted:
    /// </para>
    /// <code>
    /// ConvolutionalLayer`1 parameters do not conform to the resolved shape.
    /// Expected weights [8, 0, 3, 3] and biases [8], but received weights [8, 1, 3, 3] and biases [8].
    /// </code>
    /// <para>
    /// Guarding the roots instead of the result closes the whole class: the question "can this layer
    /// answer yet" is asked of <c>InputDepth</c>, where the sentinel still exists, instead of of
    /// <c>KernelInChannels</c>, where it has already been laundered into a plausible number. Members
    /// that are always positive -- Groups, KernelSize, OutputDepth -- cost one non-negative
    /// comparison each and can never trip it.
    /// </para>
    /// </remarks>
    /// <summary>
    /// What the walk over one declared axis learned about whether that axis can launder a sentinel.
    /// </summary>
    /// <remarks>
    /// A DIRECT read cannot launder. If <c>InputShape[0]</c> is -1 then the axis is -1, and the
    /// scan the declaration already runs rejects it. Only ARITHMETIC destroys the sentinel, by
    /// mapping a negative onto a non-negative — <c>-1 / groups</c> is 0 for any groups above 1.
    /// So an axis is a hazard only when it both computes something AND reads something the
    /// generator could not follow to a guardable dimension. Flagging every unfollowable read would
    /// condemn <c>InputShape[0]</c>, which is both idiomatic here and already safe.
    /// </remarks>
    private sealed class DeclaredAxisWalk
    {
        public bool SawArithmetic;
        public bool SawUnfollowableRead;
        public bool IsHazard => SawArithmetic && SawUnfollowableRead;
    }

    /// <summary>True when an expression applies arithmetic, ignoring the <c>*(...)</c> wildcard syntax.</summary>
    private static bool ContainsArithmetic(string expression)
    {
        for (int i = 0; i < expression.Length; i++)
        {
            char c = expression[i];
            if (c is '+' or '-' or '/' or '%') return true;

            // '*' is arithmetic EXCEPT as the adaptive-axis marker, which is stripped before this
            // point when it wraps the whole axis but can still lead a bare '*'.
            if (c == '*' && i > 0) return true;
        }

        return false;
    }

    private static void CollectDeclaredShapeSentinelRoots(
        INamedTypeSymbol classSymbol,
        string axisExpression,
        HashSet<string> roots,
        HashSet<string> visited,
        int depth,
        DeclaredAxisWalk? walk = null)
    {
        if (walk is not null && ContainsArithmetic(axisExpression)) walk.SawArithmetic = true;
        // Four hops covers every declaration in the library and stops a property that reads itself,
        // directly or through a cycle, from recursing forever. `visited` already breaks true cycles;
        // this bounds pathological chains as well.
        if (depth > 4) return;

        foreach (string identifier in ExtractIdentifiers(axisExpression))
        {
            if (!visited.Add(identifier)) continue;

            ISymbol? member = null;
            for (INamedTypeSymbol? t = classSymbol; t is not null && member is null; t = t.BaseType)
            {
                member = t.GetMembers(identifier)
                          .FirstOrDefault(m => m is IFieldSymbol or IPropertySymbol);
            }

            // Not a member of this layer: a type name, a method, a `Math` qualifier or a literal.
            // Nothing to guard, and nothing to recurse into -- but if arithmetic is being applied
            // around it, the walk can no longer prove this axis is resolved.
            if (member is null)
            {
                if (walk is not null && !IsKnownSafeIdentifier(identifier)) walk.SawUnfollowableRead = true;
                continue;
            }

            ITypeSymbol? memberType = member switch
            {
                IFieldSymbol f => f.Type,
                IPropertySymbol p => p.Type,
                _ => null,
            };

            // Only integral dimensions can carry the sentinel. A bool condition or a tensor field
            // reached through an expression is not a dimension and must not become a guard.
            if (memberType is null ||
                (memberType.SpecialType != SpecialType.System_Int32 &&
                 memberType.SpecialType != SpecialType.System_Int64))
            {
                continue;
            }

            // A computed member is a conduit, not a root -- guarding it is exactly the mistake this
            // method exists to avoid. Follow it to whatever it reads and guard THAT. A member with
            // no body we can read (auto-property, plain field, or one declared in another assembly)
            // is where the recursion bottoms out and where the sentinel is still intact.
            string? body = TryGetComputedMemberBody(member);
            if (body is not null)
            {
                CollectDeclaredShapeSentinelRoots(classSymbol, body, roots, visited, depth + 1, walk);
                continue;
            }

            // Constants are fixed at compile time and cannot hold a runtime sentinel.
            if (member is IFieldSymbol { HasConstantValue: true }) continue;

            roots.Add(identifier);
        }
    }

    /// <summary>
    /// The expression a property or field computes, when the generator can see one: an expression
    /// body, a get-accessor that is a single return, or a field initializer.
    /// </summary>
    private static string? TryGetComputedMemberBody(ISymbol member)
    {
        foreach (var node in member.DeclaringSyntaxReferences.Select(r => r.GetSyntax()))
        {

            if (node is PropertyDeclarationSyntax property)
            {
                if (property.ExpressionBody is not null)
                    return property.ExpressionBody.Expression.ToString();

                var getter = property.AccessorList?.Accessors
                    .FirstOrDefault(a => a.IsKind(SyntaxKind.GetAccessorDeclaration));
                if (getter?.ExpressionBody is not null)
                    return getter.ExpressionBody.Expression.ToString();
                if (getter?.Body?.Statements.Count == 1 &&
                    getter.Body.Statements[0] is ReturnStatementSyntax { Expression: { } returned })
                {
                    return returned.ToString();
                }

                // An auto-property, or a getter with real control flow. Neither is a conduit we can
                // follow, so the property itself is the root.
                return null;
            }

            // A field whose initializer is a bare literal is a root, not a conduit -- following it
            // would add nothing and `int _n = 8;` has no members to guard.
            //
            // Unwrap unary +/- first. `private int _inputDepth = -1;` is a
            // PrefixUnaryExpressionSyntax wrapping the literal 1, NOT a LiteralExpressionSyntax, so
            // testing the outer node alone treated it as a conduit and returned "-1" as a body.
            // Recursing into "-1" finds no identifier, so no guard was emitted -- for the exact
            // sentinel this whole feature exists to catch, merely because the layer wrote it as a
            // field initializer instead of a constructor assignment.
            if (node is VariableDeclaratorSyntax { Initializer.Value: { } initializer } &&
                Unwrap(initializer) is not LiteralExpressionSyntax)
            {
                return initializer.ToString();
            }
        }

        return null;
    }

    /// <summary>
    /// Identifiers that appear inside declared axes but can never carry the lazy sentinel: language
    /// keywords, and the numeric helpers axes are routinely written over.
    /// </summary>
    private static bool IsKnownSafeIdentifier(string identifier) => identifier switch
    {
        "Math" or "Max" or "Min" or "Abs" or "Ceiling" or "Floor" or "Round" or "Pow" or "Sqrt" => true,
        "int" or "long" or "checked" or "unchecked" or "this" or "new" or "true" or "false" => true,
        _ => false,
    };

    /// <summary>
    /// Strips parentheses and unary +/- so a signed literal is recognised as the literal it is.
    /// </summary>
    private static ExpressionSyntax Unwrap(ExpressionSyntax expression)
    {
        while (true)
        {
            switch (expression)
            {
                case ParenthesizedExpressionSyntax parenthesized:
                    expression = parenthesized.Expression;
                    continue;
                case PrefixUnaryExpressionSyntax unary when unary.IsKind(SyntaxKind.UnaryMinusExpression)
                                                         || unary.IsKind(SyntaxKind.UnaryPlusExpression):
                    expression = unary.Operand;
                    continue;
                default:
                    return expression;
            }
        }
    }

    /// <summary>C# identifiers in an expression, in source order, without allocating a regex.</summary>
    private static IEnumerable<string> ExtractIdentifiers(string expression)
    {
        for (int i = 0; i < expression.Length;)
        {
            char c = expression[i];
            if (!char.IsLetter(c) && c != '_') { i++; continue; }

            int start = i;
            while (i < expression.Length && (char.IsLetterOrDigit(expression[i]) || expression[i] == '_')) i++;

            // Skip a member access tail: in `Math.Max(a, b)` the interesting names are `a` and `b`,
            // and `Max` is not a member of the layer anyway. Skipping it keeps `visited` honest.
            if (start > 0 && expression[start - 1] == '.') continue;

            yield return expression.Substring(start, i - start);
        }
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

    /// <summary>
    /// Recognizes an explicit, ordered <c>GetAllTensors()</c> convention only when it closes a
    /// real declaration gap. A helper whose referenced tensor fields are all already classified
    /// does not change generated ordering; one that deliberately includes unclassified storage is
    /// the authoritative compatibility surface for that legacy layer.
    /// </summary>
    private static bool HasUnclassifiedConventionalTensorEnumerator(
        Compilation compilation,
        INamedTypeSymbol classSymbol)
    {
        var method = classSymbol.GetMembers("GetAllTensors")
            .OfType<IMethodSymbol>()
            .FirstOrDefault(candidate =>
                !candidate.IsStatic
                && candidate.Parameters.Length == 0
                && candidate.ReturnType is IArrayTypeSymbol array
                && IsTensorOfLayerElement(array.ElementType, classSymbol));
        if (method is null) return false;

        foreach (var syntaxReference in method.DeclaringSyntaxReferences)
        {
            var syntax = syntaxReference.GetSyntax();
            var semanticModel = compilation.GetSemanticModel(syntax.SyntaxTree);
            foreach (var identifier in syntax.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>())
            {
                if (semanticModel.GetSymbolInfo(identifier).Symbol is not IFieldSymbol field
                    || !SymbolEqualityComparer.Default.Equals(field.ContainingType, classSymbol)
                    || !ParameterMemberSemanticModel.IsNumericStateStorage(field.Type))
                {
                    continue;
                }

                if (ParameterMemberSemanticModel.Classify(field).Kind
                    == ParameterMemberSemanticModel.Kind.Unclassified)
                {
                    return true;
                }
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
    /// Recognizes a collection that is only an alternate traversal of child fields already owned
    /// by the same layer. Publishing both the fields and the aggregate gives one object several
    /// manifest slots; runtime registration cannot make that compile-time declaration unambiguous.
    /// </summary>
    private static bool IsAliasLayerCollection(
        Compilation compilation,
        INamedTypeSymbol owner,
        IFieldSymbol collectionField)
    {
        foreach (var reference in owner.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            var semanticModel = compilation.GetSemanticModel(declaration.SyntaxTree);

            foreach (var assignment in declaration.DescendantNodes().OfType<AssignmentExpressionSyntax>())
            {
                if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)
                    || !SymbolEqualityComparer.Default.Equals(
                        semanticModel.GetSymbolInfo(assignment.Left).Symbol, collectionField)
                    || assignment.Right is not CollectionExpressionSyntax collection
                    || collection.Elements.Count == 0)
                {
                    continue;
                }

                bool aliasesOwnedFields = true;
                foreach (var element in collection.Elements)
                {
                    if (element is not ExpressionElementSyntax expressionElement
                        || semanticModel.GetSymbolInfo(expressionElement.Expression).Symbol
                            is not IFieldSymbol childField
                        || SymbolEqualityComparer.Default.Equals(childField, collectionField)
                        || !SymbolEqualityComparer.Default.Equals(childField.ContainingType, owner)
                        || !IsLayerType(childField.Type))
                    {
                        aliasesOwnedFields = false;
                        break;
                    }
                }

                if (aliasesOwnedFields) return true;
            }
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
                    or "RegisterParameterComponent"
                    or "RegisterSubLayer")
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

            // A local that this method then STORES INTO one of the class's own fields is fully
            // mappable -- the field is the tensor's home, the local is just the expression that
            // built it. Allocate-then-register-then-assign is the ordinary way to swap a lazily
            // sized weight (BatchNorm and LayerNorm both resize gamma/beta this way), and treating
            // it as dynamic made the generator fall back to the runtime registry, which SILENTLY
            // drops DeclaredParameterShapes and the whole declared-shape surface with it -- 180
            // generated lines down to 21 for LayerNormalizationLayer. Restore then has no declared
            // shape to validate against, so a clone quietly keeps its freshly initialized weights
            // instead of the trained ones.
            if (symbol is ILocalSymbol local && LocalIsStoredIntoOwnField(invocation, model, classSymbol, local))
                continue;

            return true;
        }
        return false;
    }

    /// <summary>
    /// True when <paramref name="local"/> is assigned to a field of <paramref name="classSymbol"/>
    /// somewhere in the member that contains <paramref name="invocation"/>.
    /// </summary>
    /// <remarks>
    /// Deliberately scoped to the containing member and to a direct <c>_field = local</c> store.
    /// Anything less direct -- a collection element, a conditional store, a hand-off to another
    /// method -- stays unmappable, because the generator could not then name a single field for
    /// the registration and the runtime registry really is the only complete source of truth.
    /// </remarks>
    private static bool LocalIsStoredIntoOwnField(
        SyntaxNode invocation, SemanticModel model, INamedTypeSymbol classSymbol, ILocalSymbol local)
    {
        SyntaxNode? member = invocation.FirstAncestorOrSelf<MemberDeclarationSyntax>();
        if (member is null) return false;

        foreach (var assignment in member.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)) continue;
            if (model.SyntaxTree != assignment.SyntaxTree) continue;

            if (model.GetSymbolInfo(assignment.Right).Symbol is not ILocalSymbol assignedFrom
                || !SymbolEqualityComparer.Default.Equals(assignedFrom, local))
                continue;

            if (model.GetSymbolInfo(assignment.Left).Symbol is IFieldSymbol target
                && SymbolEqualityComparer.Default.Equals(target.ContainingType, classSymbol))
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
        string? Condition = null,
        string? LowPrecisionBacking = null,
        bool IsReadOnly = false);
    private record struct GradientFieldInfo(string Name, bool IsNullable);
    private record struct SubLayerFieldInfo(string Name, bool IsNullable, bool IsCollection, string? InputShape = null);
}
