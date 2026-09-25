using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

namespace AiDotNet.Generators;

/// <summary>
/// Proves the narrow case where a layer's initializer cannot construct its optional children.
/// Unknown paths retain ordinary initialization; this is not whole-program side-effect analysis.
/// </summary>
internal static class LayerStructureInitializationAnalysis
{
    internal static bool IsChildIndependent(
        Compilation compilation, INamedTypeSymbol owner, IReadOnlyList<IFieldSymbol> children)
    {
        var layerBase = compilation.GetTypeByMetadataName("AiDotNet.NeuralNetworks.Layers.LayerBase`1");
        if (layerBase is null || !SymbolEqualityComparer.Default.Equals(owner.BaseType?.OriginalDefinition, layerBase))
            return false;
        if (children.Count == 0 || children.Any(child => child.DeclaredAccessibility != Accessibility.Private
                || child.NullableAnnotation != NullableAnnotation.Annotated))
            return false;
        var initializer = owner.GetMembers("EnsureInitialized").OfType<IMethodSymbol>()
            .SingleOrDefault(method => !method.IsStatic && method.Parameters.Length == 0);
        if (initializer is null) return false;
        var proof = new Proof(compilation, owner, children);
        foreach (var constructor in owner.InstanceConstructors.Where(constructor => !constructor.IsImplicitlyDeclared))
            if (!proof.VisitMethod(constructor)) return false;
        return proof.VisitMethod(initializer);
    }

    private sealed class Proof : OperationWalker
    {
        private const int MaximumMethods = 128;
        private readonly Compilation _compilation;
        private readonly HashSet<ISymbol> _children;
        private readonly HashSet<ISymbol> _ownerTypes = new(SymbolEqualityComparer.Default);
        private readonly HashSet<ISymbol> _valueContracts = new(SymbolEqualityComparer.Default);
        private readonly HashSet<ISymbol> _exceptionContracts = new(SymbolEqualityComparer.Default);
        private readonly IPropertySymbol? _memberName;
        private readonly INamedTypeSymbol? _runtimeType;
        private readonly Dictionary<ISymbol, HashSet<ulong>> _visited = new(SymbolEqualityComparer.Default);
        private IMethodSymbol? _currentMethod;
        private ulong _knownNullParameters;
        private int _methodContexts;
        private bool _independent = true;

        internal Proof(Compilation compilation, INamedTypeSymbol owner, IEnumerable<IFieldSymbol> children)
        {
            _compilation = compilation;
            _memberName = compilation.GetTypeByMetadataName("System.Reflection.MemberInfo")?
                .GetMembers(nameof(System.Reflection.MemberInfo.Name)).OfType<IPropertySymbol>().SingleOrDefault();
            _runtimeType = compilation.GetTypeByMetadataName("System.Type");
            _children = new HashSet<ISymbol>(children.Select(child => child.OriginalDefinition), SymbolEqualityComparer.Default);
            for (var type = owner; type is not null && type.SpecialType != SpecialType.System_Object; type = type.BaseType)
                _ownerTypes.Add(type.OriginalDefinition);
            foreach (var contract in owner.AllInterfaces) _ownerTypes.Add(contract.OriginalDefinition);

            // Resolve contracts to symbols, never match a model/type name at runtime. These APIs
            // operate on numeric values, shapes, tensor storage or value-only initialization. An
            // arbitrary external receiver is NOT assumed to be independent of the layer graph.
            foreach (string metadataName in new[]
            {
                "AiDotNet.Tensors.LinearAlgebra.Tensor`1", "AiDotNet.Tensors.LinearAlgebra.TensorBase`1",
                "AiDotNet.Tensors.LinearAlgebra.Vector`1",
                "AiDotNet.Tensors.LinearAlgebra.Matrix`1", "AiDotNet.Tensors.LinearAlgebra.TensorShape",
                "AiDotNet.Tensors.LinearAlgebra.WeightRegistry", "AiDotNet.Tensors.Interfaces.INumericOperations`1",
                "AiDotNet.Tensors.Interfaces.IVectorizedOperations`1", "AiDotNet.Tensors.Engines.IEngine",
                "AiDotNet.Tensors.Engines.AiDotNetEngine", "AiDotNet.Tensors.Helpers.SimdRandom",
                "AiDotNet.Tensors.Helpers.MathHelper", "AiDotNet.Helpers.MathHelper",
                "AiDotNet.Initialization.IInitializationStrategy`1", "System.Math",
                "System.Threading.Interlocked", "System.Runtime.CompilerServices.Unsafe",
                "System.MemoryExtensions", "System.Buffers.ArrayPool`1", "System.Collections.Generic.List`1",
                "System.Span`1", "System.ReadOnlySpan`1", "System.Memory`1", "System.ReadOnlyMemory`1",
                "System.Nullable`1", "System.Array"
            })
            {
                if (compilation.GetTypeByMetadataName(metadataName) is { } contract)
                    _valueContracts.Add(contract.OriginalDefinition);
            }
            foreach (string metadataName in new[]
            {
                "System.ArgumentException", "System.ArgumentNullException", "System.ArgumentOutOfRangeException",
                "System.InvalidOperationException", "System.NotSupportedException", "System.OverflowException"
            })
                if (compilation.GetTypeByMetadataName(metadataName) is { } exception)
                    _exceptionContracts.Add(exception);
        }

        internal bool VisitMethod(IMethodSymbol method, ulong knownNullParameters = 0)
        {
            method = method.OriginalDefinition;
            if (!_independent) return false;
            if (_visited.TryGetValue(method, out var contexts) && contexts.Contains(knownNullParameters)) return true;
            if (_methodContexts++ >= MaximumMethods || method.IsAbstract || method.IsExtern || method.Parameters.Length > 64)
                return _independent = false;
            if (contexts is null) _visited.Add(method, contexts = new HashSet<ulong>());
            contexts.Add(knownNullParameters);

            // GetType cannot construct children and has no source body in the compilation.
            if (method.ContainingType.SpecialType == SpecialType.System_Object
                && method.Name == nameof(object.GetType) && method.Parameters.Length == 0)
                return true;

            if (method.DeclaringSyntaxReferences.Length != 1) return _independent = false;
            var syntax = method.DeclaringSyntaxReferences[0].GetSyntax();
            SyntaxNode? body = syntax switch
            {
                MethodDeclarationSyntax declaration => (SyntaxNode?)declaration.Body ?? declaration.ExpressionBody?.Expression,
                ConstructorDeclarationSyntax constructor => (SyntaxNode?)constructor.Body ?? constructor.ExpressionBody?.Expression,
                AccessorDeclarationSyntax accessor => (SyntaxNode?)accessor.Body ?? accessor.ExpressionBody?.Expression,
                PropertyDeclarationSyntax property => property.ExpressionBody?.Expression,
                ArrowExpressionClauseSyntax arrow => arrow.Expression,
                _ => null
            };
            // An auto-property's compiler-generated accessor only accesses its backing field.
            if (body is null && syntax is AccessorDeclarationSyntax { Body: null, ExpressionBody: null } accessorSyntax
                && accessorSyntax.Parent?.Parent is PropertyDeclarationSyntax { ExpressionBody: null })
                return true;
            if (body is null) return _independent = false;
            var operation = _compilation.GetSemanticModel(body.SyntaxTree).GetOperation(body);
            if (operation is null) return _independent = false;
            var previousMethod = _currentMethod;
            ulong previousNullParameters = _knownNullParameters;
            _currentMethod = method;
            _knownNullParameters = knownNullParameters;
            try { Visit(operation); }
            finally
            {
                _currentMethod = previousMethod;
                _knownNullParameters = previousNullParameters;
            }
            return _independent;
        }

        public override void Visit(IOperation? operation)
        {
            if (operation is null || !_independent) return;
            // A null argument is a call-site fact, not general dataflow. Any write or ref escape
            // invalidates this proof instead of incorrectly pruning a later conditional callback.
            if (operation is IAssignmentOperation assignment && ContainsKnownNullParameter(assignment.Target)
                || operation is ISimpleAssignmentOperation { IsRef: true }
                || operation is IVariableDeclaratorOperation { Symbol.RefKind: not RefKind.None }
                || operation is IArgumentOperation { Parameter.RefKind: not RefKind.None } argument
                    && ContainsKnownNullParameter(argument.Value))
                _independent = false;

            // Operators and conversions can execute arbitrary user code without an invocation node.
            // Only the same resolved value contracts may supply those methods; unknown/dynamic
            // operations and implicit disposal remain outside this narrowly bounded proof.
            IMethodSymbol? operatorMethod = operation switch
            {
                IBinaryOperation binary => binary.OperatorMethod,
                IUnaryOperation unary => unary.OperatorMethod,
                IConversionOperation conversion => conversion.OperatorMethod,
                IIncrementOrDecrementOperation increment => increment.OperatorMethod,
                ICompoundAssignmentOperation compound => compound.OperatorMethod,
                _ => null
            };
            if (operatorMethod is not null && !IsValueContract(operatorMethod.ContainingType)
                && !IsRuntimeTypeEquality(operation, operatorMethod)) _independent = false;
            if (operation is ICompoundAssignmentOperation compoundAssignment
                && (!IsValueConversion(compoundAssignment.InConversion) || !IsValueConversion(compoundAssignment.OutConversion)))
                _independent = false;
            if (operation.Type?.TypeKind == TypeKind.Dynamic
                || operation is IDynamicObjectCreationOperation or IDynamicIndexerAccessOperation
                    or IDynamicMemberReferenceOperation or ITypeParameterObjectCreationOperation
                    or IUsingOperation or IUsingDeclarationOperation or IAwaitOperation
                    or IForEachLoopOperation or IEventAssignmentOperation or IEventReferenceOperation
                    or ISpreadOperation or IInterpolatedStringHandlerCreationOperation)
                _independent = false;
            if (operation is ICollectionExpressionOperation { Type: not IArrayTypeSymbol }) _independent = false;
            if (operation is IInterpolationOperation interpolation && !IsPrimitiveFormatting(interpolation.Expression.Type))
                _independent = false;
            if (_independent) base.Visit(operation);
        }

        private static bool IsPrimitiveFormatting(ITypeSymbol? type) =>
            type is not null && (type.TypeKind == TypeKind.Enum || IsPrimitiveValueType(type.SpecialType));

        private static bool IsPrimitiveValueType(SpecialType type) => type is
            SpecialType.System_Boolean or SpecialType.System_Char or SpecialType.System_String
            or SpecialType.System_SByte or SpecialType.System_Byte or SpecialType.System_Int16
            or SpecialType.System_UInt16 or SpecialType.System_Int32 or SpecialType.System_UInt32
            or SpecialType.System_Int64 or SpecialType.System_UInt64 or SpecialType.System_IntPtr
            or SpecialType.System_UIntPtr or SpecialType.System_Single or SpecialType.System_Double
            or SpecialType.System_Decimal or SpecialType.System_Void;

        private bool IsValueConversion(CommonConversion conversion) =>
            conversion.MethodSymbol is null || IsValueContract(conversion.MethodSymbol.ContainingType);

        private bool IsRuntimeTypeEquality(IOperation operation, IMethodSymbol method) =>
            SymbolEqualityComparer.Default.Equals(method.ContainingType, _runtimeType)
            && operation is IBinaryOperation
            {
                OperatorKind: BinaryOperatorKind.Equals or BinaryOperatorKind.NotEquals,
                LeftOperand: ITypeOfOperation,
                RightOperand: ITypeOfOperation
            };

        private bool ContainsKnownNullParameter(IOperation operation) =>
            operation is IParameterReferenceOperation && IsKnownNull(operation)
            || operation.ChildOperations.Any(ContainsKnownNullParameter);

        public override void VisitObjectCreation(IObjectCreationOperation operation)
        {
            if (operation.Constructor is not { } constructor
                || !IsValueContract(constructor.ContainingType) && !_exceptionContracts.Contains(constructor.ContainingType)
                || !ArgumentsAreValues(operation.Arguments))
                _independent = false;
            base.VisitObjectCreation(operation);
        }

        public override void VisitFieldReference(IFieldReferenceOperation operation)
        {
            // Even reading a child is rejected: it could be aliased into a later mutation.
            if (_children.Contains(operation.Field.OriginalDefinition)) _independent = false;
            base.VisitFieldReference(operation);
        }

        public override void VisitInvocation(IInvocationOperation operation)
        {
            var method = operation.TargetMethod;
            if (method.MethodKind is MethodKind.DelegateInvoke or MethodKind.LocalFunction)
            {
                _independent = false;
                return;
            }
            if (operation.Instance is IInstanceReferenceOperation { ReferenceKind: InstanceReferenceKind.ContainingTypeInstance })
            {
                bool explicitBase = operation.Syntax is InvocationExpressionSyntax
                    { Expression: MemberAccessExpressionSyntax { Expression: BaseExpressionSyntax } };
                if (!explicitBase && !method.IsSealed && (method.IsVirtual || method.IsOverride || method.IsAbstract))
                    _independent = false;
                else
                    VisitMethod(method, NullArguments(operation));
            }
            else if (method.IsStatic && _ownerTypes.Contains(method.ContainingType.OriginalDefinition))
                VisitMethod(method, NullArguments(operation));
            else if (method.ContainingType.SpecialType == SpecialType.System_Object
                && method.IsStatic && method.Name == nameof(object.ReferenceEquals))
            {
                // Object identity is a framework intrinsic, not a callback into either operand.
            }
            else if (!IsValueContract(method.ContainingType) || !ArgumentsAreValues(operation.Arguments))
                _independent = false;
            // Calls on external tensors/strategies operate only on their typed arguments. A strategy
            // secretly capturing the owner to build its graph is not a supported structure contract.
            // Owner/child/delegate arguments are rejected by the ordinary descendant walk below.
            base.VisitInvocation(operation);
        }

        public override void VisitPropertyReference(IPropertyReferenceOperation operation)
        {
            if (operation.Instance is IInstanceReferenceOperation { ReferenceKind: InstanceReferenceKind.ContainingTypeInstance }
                || operation.Property.IsStatic && _ownerTypes.Contains(operation.Property.ContainingType.OriginalDefinition))
            {
                var property = operation.Property;
                if (!property.IsSealed && (property.IsVirtual || property.IsOverride || property.IsAbstract))
                    _independent = false;
                else
                {
                    if (property.GetMethod is { } getter) VisitMethod(getter);
                    // Inspect both accessors even for a read: ++, deconstruction and ref patterns
                    // must never hide a structural setter behind a non-assignment parent node.
                    if (property.SetMethod is { } setter) VisitMethod(setter);
                }
            }
            else if (!IsRuntimeTypeName(operation) && !IsValueContract(operation.Property.ContainingType))
                _independent = false;
            base.VisitPropertyReference(operation);
        }

        private bool IsRuntimeTypeName(IPropertyReferenceOperation operation) =>
            SymbolEqualityComparer.Default.Equals(operation.Property.OriginalDefinition, _memberName)
            && operation.Instance is IInvocationOperation
            {
                TargetMethod.ContainingType.SpecialType: SpecialType.System_Object,
                TargetMethod.Name: nameof(object.GetType),
                Arguments.Length: 0
            };

        private bool IsValueContract(ITypeSymbol? type)
        {
            if (type is null || type.TypeKind == TypeKind.Error) return false;
            if (_ownerTypes.Contains(type.OriginalDefinition)) return false;
            if (type is IArrayTypeSymbol array) return IsValueContract(array.ElementType);
            if (type.TypeKind is TypeKind.Enum or TypeKind.TypeParameter) return true;
            if (IsPrimitiveValueType(type.SpecialType)) return true;
            return type is INamedTypeSymbol named && _valueContracts.Contains(named.OriginalDefinition)
                && named.TypeArguments.All(IsValueContract);
        }

        private bool ArgumentsAreValues(IEnumerable<IArgumentOperation> arguments)
        {
            foreach (var argument in arguments)
            {
                if (IsKnownNull(argument.Value)) continue;
                IOperation value = argument.Value;
                while (value is IConversionOperation conversion) value = conversion.Operand;
                if (!IsValueContract(value.Type)) return false;
            }
            return true;
        }

        private ulong NullArguments(IInvocationOperation operation)
        {
            ulong mask = 0;
            foreach (var argument in operation.Arguments)
                if (argument.Parameter is { Ordinal: < 64 } parameter && IsKnownNull(argument.Value))
                    mask |= 1UL << parameter.Ordinal;
            return mask;
        }

        private bool IsKnownNull(IOperation operation)
        {
            if (operation.ConstantValue is { HasValue: true, Value: null }) return true;
            if (operation is IConversionOperation conversion) return IsKnownNull(conversion.Operand);
            return operation is IParameterReferenceOperation reference && reference.Parameter.Ordinal < 64
                && SymbolEqualityComparer.Default.Equals(reference.Parameter.ContainingSymbol.OriginalDefinition, _currentMethod)
                && (_knownNullParameters & (1UL << reference.Parameter.Ordinal)) != 0;
        }

        public override void VisitConditionalAccess(IConditionalAccessOperation operation)
        {
            // The shared tensor allocator's optional callback is null at this call site. Track
            // that explicit/default constant per method context, never suppress unknown callbacks.
            if (IsKnownNull(operation.Operation)) Visit(operation.Operation);
            else base.VisitConditionalAccess(operation);
        }

        public override void VisitInstanceReference(IInstanceReferenceOperation operation)
        {
            if (operation.ReferenceKind == InstanceReferenceKind.ContainingTypeInstance)
            {
                bool receiver = operation.Parent switch
                {
                    IInvocationOperation call => call.Instance == operation,
                    IFieldReferenceOperation field => field.Instance == operation,
                    IPropertyReferenceOperation property => property.Instance == operation,
                    _ => false
                };
                if (!receiver) _independent = false;
            }
            base.VisitInstanceReference(operation);
        }

        public override void VisitAnonymousFunction(IAnonymousFunctionOperation operation) => _independent = false;
        public override void VisitDelegateCreation(IDelegateCreationOperation operation) => _independent = false;
        public override void VisitDynamicInvocation(IDynamicInvocationOperation operation) => _independent = false;
        public override void VisitFunctionPointerInvocation(IFunctionPointerInvocationOperation operation) => _independent = false;
        public override void VisitInvalid(IInvalidOperation operation) => _independent = false;
    }
}
