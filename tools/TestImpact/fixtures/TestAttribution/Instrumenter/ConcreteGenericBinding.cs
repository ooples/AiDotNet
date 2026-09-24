using Mono.Cecil;

// Call graphs deliberately erase generic instantiations for reachability. A
// runtime contract must not: T in an unrelated type/method is not the same T.
internal static class ConcreteGenericBinding
{
    // An explicit one-hop base binding. This does not guess an outer generic
    // context or equate unrelated parameters which happen to print as T.
    internal static MethodReference? BaseConstructor(MethodReference constructed, MethodReference baseCall)
    {
        try
        {
            if (!constructed.HasThis || constructed.ExplicitThis || constructed.HasGenericParameters || constructed.Name != ".ctor" ||
                constructed.CallingConvention != MethodCallingConvention.Default || constructed.Resolve() is not MethodDefinition derived ||
                !derived.IsConstructor || derived.IsStatic || constructed.DeclaringType is not GenericInstanceType concrete ||
                concrete.ElementType.Resolve() != derived.DeclaringType || concrete.GenericArguments.Count != derived.DeclaringType.GenericParameters.Count ||
                derived.DeclaringType.BaseType is not GenericInstanceType declared ||
                baseCall.DeclaringType is not GenericInstanceType invoked || invoked.ElementType.Resolve() != declared.ElementType.Resolve() ||
                invoked.GenericArguments.Count != declared.GenericArguments.Count ||
                !baseCall.HasThis || baseCall.ExplicitThis || baseCall.HasGenericParameters || baseCall.Name != ".ctor" ||
                baseCall.Parameters.Count != 0 || baseCall.CallingConvention != MethodCallingConvention.Default ||
                baseCall.Resolve() is not MethodDefinition target || !target.IsConstructor || target.IsStatic || target.Parameters.Count != 0 ||
                target.DeclaringType != declared.ElementType.Resolve() || declared.GenericArguments.Count != target.DeclaringType.GenericParameters.Count)
                return null;
            var closed = new GenericInstanceType(declared.ElementType);
            for (int index = 0; index < declared.GenericArguments.Count; index++)
            {
                TypeReference? declaredArgument = Read(declared.GenericArguments[index], constructed);
                TypeReference? invokedArgument = Read(invoked.GenericArguments[index], constructed);
                if (declaredArgument is null || invokedArgument is null || declaredArgument.Resolve() is not TypeDefinition resolved ||
                    resolved != invokedArgument.Resolve()) return null;
                closed.GenericArguments.Add(declaredArgument);
            }
            return new MethodReference(target.Name, target.ReturnType, closed) { HasThis = true };
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or ArgumentException or InvalidOperationException)
        { return null; }
    }

    internal static TypeReference? Read(TypeReference argument, MethodReference? context)
    {
        if (argument is not GenericParameter parameter)
            return argument is TypeSpecification || argument.HasGenericParameters ? null : argument;
        if (context is null) return null;
        try
        {
            TypeReference? bound = null;
            if (parameter.Type == GenericParameterType.Type && parameter.Owner is TypeReference owner &&
                context.DeclaringType is GenericInstanceType type && Same(owner.Resolve(), type.ElementType.Resolve()) &&
                type.GenericArguments.Count == owner.Resolve().GenericParameters.Count &&
                parameter.Position >= 0 && parameter.Position < type.GenericArguments.Count)
                bound = type.GenericArguments[parameter.Position];
            if (parameter.Type == GenericParameterType.Method && parameter.Owner is MethodReference method &&
                context is GenericInstanceMethod instance && Same(method.Resolve(), instance.ElementMethod.Resolve()) &&
                instance.GenericArguments.Count == method.Resolve().GenericParameters.Count &&
                parameter.Position >= 0 && parameter.Position < instance.GenericArguments.Count)
                bound = instance.GenericArguments[parameter.Position];
            // Never recursively guess an outer context or erase a remaining T.
            return bound is null or GenericParameter or TypeSpecification || bound.HasGenericParameters ? null : bound;
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or
            ArgumentException or InvalidOperationException) { return null; }
    }

    private static bool Same(IMetadataTokenProvider? left, IMetadataTokenProvider? right)
    {
        // Resolved definitions must belong to the same loaded Cecil module,
        // not just share a name or a token from a different assembly.
        return left is not null && ReferenceEquals(left, right);
    }
}
