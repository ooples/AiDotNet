using Mono.Cecil;

// Call graphs deliberately erase generic instantiations for reachability. A
// runtime contract must not: T in an unrelated type/method is not the same T.
internal static class ConcreteGenericBinding
{
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
