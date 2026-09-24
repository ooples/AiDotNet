using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum NumericBaseConstructorContract { Unresolved, OwnedFieldsAndDoubleProvider }
internal sealed record NumericBaseConstructorAssessment(NumericBaseConstructorContract Contract, NumericProviderAssessment? Provider);

internal static class NumericBaseConstructorReader
{
    internal static NumericBaseConstructorAssessment Read(MethodReference constructed)
    {
        NumericBaseConstructorAssessment Unknown() => new(NumericBaseConstructorContract.Unresolved, null);
        try
        {
            if (constructed.Resolve() is not MethodDefinition derived || !derived.HasBody || derived.Body.Instructions.Count < 2 ||
                derived.Body.Instructions[0].OpCode.Code != Code.Ldarg_0 || derived.Body.Instructions[1].OpCode.Code != Code.Call ||
                derived.Body.Instructions[1].Operand is not MethodReference baseCall ||
                ConcreteGenericBinding.BaseConstructor(constructed, baseCall) is not MethodReference closed ||
                baseCall.Resolve() is not MethodDefinition target || ProviderCall(target) is not MethodReference providerCall) return Unknown();
            NumericProviderAssessment provider = ReviewedNumericProvider.Read(providerCall, closed);
            return provider.Contract == NumericProviderContract.TensorsDoubleCache
                ? new(NumericBaseConstructorContract.OwnedFieldsAndDoubleProvider, provider) : Unknown();
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or IOException or
            ArgumentException or InvalidOperationException) { return Unknown(); }
    }

    // Structural result only: the returned provider call still requires exact
    // package/runtime review and concrete type binding. Receiver writes neither
    // escape this nor make an arbitrary same-named provider safe.
    internal static MethodReference? ProviderCall(MethodDefinition method)
    {
        if (!method.IsConstructor || method.IsStatic || method.Parameters.Count != 0 || method.HasGenericParameters ||
            !method.HasBody || method.ImplAttributes != MethodImplAttributes.IL || method.IsPInvokeImpl || method.HasSecurityDeclarations ||
            method.Body.HasVariables || method.Body.HasExceptionHandlers || method.DeclaringType.GenericParameters.Count != 1 ||
            !Runtime(method.ReturnType, "System.Void") || method.DeclaringType.BaseType is null ||
            !Runtime(method.DeclaringType.BaseType, "System.Object")) return null;
        var il = method.Body.Instructions;
        if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldarg_0, Code.Call, Code.Ldarg_0,
            Code.Call, Code.Stfld, Code.Ldarg_0, Code.Ldc_I4_0, Code.Stfld, Code.Ret }) ||
            il[1].Operand is not MethodReference root || !root.HasThis || root.ExplicitThis || root.HasGenericParameters ||
            root.CallingConvention != MethodCallingConvention.Default || root.FullName != "System.Void System.Object::.ctor()" ||
            !Runtime(root.DeclaringType, "System.Object") || root.Resolve() is not MethodDefinition rootDefinition ||
            rootDefinition.FullName != root.FullName || !Runtime(root.ReturnType, "System.Void") ||
            il[3].Operand is not GenericInstanceMethod provider || provider.GenericArguments.Count != 1 ||
            provider.HasThis || provider.ExplicitThis || !ReviewedNumericProvider.HasProviderSignature(provider.ElementMethod) ||
            !Parameter(provider.GenericArguments[0], method.DeclaringType) ||
            OwnedFieldBinding.Read(il[4], method.DeclaringType) is not FieldDefinition numeric ||
            OwnedFieldBinding.Read(il[7], method.DeclaringType) is not FieldDefinition flag ||
            numeric == flag || numeric.DeclaringType != method.DeclaringType || flag.DeclaringType != method.DeclaringType ||
            numeric.IsStatic || flag.IsStatic || !numeric.IsInitOnly || !numeric.IsFamily || !flag.IsPrivate ||
            numeric.HasCustomAttributes || flag.HasCustomAttributes || !Runtime(flag.FieldType, "System.Boolean") ||
            numeric.FieldType is not GenericInstanceType operations || operations.GenericArguments.Count != 1 ||
            !Parameter(operations.GenericArguments[0], method.DeclaringType) ||
            provider.ElementMethod.ReturnType is not GenericInstanceType result ||
            result.ElementType.Resolve() != operations.ElementType.Resolve()) return null;
        return provider;
    }

    private static bool Parameter(TypeReference reference, TypeDefinition owner) => reference is GenericParameter parameter &&
        parameter.Type == GenericParameterType.Type && parameter.Position == 0 && parameter.Owner is TypeReference declaring && declaring.Resolve() == owner;
    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
