using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum ConfigurationConstructorContract { Unresolved, OwnedReceiverWithDoubleLearningRate }
internal sealed record ConfigurationConstructorAssessment(ConfigurationConstructorContract Contract, NumericProviderAssessment? Provider);

// Bounded normal-input constructor contract. The provider result is consumed
// directly through its local: an unrelated INumericOperations<double> instance
// cannot inherit the pinned DoubleOperations.FromDouble contract.
internal static class ConfigurationConstructorReader
{
    internal static ConfigurationConstructorAssessment Read(MethodReference call, bool backendNonNull, double learningRate)
    {
        ConfigurationConstructorAssessment Unknown() => new(ConfigurationConstructorContract.Unresolved, null);
        try
        {
            MethodReference? providerCall = ReadShape(call, backendNonNull, learningRate);
            if (providerCall is null) return Unknown();
            NumericProviderAssessment provider = ReviewedNumericProvider.Read(providerCall, call);
            return provider.Contract == NumericProviderContract.TensorsDoubleCache
                ? new(ConfigurationConstructorContract.OwnedReceiverWithDoubleLearningRate, provider) : Unknown();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    // A structural result is not a runtime contract. Read additionally requires
    // the package/runtime-bound concrete provider; tests can exercise this shape
    // independently without manufacturing trusted package evidence.
    internal static MethodReference? ReadShape(MethodReference call, bool backendNonNull, double learningRate)
    {
        MethodReference? Unknown() => null;
        try
        {
            if (!backendNonNull || !double.IsFinite(learningRate) || learningRate <= 0 ||
                !call.HasThis || call.ExplicitThis || call.HasGenericParameters || call.CallingConvention != MethodCallingConvention.Default ||
                call.Name != ".ctor" || call.Parameters.Count != 2 || !Runtime(call.ReturnType, "System.Void") ||
                !Runtime(call.Parameters[1].ParameterType, "System.Double") || call.DeclaringType is not GenericInstanceType concrete ||
                concrete.GenericArguments.Count != 1 || !Runtime(concrete.GenericArguments[0], "System.Double") ||
                call.Resolve() is not MethodDefinition method || concrete.ElementType.Resolve() != method.DeclaringType ||
                !method.IsConstructor || method.IsStatic || !method.HasBody ||
                method.Parameters.Count != 2 || !Runtime(method.Parameters[1].ParameterType, "System.Double") ||
                method.HasGenericParameters || method.ImplAttributes != MethodImplAttributes.IL ||
                method.IsPInvokeImpl || method.HasSecurityDeclarations || method.Body.HasExceptionHandlers || method.Body.Variables.Count != 1 ||
                method.DeclaringType.GenericParameters.Count != 1 || method.DeclaringType.BaseType is null ||
                !Runtime(method.DeclaringType.BaseType, "System.Object") ||
                method.DeclaringType.Methods.Any(candidate => candidate.IsConstructor && candidate.IsStatic || candidate.Name == "Finalize")) return Unknown();
            var il = method.Body.Instructions;
            Code[] shape = [Code.Ldarg_0, Code.Ldc_I4_1, Code.Stfld, Code.Ldarg_0, Code.Ldc_I4, Code.Stfld,
                Code.Ldarg_0, Code.Call, Code.Ldarg_0, Code.Ldarg_1, Code.Dup, Code.Brtrue_S, Code.Pop, Code.Ldstr, Code.Newobj, Code.Throw,
                Code.Stfld, Code.Ldarg_2, Code.Ldc_R8, Code.Bgt_Un_S, Code.Ldstr, Code.Ldstr, Code.Newobj, Code.Throw,
                Code.Call, Code.Stloc_0, Code.Ldarg_0, Code.Ldloc_0, Code.Ldarg_2, Code.Callvirt, Code.Call, Code.Ret];
            if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(shape) ||
                il[4].Operand is not int minimum || minimum != 1024 || il[18].Operand is not double zero || zero != 0 ||
                il[11].Operand != il[16] || il[19].Operand != il[24] ||
                !RootConstructor(il[7]) || !ExceptionConstructor(il[14], "System.ArgumentNullException", 1) ||
                !ExceptionConstructor(il[22], "System.ArgumentOutOfRangeException", 2)) return Unknown();
            FieldDefinition? auto = OwnedFieldBinding.Read(il[2], method.DeclaringType);
            FieldDefinition? group = OwnedFieldBinding.Read(il[5], method.DeclaringType);
            FieldDefinition? backend = OwnedFieldBinding.Read(il[16], method.DeclaringType);
            if (auto is null || group is null || backend is null || new[] { auto, group, backend }.Distinct().Count() != 3 ||
                new[] { auto, group, backend }.Any(field => field.IsStatic || !field.IsPrivate) ||
                !Runtime(auto.FieldType, "System.Boolean") || !Runtime(group.FieldType, "System.Int32") ||
                !BackendType(backend.FieldType, method.DeclaringType) || !BackendType(method.Parameters[0].ParameterType, method.DeclaringType) ||
                !BackendType(call.Parameters[0].ParameterType, method.DeclaringType) ||
                il[24].Operand is not MethodReference providerCall || il[29].Operand is not MethodReference fromDouble ||
                il[30].Operand is not MethodReference setter || !DoubleConversion(fromDouble, method) ||
                !LearningRateSetter(setter, method.DeclaringType)) return Unknown();
            if (providerCall is not GenericInstanceMethod supplied || supplied.GenericArguments.Count != 1 ||
                !ReviewedNumericProvider.HasProviderSignature(supplied.ElementMethod) || !Parameter(supplied.GenericArguments[0], method.DeclaringType) ||
                providerCall.Resolve() is not MethodDefinition providerDefinition || fromDouble.Resolve() is not MethodDefinition conversion ||
                providerDefinition.Module != conversion.Module || method.Body.Variables[0].VariableType is not GenericInstanceType local ||
                fromDouble.DeclaringType is not GenericInstanceType receiver || local.ElementType.Resolve() != receiver.ElementType.Resolve() ||
                local.GenericArguments.Count != 1 || !Parameter(local.GenericArguments[0], method.DeclaringType)) return Unknown();
            return providerCall;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    private static bool DoubleConversion(MethodReference reference, MethodDefinition constructor)
    {
        if (!reference.HasThis || reference.ExplicitThis || reference.HasGenericParameters || reference.CallingConvention != MethodCallingConvention.Default ||
            reference.Name != "FromDouble" || reference.Parameters.Count != 1 || !Runtime(reference.Parameters[0].ParameterType, "System.Double") ||
            reference.DeclaringType is not GenericInstanceType type || type.GenericArguments.Count != 1 ||
            !Parameter(type.GenericArguments[0], constructor.DeclaringType) || reference.ReturnType is not GenericParameter returned ||
            returned.Type != GenericParameterType.Type || returned.Position != 0 || returned.Owner is not TypeReference returnedOwner ||
            returnedOwner.Resolve() != type.ElementType.Resolve() || type.ElementType.FullName != "AiDotNet.Tensors.Interfaces.INumericOperations`1" ||
            reference.Resolve() is not MethodDefinition definition || definition.DeclaringType != type.ElementType.Resolve()) return false;
        return true;
    }

    private static bool LearningRateSetter(MethodReference reference, TypeDefinition owner)
    {
        if (!reference.HasThis || reference.ExplicitThis || reference.HasGenericParameters || reference.Parameters.Count != 1 ||
            reference.CallingConvention != MethodCallingConvention.Default || !Runtime(reference.ReturnType, "System.Void") ||
            !Parameter(reference.Parameters[0].ParameterType, owner) || !OwnedFieldBinding.SelfType(reference.DeclaringType, owner) ||
            reference.Resolve() is not MethodDefinition method ||
            method.DeclaringType != owner || method.IsStatic || method.ImplAttributes != MethodImplAttributes.IL ||
            !method.HasBody || method.Body.HasExceptionHandlers || method.Body.HasVariables || method.HasSecurityDeclarations || method.IsPInvokeImpl ||
            !method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldarg_0, Code.Ldarg_1, Code.Stfld, Code.Ret }) ||
            OwnedFieldBinding.Read(method.Body.Instructions[2], owner) is not FieldDefinition field || field.IsStatic || !field.IsPrivate ||
            !Parameter(field.FieldType, owner)) return false;
        return true;
    }

    internal static bool BackendType(TypeReference reference, TypeDefinition owner) => reference is GenericInstanceType type &&
        type.GenericArguments.Count == 1 && Parameter(type.GenericArguments[0], owner) &&
        type.ElementType.Resolve() is TypeDefinition definition && definition.IsInterface &&
        definition.Module == owner.Module && definition.FullName == "AiDotNet.DistributedTraining.ICommunicationBackend`1";
    private static bool Parameter(TypeReference reference, TypeDefinition owner) => reference is GenericParameter parameter &&
        parameter.Type == GenericParameterType.Type && parameter.Position == 0 && parameter.Owner is TypeReference declared && declared.Resolve() == owner;
    private static bool RootConstructor(Instruction instruction) => ExceptionConstructor(instruction, "System.Object", 0);
    private static bool ExceptionConstructor(Instruction instruction, string type, int count) => instruction.Operand is MethodReference call &&
        call.HasThis && !call.ExplicitThis && !call.HasGenericParameters && call.CallingConvention == MethodCallingConvention.Default &&
        call.Name == ".ctor" && call.Parameters.Count == count && call.Parameters.All(parameter => Runtime(parameter.ParameterType, "System.String")) &&
        Runtime(call.ReturnType, "System.Void") && Runtime(call.DeclaringType, type) && call.Resolve() is MethodDefinition definition &&
        definition.IsConstructor && !definition.IsStatic && definition.FullName == call.FullName;
    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
