using Mono.Cecil;
using Mono.Cecil.Cil;

internal sealed record BuildKeyOverrideShape(string Gate, string Key, string Loaded);

// Complete lock-protected byte-array copy, including the gate initializer.
// This writes process-global state: callers must still establish exclusive
// startup and exclude later external access. It is not a pure-method contract.
internal static class BuildKeyOverrideReader
{
    internal static BuildKeyOverrideShape? ReadShape(MethodDefinition method)
    {
        try
        {
            TypeDefinition owner = method.DeclaringType;
            if (!owner.IsAbstract || !owner.IsSealed || owner.HasGenericParameters || owner.HasInterfaces || owner.BaseType is null ||
                !SignedLicenseReader.Type(owner.BaseType, typeof(object)) || !method.IsStatic || method.HasThis || method.ExplicitThis ||
                method.HasGenericParameters || method.CallingConvention != MethodCallingConvention.Default || method.IsPInvokeImpl ||
                method.HasSecurityDeclarations || method.ImplAttributes != MethodImplAttributes.IL || !method.HasBody ||
                method.Parameters.Count != 1 || !SignedLicenseReader.Type(method.Parameters[0].ParameterType, typeof(byte[])) ||
                !SignedLicenseReader.Type(method.ReturnType, typeof(void))) return null;
            var body = method.Body;
            if (body.Variables.Count != 2 || !SignedLicenseReader.Type(body.Variables[0].VariableType, typeof(object)) ||
                !SignedLicenseReader.Type(body.Variables[1].VariableType, typeof(bool))) return null;
            Code[] codes = [Code.Ldsfld, Code.Stloc_0, Code.Ldc_I4_0, Code.Stloc_1, Code.Ldloc_0, Code.Ldloca_S, Code.Call,
                Code.Ldarg_0, Code.Brfalse_S, Code.Ldarg_0, Code.Ldlen, Code.Conv_I4, Code.Ldc_I4_0, Code.Bgt_S, Code.Ldnull, Code.Br_S,
                Code.Ldarg_0, Code.Callvirt, Code.Castclass, Code.Stsfld, Code.Ldc_I4_1, Code.Stsfld, Code.Leave_S, Code.Ldloc_1,
                Code.Brfalse_S, Code.Ldloc_0, Code.Call, Code.Endfinally, Code.Ret];
            var il = body.Instructions;
            if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes) || il[5].Operand != body.Variables[1] ||
                il[8].Operand != il[14] || il[13].Operand != il[16] || il[15].Operand != il[19] || il[22].Operand != il[28] ||
                il[24].Operand != il[27] || il[18].Operand is not TypeReference cast || !SignedLicenseReader.Type(cast, typeof(byte[])) ||
                OwnedFieldBinding.Read(il[0], owner) is not FieldDefinition gate || OwnedFieldBinding.Read(il[19], owner) is not FieldDefinition key ||
                OwnedFieldBinding.Read(il[21], owner) is not FieldDefinition loaded || gate == key || gate == loaded || key == loaded ||
                !gate.IsInitOnly || key.IsInitOnly || loaded.IsInitOnly || !SignedLicenseReader.Type(gate.FieldType, typeof(object)) ||
                !SignedLicenseReader.Type(key.FieldType, typeof(byte[])) || !SignedLicenseReader.Type(loaded.FieldType, typeof(bool)) ||
                new[] { gate, key, loaded }.Any(field => !field.IsPrivate || !field.IsStatic || field.IsLiteral || field.InitialValue.Length != 0 ||
                    field.CustomAttributes.Any(attribute => !SignedLicenseReader.Type(attribute.AttributeType, typeof(System.Runtime.CompilerServices.NullableAttribute)))) ||
                owner.Fields.Any(field => !field.IsLiteral && field != gate && field != key && field != loaded)) return null;
            if (body.ExceptionHandlers.Count != 1) return null;
            var handler = body.ExceptionHandlers[0];
            if (handler.HandlerType != ExceptionHandlerType.Finally || handler.TryStart != il[4] || handler.TryEnd != il[23] ||
                handler.HandlerStart != il[23] || handler.HandlerEnd != il[28] || handler.CatchType is not null || handler.FilterStart is not null) return null;
            if (!Call(il[6], typeof(System.Threading.Monitor), "Enter", [typeof(object), typeof(bool).MakeByRefType()]) ||
                !Call(il[17], typeof(Array), "Clone", []) || !Call(il[26], typeof(System.Threading.Monitor), "Exit", [typeof(object)])) return null;
            MethodDefinition[] constructors = owner.Methods.Where(candidate => candidate.IsConstructor).ToArray();
            if (constructors.Length != 1) return null;
            MethodDefinition initializer = constructors[0];
            if (!initializer.IsStatic || initializer.HasThis || initializer.ExplicitThis || initializer.HasGenericParameters || initializer.Parameters.Count != 0 ||
                !initializer.HasBody || initializer.ImplAttributes != MethodImplAttributes.IL || initializer.IsPInvokeImpl || initializer.HasSecurityDeclarations ||
                initializer.Body.HasVariables || initializer.Body.HasExceptionHandlers || !SignedLicenseReader.Type(initializer.ReturnType, typeof(void))) return null;
            var init = initializer.Body.Instructions;
            if (!init.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Newobj, Code.Stsfld, Code.Ret }) ||
                OwnedFieldBinding.Read(init[1], owner) != gate || !Call(init[0], typeof(object), ".ctor", [])) return null;
            return new(gate.FullName, key.FullName, loaded.FullName);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }

    private static bool Call(Instruction instruction, Type owner, string name, Type[] arguments)
    {
        System.Reflection.MethodBase expected = name == ".ctor"
            ? owner.GetConstructor(arguments) ?? throw new InvalidOperationException("Missing constructor.")
            : owner.GetMethod(name, arguments) ?? throw new InvalidOperationException("Missing method.");
        if (instruction.Operand is not MethodReference call || call is MethodSpecification || call.HasGenericParameters || call.ExplicitThis ||
            call.CallingConvention != MethodCallingConvention.Default || call.HasThis == expected.IsStatic ||
            !SignedLicenseReader.Type(call.DeclaringType, owner) || call.Resolve() is not MethodDefinition resolved ||
            resolved.MetadataToken.ToInt32() != expected.MetadataToken || !OwnedFieldBinding.SameType(call.ReturnType, resolved.ReturnType) ||
            call.Parameters.Count != resolved.Parameters.Count) return false;
        return !call.Parameters.Where((parameter, index) => !Parameter(parameter.ParameterType, resolved.Parameters[index].ParameterType)).Any();
    }

    private static bool Parameter(TypeReference first, TypeReference second) => first is ByReferenceType reference
        ? second is ByReferenceType other && OwnedFieldBinding.SameType(reference.ElementType, other.ElementType)
        : OwnedFieldBinding.SameType(first, second);
}
