using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal sealed record ConstructorCallAssessment(int Instruction, ConstructorPrefixAssessment Prefix,
    LockedInitializationAssessment LockedTail, string KeyProvider, NumericBaseConstructorAssessment NumericBase);

// Supplies input facts from actual IL operands, never hard-coded test names or
// assumed rank/world-size defaults. The Guid contract proves string shape on
// normal return, NOT uniqueness, RNG purity, or absence of host side effects.
internal static class ConstructorCallReader
{
    internal static ConstructorCallAssessment? Read(MethodDefinition caller, int index)
    {
        try
        {
            if (!caller.HasBody || index < 3 || index >= caller.Body.Instructions.Count) return null;
            var il = caller.Body.Instructions;
            if (il[index].OpCode.Code != Code.Newobj || il[index].Operand is not MethodReference constructor ||
                !constructor.HasThis || constructor.ExplicitThis || constructor.HasGenericParameters || constructor.Parameters.Count != 3 ||
                constructor.CallingConvention != MethodCallingConvention.Default || constructor.Name != ".ctor" ||
                !Runtime(constructor.ReturnType, "System.Void") || !Runtime(constructor.Parameters[0].ParameterType, "System.Int32") ||
                !Runtime(constructor.Parameters[1].ParameterType, "System.Int32") || !Runtime(constructor.Parameters[2].ParameterType, "System.String") ||
                constructor.Resolve() is not MethodDefinition definition || !definition.IsConstructor || definition.IsStatic ||
                !Integer(il[index - 3], out int rank) || !Integer(il[index - 2], out int worldSize) ||
                il[index - 1].OpCode.Code != Code.Call || il[index - 1].Operand is not MethodReference key ||
                !GuidKey(key)) return null;
            // Branches may not enter halfway through the argument pushes.
            if (il.Any(instruction => instruction.Operand is Instruction target && Inside(target) ||
                instruction.Operand is Instruction[] targets && targets.Any(Inside)) ||
                caller.Body.ExceptionHandlers.Any(handler => Inside(handler.HandlerStart) ||
                    handler.FilterStart is not null && Inside(handler.FilterStart))) return null;
            bool Inside(Instruction target) => il.IndexOf(target) is int position && position > index - 3 && position <= index;
            ConstructorPrefixAssessment prefix = ConstructorPrefixReader.Read(definition, rank, worldSize, ConstructorKeyFact.NonWhitespaceString);
            return prefix.Contract == ConstructorPrefixContract.Unresolved ? null :
                new(index, prefix, LockedInitializationReader.Read(definition), key.FullName, NumericBaseConstructorReader.Read(constructor));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException or InvalidCastException)
        { return null; }
    }

    private static bool GuidKey(MethodReference call)
    {
        if (call.HasThis || call.ExplicitThis || call.HasGenericParameters || call.Parameters.Count != 0 ||
            call.CallingConvention != MethodCallingConvention.Default || !Runtime(call.ReturnType, "System.String") ||
            call.Resolve() is not MethodDefinition method || !method.IsStatic || !method.HasBody || method.HasGenericParameters ||
            method.ImplAttributes != MethodImplAttributes.IL || method.IsPInvokeImpl || method.HasSecurityDeclarations ||
            method.Body.HasExceptionHandlers || method.Body.Variables.Count != 1 || !Runtime(method.Body.Variables[0].VariableType, "System.Guid")) return false;
        var il = method.Body.Instructions;
        if (il.Count != 6 || il[0].OpCode.Code != Code.Call || il[1].OpCode.Code != Code.Stloc_0 ||
            il[2].OpCode.Code is not (Code.Ldloca or Code.Ldloca_S) || il[2].Operand != method.Body.Variables[0] ||
            il[3].OpCode.Code != Code.Ldstr || il[3].Operand is not "N" || il[4].OpCode.Code != Code.Call || il[5].OpCode.Code != Code.Ret ||
            !GuidCall(il[0], "System.Guid System.Guid::NewGuid()", false) ||
            !GuidCall(il[4], "System.String System.Guid::ToString(System.String)", true)) return false;
        using var stream = File.OpenRead(typeof(object).Assembly.Location);
        return Convert.ToHexStringLower(SHA256.HashData(stream)) == ReviewedOwnerCompletion.RuntimeHash;
    }

    private static bool GuidCall(Instruction instruction, string signature, bool instance) => instruction.Operand is MethodReference call &&
        call.HasThis == instance && !call.ExplicitThis && !call.HasGenericParameters && call.CallingConvention == MethodCallingConvention.Default &&
        call.FullName == signature && Runtime(call.DeclaringType, "System.Guid") &&
        call.Resolve() is MethodDefinition definition && definition.FullName == signature && definition.HasThis == instance &&
        Runtime(call.ReturnType, instance ? "System.String" : "System.Guid") &&
        call.Parameters.All(parameter => Runtime(parameter.ParameterType, "System.String"));

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static bool Integer(Instruction instruction, out int value)
    {
        value = instruction.OpCode.Code switch
        {
            Code.Ldc_I4_M1 => -1, Code.Ldc_I4_0 => 0, Code.Ldc_I4_1 => 1, Code.Ldc_I4_2 => 2, Code.Ldc_I4_3 => 3,
            Code.Ldc_I4_4 => 4, Code.Ldc_I4_5 => 5, Code.Ldc_I4_6 => 6, Code.Ldc_I4_7 => 7, Code.Ldc_I4_8 => 8,
            Code.Ldc_I4_S => (sbyte)instruction.Operand, Code.Ldc_I4 => (int)instruction.Operand, _ => 0
        };
        return instruction.OpCode.Code is Code.Ldc_I4_M1 or Code.Ldc_I4_0 or Code.Ldc_I4_1 or Code.Ldc_I4_2 or Code.Ldc_I4_3 or
            Code.Ldc_I4_4 or Code.Ldc_I4_5 or Code.Ldc_I4_6 or Code.Ldc_I4_7 or Code.Ldc_I4_8 or Code.Ldc_I4_S or Code.Ldc_I4;
    }
}
