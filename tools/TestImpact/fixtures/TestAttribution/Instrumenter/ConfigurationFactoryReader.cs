using Mono.Cecil;
using Mono.Cecil.Cil;

internal sealed record ConfigurationFactoryShape(MethodReference Constructor, double LearningRate, string[] WrittenFields);

internal static class ConfigurationFactoryReader
{
    // Normal non-null argument path only. A null-input test still requires its
    // separate exception/observer proof; factory ownership cannot suppress it.
    internal static ConfigurationFactoryShape? ReadShape(MethodReference call)
    {
        try
        {
            if (call.HasThis || call.ExplicitThis || call.HasGenericParameters || call.Parameters.Count != 1 ||
                call.CallingConvention != MethodCallingConvention.Default || call.DeclaringType is not GenericInstanceType concrete ||
                concrete.GenericArguments.Count != 1 || call.Resolve() is not MethodDefinition method || !method.IsStatic ||
                !method.HasBody || method.ImplAttributes != MethodImplAttributes.IL || method.IsPInvokeImpl ||
                method.HasSecurityDeclarations || method.HasGenericParameters || method.Parameters.Count != 1 ||
                method.Body.HasExceptionHandlers || method.Body.HasVariables || concrete.ElementType.Resolve() != method.DeclaringType ||
                !OwnedFieldBinding.SelfType(method.ReturnType, method.DeclaringType) || !OwnedFieldBinding.SelfType(call.ReturnType, method.DeclaringType) ||
                !ConfigurationConstructorReader.BackendType(method.Parameters[0].ParameterType, method.DeclaringType) ||
                !ConfigurationConstructorReader.BackendType(call.Parameters[0].ParameterType, method.DeclaringType)) return null;
            var il = method.Body.Instructions;
            if (il.Count < 12 || (il.Count - 9) % 3 != 0 ||
                !il.Take(8).Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldarg_0, Code.Brtrue_S, Code.Ldstr,
                    Code.Newobj, Code.Throw, Code.Ldarg_0, Code.Ldc_R8, Code.Newobj }) ||
                il[1].Operand != il[5] || il[^1].OpCode.Code != Code.Ret || il[6].Operand is not double rate ||
                il[7].Operand is not MethodReference constructor || !OwnedFieldBinding.SelfType(constructor.DeclaringType, method.DeclaringType) ||
                constructor.Parameters.Count != 2 || il[3].Operand is not MethodReference failure ||
                failure.FullName != "System.Void System.ArgumentNullException::.ctor(System.String)" ||
                failure.Resolve() is not MethodDefinition exception ||
                !string.Equals(Path.GetFullPath(exception.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
                    OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)) return null;
            var closed = new MethodReference(constructor.Name, constructor.ReturnType, concrete) { HasThis = constructor.HasThis,
                ExplicitThis = constructor.ExplicitThis, CallingConvention = constructor.CallingConvention };
            foreach (var parameter in constructor.Parameters) closed.Parameters.Add(new(parameter.ParameterType));
            if (ConfigurationConstructorReader.ReadShape(closed, true, rate) is null) return null;
            var fields = new List<string>();
            for (int index = 8; index < il.Count - 1; index += 3)
            {
                if (il[index].OpCode.Code != Code.Dup || il[index + 1].OpCode.Code is not (Code.Ldc_I4_0 or Code.Ldc_I4_1) ||
                    il[index + 2].OpCode.Code is not (Code.Call or Code.Callvirt) || il[index + 2].Operand is not MethodReference setter ||
                    BooleanPropertyReader.ReadSelf(setter, method.DeclaringType, BooleanAccessorKind.Write) is not FieldDefinition field) return null;
                fields.Add(field.FullName);
            }
            return new(closed, rate, fields.ToArray());
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }
}
