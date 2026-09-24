using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Bounded compiler-pattern proof, not a claim that the test framework observes
// the returned task. Unsupported kickoff shapes remain unresolved.
internal static class AsyncOwnerReader
{
    internal static AsyncOwnerBinding Read(MethodDefinition entry, MethodDefinition moveNext)
    {
        try { return Matches(entry, moveNext) ? AsyncOwnerBinding.ReturnsStateMachineTask : AsyncOwnerBinding.Unresolved; }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or
            IOException or ArgumentException or InvalidOperationException)
        {
            return AsyncOwnerBinding.Unresolved;
        }
    }

    private static bool Matches(MethodDefinition entry, MethodDefinition moveNext)
    {
        if (!entry.HasBody || entry.HasGenericParameters || entry.Parameters.Count != 0 || entry.IsVirtual ||
            entry.Body.ExceptionHandlers.Count != 0 || entry.Body.Variables.Count != 1 || !entry.Body.InitLocals ||
            entry.ReturnType.FullName != "System.Threading.Tasks.Task" || !Runtime(entry.ReturnType.Resolve()) ||
            !moveNext.HasBody || moveNext.IsStatic || moveNext.Parameters.Count != 0 || moveNext.Name != "MoveNext" ||
            moveNext.ReturnType.MetadataType != MetadataType.Void || entry.Module != moveNext.Module) return false;
        TypeDefinition state = moveNext.DeclaringType;
        if (!state.IsValueType || state.HasGenericParameters || state.DeclaringType != entry.DeclaringType ||
            state.Methods.Any(method => method.IsConstructor && method.IsStatic) ||
            entry.Body.Variables[0].VariableType.Resolve() != state) return false;
        CustomAttribute[] markers = entry.CustomAttributes.Where(attribute =>
            attribute.AttributeType.FullName == "System.Runtime.CompilerServices.AsyncStateMachineAttribute").ToArray();
        if (markers.Length != 1 || !Runtime(markers[0].AttributeType.Resolve()) || markers[0].ConstructorArguments.Count != 1 ||
            markers[0].ConstructorArguments[0].Value is not TypeReference declaredState || declaredState.Resolve() != state) return false;
        InterfaceImplementation[] interfaces = state.Interfaces.Where(item =>
            item.InterfaceType.FullName == "System.Runtime.CompilerServices.IAsyncStateMachine").ToArray();
        if (interfaces.Length != 1 || !Runtime(interfaces[0].InterfaceType.Resolve())) return false;
        // Require actual interface dispatch to this MoveNext, not an explicit
        // implementation pointing elsewhere under a familiar method name.
        if (!moveNext.IsVirtual || !moveNext.IsFinal ||
            (moveNext.HasOverrides ? moveNext.Overrides.Count != 1 || moveNext.Overrides[0].Name != "MoveNext" ||
                moveNext.Overrides[0].DeclaringType.Resolve() != interfaces[0].InterfaceType.Resolve() : !moveNext.IsPublic) ||
            state.Methods.Any(method => method != moveNext && method.Overrides.Any(reference =>
                reference.DeclaringType.FullName == "System.Runtime.CompilerServices.IAsyncStateMachine" && reference.Name == "MoveNext"))) return false;
        Instruction[] body = entry.Body.Instructions.ToArray();
        Code[] expected = [Code.Ldloca_S, Code.Call, Code.Stfld, Code.Ldloca_S, Code.Ldc_I4_M1, Code.Stfld,
            Code.Ldloca_S, Code.Ldflda, Code.Ldloca_S, Code.Call, Code.Ldloca_S, Code.Ldflda, Code.Call, Code.Ret];
        if (body.Length != expected.Length) return false;
        for (int index = 0; index < body.Length; index++)
        {
            Code actual = body[index].OpCode.Code;
            if (actual == Code.Ldloca) actual = Code.Ldloca_S;
            if (actual != expected[index]) return false;
            if (actual == Code.Ldloca_S && (body[index].Operand is not VariableDefinition local || local != entry.Body.Variables[0])) return false;
        }
        if (body[2].Operand is not FieldReference builderReference || body[5].Operand is not FieldReference stateReference ||
            body[7].Operand is not FieldReference startBuilder || body[11].Operand is not FieldReference returnedBuilder ||
            body[1].Operand is not MethodReference create || body[9].Operand is not GenericInstanceMethod start ||
            body[12].Operand is not MethodReference getTask) return false;
        FieldDefinition? builder = builderReference.Resolve(), stateField = stateReference.Resolve();
        if (builder is null || stateField is null || builder.IsStatic || stateField.IsStatic ||
            builder.DeclaringType != state || stateField.DeclaringType != state || startBuilder.Resolve() != builder ||
            returnedBuilder.Resolve() != builder || stateField.FieldType.MetadataType != MetadataType.Int32 ||
            builder.FieldType.FullName != "System.Runtime.CompilerServices.AsyncTaskMethodBuilder" ||
            !Runtime(builder.FieldType.Resolve()) || start.GenericArguments.Count != 1 || start.GenericArguments[0].Resolve() != state)
            return false;
        if (create.FullName != "System.Runtime.CompilerServices.AsyncTaskMethodBuilder System.Runtime.CompilerServices.AsyncTaskMethodBuilder::Create()" ||
            start.ElementMethod.FullName != "System.Void System.Runtime.CompilerServices.AsyncTaskMethodBuilder::Start(!!0&)" ||
            getTask.FullName != "System.Threading.Tasks.Task System.Runtime.CompilerServices.AsyncTaskMethodBuilder::get_Task()" ||
            !Runtime(create.Resolve()?.DeclaringType) || !Runtime(start.Resolve()?.DeclaringType) || !Runtime(getTask.Resolve()?.DeclaringType)) return false;
        // The exceptional-exit reader separately checks the catch's argument
        // and control flow. Here require that its builder is the one returned.
        var faultCalls = moveNext.Body.Instructions.Where(instruction => instruction.OpCode.Code == Code.Call &&
            instruction.Operand is MethodReference reference && reference.FullName ==
            "System.Void System.Runtime.CompilerServices.AsyncTaskMethodBuilder::SetException(System.Exception)").ToArray();
        if (faultCalls.Length != 1) return false;
        int faultIndex = moveNext.Body.Instructions.IndexOf(faultCalls[0]);
        return faultIndex >= 2 && moveNext.Body.Instructions[faultIndex - 2].OpCode.Code == Code.Ldflda &&
            moveNext.Body.Instructions[faultIndex - 2].Operand is FieldReference faultBuilder && faultBuilder.Resolve() == builder;
    }

    private static bool Runtime(TypeDefinition? type) => type is not null && string.Equals(
        Path.GetFullPath(type.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
