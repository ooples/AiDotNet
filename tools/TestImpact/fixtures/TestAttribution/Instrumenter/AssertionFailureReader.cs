using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// This proves only the immediate caller's exceptional exit. It deliberately
// does not infer that an outer caller awaits the task or propagates the failure.
internal static class AssertionFailureReader
{
    internal static AssertionFailurePropagation Read(MethodDefinition caller, int callIndex)
    {
        if (!caller.HasBody || callIndex < 0 || callIndex >= caller.Body.Instructions.Count ||
            caller.Body.Instructions[callIndex].OpCode.Code != Code.Call ||
            caller.Body.Instructions[callIndex].Operand is not MethodReference call ||
            ReviewedRuntimeContracts.Assess(call).Status != RuntimeContractStatus.ReviewedConditional)
            return AssertionFailurePropagation.Unresolved;
        if (caller.Body.ExceptionHandlers.Count == 0) return AssertionFailurePropagation.LeavesMethod;
        // No nested catches, filters, or finally blocks. In particular a user
        // catch which swallows an assertion must never inherit the async rule.
        if (caller.Body.ExceptionHandlers.Count != 1) return AssertionFailurePropagation.Unresolved;
        ExceptionHandler handler = caller.Body.ExceptionHandlers[0];
        var instructions = caller.Body.Instructions;
        int start = instructions.IndexOf(handler.HandlerStart);
        int end = handler.HandlerEnd is null ? instructions.Count : instructions.IndexOf(handler.HandlerEnd);
        int tryStart = instructions.IndexOf(handler.TryStart);
        int tryEnd = handler.TryEnd is null ? instructions.Count : instructions.IndexOf(handler.TryEnd);
        if (handler.HandlerType != ExceptionHandlerType.Catch || handler.CatchType?.FullName != "System.Exception" ||
            tryStart < 0 || callIndex < tryStart || callIndex >= tryEnd || start < 0 || end - start != 9 ||
            !caller.HasThis || caller.IsStatic || caller.Parameters.Count != 0 || caller.ReturnType.MetadataType != MetadataType.Void ||
            !caller.DeclaringType.IsValueType) return AssertionFailurePropagation.Unresolved;
        Instruction[] tail = instructions.Skip(start).Take(9).ToArray();
        int Local(Instruction instruction, bool load) => instruction.Operand is VariableDefinition local ? local.Index :
            load && instruction.OpCode.Code is (Code.Ldloc_0 or Code.Ldloc_1 or Code.Ldloc_2 or Code.Ldloc_3) ? (int)instruction.OpCode.Code - (int)Code.Ldloc_0 :
            !load && instruction.OpCode.Code is (Code.Stloc_0 or Code.Stloc_1 or Code.Stloc_2 or Code.Stloc_3) ? (int)instruction.OpCode.Code - (int)Code.Stloc_0 : -1;
        bool Store(Instruction instruction) => instruction.OpCode.Code is Code.Stloc or Code.Stloc_S or Code.Stloc_0 or Code.Stloc_1 or Code.Stloc_2 or Code.Stloc_3;
        bool Load(Instruction instruction) => instruction.OpCode.Code is Code.Ldloc or Code.Ldloc_S or Code.Ldloc_0 or Code.Ldloc_1 or Code.Ldloc_2 or Code.Ldloc_3;
        if (!Store(tail[0]) || Local(tail[0], false) < 0 || tail[1].OpCode.Code != Code.Ldarg_0 ||
            tail[2].OpCode.Code != Code.Ldc_I4_S || tail[2].Operand is not sbyte state || state != -2 ||
            tail[3].OpCode.Code != Code.Stfld || tail[3].Operand is not FieldReference stateField ||
            tail[4].OpCode.Code != Code.Ldarg_0 || tail[5].OpCode.Code != Code.Ldflda || tail[5].Operand is not FieldReference builder ||
            !Load(tail[6]) || Local(tail[6], true) != Local(tail[0], false) || tail[7].OpCode.Code != Code.Call ||
            tail[7].Operand is not MethodReference setException ||
            tail[8].OpCode.Code is not (Code.Leave or Code.Leave_S) || tail[8].Operand is not Instruction exit ||
            !instructions.Contains(exit) || exit.OpCode.Code != Code.Ret)
            return AssertionFailurePropagation.Unresolved;
        try
        {
            FieldDefinition? stateDefinition = stateField.Resolve(), builderDefinition = builder.Resolve();
            MethodDefinition? setter = setException.Resolve();
            TypeDefinition? caught = handler.CatchType.Resolve();
            string runtime = Path.GetFullPath(typeof(object).Assembly.Location);
            bool Runtime(ModuleDefinition module) => string.Equals(Path.GetFullPath(module.FileName), runtime,
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
            if (stateDefinition is null || builderDefinition is null || setter is null || caught is null ||
                stateDefinition.IsStatic || builderDefinition.IsStatic || stateDefinition.DeclaringType != caller.DeclaringType ||
                builderDefinition.DeclaringType != caller.DeclaringType || stateDefinition.FieldType.MetadataType != MetadataType.Int32 ||
                builderDefinition.FieldType.FullName != "System.Runtime.CompilerServices.AsyncTaskMethodBuilder" ||
                setException.FullName != "System.Void System.Runtime.CompilerServices.AsyncTaskMethodBuilder::SetException(System.Exception)" ||
                !Runtime(setter.Module) || !Runtime(caught.Module) || builderDefinition.FieldType.Resolve() != setter.DeclaringType)
                return AssertionFailurePropagation.Unresolved;
            return AssertionFailurePropagation.ForwardsToTaskBuilder;
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or IOException or ArgumentException)
        {
            return AssertionFailurePropagation.Unresolved;
        }
    }
}
