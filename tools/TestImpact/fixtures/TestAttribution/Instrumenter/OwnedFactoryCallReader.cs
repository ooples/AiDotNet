using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Path-sensitive counterpart to returned-object ownership. A null-check lambda
// never receives the new object; it must not be mislabeled as a proven escape.
// This is a comparative path fact, not a purity claim about throwing exceptions.
internal static class OwnedFactoryCallReader
{
    internal static OwnedFactoryCallPath Read(MethodDefinition caller, MethodDefinition before, MethodDefinition after)
    {
        try
        {
            if (!caller.HasBody || caller.Body.ExceptionHandlers.Count != 0 || caller.Body.Instructions.Count != 3 ||
                caller.Parameters.Count != 0 || caller.Body.Instructions[0].OpCode.Code != Code.Ldnull ||
                caller.Body.Instructions[1].OpCode.Code != Code.Call || caller.Body.Instructions[1].Operand is not MethodReference call ||
                caller.Body.Instructions[2].OpCode.Code != Code.Ret || call.HasThis || call.Parameters.Count != 1 ||
                call.ReturnType.MetadataType == MetadataType.Void || !MatchesTarget(call, after)) return OwnedFactoryCallPath.Unresolved;
            OwnedResultEffect? old = new OwnedResultEffects().Read(before), next = new OwnedResultEffects().Read(after);
            if (old is null || next is null || old.ShapeHash != next.ShapeHash ||
                !old.OwnershipDependencies.SequenceEqual(next.OwnershipDependencies) || !Guard(before) || !Guard(after))
                return OwnedFactoryCallPath.Unresolved;
            return OwnedFactoryCallPath.NullGuardPrecedesOwnedChange;
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or IOException or
            ArgumentException or InvalidOperationException)
        {
            return OwnedFactoryCallPath.Unresolved;
        }
    }

    private static bool MatchesTarget(MethodReference call, MethodDefinition target)
    {
        MethodDefinition? resolved = call.Resolve();
        return resolved is not null && resolved.MetadataToken == target.MetadataToken &&
            resolved.Module.Mvid == target.Module.Mvid && resolved.FullName == target.FullName &&
            string.Equals(Path.GetFullPath(resolved.Module.FileName), Path.GetFullPath(target.Module.FileName),
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    }

    private static bool Guard(MethodDefinition factory)
    {
        if (!factory.IsStatic || factory.Parameters.Count != 1 || factory.Parameters[0].ParameterType.IsValueType ||
            factory.Parameters[0].ParameterType is ByReferenceType or PointerType || !factory.HasBody ||
            factory.Body.ExceptionHandlers.Count != 0 || factory.Body.Instructions.Count < 6) return false;
        var body = factory.Body.Instructions;
        if (body[0].OpCode.Code != Code.Ldarg_0 || body[1].OpCode.Code is not (Code.Brtrue or Code.Brtrue_S) ||
            body[1].Operand != body[5] || body[2].OpCode.Code != Code.Ldstr || body[2].Operand is not string ||
            body[3].OpCode.Code != Code.Newobj || body[3].Operand is not MethodReference constructor ||
            constructor.FullName != "System.Void System.ArgumentNullException::.ctor(System.String)" || body[4].OpCode.Code != Code.Throw)
            return false;
        MethodDefinition? exception = constructor.Resolve();
        return exception is not null && string.Equals(Path.GetFullPath(exception.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    }
}
