using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Factory ownership ends at return. This second check follows the value in a
// caller; passing a derived bool to an assertion is NOT automatically harmless
// (throwing may invoke first-chance handlers). No external sink is whitelisted.
internal static class OwnedReturnUseReader
{
    private enum Value { Unrelated, OwnedReference, DerivedValue }

    internal static OwnedReturnUse Read(MethodDefinition caller, int factoryCall)
    {
        var unresolved = new HashSet<string>(StringComparer.Ordinal);
        OwnedReturnUse Result(OwnedReturnUseStatus status, int instruction) => new(status, instruction, unresolved.Order(StringComparer.Ordinal).ToArray());
        if (!caller.HasBody || factoryCall < 0 || factoryCall >= caller.Body.Instructions.Count ||
            caller.Body.Instructions[factoryCall].Operand is not MethodReference factory ||
            caller.Body.Instructions[factoryCall].OpCode.Code is not (Code.Call or Code.Callvirt) ||
            factory.ReturnType.MetadataType == MetadataType.Void)
            return Result(OwnedReturnUseStatus.NeedsBoundaryProof, factoryCall);
        var stack = new Stack<Value>();
        var locals = new Value[caller.Body.Variables.Count];
        stack.Push(Value.OwnedReference);
        bool boundary = caller.Body.ExceptionHandlers.Count != 0;
        bool Pop(out Value value) => stack.TryPop(out value);
        for (int index = factoryCall + 1; index < caller.Body.Instructions.Count; index++)
        {
            Instruction instruction = caller.Body.Instructions[index];
            switch (instruction.OpCode.Code)
            {
                case Code.Nop: break;
                case Code.Dup:
                    if (!Pop(out Value duplicate)) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    stack.Push(duplicate); stack.Push(duplicate); break;
                case Code.Pop:
                    if (!Pop(out _)) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    break;
                case Code.Ldloc_0: case Code.Ldloc_1: case Code.Ldloc_2: case Code.Ldloc_3: case Code.Ldloc: case Code.Ldloc_S:
                {
                    int local = instruction.Operand is VariableDefinition variable ? variable.Index : (int)instruction.OpCode.Code - (int)Code.Ldloc_0;
                    if ((uint)local >= locals.Length) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    stack.Push(locals[local]); break;
                }
                case Code.Stloc_0: case Code.Stloc_1: case Code.Stloc_2: case Code.Stloc_3: case Code.Stloc: case Code.Stloc_S:
                {
                    int local = instruction.Operand is VariableDefinition variable ? variable.Index : (int)instruction.OpCode.Code - (int)Code.Stloc_0;
                    if ((uint)local >= locals.Length || !Pop(out Value value)) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    locals[local] = value; break;
                }
                case Code.Ldnull: case Code.Ldstr: case Code.Ldc_I4_0: case Code.Ldc_I4_1:
                case Code.Ldarg_0: case Code.Ldarg_1: case Code.Ldarg_2: case Code.Ldarg_3:
                    stack.Push(Value.Unrelated); break;
                case Code.Stsfld: case Code.Stfld: case Code.Stelem_Ref: case Code.Stobj:
                    // Even an unrelated write after an assertion may become
                    // control-dependent on whether the changed assertion throws.
                    return Result(OwnedReturnUseStatus.Escapes, index);
                case Code.Call: case Code.Callvirt:
                {
                    if (instruction.Operand is not MethodReference call) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    var arguments = new Value[call.Parameters.Count];
                    for (int i = arguments.Length - 1; i >= 0; i--)
                        if (!Pop(out arguments[i])) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    Value receiver = Value.Unrelated;
                    if (call.HasThis && !Pop(out receiver)) return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
                    if (arguments.Contains(Value.OwnedReference)) return Result(OwnedReturnUseStatus.Escapes, index);
                    if (receiver == Value.OwnedReference && IsTrivialPrimitiveGetter(call, factory.ReturnType))
                    {
                        stack.Push(Value.DerivedValue); break;
                    }
                    if (receiver == Value.OwnedReference) return Result(OwnedReturnUseStatus.Escapes, index);
                    unresolved.Add(call.FullName + "@" + call.DeclaringType.Scope);
                    boundary = true;
                    if (call.ReturnType.MetadataType != MetadataType.Void) stack.Push(Value.DerivedValue);
                    break;
                }
                case Code.Ret:
                    if (caller.ReturnType.MetadataType != MetadataType.Void && (!Pop(out Value returned) || returned != Value.Unrelated))
                        return Result(OwnedReturnUseStatus.Escapes, index);
                    return Result(boundary || stack.Count != 0 ? OwnedReturnUseStatus.NeedsBoundaryProof : OwnedReturnUseStatus.DiscardedLocally, index);
                default:
                    // Branches, exception exits, await/address/boxing/delegate
                    // operations require a stronger path/lifetime proof.
                    return Result(OwnedReturnUseStatus.NeedsBoundaryProof, index);
            }
        }
        return Result(OwnedReturnUseStatus.NeedsBoundaryProof, caller.Body.Instructions.Count);
    }

    private static bool IsTrivialPrimitiveGetter(MethodReference reference, TypeReference allocated)
    {
        try
        {
            MethodDefinition? method = reference.Resolve();
            TypeDefinition? owner = allocated.Resolve();
            if (method is null || owner is null || method.DeclaringType != owner || !method.HasBody || method.IsStatic ||
                method.Parameters.Count != 0 || !method.ReturnType.IsPrimitive || method.Body.ExceptionHandlers.Count != 0)
                return false;
            Instruction[] body = method.Body.Instructions.Where(instruction => instruction.OpCode.Code != Code.Nop).ToArray();
            return body.Length == 3 && body[0].OpCode.Code == Code.Ldarg_0 && body[1].OpCode.Code == Code.Ldfld &&
                body[1].Operand is FieldReference field && field.Resolve() is FieldDefinition definition &&
                !definition.IsStatic && definition.DeclaringType == owner && definition.FieldType.FullName == method.ReturnType.FullName &&
                body[2].OpCode.Code == Code.Ret;
        }
        catch (AssemblyResolutionException) { return false; }
        catch (ResolutionException) { return false; }
    }
}
