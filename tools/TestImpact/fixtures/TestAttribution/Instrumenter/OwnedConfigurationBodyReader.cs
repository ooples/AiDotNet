using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal sealed record ConfigurationAllocation(MethodReference Constructor, double LearningRate);
internal sealed record OwnedConfigurationBodyShape(ConstructorCallAssessment Backend, ConfigurationAllocation[] Configurations,
    int[] Assertions, string[] AccessedFields);

// Exhaustive interpretation of the user window, not a search for interesting
// calls. Unsupported instructions, escaped objects, extra calls and workers
// invalidate the entire shape. Provider/initializer/lifetime proofs are retained
// separately; this shape alone never authorizes reuse.
internal static class OwnedConfigurationBodyReader
{
    private enum Kind { Backend, Configuration, Boolean, Number, Message }
    private sealed record Value(Kind Kind, TypeReference? Type = null, double Number = 0);

    internal static OwnedConfigurationBodyShape? ReadShape(MethodDefinition entry, MethodDefinition body)
    {
        try
        {
            YieldBodyWindow window = YieldBodyReader.Read(entry, body);
            if (window.Contract != YieldBodyContract.SingleYieldTaskBody || window.Length < 4) return null;
            int backendIndex = window.Start + 3;
            ConstructorCallAssessment? backend = ConstructorCallReader.Read(body, backendIndex);
            if (backend is null || body.Body.Instructions[backendIndex].Operand is not MethodReference backendConstructor) return null;
            var stack = new Stack<Value>();
            var locals = new Value?[body.Body.Variables.Count];
            stack.Push(new(Kind.Backend, backendConstructor.DeclaringType));
            var allocations = new List<ConfigurationAllocation>();
            var assertions = new List<int>();
            var fields = new HashSet<string>(StringComparer.Ordinal);
            for (int index = backendIndex + 1; index < window.Start + window.Length; index++)
            {
                Instruction instruction = body.Body.Instructions[index];
                switch (instruction.OpCode.Code)
                {
                    case Code.Stloc_0: case Code.Stloc_1: case Code.Stloc_2: case Code.Stloc_3: case Code.Stloc: case Code.Stloc_S:
                    {
                        int slot = Local(instruction);
                        if ((uint)slot >= locals.Length || !stack.TryPop(out Value? stored) || !LocalType(body.Body.Variables[slot].VariableType, stored)) return null;
                        locals[slot] = stored;
                        break;
                    }
                    case Code.Ldloc_0: case Code.Ldloc_1: case Code.Ldloc_2: case Code.Ldloc_3: case Code.Ldloc: case Code.Ldloc_S:
                    {
                        int slot = Local(instruction);
                        if ((uint)slot >= locals.Length || locals[slot] is not Value stored) return null;
                        stack.Push(stored);
                        break;
                    }
                    case Code.Dup:
                        if (!stack.TryPeek(out Value? copy)) return null;
                        stack.Push(copy); break;
                    case Code.Ldc_I4_0: case Code.Ldc_I4_1: stack.Push(new(Kind.Boolean)); break;
                    case Code.Ldc_R8:
                        if (instruction.Operand is not double number) return null;
                        stack.Push(new(Kind.Number, Number: number)); break;
                    case Code.Ldnull: case Code.Ldstr: stack.Push(new(Kind.Message)); break;
                    case Code.Newobj:
                    {
                        if (instruction.Operand is not MethodReference constructor || !stack.TryPop(out Value? rate) || rate.Kind != Kind.Number ||
                            !stack.TryPop(out Value? source) || source.Kind != Kind.Backend ||
                            !BackendMatches(backendConstructor, constructor) ||
                            ConfigurationConstructorReader.ReadShape(constructor, true, rate.Number) is null) return null;
                        allocations.Add(new(constructor, rate.Number));
                        stack.Push(new(Kind.Configuration, constructor.DeclaringType));
                        break;
                    }
                    case Code.Call: case Code.Callvirt:
                    {
                        if (instruction.Operand is not MethodReference call) return null;
                        if (!call.HasThis && ConfigurationFactoryReader.ReadShape(call) is ConfigurationFactoryShape factory)
                        {
                            if (instruction.OpCode.Code != Code.Call || !stack.TryPop(out Value? source) || source.Kind != Kind.Backend ||
                                !BackendMatches(backendConstructor, factory.Constructor)) return null;
                            allocations.Add(new(factory.Constructor, factory.LearningRate));
                            fields.UnionWith(factory.WrittenFields);
                            stack.Push(new(Kind.Configuration, factory.Constructor.DeclaringType));
                            break;
                        }
                        if (call.HasThis)
                        {
                            bool write = call.Parameters.Count == 1;
                            if (write && (!stack.TryPop(out Value? assigned) || assigned.Kind != Kind.Boolean)) return null;
                            if (!stack.TryPop(out Value? receiver) || receiver.Kind != Kind.Configuration || receiver.Type is null ||
                                BooleanPropertyReader.Read(call, receiver.Type, write ? BooleanAccessorKind.Write : BooleanAccessorKind.Read) is not FieldDefinition field)
                                return null;
                            fields.Add(field.FullName);
                            if (!write) stack.Push(new(Kind.Boolean));
                            break;
                        }
                        if (instruction.OpCode.Code != Code.Call || ReviewedRuntimeContracts.Assess(call).Status != RuntimeContractStatus.ReviewedConditional ||
                            AssertionFailureReader.Read(body, index) != AssertionFailurePropagation.ForwardsToTaskBuilder ||
                            call.Parameters.Count == 2 && (!stack.TryPop(out Value? message) || message.Kind != Kind.Message) ||
                            !stack.TryPop(out Value? condition) || condition.Kind != Kind.Boolean) return null;
                        assertions.Add(index);
                        break;
                    }
                    default: return null;
                }
            }
            return stack.Count == 0 && allocations.Count == 1 && assertions.Count > 0
                ? new(backend, allocations.ToArray(), assertions.ToArray(), fields.Order(StringComparer.Ordinal).ToArray()) : null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }

    private static bool BackendMatches(MethodReference backend, MethodReference configuration)
    {
        if (configuration.DeclaringType is not GenericInstanceType configurationType || configurationType.GenericArguments.Count != 1 ||
            configuration.Parameters.Count != 2 || configuration.Parameters[0].ParameterType is not GenericInstanceType expected ||
            expected.ElementType.Resolve() is not TypeDefinition target || backend.Resolve() is not MethodDefinition constructor ||
            !constructor.HasBody || constructor.DeclaringType.Methods.Any(method => method.Name == "Finalize") ||
            constructor.Body.Instructions.Count < 2 || constructor.Body.Instructions[1].Operand is not MethodReference baseCall ||
            ConcreteGenericBinding.BaseConstructor(backend, baseCall) is not MethodReference closed ||
            closed.DeclaringType is not GenericInstanceType parent || parent.GenericArguments.Count != 1 ||
            parent.GenericArguments[0].Resolve() is not TypeDefinition scalar || scalar != configurationType.GenericArguments[0].Resolve() ||
            parent.ElementType.Resolve() is not TypeDefinition definition || definition.Methods.Any(method => method.Name == "Finalize")) return false;
        return definition.Interfaces.Any(implementation => implementation.InterfaceType is GenericInstanceType contract &&
            contract.ElementType.Resolve() == target && contract.GenericArguments.Count == 1 &&
            contract.GenericArguments[0] is GenericParameter parameter && parameter.Type == GenericParameterType.Type &&
            parameter.Position == 0 && parameter.Owner is TypeReference owner && owner.Resolve() == definition);
    }

    private static int Local(Instruction instruction) => instruction.Operand is VariableDefinition local ? local.Index : instruction.OpCode.Code switch
    {
        Code.Ldloc_0 or Code.Stloc_0 => 0, Code.Ldloc_1 or Code.Stloc_1 => 1,
        Code.Ldloc_2 or Code.Stloc_2 => 2, Code.Ldloc_3 or Code.Stloc_3 => 3, _ => -1
    };
    private static bool LocalType(TypeReference type, Value value)
    {
        if (value.Kind is Kind.Backend or Kind.Configuration)
            return type is GenericInstanceType actual && value.Type is GenericInstanceType expected &&
                actual.ElementType.Resolve() == expected.ElementType.Resolve() && actual.GenericArguments.Count == 1 && expected.GenericArguments.Count == 1 &&
                actual.GenericArguments[0].Resolve() is TypeDefinition scalar && scalar == expected.GenericArguments[0].Resolve();
        string name = value.Kind switch { Kind.Boolean => "System.Boolean", Kind.Number => "System.Double", Kind.Message => "System.String", _ => "" };
        return type is not TypeSpecification && type.FullName == name && type.Resolve() is TypeDefinition definition &&
            string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    }
}
