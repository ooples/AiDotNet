using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal sealed record NullBackendBodyShape(string[] Factories, string[] DelegateCaches);

// Expected exceptions are not pure: first-chance observers and shared type
// initialization still require workload/lifetime contracts. This reader proves
// only the exhaustive body path and its exact, non-escaping callback targets.
internal static class NullBackendBodyReader
{
    internal static NullBackendBodyShape? ReadShape(MethodDefinition entry, MethodDefinition body)
    {
        try
        {
            YieldBodyWindow window = YieldBodyReader.Read(entry, body);
            if (window.Contract != YieldBodyContract.SingleYieldTaskBody || window.Length == 0 || window.Length % 11 != 0) return null;
            var factories = new List<string>();
            var caches = new HashSet<FieldDefinition>();
            TypeDefinition? closure = null;
            FieldDefinition? singleton = null;
            var il = body.Body.Instructions;
            for (int start = window.Start; start < window.Start + window.Length; start += 11)
            {
                var sequence = il.Skip(start).Take(11).ToArray();
                if (!sequence.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldsfld, Code.Dup, Code.Brtrue_S,
                    Code.Pop, Code.Ldsfld, Code.Ldftn, Code.Newobj, Code.Dup, Code.Stsfld, Code.Call, Code.Pop }) ||
                    sequence[2].Operand != sequence[9] || sequence[0].Operand is not FieldReference cacheReference ||
                    cacheReference.Resolve() is not FieldDefinition cache || !cache.IsStatic || cache.IsInitOnly ||
                    !caches.Add(cache) || !FuncObject(cache.FieldType) ||
                    OwnedFieldBinding.Read(sequence[0], cache.DeclaringType) != cache || OwnedFieldBinding.Read(sequence[8], cache.DeclaringType) != cache ||
                    OwnedFieldBinding.Read(sequence[4], cache.DeclaringType) is not FieldDefinition instance ||
                    !instance.IsStatic || !instance.IsInitOnly || instance.FieldType.Resolve() != cache.DeclaringType ||
                    sequence[5].Operand is not MethodReference callback || callback.Resolve() is not MethodDefinition lambda ||
                    callback.DeclaringType.Resolve() != cache.DeclaringType || !callback.HasThis || callback.ExplicitThis || callback.HasGenericParameters ||
                    callback.CallingConvention != MethodCallingConvention.Default || callback.Parameters.Count != 0 || !Runtime(callback.ReturnType, "System.Object") ||
                    lambda.IsStatic || lambda.IsVirtual || !lambda.HasBody || lambda.Body.HasVariables || lambda.Body.HasExceptionHandlers ||
                    lambda.ImplAttributes != MethodImplAttributes.IL || lambda.IsPInvokeImpl || lambda.HasSecurityDeclarations ||
                    !lambda.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldnull, Code.Call, Code.Ret }) ||
                    lambda.Body.Instructions[1].Operand is not MethodReference factory || ConfigurationFactoryReader.ReadShape(factory) is null ||
                    !DelegateConstructor(sequence[6]) || !Assertion(sequence[9])) return null;
                if (closure is not null && (closure != cache.DeclaringType || singleton != instance)) return null;
                closure = cache.DeclaringType; singleton = instance;
                factories.Add(DependencyGraph.Stable(factory.Resolve()));
            }
            if (closure is null || singleton is null || !Closure(closure, singleton, caches, entry.DeclaringType)) return null;
            return new(factories.ToArray(), caches.Select(field => field.FullName).Order(StringComparer.Ordinal).ToArray());
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }

    private static bool Closure(TypeDefinition type, FieldDefinition singleton, HashSet<FieldDefinition> caches, TypeDefinition owner)
    {
        if (type.DeclaringType != owner || !type.IsNestedPrivate || type.HasGenericParameters || type.BaseType is null || !Runtime(type.BaseType, "System.Object") ||
            type.HasInterfaces || type.Methods.Any(method => method.Name == "Finalize") ||
            type.Fields.Any(field => field != singleton && (!field.IsStatic || field.IsInitOnly ||
                field.FieldType.Resolve() is not TypeDefinition delegateType || delegateType.BaseType is null ||
                !Runtime(delegateType.BaseType, "System.MulticastDelegate")))) return false;
        MethodDefinition[] constructors = type.Methods.Where(method => method.IsConstructor).ToArray();
        MethodDefinition[] instances = constructors.Where(method => !method.IsStatic).ToArray();
        MethodDefinition[] statics = constructors.Where(method => method.IsStatic).ToArray();
        bool Plain(MethodDefinition method) => method.HasBody && !method.HasGenericParameters && method.Parameters.Count == 0 &&
            !method.Body.HasVariables && !method.Body.HasExceptionHandlers && method.ImplAttributes == MethodImplAttributes.IL &&
            !method.IsPInvokeImpl && !method.HasSecurityDeclarations;
        if (instances.Length != 1 || statics.Length != 1 || !Plain(instances[0]) || !Plain(statics[0])) return false;
        var init = instances[0].Body.Instructions;
        var shared = statics[0].Body.Instructions;
        return init.Count == 3 && init[0].OpCode.Code == Code.Ldarg_0 && init[1].OpCode.Code == Code.Call &&
            init[1].Operand is MethodReference baseCall && baseCall.FullName == "System.Void System.Object::.ctor()" &&
            Runtime(baseCall.DeclaringType, "System.Object") && init[2].OpCode.Code == Code.Ret && shared.Count == 3 &&
            shared[0].OpCode.Code == Code.Newobj && shared[0].Operand is MethodReference allocation && allocation.Resolve() == instances[0] &&
            shared[1].OpCode.Code == Code.Stsfld && OwnedFieldBinding.Read(shared[1], type) == singleton && shared[2].OpCode.Code == Code.Ret;
    }

    private static bool Assertion(Instruction instruction)
    {
        if (instruction.Operand is not GenericInstanceMethod call || call.GenericArguments.Count != 1 ||
            !Runtime(call.GenericArguments[0], "System.ArgumentNullException") || call.HasThis || call.ExplicitThis ||
            call.CallingConvention != MethodCallingConvention.Generic || call.Parameters.Count != 1 || !FuncObject(call.Parameters[0].ParameterType) ||
            call.Resolve() is not MethodDefinition definition || definition.FullName != "T Xunit.Assert::Throws(System.Func`1<System.Object>)") return false;
        // The pinned implementation synchronously invokes the known Func once,
        // catches System.Exception, checks the exact runtime type and returns it.
        // The following pop prevents the exception object escaping this body.
        return Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(definition.Module.FileName))) == ReviewedRuntimeContracts.AssertionHash;
    }

    private static bool DelegateConstructor(Instruction instruction) => instruction.Operand is MethodReference call && call.Name == ".ctor" &&
        call.HasThis && !call.ExplicitThis && !call.HasGenericParameters && call.CallingConvention == MethodCallingConvention.Default &&
        FuncObject(call.DeclaringType) && Runtime(call.ReturnType, "System.Void") && call.Parameters.Count == 2 &&
        Runtime(call.Parameters[0].ParameterType, "System.Object") && Runtime(call.Parameters[1].ParameterType, "System.IntPtr") &&
        call.Resolve() is MethodDefinition definition && definition.IsConstructor && !definition.IsStatic;
    private static bool FuncObject(TypeReference type) => type is GenericInstanceType function && function.GenericArguments.Count == 1 &&
        Runtime(function.ElementType, "System.Func`1") && Runtime(function.GenericArguments[0], "System.Object");
    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
