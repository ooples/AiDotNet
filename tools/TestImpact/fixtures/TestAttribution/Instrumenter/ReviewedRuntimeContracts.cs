using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Deliberately an explicit, code-reviewed catalog, not a configuration allowlist
// and not a policy learned from successful test executions. Changes to package
// bytes require another review even if the assembly version stays unchanged.
internal static class ReviewedRuntimeContracts
{
    internal const string AssertionHash = "bcd9711b22d227ac8cd4c1568e72433a0f576d2ec8aae64ac508081892015b1d";

    internal static RuntimeContractAssessment Assess(MethodReference reference)
    {
        RuntimeContractAssessment Unknown() => new(RuntimeContractStatus.Unknown, null,
            reference.FullName, "", "", [], []);
        // Names select a candidate only. Actual loaded bytes, module identity,
        // signature, and the reviewed implementation below establish the match.
        if (reference.FullName is not ("System.Void Xunit.Assert::True(System.Boolean,System.String)" or
            "System.Void Xunit.Assert::False(System.Boolean,System.String)")) return Unknown();
        try
        {
            MethodDefinition? definition = reference.Resolve();
            if (definition is null || definition.Module.Assembly.Name.FullName !=
                "xunit.assert, Version=2.9.3.0, Culture=neutral, PublicKeyToken=8d05b1bb7a6fdb6c" ||
                definition.Module.Assembly.Modules.Count != 1 || !definition.IsStatic ||
                definition.HasGenericParameters || reference.HasThis || reference.HasGenericParameters ||
                definition.FullName != reference.FullName) return Unknown();
            string path = definition.Module.FileName;
            if (string.IsNullOrEmpty(path) || HasLinkedComponent(path)) return Unknown();
            byte[] bytes = File.ReadAllBytes(path);
            string hash = Convert.ToHexStringLower(SHA256.HashData(bytes));
            if (hash != AssertionHash) return Unknown();
            // Resolve the transitive Nullable<bool> implementation using the
            // very same resolver as this call, not just the host runtime name.
            using var stream = new MemoryStream(bytes, writable: false);
            using var bound = AssemblyDefinition.ReadAssembly(stream, new ReaderParameters
            {
                AssemblyResolver = definition.Module.AssemblyResolver
            });
            MethodDefinition reviewed = bound.MainModule.Types.Single(type => type.FullName == "Xunit.Assert")
                .Methods.Single(method => method.FullName == reference.FullName);
            MethodReference nullableConstructor = reviewed.Body.Instructions.Select(instruction => instruction.Operand)
                .OfType<MethodReference>().Single(method => method.Name == ".ctor");
            MethodDefinition? resolvedNullable = nullableConstructor.Resolve();
            string runtime = typeof(bool).Assembly.Location;
            if (resolvedNullable is null || HasLinkedComponent(runtime) ||
                !string.Equals(Path.GetFullPath(resolvedNullable.Module.FileName), Path.GetFullPath(runtime),
                    OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)) return Unknown();
            byte[] runtimeBytes = File.ReadAllBytes(runtime);
            string runtimeHash = Convert.ToHexStringLower(SHA256.HashData(runtimeBytes));
            using var runtimeStream = new MemoryStream(runtimeBytes, writable: false);
            using var runtimeAssembly = AssemblyDefinition.ReadAssembly(runtimeStream);
            TypeDefinition? nullable = runtimeAssembly.MainModule.Types.SingleOrDefault(type => type.FullName == "System.Nullable`1");
            if (nullable is null || !IsReviewedNullableImplementation(nullable)) return Unknown();
            if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))) != hash) return Unknown();
            if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(runtime))) != runtimeHash) return Unknown();
            // Reviewed IL: bool is copied into Nullable<bool>; the passing branch
            // only reads HasValue/GetValueOrDefault and returns. The failing
            // branch constructs and throws TrueException/FalseException. That
            // branch can invoke process-wide first-chance handlers, so passing
            // the enclosing test does NOT establish SuccessfulAssertionCall.
            return new(RuntimeContractStatus.ReviewedConditional, RuntimeContractId.XunitBooleanAssertion293,
                reference.FullName, hash, runtimeHash,
                [RuntimeContractEffect.ReadScalarArguments, RuntimeContractEffect.ThrowOnFailure],
                [RuntimeContractRequirement.ExactBinaryAndRuntimeBinding,
                 RuntimeContractRequirement.InitializationEffectsProven,
                 RuntimeContractRequirement.SuccessfulAssertionCall,
                 RuntimeContractRequirement.FailurePropagatesToObservedOwner,
                 RuntimeContractRequirement.SuccessfulOwnerExecution]);
        }
        catch (Exception error) when (error is AssemblyResolutionException or ResolutionException or
            IOException or UnauthorizedAccessException or BadImageFormatException or InvalidOperationException or ArgumentException)
        {
            return Unknown();
        }
    }

    // The contract includes exactly the three transitive runtime operations on
    // its passing path. Do not assume an arbitrary same-named runtime is pure.
    internal static bool IsReviewedNullableImplementation(TypeDefinition type)
    {
        if (!type.IsValueType || type.GenericParameters.Count != 1 ||
            type.Methods.Any(method => method.IsConstructor && method.IsStatic)) return false;
        MethodDefinition[] constructors = type.Methods.Where(method => method.IsConstructor && !method.IsStatic).ToArray();
        MethodDefinition[] hasValues = type.Methods.Where(method => method.Name == "get_HasValue").ToArray();
        MethodDefinition[] getValues = type.Methods.Where(method => method.Name == "GetValueOrDefault" && method.Parameters.Count == 0).ToArray();
        if (constructors.Length != 1 || hasValues.Length != 1 || getValues.Length != 1) return false;
        MethodDefinition constructor = constructors[0], hasValue = hasValues[0], getValue = getValues[0];
        bool Shape(MethodDefinition method, params Code[] codes) => method.HasBody && !method.IsStatic &&
            !method.IsVirtual && !method.HasGenericParameters && method.Body.ExceptionHandlers.Count == 0 &&
            method.Body.Variables.Count == 0 && method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes);
        if (!Shape(constructor, Code.Ldarg_0, Code.Ldarg_1, Code.Stfld, Code.Ldarg_0, Code.Ldc_I4_1, Code.Stfld, Code.Ret) ||
            !Shape(hasValue, Code.Ldarg_0, Code.Ldfld, Code.Ret) || !Shape(getValue, Code.Ldarg_0, Code.Ldfld, Code.Ret) ||
            constructor.Parameters.Count != 1 || constructor.Parameters[0].ParameterType != type.GenericParameters[0] ||
            constructor.ReturnType.MetadataType != MetadataType.Void || hasValue.Parameters.Count != 0 ||
            hasValue.ReturnType.MetadataType != MetadataType.Boolean || getValue.ReturnType != type.GenericParameters[0]) return false;
        FieldDefinition? Field(Instruction instruction) => instruction.Operand is FieldReference reference ? reference.Resolve() : null;
        FieldDefinition? value = Field(constructor.Body.Instructions[2]);
        FieldDefinition? flag = Field(constructor.Body.Instructions[5]);
        return value is not null && flag is not null && value.DeclaringType == type && flag.DeclaringType == type &&
            !value.IsStatic && !flag.IsStatic && value.FieldType == type.GenericParameters[0] &&
            flag.FieldType.MetadataType == MetadataType.Boolean && Field(hasValue.Body.Instructions[1]) == flag &&
            Field(getValue.Body.Instructions[1]) == value;
    }

    private static bool HasLinkedComponent(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if ((entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null) return true;
        return false;
    }
}
