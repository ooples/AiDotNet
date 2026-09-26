using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum SignedLicenseContract { Unresolved, OwnedHmac }
internal enum SignedLicenseRequirement { HelperInitialization, StableKeySlot, SuccessfulInitialization, NoExceptionObservers }
internal sealed record SignedLicenseAssessment(SignedLicenseContract Contract, string KeySlot, SignedLicenseRequirement[] Requirements);

// This is the complete synchronous signing protocol, including disposal. The
// virtual hash/dispose calls operate on the exact locally allocated HMACSHA256,
// never a caller-provided HashAlgorithm. The helper's cctor is a separate proof.
internal static class SignedLicenseReader
{
    internal static SignedLicenseAssessment Read(MethodDefinition method)
    {
        SignedLicenseAssessment Unknown() => new(SignedLicenseContract.Unresolved, "", []);
        try
        {
            FieldDefinition? key = ReadShape(method);
            if (key is null || Hash(typeof(object).Assembly.Location) != ReviewedOwnerCompletion.RuntimeHash ||
                Hash(typeof(HMACSHA256).Assembly.Location) != ReviewedCryptoStartup.CryptoRuntimeHash) return Unknown();
            return new(SignedLicenseContract.OwnedHmac, key.FullName, Enum.GetValues<SignedLicenseRequirement>());
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    internal static FieldDefinition? ReadShape(MethodDefinition method)
    {
        try
        {
            if (!method.IsStatic || method.HasThis || method.ExplicitThis || method.HasGenericParameters || method.DeclaringType.HasGenericParameters ||
                method.CallingConvention != MethodCallingConvention.Default || method.IsPInvokeImpl || method.HasSecurityDeclarations ||
                method.ImplAttributes != MethodImplAttributes.IL || !method.HasBody || method.Parameters.Count != 2 ||
                !Type(method.ReturnType, typeof(string)) || !Type(method.Parameters[0].ParameterType, typeof(string)) ||
                !Type(method.Parameters[1].ParameterType, typeof(byte[]))) return null;
            var body = method.Body;
            System.Type[] locals = [typeof(string), typeof(HMACSHA256), typeof(byte[]), typeof(string), typeof(string)];
            if (body.Variables.Count != locals.Length || body.Variables.Where((local, index) => !Type(local.VariableType, locals[index])).Any()) return null;
            Code[] codes = [Code.Ldstr, Code.Ldarg_0, Code.Call, Code.Stloc_0, Code.Ldarg_1, Code.Dup, Code.Brtrue_S, Code.Pop,
                Code.Ldsfld, Code.Newobj, Code.Stloc_1, Code.Ldloc_1, Code.Call, Code.Ldloc_0, Code.Callvirt, Code.Callvirt,
                Code.Stloc_2, Code.Ldloc_2, Code.Call, Code.Ldc_I4_S, Code.Ldc_I4_S, Code.Callvirt, Code.Ldc_I4_S, Code.Ldc_I4_S,
                Code.Callvirt, Code.Ldc_I4_S, Code.Callvirt, Code.Stloc_3, Code.Ldloc_0, Code.Ldstr, Code.Ldloc_3, Code.Call,
                Code.Stloc_S, Code.Leave_S, Code.Ldloc_1, Code.Brfalse_S, Code.Ldloc_1, Code.Callvirt, Code.Endfinally, Code.Ldloc_S, Code.Ret];
            var il = body.Instructions;
            if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes) ||
                il[0].Operand is not string prefix || prefix != "aidn." || il[29].Operand is not string separator || separator != "." ||
                il[6].Operand != il[9] || il[33].Operand != il[39] || il[35].Operand != il[38] ||
                il[32].Operand != body.Variables[4] || il[39].Operand != body.Variables[4] ||
                OwnedFieldBinding.Read(il[8], method.DeclaringType) is not FieldDefinition key || !key.IsStatic || !key.IsInitOnly ||
                key.IsLiteral || !Type(key.FieldType, typeof(byte[])) ||
                key.CustomAttributes.Any(attribute => !Type(attribute.AttributeType, typeof(System.Runtime.CompilerServices.NullableAttribute)))) return null;
            int[] positions = [19, 20, 22, 23, 25];
            int[] values = [43, 45, 47, 95, 61];
            if (positions.Where((position, index) => il[position].Operand is not sbyte value || value != values[index]).Any()) return null;
            if (body.ExceptionHandlers.Count != 1) return null;
            var handler = body.ExceptionHandlers[0];
            if (handler.HandlerType != ExceptionHandlerType.Finally || handler.TryStart != il[11] || handler.TryEnd != il[34] ||
                handler.HandlerStart != il[34] || handler.HandlerEnd != il[39] || handler.FilterStart is not null || handler.CatchType is not null) return null;
            (int Index, System.Type Owner, string Name, System.Type[] Arguments)[] calls =
            [
                (2, typeof(string), nameof(string.Concat), [typeof(string), typeof(string)]),
                (9, typeof(HMACSHA256), ".ctor", [typeof(byte[])]),
                (12, typeof(System.Text.Encoding), "get_UTF8", []),
                (14, typeof(System.Text.Encoding), nameof(System.Text.Encoding.GetBytes), [typeof(string)]),
                (15, typeof(HashAlgorithm), nameof(HashAlgorithm.ComputeHash), [typeof(byte[])]),
                (18, typeof(Convert), nameof(Convert.ToBase64String), [typeof(byte[])]),
                (21, typeof(string), nameof(string.Replace), [typeof(char), typeof(char)]),
                (24, typeof(string), nameof(string.Replace), [typeof(char), typeof(char)]),
                (26, typeof(string), nameof(string.TrimEnd), [typeof(char)]),
                (31, typeof(string), nameof(string.Concat), [typeof(string), typeof(string), typeof(string)]),
                (37, typeof(IDisposable), nameof(IDisposable.Dispose), [])
            ];
            foreach (var site in calls)
            {
                System.Reflection.MethodBase expected = site.Name == ".ctor"
                    ? site.Owner.GetConstructor(site.Arguments) ?? throw new InvalidOperationException("Missing runtime constructor.")
                    : site.Owner.GetMethod(site.Name, site.Arguments) ?? throw new InvalidOperationException("Missing runtime method.");
                if (il[site.Index].Operand is not MethodReference call || call is MethodSpecification || call.HasGenericParameters || call.ExplicitThis ||
                    call.CallingConvention != MethodCallingConvention.Default || call.HasThis == expected.IsStatic ||
                    call.Resolve() is not MethodDefinition resolved || resolved.MetadataToken.ToInt32() != expected.MetadataToken ||
                    !SamePath(resolved.Module.FileName, expected.Module.Assembly.Location) || !Type(call.DeclaringType, site.Owner) ||
                    !OwnedFieldBinding.SameType(call.ReturnType, resolved.ReturnType) || call.Parameters.Count != resolved.Parameters.Count ||
                    call.Parameters.Where((parameter, index) => !OwnedFieldBinding.SameType(parameter.ParameterType, resolved.Parameters[index].ParameterType)).Any()) return null;
            }
            return key;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }

    internal static bool Type(TypeReference reference, System.Type expected)
    {
        if (expected.IsArray)
            return reference is ArrayType array && array.IsVector && array.Rank == 1 && expected.GetElementType() is System.Type element && Type(array.ElementType, element);
        return reference is not TypeSpecification && reference.FullName == expected.FullName && reference.Resolve() is TypeDefinition resolved &&
            resolved.MetadataToken.ToInt32() == expected.MetadataToken && SamePath(resolved.Module.FileName, expected.Assembly.Location);
    }

    private static bool SamePath(string first, string second) => string.Equals(Path.GetFullPath(first), Path.GetFullPath(second),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
            entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists || (entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Missing or linked signing-contract binary.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
