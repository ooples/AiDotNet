using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum LicenseSupportInitializerContract { Unresolved, OwnedTestKeys }
internal enum LicenseSupportInitializerRequirement { NoExternalKeySlotMutation }
internal sealed record LicenseSupportInitializerAssessment(LicenseSupportInitializerContract Contract, string ByteKeySlot, string PairSlot,
    CryptoStartupAssessment? Crypto, LicenseSupportInitializerRequirement[] Requirements);

// Proves the complete helper cctor, including the otherwise hidden Ed25519
// allocation. Later license validation/signing and external accesses to these
// fields are not part of this initializer contract.
internal static class LicenseSupportInitializerReader
{
    internal static LicenseSupportInitializerAssessment Read(TypeDefinition owner)
    {
        LicenseSupportInitializerAssessment Unknown() => new(LicenseSupportInitializerContract.Unresolved, "", "", null, []);
        try
        {
            MethodDefinition? factory = ReadShape(owner);
            if (factory is null) return Unknown();
            CryptoStartupAssessment crypto = ReviewedCryptoStartup.Read(factory);
            if (crypto.Contract != CryptoStartupContract.OwnedEd25519KeyPair) return Unknown();
            var il = owner.Methods.Single(method => method.IsConstructor && method.IsStatic).Body.Instructions;
            return new(LicenseSupportInitializerContract.OwnedTestKeys, ((FieldReference)il[3].Operand).FullName,
                ((FieldReference)il[5].Operand).FullName, crypto with
                { Requirements = crypto.Requirements.Where(requirement => requirement != CryptoStartupRequirement.DeclaringTypeInitialization).ToArray() },
                [LicenseSupportInitializerRequirement.NoExternalKeySlotMutation]);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    internal static MethodDefinition? ReadShape(TypeDefinition owner)
    {
        try
        {
            if (!owner.IsAbstract || !owner.IsSealed || owner.HasGenericParameters || owner.HasInterfaces || owner.BaseType is null ||
                !Runtime(owner.BaseType, "System.Object")) return null;
            MethodDefinition[] constructors = owner.Methods.Where(method => method.IsConstructor).ToArray();
            if (constructors.Length != 1) return null;
            MethodDefinition initializer = constructors[0];
            if (!initializer.IsStatic || initializer.HasThis || initializer.ExplicitThis || initializer.HasGenericParameters ||
                initializer.Parameters.Count != 0 || !initializer.HasBody || initializer.ImplAttributes != MethodImplAttributes.IL ||
                initializer.IsPInvokeImpl || initializer.HasSecurityDeclarations || initializer.Body.HasVariables || initializer.Body.HasExceptionHandlers ||
                !Runtime(initializer.ReturnType, "System.Void")) return null;
            var il = initializer.Body.Instructions;
            if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Call, Code.Ldstr, Code.Callvirt, Code.Stsfld,
                Code.Call, Code.Stsfld, Code.Ret }) || il[1].Operand is not string ||
                il[0].Operand is not MethodReference encoding || encoding.FullName != "System.Text.Encoding System.Text.Encoding::get_UTF8()" ||
                encoding.HasThis || encoding.ExplicitThis || encoding.HasGenericParameters || encoding.Parameters.Count != 0 ||
                encoding.CallingConvention != MethodCallingConvention.Default ||
                !Runtime(encoding.ReturnType, "System.Text.Encoding") || !Runtime(encoding.DeclaringType, "System.Text.Encoding") ||
                encoding.Resolve() is not MethodDefinition getter || !getter.IsStatic || getter.FullName != encoding.FullName ||
                il[2].Operand is not MethodReference convert || convert.FullName != "System.Byte[] System.Text.Encoding::GetBytes(System.String)" ||
                !convert.HasThis || convert.ExplicitThis || convert.HasGenericParameters || convert.Parameters.Count != 1 ||
                convert.CallingConvention != MethodCallingConvention.Default ||
                !Runtime(convert.Parameters[0].ParameterType, "System.String") || !Runtime(convert.DeclaringType, "System.Text.Encoding") ||
                convert.ReturnType is not ArrayType array || array.Rank != 1 || !array.IsVector || !Runtime(array.ElementType, "System.Byte") ||
                convert.Resolve() is not MethodDefinition conversion || conversion.IsStatic || conversion.FullName != convert.FullName ||
                OwnedFieldBinding.Read(il[3], owner) is not FieldDefinition bytes || OwnedFieldBinding.Read(il[5], owner) is not FieldDefinition pair ||
                bytes == pair || bytes.FieldType is not ArrayType byteArray || !byteArray.IsVector || byteArray.Rank != 1 ||
                !Runtime(byteArray.ElementType, "System.Byte") ||
                new[] { bytes, pair }.Any(field => !field.IsStatic || !field.IsInitOnly || field.IsLiteral || field.InitialValue.Length != 0 ||
                    field.CustomAttributes.Any(attribute => !Runtime(attribute.AttributeType, "System.Runtime.CompilerServices.NullableAttribute"))) ||
                owner.Fields.Any(field => !field.IsLiteral && field != bytes && field != pair) ||
                il[4].Operand is not MethodReference call || call.HasThis || call.ExplicitThis || call.HasGenericParameters || call.Parameters.Count != 0 ||
                call.CallingConvention != MethodCallingConvention.Default || !OwnedFieldBinding.SelfType(call.DeclaringType, owner) ||
                call.Resolve() is not MethodDefinition factory || factory.DeclaringType != owner ||
                !OwnedFieldBinding.SameType(call.ReturnType, factory.ReturnType) || !OwnedFieldBinding.SameType(pair.FieldType, factory.ReturnType) ||
                !ReviewedCryptoStartup.ReadShape(factory)) return null;
            return factory;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return null; }
    }

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
