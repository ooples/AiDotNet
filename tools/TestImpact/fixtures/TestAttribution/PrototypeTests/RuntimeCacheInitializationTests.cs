using System.Collections.Concurrent;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class RuntimeCacheInitializationTests
{
    public enum Mutation { MutableField, PublicField, InstanceField, ForeignField, ExtraCall, ExceptionHandler,
        Synchronized, CustomKey, CustomValue, ComparerArgument, GenericConstructor, ThreadStatic, GenericOwner, WrongBase }

    [Fact]
    public void PrivateTypeCacheAllocationHasAnExactRuntimeBinding()
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        RuntimeCacheInitializationAssessment result = RuntimeCacheInitializationReader.Read(Fixture(assembly));
        bool supported = Hash(typeof(object).Assembly.Location) == ReviewedOwnerCompletion.RuntimeHash &&
            Hash(typeof(ConcurrentDictionary<,>).Assembly.Location) == RuntimeCacheInitializationReader.CollectionsHash;
        Assert.Equal(supported ? RuntimeCacheInitializationContract.PrivateTypeAccelerationCache : RuntimeCacheInitializationContract.Unresolved, result.Contract);
        if (supported)
        {
            Assert.NotEmpty(result.Field);
            Assert.Equal(RuntimeCacheInitializationReader.CollectionsHash, result.CollectionsHash);
            Assert.Equal(ReviewedOwnerCompletion.RuntimeHash, result.RuntimeHash);
        }
    }

    [Theory]
    [InlineData(Mutation.MutableField)]
    [InlineData(Mutation.PublicField)]
    [InlineData(Mutation.InstanceField)]
    [InlineData(Mutation.ForeignField)]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.ExceptionHandler)]
    [InlineData(Mutation.Synchronized)]
    [InlineData(Mutation.CustomKey)]
    [InlineData(Mutation.CustomValue)]
    [InlineData(Mutation.ComparerArgument)]
    [InlineData(Mutation.GenericConstructor)]
    [InlineData(Mutation.ThreadStatic)]
    [InlineData(Mutation.GenericOwner)]
    [InlineData(Mutation.WrongBase)]
    public void ChangedAllocationCannotInheritTheCacheContract(Mutation mutation)
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        TypeDefinition type = Fixture(assembly);
        FieldDefinition field = Assert.Single(type.Fields);
        MethodDefinition initializer = Assert.Single(type.Methods, method => method.IsConstructor && method.IsStatic);
        var body = initializer.Body.Instructions;
        var constructor = (MethodReference)body[0].Operand;
        var cache = (GenericInstanceType)constructor.DeclaringType;
        switch (mutation)
        {
            case Mutation.MutableField: field.IsInitOnly = false; break;
            case Mutation.PublicField: field.IsPublic = true; break;
            case Mutation.InstanceField: field.IsStatic = false; break;
            case Mutation.ForeignField:
                body[1].Operand = new FieldDefinition("foreign", Mono.Cecil.FieldAttributes.Private | Mono.Cecil.FieldAttributes.Static | Mono.Cecil.FieldAttributes.InitOnly, field.FieldType); break;
            case Mutation.ExtraCall: body.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.ExceptionHandler: initializer.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.Synchronized: initializer.ImplAttributes |= Mono.Cecil.MethodImplAttributes.Synchronized; break;
            case Mutation.CustomKey: cache.GenericArguments[0] = assembly.MainModule.TypeSystem.String; break;
            case Mutation.CustomValue: cache.GenericArguments[1] = assembly.MainModule.TypeSystem.Object; break;
            case Mutation.ComparerArgument: constructor.Parameters.Add(new(assembly.MainModule.TypeSystem.Object)); break;
            case Mutation.GenericConstructor: body[0].Operand = new GenericInstanceMethod(constructor); break;
            case Mutation.ThreadStatic:
                var marker = typeof(ThreadStaticAttribute).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException("Missing thread-static marker.");
                field.CustomAttributes.Add(new(assembly.MainModule.ImportReference(marker))); break;
            case Mutation.GenericOwner: type.GenericParameters.Add(new("T", type)); break;
            case Mutation.WrongBase: type.BaseType = assembly.MainModule.TypeSystem.String; break;
        }
        var result = RuntimeCacheInitializationReader.Read(type);
        Assert.Equal(RuntimeCacheInitializationContract.Unresolved, result.Contract);
        Assert.Empty(result.Field);
        Assert.Empty(result.RuntimeHash);
        Assert.Empty(result.CollectionsHash);
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(RuntimeCacheInitializationTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }

    private static AssemblyDefinition Read(DefaultAssemblyResolver resolver) => AssemblyDefinition.ReadAssembly(
        typeof(RuntimeCacheInitializationTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });

    private static TypeDefinition Fixture(AssemblyDefinition assembly) => assembly.MainModule.Types
        .Single(type => type.FullName == typeof(RuntimeCacheInitializationTests).FullName).NestedTypes.Single(type => type.Name == nameof(CacheFixture));

    private static string Hash(string file) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));

    private static class CacheFixture
    {
        private static readonly ConcurrentDictionary<Type, (bool Simd, bool Gpu)> Cache = new();
        internal static int Count => Cache.Count;
    }
}
