using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ScopedAsyncLocalTests
{
    public enum Mutation { FactoryCallback, WrongGetSlot, WrongSetSlot, WrongRestoreSlot, MutablePrevious,
        SharedPrevious, WrongPrevious, ExtraDispose, NotSealed, AddedFinalizer, SwallowHandler, CallbackInitializer,
        WrongScopeInterface, ExternalScopeMethod, MutableSlot, MissingInitializer, SynchronizedFactory, SynchronizedDispose, InternalCall }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SaveSetRestoreStillRequiresLifetimeAndContextProof(bool stackForm)
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        TypeDefinition type = Fixture(assembly);
        MethodDefinition factory = type.Methods.Single(method => method.Name == "Set");
        if (stackForm)
        {
            Assert.Single(factory.Body.Variables);
            factory.Body.Instructions.RemoveAt(6);
            factory.Body.Instructions.RemoveAt(2);
            factory.Body.Variables.Clear();
        }
        AsyncLocalScopeAssessment result = ScopedAsyncLocalReader.Read(factory,
            type.Methods.Single(method => method.Name == "get_Current"));
        if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) != ReviewedOwnerCompletion.RuntimeHash)
        {
            Assert.Equal(AsyncLocalScopeContract.Unresolved, result.Contract);
            return;
        }
        Assert.Equal(AsyncLocalScopeContract.RestoresPreviousString, result.Contract);
        Assert.Equal(Enum.GetValues<AsyncLocalScopeRequirement>(), result.Requirements);
        Assert.Equal(ReviewedOwnerCompletion.RuntimeHash, result.RuntimeHash);
        Assert.Contains("Slot", result.Slot, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Mutation.FactoryCallback)]
    [InlineData(Mutation.WrongGetSlot)]
    [InlineData(Mutation.WrongSetSlot)]
    [InlineData(Mutation.WrongRestoreSlot)]
    [InlineData(Mutation.MutablePrevious)]
    [InlineData(Mutation.SharedPrevious)]
    [InlineData(Mutation.WrongPrevious)]
    [InlineData(Mutation.ExtraDispose)]
    [InlineData(Mutation.NotSealed)]
    [InlineData(Mutation.AddedFinalizer)]
    [InlineData(Mutation.SwallowHandler)]
    [InlineData(Mutation.CallbackInitializer)]
    [InlineData(Mutation.WrongScopeInterface)]
    [InlineData(Mutation.ExternalScopeMethod)]
    [InlineData(Mutation.MutableSlot)]
    [InlineData(Mutation.MissingInitializer)]
    [InlineData(Mutation.SynchronizedFactory)]
    [InlineData(Mutation.SynchronizedDispose)]
    [InlineData(Mutation.InternalCall)]
    public void AlteredScopeSemanticsRemainUnresolved(Mutation mutation)
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        TypeDefinition type = Fixture(assembly);
        MethodDefinition factory = type.Methods.Single(method => method.Name == "Set");
        MethodDefinition getter = type.Methods.Single(method => method.Name == "get_Current");
        TypeDefinition scope = type.NestedTypes.Single();
        MethodDefinition dispose = scope.Methods.Single(method => method.Name == "Dispose");
        FieldDefinition slot = type.Fields.Single(field => field.Name == "Slot");
        FieldDefinition previous = scope.Fields.Single();
        var otherSlot = new FieldDefinition("OtherSlot", FieldAttributes.Private | FieldAttributes.Static | FieldAttributes.InitOnly, slot.FieldType);
        type.Fields.Add(otherSlot);
        switch (mutation)
        {
            case Mutation.FactoryCallback:
                factory.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Call, type.Module.ImportReference(
                    typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException("Missing Collect")))); break;
            case Mutation.WrongGetSlot: getter.Body.Instructions[0].Operand = otherSlot; break;
            case Mutation.WrongSetSlot: factory.Body.Instructions.Single(instruction => instruction.OpCode.Code == Code.Ldsfld && instruction != factory.Body.Instructions[0]).Operand = otherSlot; break;
            case Mutation.WrongRestoreSlot: dispose.Body.Instructions[0].Operand = otherSlot; break;
            case Mutation.MutablePrevious: previous.IsInitOnly = false; break;
            case Mutation.SharedPrevious: previous.IsStatic = true; break;
            case Mutation.WrongPrevious:
                dispose.Body.Instructions[2].Operand = new FieldDefinition("Other", FieldAttributes.Private, previous.FieldType); break;
            case Mutation.ExtraDispose: dispose.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.NotSealed: scope.IsSealed = false; break;
            case Mutation.AddedFinalizer:
                scope.Methods.Add(new("Finalize", MethodAttributes.Family | MethodAttributes.Virtual, type.Module.TypeSystem.Void)); break;
            case Mutation.SwallowHandler: dispose.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Catch)); break;
            case Mutation.CallbackInitializer:
                MethodDefinition initializer = type.Methods.Single(method => method.IsConstructor && method.IsStatic);
                initializer.Body.Instructions[0].Operand = type.Module.ImportReference(typeof(AsyncLocal<string>).GetConstructors()
                    .Single(constructor => constructor.GetParameters().Length == 1)); break;
            case Mutation.WrongScopeInterface: scope.Interfaces.Clear(); break;
            case Mutation.ExternalScopeMethod: dispose.Overrides.Add(type.Module.ImportReference(typeof(IDisposable).GetMethod("Dispose")
                ?? throw new InvalidOperationException("Missing Dispose"))); break;
            case Mutation.MutableSlot: slot.IsInitOnly = false; break;
            case Mutation.MissingInitializer: type.Methods.Remove(type.Methods.Single(method => method.IsConstructor && method.IsStatic)); break;
            case Mutation.SynchronizedFactory: factory.ImplAttributes |= MethodImplAttributes.Synchronized; break;
            case Mutation.SynchronizedDispose: dispose.ImplAttributes |= MethodImplAttributes.Synchronized; break;
            case Mutation.InternalCall: factory.ImplAttributes |= MethodImplAttributes.InternalCall; break;
        }
        AsyncLocalScopeAssessment result = ScopedAsyncLocalReader.Read(factory, getter);
        Assert.Equal(AsyncLocalScopeContract.Unresolved, result.Contract);
        Assert.Empty(result.Requirements);
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(ScopedAsyncLocalTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }

    private static AssemblyDefinition Read(DefaultAssemblyResolver resolver) => AssemblyDefinition.ReadAssembly(
        typeof(ScopedAsyncLocalTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });

    private static TypeDefinition Fixture(AssemblyDefinition assembly) => assembly.MainModule.Types
        .Single(type => type.FullName == typeof(ScopedAsyncLocalTests).FullName).NestedTypes.Single(type => type.Name == "ScopeFixture");

    private static class ScopeFixture
    {
        private static readonly AsyncLocal<string?> Slot = new();
        internal static string? Current => Slot.Value;
        internal static IDisposable Set(string? value)
        {
            string? previous = Slot.Value;
            Slot.Value = value;
            return new Scope(previous);
        }
        private sealed class Scope : IDisposable
        {
            private readonly string? previous;
            internal Scope(string? previous) { this.previous = previous; }
            public void Dispose() { Slot.Value = previous; }
        }
    }
}
