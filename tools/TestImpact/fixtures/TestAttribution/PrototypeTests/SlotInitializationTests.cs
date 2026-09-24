using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class SlotInitializationTests
{
    public enum Mutation { Callback, ExtraCall, MutableField, PublicField, ForeignField, RepeatedWrite,
        WrongFieldType, MissingInitializer, Handler, Local, Synchronized, GenericOwner, UnexpectedConstructor }

    [Fact]
    public void PrivateSlotAndLockAllocationsHaveNoUserCallback()
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        var expected = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) ==
            ReviewedOwnerCompletion.RuntimeHash ? SlotInitializationContract.CallbackFreeAllocations : SlotInitializationContract.Unresolved;
        Assert.Equal(expected, SlotInitializationReader.Read(Fixture(assembly)));
    }

    [Theory]
    [InlineData(Mutation.Callback)]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.MutableField)]
    [InlineData(Mutation.PublicField)]
    [InlineData(Mutation.ForeignField)]
    [InlineData(Mutation.RepeatedWrite)]
    [InlineData(Mutation.WrongFieldType)]
    [InlineData(Mutation.MissingInitializer)]
    [InlineData(Mutation.Handler)]
    [InlineData(Mutation.Local)]
    [InlineData(Mutation.Synchronized)]
    [InlineData(Mutation.GenericOwner)]
    [InlineData(Mutation.UnexpectedConstructor)]
    public void InitializerEffectsOutsideTheReviewedShapeRemainOpen(Mutation mutation)
    {
        using var resolver = Resolver();
        using var assembly = Read(resolver);
        TypeDefinition type = Fixture(assembly);
        MethodDefinition initializer = type.Methods.Single(method => method.IsConstructor && method.IsStatic);
        var instructions = initializer.Body.Instructions;
        FieldDefinition slot = type.Fields.Single(field => field.Name == "Slot");
        switch (mutation)
        {
            case Mutation.Callback:
                instructions[0].Operand = type.Module.ImportReference(typeof(AsyncLocal<string>).GetConstructors()
                    .Single(constructor => constructor.GetParameters().Length == 1)); break;
            case Mutation.ExtraCall:
                instructions.Insert(0, Instruction.Create(OpCodes.Call, type.Module.ImportReference(
                    typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException("Missing Collect")))); break;
            case Mutation.MutableField: slot.IsInitOnly = false; break;
            case Mutation.PublicField: slot.IsPublic = true; break;
            case Mutation.ForeignField:
                type.Fields.Remove(slot); type.DeclaringType.Fields.Add(slot); break;
            case Mutation.RepeatedWrite:
                instructions.Insert(instructions.Count - 1, Instruction.Create(OpCodes.Newobj, (MethodReference)instructions[0].Operand));
                instructions.Insert(instructions.Count - 1, Instruction.Create(OpCodes.Stsfld, slot)); break;
            case Mutation.WrongFieldType: slot.FieldType = type.Module.TypeSystem.Object; break;
            case Mutation.MissingInitializer: type.Methods.Remove(initializer); break;
            case Mutation.Handler: initializer.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.Local: initializer.Body.Variables.Add(new(type.Module.TypeSystem.String)); break;
            case Mutation.Synchronized: initializer.ImplAttributes |= MethodImplAttributes.Synchronized; break;
            case Mutation.GenericOwner: type.GenericParameters.Add(new("T", type)); break;
            case Mutation.UnexpectedConstructor:
                instructions[0].Operand = type.Module.ImportReference(typeof(MemoryStream).GetConstructor(Type.EmptyTypes)
                    ?? throw new InvalidOperationException("Missing MemoryStream constructor")); break;
        }
        Assert.Equal(SlotInitializationContract.Unresolved, SlotInitializationReader.Read(type));
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(SlotInitializationTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }

    private static AssemblyDefinition Read(DefaultAssemblyResolver resolver) => AssemblyDefinition.ReadAssembly(
        typeof(SlotInitializationTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });

    private static TypeDefinition Fixture(AssemblyDefinition assembly) => assembly.MainModule.Types
        .Single(type => type.FullName == typeof(SlotInitializationTests).FullName).NestedTypes.Single(type => type.Name == "SlotFixture");

    private static class SlotFixture
    {
        private static readonly AsyncLocal<string?> Slot = new();
        private static readonly object Gate = new();
        internal static string? Current { get { lock (Gate) return Slot.Value; } }
    }
}
