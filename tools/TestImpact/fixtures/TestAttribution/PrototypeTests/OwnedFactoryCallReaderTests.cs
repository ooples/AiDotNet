using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class OwnedFactoryCallReaderTests
{
    public enum Mutation { None, NonNullArgument, AdditionalCallerInstruction, Catch, ReversedGuard, DifferentException, PrefixEffect }

    [Theory]
    [InlineData(Mutation.None, OwnedFactoryCallPath.NullGuardPrecedesOwnedChange)]
    [InlineData(Mutation.NonNullArgument, OwnedFactoryCallPath.Unresolved)]
    [InlineData(Mutation.AdditionalCallerInstruction, OwnedFactoryCallPath.Unresolved)]
    [InlineData(Mutation.Catch, OwnedFactoryCallPath.Unresolved)]
    [InlineData(Mutation.ReversedGuard, OwnedFactoryCallPath.Unresolved)]
    [InlineData(Mutation.DifferentException, OwnedFactoryCallPath.Unresolved)]
    [InlineData(Mutation.PrefixEffect, OwnedFactoryCallPath.Unresolved)]
    public void OnlyTheExactNullGuardPathIsRecognized(Mutation mutation, OwnedFactoryCallPath expected)
    {
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(OwnedFactoryCallReaderTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(OwnedFactoryCallReaderTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        var type = assembly.MainModule.Types.Single(type => type.FullName == typeof(OwnedFactoryCallReaderTests).FullName);
        MethodDefinition factory = type.Methods.Single(method => method.Name == nameof(Create));
        MethodDefinition caller = type.Methods.Single(method => method.Name == nameof(NullCall));
        switch (mutation)
        {
            case Mutation.NonNullArgument: caller.Body.Instructions[0] = Instruction.Create(OpCodes.Ldstr, "not null"); break;
            case Mutation.AdditionalCallerInstruction: caller.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.Catch: caller.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Catch)); break;
            case Mutation.ReversedGuard: factory.Body.Instructions[1].OpCode = OpCodes.Brfalse_S; break;
            case Mutation.DifferentException:
                factory.Body.Instructions[3].Operand = assembly.MainModule.ImportReference(typeof(InvalidOperationException).GetConstructor([typeof(string)])
                    ?? throw new InvalidOperationException("Missing exception constructor")); break;
            case Mutation.PrefixEffect: factory.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
        }
        Assert.Equal(expected, OwnedFactoryCallReader.Read(caller, factory, factory));
    }

    private sealed class Configuration { public bool Flag { get; set; } }
    private static Configuration Create(object? input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return new Configuration { Flag = true };
    }
    private static object NullCall() => Create(null);
}
