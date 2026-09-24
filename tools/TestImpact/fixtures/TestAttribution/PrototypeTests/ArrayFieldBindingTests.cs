using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ArrayFieldBindingTests
{
    public enum Mutation { Element, Rank, LowerBound, UpperBound }

    [Fact]
    public void IdenticalVectorElementTypesBind()
    {
        using var resolver = Resolver();
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(ArrayFieldBindingTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
        var element = assembly.MainModule.ImportReference(typeof(byte));
        Assert.True(OwnedFieldBinding.SameType(new ArrayType(element), new ArrayType(element)));
    }

    [Theory]
    [InlineData(Mutation.Element)]
    [InlineData(Mutation.Rank)]
    [InlineData(Mutation.LowerBound)]
    [InlineData(Mutation.UpperBound)]
    public void ArrayStorageShapesCannotBeConfused(Mutation mutation)
    {
        using var resolver = Resolver();
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(ArrayFieldBindingTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
        var element = assembly.MainModule.ImportReference(typeof(byte));
        var first = new ArrayType(element);
        var second = new ArrayType(element);
        switch (mutation)
        {
            case Mutation.Element: second = new ArrayType(assembly.MainModule.ImportReference(typeof(int))); break;
            case Mutation.Rank: second = new ArrayType(element, 2); break;
            case Mutation.LowerBound: second.Dimensions[0] = new ArrayDimension(1, null); break;
            case Mutation.UpperBound: second.Dimensions[0] = new ArrayDimension(null, 4); break;
        }
        Assert.False(OwnedFieldBinding.SameType(first, second));
        Assert.False(OwnedFieldBinding.SameType(second, first));
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }
}
