using System.Reflection;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serialization;

public sealed class ClonePartialGenericOwnerTests
{
    public ClonePartialGenericOwnerTests() => TestModuleInitializer.EnsureInitialized();

    public enum OwnerShape { FixedArgument, NestedFixedArgument, RepeatedParameter, ArrayParameter }

    [Theory]
    [InlineData(OwnerShape.FixedArgument, false)]
    [InlineData(OwnerShape.FixedArgument, true)]
    [InlineData(OwnerShape.NestedFixedArgument, false)]
    [InlineData(OwnerShape.NestedFixedArgument, true)]
    [InlineData(OwnerShape.RepeatedParameter, false)]
    [InlineData(OwnerShape.RepeatedParameter, true)]
    [InlineData(OwnerShape.ArrayParameter, false)]
    [InlineData(OwnerShape.ArrayParameter, true)]
    public void PublicBinding_PreservesFixedAndRepeatedGenericArguments(OwnerShape shape, bool compatible)
    {
        var definition = typeof(GenericPairOwner<,>);
        var placeholder = definition.GetGenericArguments()[0];
        Type firstActual = typeof(int);
        Type secondActual = compatible ? typeof(int) : typeof(float);
        Type firstDeclared;
        Type secondDeclared;
        switch (shape)
        {
            case OwnerShape.FixedArgument:
                firstDeclared = compatible ? typeof(int) : typeof(string);
                secondDeclared = placeholder;
                break;
            case OwnerShape.NestedFixedArgument:
                firstActual = typeof(List<int>);
                firstDeclared = compatible ? typeof(List<int>) : typeof(List<string>);
                secondDeclared = placeholder;
                break;
            case OwnerShape.RepeatedParameter:
                firstDeclared = placeholder;
                secondDeclared = placeholder;
                break;
            case OwnerShape.ArrayParameter:
                firstActual = typeof(int[]);
                firstDeclared = placeholder.MakeArrayType();
                secondDeclared = placeholder;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(shape));
        }

        var declaredOwner = definition.MakeGenericType(firstDeclared, secondDeclared);
        var member = declaredOwner.GetField("_options", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
            ?? throw new InvalidOperationException("Missing public-binding fixture member.");
        var modelType = typeof(GenericPairModel<,>).MakeGenericType(firstActual, secondActual);
        CloneRegistry.Register(new ClonePlan(modelType, Array.Empty<ClonePlanEntry>(), null, null,
            new[] { new[] { new CloneConstructorArgumentBinding("options", new[] { member }) } }));
        var original = Activator.CreateInstance(modelType, new CloneDeclaredOwnerRuntimeTests.OwnSettings { Width = 23 })
            ?? throw new InvalidOperationException("Missing actual generic fixture instance.");

        if (!compatible)
        {
            var error = Assert.Throws<InvalidOperationException>(() => CloneEngine.CopyConfiguration(original));
            Assert.Contains("explicit clone owner", error.Message);
            return;
        }
        var clone = CloneEngine.CopyConfiguration(original);
        Assert.NotSame(original, clone);
        Assert.Equal(23, Assert.IsType<int>(modelType.GetProperty("Width")?.GetValue(clone)));
    }

    public abstract class GenericPairOwner<TFirst, TSecond>
    {
        protected readonly CloneDeclaredOwnerRuntimeTests.OwnSettings _options;
        protected GenericPairOwner(CloneDeclaredOwnerRuntimeTests.OwnSettings? options)
            => _options = options ?? new CloneDeclaredOwnerRuntimeTests.OwnSettings();
        public int Width => _options.Width;
    }

    public sealed class GenericPairModel<TFirst, TSecond> : GenericPairOwner<TFirst, TSecond>
    {
        public GenericPairModel(CloneDeclaredOwnerRuntimeTests.OwnSettings? options = null) : base(options) { }
    }
}
