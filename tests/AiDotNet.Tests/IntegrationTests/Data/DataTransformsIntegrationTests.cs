using AiDotNet.Data.Transforms;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

/// <summary>
/// Integration tests for data transform classes:
/// Compose, IdentityTransform, LambdaTransform.
/// </summary>
public class DataTransformsIntegrationTests
{
    #region IdentityTransform

    [Fact]
    public void IdentityTransform_ReturnsInputUnchanged_Double()
    {
        var transform = new IdentityTransform<double>();
        Assert.Equal(42.0, transform.Apply(42.0));
    }

    [Fact]
    public void IdentityTransform_ReturnsInputUnchanged_String()
    {
        var transform = new IdentityTransform<string>();
        Assert.Equal("hello", transform.Apply("hello"));
    }

    [Fact]
    public void IdentityTransform_Null_ReturnsNull()
    {
        var transform = new IdentityTransform<string>();
        Assert.Null(transform.Apply(null));
    }

    [Fact]
    public void IdentityTransform_ReturnsInputUnchanged_Array()
    {
        var transform = new IdentityTransform<int[]>();
        var input = new[] { 1, 2, 3 };
        var result = transform.Apply(input);
        Assert.Same(input, result);
    }

    #endregion

    #region LambdaTransform

    [Fact]
    public void LambdaTransform_AppliesFunction()
    {
        var transform = new LambdaTransform<double, double>(x => x * 2.0);
        Assert.Equal(10.0, transform.Apply(5.0));
    }

    [Fact]
    public void LambdaTransform_DifferentInputOutput()
    {
        var transform = new LambdaTransform<int, string>(x => x.ToString());
        Assert.Equal("42", transform.Apply(42));
    }

    [Fact]
    public void LambdaTransform_NullFunc_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new LambdaTransform<int, int>(null));
    }

    [Fact]
    public void LambdaTransform_ComplexTransformation()
    {
        var transform = new LambdaTransform<double[], double>(arr => arr.Sum());
        Assert.Equal(6.0, transform.Apply(new[] { 1.0, 2.0, 3.0 }));
    }

    #endregion

    #region Compose

    [Fact]
    public void Compose_SingleTransform_AppliesIt()
    {
        var doubler = new LambdaTransform<double, double>(x => x * 2.0);
        var composed = new Compose<double>(doubler);
        Assert.Equal(10.0, composed.Apply(5.0));
        Assert.Equal(1, composed.Count);
    }

    [Fact]
    public void Compose_MultipleTransforms_AppliesInOrder()
    {
        var add1 = new LambdaTransform<int, int>(x => x + 1);
        var multiply2 = new LambdaTransform<int, int>(x => x * 2);
        var composed = new Compose<int>(add1, multiply2);

        // (5 + 1) * 2 = 12
        Assert.Equal(12, composed.Apply(5));
        Assert.Equal(2, composed.Count);
    }

    [Fact]
    public void Compose_OrderMatters()
    {
        var add1 = new LambdaTransform<int, int>(x => x + 1);
        var multiply2 = new LambdaTransform<int, int>(x => x * 2);

        var addFirst = new Compose<int>(add1, multiply2);
        var mulFirst = new Compose<int>(multiply2, add1);

        // (5 + 1) * 2 = 12
        Assert.Equal(12, addFirst.Apply(5));
        // (5 * 2) + 1 = 11
        Assert.Equal(11, mulFirst.Apply(5));
    }

    [Fact]
    public void Compose_WithIdentity_NoEffect()
    {
        var identity = new IdentityTransform<int>();
        var add1 = new LambdaTransform<int, int>(x => x + 1);
        var composed = new Compose<int>(identity, add1, identity);

        Assert.Equal(6, composed.Apply(5));
        Assert.Equal(3, composed.Count);
    }

    [Fact]
    public void Compose_EmptyTransformArray_PassesThrough()
    {
        var composed = new Compose<int>(Array.Empty<ITransform<int, int>>());
        Assert.Equal(42, composed.Apply(42));
        Assert.Equal(0, composed.Count);
    }

    [Fact]
    public void Compose_NullTransforms_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new Compose<int>((ITransform<int, int>[])null));
    }

    [Fact]
    public void Compose_FromEnumerable_Works()
    {
        var transforms = new List<ITransform<int, int>>
        {
            new LambdaTransform<int, int>(x => x + 10),
            new LambdaTransform<int, int>(x => x * 3)
        };
        var composed = new Compose<int>(transforms);

        // (5 + 10) * 3 = 45
        Assert.Equal(45, composed.Apply(5));
    }

    [Fact]
    public void Compose_NullEnumerable_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new Compose<int>((IEnumerable<ITransform<int, int>>)null));
    }

    [Fact]
    public void Compose_ChainedCompositions()
    {
        var inner = new Compose<int>(
            new LambdaTransform<int, int>(x => x + 1),
            new LambdaTransform<int, int>(x => x * 2));
        var outer = new Compose<int>(inner, new LambdaTransform<int, int>(x => x - 3));

        // ((5 + 1) * 2) - 3 = 9
        Assert.Equal(9, outer.Apply(5));
    }

    #endregion
}
