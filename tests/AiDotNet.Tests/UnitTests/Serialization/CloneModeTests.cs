using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// The three sharing modes have to differ under mutation, or they are one mode with three names.
/// </summary>
/// <remarks>
/// Each test writes through the copy and then reads the ORIGINAL. That is the only observation that
/// separates them: all three produce a copy with identical values, and what happens next is the
/// whole distinction.
/// </remarks>
public class CloneModeTests
{
    private static DenseLayer<double> Trained()
    {
        // DenseLayer's input width is lazy, so it owns no parameters until something flows
        // through it. Forward once, then the weights exist to be set and compared.
        var layer = new DenseLayer<double>(2);
        layer.Forward(new Tensor<double>(new[] { 1, 3 }));

        var p = layer.GetParameters();
        for (var i = 0; i < p.Length; i++) p[i] = i + 1.0;
        layer.UpdateParameters(p);
        return layer;
    }

    [Fact]
    public void Deep_leaves_the_original_alone()
    {
        var original = Trained();
        var clone = (DenseLayer<double>)original.Clone(CloneOptions.Full);

        Mutate(clone);

        Assert.Equal(1.0, original.GetParameters()[0], precision: 10);
    }

    [Fact]
    public void CopyOnWrite_reads_the_same_and_still_splits_on_write()
    {
        var original = Trained();
        var clone = (DenseLayer<double>)original.Clone(CloneOptions.CopyOnWrite);

        // Identical before anybody writes -- that is the point of it being free.
        Assert.Equal(original.GetParameters()[0], clone.GetParameters()[0], precision: 10);

        Mutate(clone);

        // The write splits them, so this is a copy despite having shared storage a moment ago.
        Assert.Equal(1.0, original.GetParameters()[0], precision: 10);
    }

    [Fact]
    public void Shared_is_an_alias_and_writes_reach_the_original()
    {
        var original = Trained();
        var clone = (DenseLayer<double>)original.Clone(CloneOptions.Shared);

        Mutate(clone);

        // NOT a copy. This asserts the footgun on purpose: if this ever starts passing as 1.0,
        // Shared has silently become CopyOnWrite and callers relying on the alias are broken.
        Assert.Equal(99.0, original.GetParameters()[0], precision: 10);
    }

    [Fact]
    public void ShareRandomState_carries_the_seed_only_when_asked()
    {
        var original = Trained();
        original.RandomSeed = 4242;

        var derived = (DenseLayer<double>)original.Clone(CloneOptions.Full);
        var shared = (DenseLayer<double>)original.Clone(
            new CloneOptions { ShareRandomState = true });

        Assert.Equal(4242, shared.RandomSeed);
        Assert.NotEqual(4242, derived.RandomSeed ?? 0);
    }

    private static void Mutate(DenseLayer<double> layer)
    {
        var p = layer.GetParameters();
        p[0] = 99.0;
        layer.UpdateParameters(p);
    }
}
