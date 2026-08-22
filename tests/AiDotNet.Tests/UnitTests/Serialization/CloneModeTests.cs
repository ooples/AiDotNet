using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

[AiDotNet.Attributes.ElementWiseShape]
[AiDotNet.Attributes.AutoParameters]
public sealed partial class CloneStateProbeLayer<T> : LayerBase<T>
{
    [AiDotNet.Attributes.Buffer(Name = "running", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _running = new([1]);

    [AiDotNet.Attributes.Buffer(Name = "optimizer", Role = PersistentTensorRole.OptimizerState)]
    private Tensor<T> _optimizer = new([1]);

    public CloneStateProbeLayer()
        : base([1], [1])
    {
        _running[0] = NumOps.FromDouble(-1);
        _optimizer[0] = NumOps.FromDouble(-2);
    }

    public override bool SupportsTraining => false;

    public double Running
    {
        get => NumOps.ToDouble(_running[0]);
        set => _running[0] = NumOps.FromDouble(value);
    }

    public double Optimizer
    {
        get => NumOps.ToDouble(_optimizer[0]);
        set => _optimizer[0] = NumOps.FromDouble(value);
    }

    public override void ResetState()
    {
    }
}

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

    [Fact]
    public void ShareRandomState_preserves_dropout_progress_while_default_derives_a_new_stream()
    {
        var original = new DropoutLayer<double>(0.5) { RandomSeed = 4242 };
        var input = new Tensor<double>(new[] { 1, 256 });
        input.Fill(1.0);

        // Advance the source once before cloning. Merely copying the seed would restart the clone
        // at mask zero and fail the next-output equality below.
        _ = original.Forward(input);
        var shared = (DropoutLayer<double>)original.Clone(
            new CloneOptions { ShareRandomState = true });
        var derived = (DropoutLayer<double>)original.Clone(CloneOptions.Full);

        var expectedNext = original.Forward(input);
        var sharedNext = shared.Forward(input);
        var derivedNext = derived.Forward(input);

        bool derivedDiffers = false;
        for (int i = 0; i < expectedNext.Length; i++)
        {
            Assert.Equal(expectedNext[i], sharedNext[i]);
            derivedDiffers |= expectedNext[i] != derivedNext[i];
        }

        Assert.True(derivedDiffers,
            "The default clone reused the source dropout stream instead of deriving an independent one.");
    }

    [Fact]
    public void Bare_clone_uses_the_generated_full_clone_path()
    {
        var original = Trained();

        var clone = (DenseLayer<double>)original.Clone();
        Mutate(clone);

        Assert.Equal(1.0, original.GetParameters()[0], precision: 10);
        Assert.Equal(99.0, clone.GetParameters()[0], precision: 10);
    }

    [Fact]
    public void Buffer_and_optimizer_state_flags_are_independent()
    {
        var original = new CloneStateProbeLayer<double>
        {
            Running = 17,
            Optimizer = 29,
        };

        var buffersOnly = (CloneStateProbeLayer<double>)original.Clone(new CloneOptions
        {
            IncludeParameters = false,
            IncludeBuffers = true,
            IncludeOptimizerState = false,
        });
        var optimizerOnly = (CloneStateProbeLayer<double>)original.Clone(new CloneOptions
        {
            IncludeParameters = false,
            IncludeBuffers = false,
            IncludeOptimizerState = true,
        });

        Assert.Equal(17, buffersOnly.Running);
        Assert.Equal(-2, buffersOnly.Optimizer);
        Assert.Equal(-1, optimizerOnly.Running);
        Assert.Equal(29, optimizerOnly.Optimizer);
    }

    [Fact]
    public void Shared_mode_aliases_registered_state_while_full_is_independent()
    {
        var original = new CloneStateProbeLayer<double>
        {
            Running = 17,
            Optimizer = 29,
        };

        var full = (CloneStateProbeLayer<double>)original.Clone(CloneOptions.Full);
        var shared = (CloneStateProbeLayer<double>)original.Clone(CloneOptions.Shared);

        full.Running = 41;
        full.Optimizer = 43;
        Assert.Equal(17, original.Running);
        Assert.Equal(29, original.Optimizer);

        shared.Running = 47;
        shared.Optimizer = 53;
        Assert.Equal(47, original.Running);
        Assert.Equal(53, original.Optimizer);
    }

    [Fact]
    public void CopyOnWrite_mode_splits_registered_state_on_write()
    {
        var original = new CloneStateProbeLayer<double>
        {
            Running = 17,
            Optimizer = 29,
        };

        var clone = (CloneStateProbeLayer<double>)original.Clone(CloneOptions.CopyOnWrite);
        Assert.Equal(original.Running, clone.Running);
        Assert.Equal(original.Optimizer, clone.Optimizer);

        clone.Running = 41;
        clone.Optimizer = 43;

        Assert.Equal(17, original.Running);
        Assert.Equal(29, original.Optimizer);
        Assert.Equal(41, clone.Running);
        Assert.Equal(43, clone.Optimizer);
    }

    private static void Mutate(DenseLayer<double> layer)
    {
        var p = layer.GetParameters();
        p[0] = 99.0;
        layer.UpdateParameters(p);
    }
}
