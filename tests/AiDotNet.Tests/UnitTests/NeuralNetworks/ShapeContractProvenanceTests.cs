using System;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Guards the shape contract against being silently weakened or reverted.
/// </summary>
/// <remarks>
/// <para>
/// This problem has been fixed and undone repeatedly. Previous fixes corrected the declared numbers
/// on individual models, so the next model with a custom forward reintroduced it. These tests pin
/// the MECHANISM instead: layer types state a shape relation, the sequential walk marks the axes it
/// merely inferred, and the contract asserts only what a layer genuinely claims. If any of that is
/// removed, one of these fails and names what was lost — rather than the failure reappearing months
/// later as an unrelated model going red.
/// </para>
/// </remarks>
public class ShapeContractProvenanceTests
{
    private const BindingFlags Any = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    /// <summary>Shape-preserving layers must say so, or nothing can tell a wrong layer from a wrong declaration.</summary>
    [Theory]
    [InlineData(typeof(BatchNormalizationLayer<double>), ShapeRelationKind.Identity)]
    [InlineData(typeof(ActivationLayer<double>), ShapeRelationKind.Identity)]
    [InlineData(typeof(ConvolutionalLayer<double>), ShapeRelationKind.Convolutional)]
    public void LayerTypesDeclareTheirShapeRelation(Type layerType, ShapeRelationKind expected)
    {
        var prop = layerType.GetProperty("OutputShapeRelation", Any);
        Assert.True(prop is not null,
            $"{layerType.Name} no longer exposes OutputShapeRelation. Without a relation a shape " +
            "mismatch can only report two disagreeing numbers, which reads as 'this layer is broken' " +
            "even when the layer computed correctly and the DECLARATION was wrong.");

        var instance = Instantiate(layerType);
        if (instance is null) return;   // ctor shape varies; the declaration check above is the point

        var actual = (ShapeRelationKind)prop!.GetValue(instance)!;
        Assert.True(actual == expected,
            $"{layerType.Name} declares {actual} but should declare {expected}. The relation is what " +
            "lets a declared output be read as a claim about the input, so weakening it to Unknown " +
            "silently disables the diagnosis for every model using this layer.");
    }

    /// <summary>
    /// A declared shape whose relation is Identity is a claim about the input, and must be
    /// invertible — that inversion is what identifies the propagation as the author.
    /// </summary>
    [Fact]
    public void IdentityRelationInvertsToTheClaimedInput()
    {
        var layer = MakeDeclaring([8, 32, 32]);
        var implied = typeof(LayerBase<double>)
            .GetMethod("ImpliedInputShape", Any)!
            .Invoke(layer, new object[] { new[] { 8, 32, 32 } }) as int[];

        Assert.True(implied is not null,
            "An Identity layer must be able to state the input its declared output implies; without " +
            "that inversion the failure text cannot distinguish a wrong layer from a wrong declaration.");
        Assert.Equal(new[] { 8, 32, 32 }, implied!);
    }

    /// <summary>
    /// The sequential walk must not leave layers asserting spatial dimensions it merely inferred.
    /// </summary>
    /// <remarks>
    /// This is the regression that produced 27 failures across MaskAdapter, Mask2Former and SlimSAM:
    /// each declared [C, 32, 32] from the architecture's input while its real forward received
    /// [C, 8, 8]. The layers were correct; ResolveLazyLayerShapes had written a shape describing a
    /// chain those models do not run. If the marking is removed, those models go red again.
    /// </remarks>
    [Fact]
    public void PropagatedSpatialAxesAreNotAssertedAgainstTheForward()
    {
        var relax = typeof(LayerBase<double>).GetMethod("RelaxPropagatedSpatialAxes", Any);
        Assert.True(relax is not null,
            "LayerBase.RelaxPropagatedSpatialAxes is gone. ResolveLazyLayerShapes uses it to record " +
            "that a layer's spatial axes were inferred by the sequential walk rather than determined " +
            "by the layer. Without it the walk's inference is asserted as the layer's own claim, and " +
            "every model whose forward is not a plain sequential chain fails the shape contract.");

        // A shape-preserving layer told it outputs [8, 32, 32] by the walk, then handed [8, 8, 8].
        var layer = MakeDeclaring([8, 32, 32]);
        relax!.Invoke(layer, null);

        var verify = typeof(LayerBase<double>)
            .GetMethods(Any)
            .First(m => m.Name == "VerifyReportedOutputShape" && m.GetParameters().Length == 2);

        var produced = new Tensor<double>([1, 8, 8, 8]);
        var received = new Tensor<double>([1, 8, 8, 8]);

        // Must NOT throw: the spatial mismatch is the walk's inference, not this layer's claim.
        verify.Invoke(layer, new object[] { produced, received });
    }

    /// <summary>The channel axis stays enforced — relaxing spatial dims must not disable the contract.</summary>
    [Fact]
    public void ChannelAxisIsStillEnforcedAfterRelaxing()
    {
        var layer = MakeDeclaring([8, 32, 32]);
        typeof(LayerBase<double>).GetMethod("RelaxPropagatedSpatialAxes", Any)!.Invoke(layer, null);

        var verify = typeof(LayerBase<double>)
            .GetMethods(Any)
            .First(m => m.Name == "VerifyReportedOutputShape" && m.GetParameters().Length == 2);

        // Channel axis disagrees (declared 8, produced 16) — that IS the layer's own claim.
        var produced = new Tensor<double>([1, 16, 8, 8]);
        var ex = Assert.ThrowsAny<Exception>(() =>
            verify.Invoke(layer, new object[] { produced, produced }));

        Assert.True(ex.GetBaseException() is InvalidOperationException,
            "Relaxing the inferred SPATIAL axes must not stop the contract enforcing the channel " +
            $"axis, which the layer genuinely determines. Got {ex.GetBaseException().GetType().Name}.");
    }

    /// <summary>
    /// A layer defined to preserve an axis must still preserve it after resolution.
    /// </summary>
    /// <remarks>
    /// A pre-LN block is x + Attn(LN(x)) then x + FFN(LN(x)) — every term keeps S, and none of its
    /// parameters are sized by it. Resolution used to pin whichever length arrived first, so the
    /// block's own metadata then contradicted every later batch of a different length. Variable
    /// sequence length is the ordinary case for text, so this must not regress.
    /// </remarks>
    [Fact]
    public void ResolutionKeepsAnAxisTheLayerIsDefinedToPreserve()
    {
        var block = NewBlock();
        block.ResolveShapesOnly([8, HiddenSize]);   // walk hands it a concrete sequence length

        Assert.Equal([-1, HiddenSize], block.GetInputShape());
        Assert.Equal([-1, HiddenSize], block.GetOutputShape());
    }

    /// <summary>
    /// Keeping a free axis must not read as "never resolved" — those are different states that
    /// both show up in the shape array as a bare -1.
    /// </summary>
    [Fact]
    public void KeepingAFreeAxisStillCountsAsResolved()
    {
        var block = NewBlock();
        Assert.False(block.IsShapeResolved, "precondition: unresolved before the first shape probe");

        block.ResolveShapesOnly([8, HiddenSize]);

        Assert.True(block.IsShapeResolved,
            "A layer that resolved while deliberately keeping a declared-free axis dynamic IS " +
            "resolved. Reading the surviving -1 as 'not resolved yet' re-runs first-forward setup " +
            "on every pass and leaves a sequence-agnostic block permanently unresolved.");
    }

    /// <summary>
    /// The layer's OWN width can never be declared free — otherwise "this axis is genuinely
    /// dynamic" would become a way to opt out of the shape contract entirely.
    /// </summary>
    [Fact]
    public void TheParameterizedAxisCannotBeLeftFree()
    {
        var block = NewBlock();
        var resolve = typeof(LayerBase<double>)
            .GetMethods(Any)
            .First(m => m.Name == "ResolveShapes" && m.GetParameters().Length == 2);

        // Axis 1 is the feature width — sized by this block's parameters, so not free.
        var ex = Assert.ThrowsAny<Exception>(() =>
            resolve.Invoke(block, [new[] { -1, -1 }, new[] { -1, -1 }]));

        Assert.True(ex.GetBaseException() is ArgumentException,
            $"Expected the contract to reject a free parameterized axis; got {ex.GetBaseException().GetType().Name}.");
    }

    /// <summary>
    /// FeatureOnly is the mirror of ChannelOnly, and reading one with the other's convention
    /// inverts exactly which axis is a real claim.
    /// </summary>
    /// <remarks>
    /// Three of the four relations are channel-first. Applying that assumption to a feature-last
    /// layer asserts the sequence length, which no transformer block fixes, and exempts the
    /// feature width, which every one of them does — the contract still runs, but backwards.
    /// </remarks>
    [Fact]
    public void FeatureOnlyPutsTheLayersOwnAxisLast()
    {
        var implied = typeof(LayerBase<double>)
            .GetMethod("ImpliedInputShape", Any)!
            .Invoke(NewBlock(), [new[] { 12, HiddenSize }]) as int[];

        Assert.True(implied is not null, "FeatureOnly must be invertible: it preserves every leading axis.");
        Assert.Equal([12, -1], implied!);
    }

    private const int HiddenSize = 16;

    private static PreLNTransformerBlock<double> NewBlock() =>
        new(HiddenSize, 4 * HiddenSize, new MultiHeadAttentionLayer<double>(headCount: 4, headDimension: 4));

    /// <summary>
    /// An Identity-relation layer whose DECLARED output shape is set the way the sequential walk
    /// sets it — directly, rather than from anything the layer computed.
    /// </summary>
    private static ActivationLayer<double> MakeDeclaring(int[] declared)
    {
        var layer = new ActivationLayer<double>((AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.ReLUActivation<double>());
        typeof(LayerBase<double>)
            .GetProperty("OutputShape", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(layer, declared);
        return layer;
    }

    private static object? Instantiate(Type layerType)
    {
        foreach (var ctor in layerType.GetConstructors().OrderBy(c => c.GetParameters().Length))
        {
            try
            {
                var args = ctor.GetParameters()
                    .Select(p => p.HasDefaultValue ? p.DefaultValue
                               : p.ParameterType == typeof(int[]) ? new[] { 8, 32, 32 }
                               : p.ParameterType == typeof(int) ? (object)8
                               : null)
                    .ToArray();
                return ctor.Invoke(args);
            }
            catch { }
        }
        return null;
    }
}
