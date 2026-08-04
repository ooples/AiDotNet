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
