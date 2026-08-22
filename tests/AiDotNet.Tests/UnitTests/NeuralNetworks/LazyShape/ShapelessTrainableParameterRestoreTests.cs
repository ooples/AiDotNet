using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.LazyShape;

/// <summary>
/// Restore round trips for the two layers whose <c>[TrainableParameter]</c> declarations carry a role
/// but no <c>Shape</c>.
/// </summary>
/// <remarks>
/// <para>
/// A shape-less declaration produces a <c>DeclaredParameterTensors()</c> override but no shape
/// declaration, so <c>HasActiveDeclaredParameterShapes</c> stayed at its <c>false</c> default and
/// <c>TryAdoptRestoredParameters</c> returned before consulting the checkpoint. 114 of the 293
/// generated layers were in that state. <c>SVTRThinPlateSplineLayer</c> was worse off still: it was
/// not <c>partial</c>, so the generator emitted nothing for it at all and its attributes were inert.
/// </para>
/// <para>
/// These tests assert the property that matters to a caller, not the mechanism: parameters written
/// into a fresh layer are the parameters that layer reports back, and a forward pass does not replace
/// them with freshly initialized values.
/// </para>
/// </remarks>
public class ShapelessTrainableParameterRestoreTests
{
    private static Tensor<double> Ramp(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.5 + (i % 7) * 0.125;
        return t;
    }

    [Fact]
    public void SubpixelConvolutional_RestoredParameters_SurviveTheFirstForward()
    {
        var input = Ramp([1, 2, 4, 4]);

        // Donor: resolve it, then give every parameter a value no initializer would produce.
        var donor = new SubpixelConvolutionalLayer<double>(2, 2, 3, new ReLUActivation<double>() as IActivationFunction<double>);
        donor.Forward(input);
        var donorParams = donor.GetParameters();
        for (int i = 0; i < donorParams.Length; i++) donorParams[i] = 0.25 + i * 0.5;
        donor.SetParameters(donorParams);

        // Recipient: fresh and unresolved, restored from the donor, then driven forward. The forward is
        // the whole point -- it is what used to reallocate and re-initialize over the restore.
        var restored = new SubpixelConvolutionalLayer<double>(2, 2, 3, new ReLUActivation<double>() as IActivationFunction<double>);
        restored.SetParameters(donorParams);
        restored.Forward(input);

        var after = restored.GetParameters();
        Assert.Equal(donorParams.Length, after.Length);
        for (int i = 0; i < donorParams.Length; i++)
        {
            Assert.True(donorParams[i] == after[i],
                $"parameter {i} came back as {after[i]} instead of the restored {donorParams[i]}. " +
                "The restore was discarded and the layer re-initialized itself.");
        }
    }

    [Fact]
    public void SvtrThinPlateSpline_RestoredParameters_SurviveTheFirstForward()
    {
        var input = Ramp([1, 3, 32, 100]);

        var donor = new SVTRThinPlateSplineLayer<double>();
        donor.Forward(input);
        var donorParams = donor.GetParameters();
        for (int i = 0; i < donorParams.Length; i++) donorParams[i] = 0.125 + (i % 11) * 0.0625;
        donor.SetParameters(donorParams);

        var restored = new SVTRThinPlateSplineLayer<double>();
        restored.SetParameters(donorParams);
        restored.Forward(input);

        var after = restored.GetParameters();
        Assert.Equal(donorParams.Length, after.Length);
        int mismatches = 0;
        for (int i = 0; i < donorParams.Length; i++)
            if (donorParams[i] != after[i]) mismatches++;

        Assert.True(mismatches == 0,
            $"{mismatches} of {donorParams.Length} parameters did not survive the round trip. " +
            "SVTR's [TrainableParameter] declarations were inert while the class was non-partial, " +
            "so the restore had nothing to bind to.");
    }
}
