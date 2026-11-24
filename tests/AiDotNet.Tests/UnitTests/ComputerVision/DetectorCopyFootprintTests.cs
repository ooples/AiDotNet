using System;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Cascade R-CNN's WithParameters ran out of memory in the heavy nightly (#2447). Two things made a detector copy
/// several times its size: the generated state persisted the backbone a second time beside the parameter vector
/// that already carries it (189 MB of a 745 MB payload), and WithParameters serialized the source's parameters
/// only to overwrite them, allocating 13 times the parameter bytes. These pin both on a smaller two-stage detector
/// with the same ResNet-50 backbone.
/// </summary>
[Collection("FusedTrainingSerial")]
public class DetectorCopyFootprintTests
{
    private static FasterRCNN<double> ResolvedDetector()
    {
        var options = new ObjectDetectionOptions<double>
        {
            Size = ModelSize.Small, NumClasses = 3, InputSize = new[] { 64, 64 }, Seed = 7
        };
        var model = new FasterRCNN<double>(options);
        model.Predict(new Tensor<double>(new[] { 1, 3, 64, 64 }));
        return model;
    }

    [Fact]
    public void Serialize_CarriesTheBackboneOnce()
    {
        using var model = ResolvedDetector();
        long parameterBytes = 8L * model.GetParameters().Length;

        long payload = model.Serialize().LongLength;

        // The parameter vector as doubles plus a small header and declared state. A backbone persisted again as
        // state adds its own weights on top (a ResNet-50 is 23.5 million values).
        Assert.True(payload < parameterBytes + (1L << 20),
            $"Serialize wrote {payload:N0} bytes for {parameterBytes:N0} bytes of parameters: something beyond the "
            + "vector (the backbone, persisted again as state?) is being written.");
    }

    [Fact]
    public void WithParameters_RoundTripsAndIsIndependent()
    {
        using var model = ResolvedDetector();
        var parameters = model.GetParameters();
        var shifted = new Vector<double>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) shifted[i] = parameters[i] + 0.5;

        var copy = model.WithParameters(shifted);
        try
        {
            Assert.NotSame(model, copy);
            var restored = ((IParameterizable<double, Tensor<double>, Tensor<double>>)copy).GetParameters();
            Assert.Equal(shifted.Length, restored.Length);
            for (int i = 0; i < shifted.Length; i++) Assert.Equal(shifted[i], restored[i]);

            var source = model.GetParameters();
            for (int i = 0; i < parameters.Length; i++) Assert.Equal(parameters[i], source[i]);
        }
        finally
        {
            (copy as IDisposable)?.Dispose();
        }
    }

#if NETCOREAPP
    [Fact]
    public void WithParameters_DoesNotRoundTripTheSourceParametersThroughSerialize()
    {
        using var model = ResolvedDetector();
        var parameters = model.GetParameters();
        // Warm both paths once, so one-time costs (reflection, plan caches) are not counted.
        (model.WithParameters(parameters) as IDisposable)?.Dispose();
        (model.DeepCopy() as IDisposable)?.Dispose();

        long start = GC.GetAllocatedBytesForCurrentThread();
        var deepCopy = model.DeepCopy();
        long deepCopyBytes = GC.GetAllocatedBytesForCurrentThread() - start;
        (deepCopy as IDisposable)?.Dispose();

        start = GC.GetAllocatedBytesForCurrentThread();
        var copy = model.WithParameters(parameters);
        long withParametersBytes = GC.GetAllocatedBytesForCurrentThread() - start;
        (copy as IDisposable)?.Dispose();

        // WithParameters was DeepCopy followed by SetParameters, so it always cost more than DeepCopy: the source's
        // vector serialized, decoded, installed, then overwritten (Cascade R-CNN: 13 times the parameter bytes, and
        // out of memory in the nightly). Building the copy without that round trip costs well under a DeepCopy.
        Assert.True(withParametersBytes < 0.75 * deepCopyBytes,
            $"WithParameters allocated {withParametersBytes:N0} bytes against DeepCopy's {deepCopyBytes:N0}: it is "
            + "still carrying the source's parameters through Serialize.");
    }
#endif
}
