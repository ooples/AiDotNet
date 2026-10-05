using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for instance segmenters (Mask R-CNN, SOLOv2, YOLO-Seg): models that map an image to a
/// set of detected objects, each with a class, a box and a mask.
/// </summary>
/// <remarks>
/// <para>
/// <c>InstanceSegmenterBase</c> is a <c>ModelBase</c>, not an <c>INeuralNetworkModel</c>, and is built
/// from <c>InstanceSegmentationOptions</c>, so neither the segmentation family (which drives a network)
/// nor the tensor-module family (a width-only constructor) can build one. These models therefore had no
/// generated coverage at all.
/// </para>
/// <para>
/// <b>The base owns the image size and class count and passes them to the factory.</b> The generated fixture
/// implements <see cref="CreateSegmenter"/> from exactly those values, and every input this base builds uses
/// the same <see cref="ImageSize"/>, so the model and its inputs cannot disagree.
/// </para>
/// </remarks>
public abstract class InstanceSegmenterTestBase<T>
{
    /// <summary>Shared numeric operations, matching the sibling family bases.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a numeric value to double for assertion.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    /// <summary>Side of the square input image; small, so the full topology runs at smoke cost.</summary>
    protected virtual int ImageSize => 32;

    /// <summary>Foreground classes the segmenter predicts.</summary>
    protected virtual int NumClasses => 3;

    /// <summary>Subclasses construct their segmenter with exactly the configuration they are handed.</summary>
    protected abstract InstanceSegmenterBase<T> CreateSegmenter(int imageSize, int numClasses);

    private InstanceSegmenterBase<T> CreateSegmenter() => CreateSegmenter(ImageSize, NumClasses);

    /// <summary>One NCHW RGB image with a fixed pattern; identical on every call.</summary>
    private Tensor<T> CreateImage()
    {
        var image = new Tensor<T>(new[] { 1, 3, ImageSize, ImageSize });
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = NumOps.FromDouble((((i * 7) % 23) / 23.0) - 0.4);
        }

        return image;
    }

    private static void AssertFinite(Tensor<T> tensor, string what)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double value = ToD(tensor[i]);
            Assert.False(double.IsNaN(value) || double.IsInfinity(value), $"{what} is not finite at {i}: {value}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();

        var output = model.Predict(CreateImage());

        Assert.True(output.Length > 0, "Predict returned an empty tensor.");
        AssertFinite(output, "Predict output");
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();

        var first = model.Predict(CreateImage()).ToArray();
        var second = model.Predict(CreateImage()).ToArray();

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(ToD(first[i]), ToD(second[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_ShouldBeNonEmptyAndMatchTheCount()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();

        var parameters = model.GetParameters();

        Assert.True(parameters.Length > 0, "An instance segmenter reports no parameters, so nothing can train or save it.");
        Assert.Equal(model.ParameterCount, parameters.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task SetParameters_ShouldRoundTripTheVector()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();
        var original = model.GetParameters();
        var shifted = new Vector<T>(original.Length);
        for (int i = 0; i < original.Length; i++)
        {
            shifted[i] = NumOps.Add(original[i], NumOps.FromDouble(0.25));
        }

        model.SetParameters(shifted);
        var readBack = model.GetParameters();

        Assert.Equal(shifted.Length, readBack.Length);
        for (int i = 0; i < shifted.Length; i++)
        {
            Assert.Equal(ToD(shifted[i]), ToD(readBack[i]), 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldPredictIdentically()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();
        var expected = model.Predict(CreateImage()).ToArray();

        var clone = Assert.IsAssignableFrom<InstanceSegmenterBase<T>>(model.Clone());
        var actual = clone.Predict(CreateImage()).ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(ToD(expected[i]), ToD(actual[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Segment_ShouldReturnWellFormedInstances()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();

        var result = model.Segment(CreateImage());

        Assert.NotNull(result);
        Assert.NotNull(result.Instances);
        foreach (var instance in result.Instances)
        {
            Assert.InRange(instance.ClassId, 0, NumClasses);
            double confidence = ToD(instance.Confidence);
            Assert.False(double.IsNaN(confidence), "An instance has a NaN confidence.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Train_ShouldChangeParameters()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateSegmenter();
        var image = CreateImage();
        var before = model.GetParameters().ToArray();

        var prediction = model.Predict(image);
        model.Train(image, new Tensor<T>(prediction._shape));

        var after = model.GetParameters().ToArray();
        Assert.All(after, value => Assert.False(double.IsNaN(ToD(value)) || double.IsInfinity(ToD(value))));
        Assert.True(before.Zip(after, (a, b) => !System.Collections.Generic.EqualityComparer<T>.Default.Equals(a, b)).Any(changed => changed),
            "A training step toward a zero target left every parameter unchanged.");
    }
}

/// <summary>Double-precision convenience form, matching the other family bases.</summary>
public abstract class InstanceSegmenterTestBase : InstanceSegmenterTestBase<double>
{
}