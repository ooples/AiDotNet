using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.ActionRecognition;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.ActionRecognition;

/// <summary>
/// Pins that TimeSformer's declared input (<c>Architecture.GetInputShape()</c>) is the video clip the model
/// is built for, and that the model accepts an input of exactly that shape.
/// </summary>
public class TimeSformerDeclaredInputTests
{
    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)rng.NextDouble();
        }
        return t;
    }

    private static void AssertIsAProbabilityVector(Tensor<float> probs, int classes)
    {
        Assert.Equal(classes, probs.Length);
        double sum = 0;
        var span = probs.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            Assert.False(float.IsNaN(span[i]) || float.IsInfinity(span[i]), $"probability {i} was {span[i]}");
            sum += span[i];
        }
        Assert.Equal(1.0, sum, 3);
    }

    [Fact]
    public void DefaultConstructor_DeclaresAnEightFrameClip_AndPredictAcceptsIt()
    {
        var model = new TimeSformer<float>();

        // The default model is built for 8-frame 224x224 RGB clips (numFrames = 8 sizes the divided
        // space-time blocks and the positional table). It used to declare InputType.ThreeDimensional,
        // i.e. a single [3, 224, 224] frame, so anything that sizes an input from the declared shape
        // (the auto-batching rank check, generated fixtures, serving) fed it a one-frame "video".
        var declared = model.Architecture.GetInputShape();
        Assert.Equal(InputType.FourDimensional, model.Architecture.InputType);
        Assert.Equal(new[] { 8, 3, 224, 224 }, declared);

        var probs = model.Predict(Random(declared, seed: 3));

        AssertIsAProbabilityVector(probs, classes: 400);
    }
}
