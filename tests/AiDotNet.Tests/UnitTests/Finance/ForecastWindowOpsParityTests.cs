using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Base;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Pins <see cref="ForecastWindowOps"/> to the element-by-element copies it replaced in the autoregressive
/// forecasters (Autoformer, Informer, DeepAR, TimesNet, TFT, Crossformer). The expected values, and the expected
/// exception types for out-of-range requests, come from those copies, written out below with plain loops.
/// </summary>
public class ForecastWindowOpsParityTests
{
    private static Tensor<T> Random<T>(int seed, params int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new T[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = ops.FromDouble(random.NextDouble() * 20.0 - 10.0);
        return new Tensor<T>(data, shape);
    }

    private static void AssertSame<T>(Tensor<T> expected, Tensor<T> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    /// <summary>The copy every ShiftInputWithPredictions override ran.</summary>
    private static Tensor<T> ReferenceShift<T>(Tensor<T> input, Tensor<T> predictions, int length, int steps)
    {
        int batch = input.Shape[0], features = input.Shape[2];
        var shifted = new Tensor<T>(new[] { batch, length, features });
        int keep = length - steps;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < keep; t++)
                for (int f = 0; f < features; f++) shifted[b, t, f] = input[b, t + steps, f];
            for (int t = 0; t < steps; t++)
                for (int f = 0; f < features; f++) shifted[b, keep + t, f] = predictions[b, t, f];
        }

        return shifted;
    }

    /// <summary>The copy every ConcatenatePredictions override ran.</summary>
    private static Tensor<T> ReferenceConcatenate<T>(List<Tensor<T>> predictions, int totalSteps, int features)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps, features });
        int batch = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batch, totalSteps, features });
        int step = 0;
        foreach (var prediction in predictions)
        {
            int add = Math.Min(prediction.Shape[1], totalSteps - step);
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < add; t++)
                    for (int f = 0; f < features; f++) result[b, step + t, f] = prediction[b, t, f];
            step += add;
        }

        return result;
    }

    private static void ShiftMatches<T>(int batch, int inputLength, int length, int features, int steps, int[] predictionShape)
    {
        var input = Random<T>(1, batch, inputLength, features);
        var predictions = Random<T>(2, predictionShape);

        AssertSame(
            ReferenceShift(input, predictions, length, steps),
            ForecastWindowOps.ShiftWindow(input, predictions, length, steps));
    }

    [Theory]
    [InlineData(2, 8, 8, 3, 2, new[] { 2, 2, 3 })]   // the usual step
    [InlineData(2, 8, 8, 3, 2, new[] { 4, 5, 6 })]   // predictions larger than needed on every axis
    [InlineData(1, 8, 8, 1, 1, new[] { 1, 1, 1 })]
    [InlineData(3, 6, 6, 2, 6, new[] { 3, 6, 2 })]   // the whole window replaced
    [InlineData(3, 6, 6, 2, 0, new[] { 3, 4, 2 })]   // no movement
    [InlineData(3, 6, 6, 2, 0, new[] { 1, 1, 1 })]   // no movement: the predictions are never read
    [InlineData(2, 9, 6, 2, 2, new[] { 2, 2, 2 })]   // input longer than the window (TimesNet, TFT, Crossformer)
    [InlineData(2, 2, 6, 2, 6, new[] { 2, 6, 2 })]   // input shorter than the window, but none of it is kept
    public void ShiftWindow_MatchesTheElementCopy_Double(int batch, int inputLength, int length, int features, int steps, int[] predictionShape)
        => ShiftMatches<double>(batch, inputLength, length, features, steps, predictionShape);

    [Theory]
    [InlineData(2, 8, 8, 3, 3, new[] { 2, 4, 3 })]
    [InlineData(1, 5, 5, 2, 5, new[] { 1, 5, 2 })]
    public void ShiftWindow_MatchesTheElementCopy_Float(int batch, int inputLength, int length, int features, int steps, int[] predictionShape)
        => ShiftMatches<float>(batch, inputLength, length, features, steps, predictionShape);

    [Theory]
    [InlineData(2, 8, 8, 3, 9, new[] { 2, 9, 3 })]    // further than the window
    [InlineData(2, 8, 8, 3, -1, new[] { 2, 2, 3 })]   // backwards
    [InlineData(2, 4, 8, 3, 2, new[] { 2, 2, 3 })]    // input shorter than the window
    [InlineData(2, 8, 8, 3, 2, new[] { 1, 2, 3 })]    // predictions: too few batches
    [InlineData(2, 8, 8, 3, 2, new[] { 2, 1, 3 })]    // predictions: too few steps
    [InlineData(2, 8, 8, 3, 2, new[] { 2, 2, 2 })]    // predictions: too few features
    [InlineData(2, 8, 8, 3, 2, new[] { 2, 6 })]       // predictions: rank 2
    [InlineData(2, 8, 8, 3, 2, new[] { 2, 2, 3, 1 })] // predictions: rank 4
    public void ShiftWindow_OutOfRange_ThrowsWhatTheElementCopyThrew(int batch, int inputLength, int length, int features, int steps, int[] predictionShape)
    {
        var input = Random<double>(1, batch, inputLength, features);
        var predictions = Random<double>(2, predictionShape);

        var expected = Record.Exception(() => ReferenceShift(input, predictions, length, steps));
        var actual = Record.Exception(() => ForecastWindowOps.ShiftWindow(input, predictions, length, steps));

        var expectedType = Assert.IsAssignableFrom<Exception>(expected).GetType();
        Assert.IsType(expectedType, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(6)]
    public void ShiftWindow_ReturnsACopy(int steps)
    {
        // The window is fed back into the model; it must not alias the tensors it was built from
        var input = Random<double>(1, 2, 6, 2);
        var predictions = Random<double>(2, 2, 6, 2);
        double[] inputBefore = input.ToArray(), predictionsBefore = predictions.ToArray();

        var shifted = ForecastWindowOps.ShiftWindow(input, predictions, 6, steps);
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < 6; t++)
                for (int f = 0; f < 2; f++) shifted[b, t, f] = 1234.5;

        Assert.Equal(inputBefore, input.ToArray());
        Assert.Equal(predictionsBefore, predictions.ToArray());
    }

    private static void ConcatenateMatches<T>(int totalSteps, int features, int[][] chunkShapes)
    {
        var chunks = chunkShapes.Select((shape, i) => Random<T>(10 + i, shape)).ToList();

        AssertSame(
            ReferenceConcatenate(chunks, totalSteps, features),
            ForecastWindowOps.ConcatenateLeadingSteps(chunks, totalSteps, features));
    }

    public static IEnumerable<object[]> ConcatenateCases()
    {
        // total steps, features, chunk shapes
        yield return new object[] { 6, 3, new[] { new[] { 2, 2, 3 }, new[] { 2, 2, 3 }, new[] { 2, 2, 3 } } };   // exact fit
        yield return new object[] { 5, 3, new[] { new[] { 2, 2, 3 }, new[] { 2, 2, 3 }, new[] { 2, 2, 3 } } };   // last chunk cut
        yield return new object[] { 3, 3, new[] { new[] { 2, 4, 3 }, new[] { 2, 4, 3 } } };                      // later chunks unused
        yield return new object[] { 9, 3, new[] { new[] { 2, 2, 3 }, new[] { 2, 3, 3 } } };                      // zero padded
        yield return new object[] { 4, 2, new[] { new[] { 2, 4, 2 } } };                                         // one chunk
        yield return new object[] { 4, 2, new[] { new[] { 2, 3, 5 }, new[] { 3, 3, 2 } } };                      // wider and deeper chunks
        yield return new object[] { 4, 2, new[] { new[] { 2, 0, 2 }, new[] { 2, 4, 2 } } };                      // an empty chunk
        yield return new object[] { 4, 2, Array.Empty<int[]>() };                                                // no chunks
        yield return new object[] { 0, 2, new[] { new[] { 2, 3, 2 } } };                                         // nothing requested
        yield return new object[] { 4, 2, new[] { new[] { 2, 4, 2 }, new[] { 1, 4, 1 } } };                      // a bad chunk that is never read
    }

    [Theory]
    [MemberData(nameof(ConcatenateCases))]
    public void ConcatenateLeadingSteps_MatchesTheElementCopy_Double(int totalSteps, int features, int[][] chunkShapes)
        => ConcatenateMatches<double>(totalSteps, features, chunkShapes);

    [Theory]
    [MemberData(nameof(ConcatenateCases))]
    public void ConcatenateLeadingSteps_MatchesTheElementCopy_Float(int totalSteps, int features, int[][] chunkShapes)
        => ConcatenateMatches<float>(totalSteps, features, chunkShapes);

    public static IEnumerable<object[]> BadConcatenateCases()
    {
        yield return new object[] { 6, 3, new[] { new[] { 2, 2, 3 }, new[] { 1, 2, 3 } } };      // too few batches
        yield return new object[] { 6, 3, new[] { new[] { 2, 2, 3 }, new[] { 2, 2, 2 } } };      // too few features
        yield return new object[] { 6, 3, new[] { new[] { 2, 2, 2 } } };                         // first chunk too narrow
        yield return new object[] { 6, 3, new[] { new[] { 2, 2 } } };                            // rank 2
        yield return new object[] { 6, 3, new[] { new[] { 2, 2, 3, 1 } } };                      // rank 4
    }

    [Theory]
    [MemberData(nameof(BadConcatenateCases))]
    public void ConcatenateLeadingSteps_OutOfRange_ThrowsWhatTheElementCopyThrew(int totalSteps, int features, int[][] chunkShapes)
    {
        var chunks = chunkShapes.Select((shape, i) => Random<double>(10 + i, shape)).ToList();

        var expected = Record.Exception(() => ReferenceConcatenate(chunks, totalSteps, features));
        var actual = Record.Exception(() => ForecastWindowOps.ConcatenateLeadingSteps(chunks, totalSteps, features));

        var expectedType = Assert.IsAssignableFrom<Exception>(expected).GetType();
        Assert.IsType(expectedType, actual);
    }

    [Fact]
    public void ConcatenateLeadingSteps_ReturnsACopy()
    {
        var chunk = Random<double>(3, 2, 4, 2);
        double[] before = chunk.ToArray();

        var forecast = ForecastWindowOps.ConcatenateLeadingSteps(new List<Tensor<double>> { chunk }, 4, 2);
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < 4; t++)
                for (int f = 0; f < 2; f++) forecast[b, t, f] = 1234.5;

        Assert.Equal(before, chunk.ToArray());
    }
}
