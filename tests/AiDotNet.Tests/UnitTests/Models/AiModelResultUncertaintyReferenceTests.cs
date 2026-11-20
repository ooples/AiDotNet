using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Uncertainty helpers of <see cref="AiModelResult{T, TInput, TOutput}"/> in Half precision, pinned where the
/// engine-op conversion (#2490) changed a result: the calibration clamps and the sample mean/variance.
/// </summary>
public class AiModelResultUncertaintyReferenceTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.Static | BindingFlags.NonPublic;

    [Fact]
    public void PlattScaling_Half_IdentityMapKeepsSmallProbability()
    {
        // Platt scaling with a = 1, b = 0 is sigmoid(logit(p)) = p. The clamp only exists to keep the logit
        // finite, so a representable probability such as 1e-4 must come back unchanged; clamping Half from
        // below at 2^-11 (4.9e-4) moved it.
        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1 }));
        var result = CreateResult(model);
        SetProperty(result, "PlattScalingA", new Vector<Half>(new[] { (Half)1 }));
        SetProperty(result, "PlattScalingB", new Vector<Half>(new[] { (Half)0 }));
        var probabilities = new Vector<Half>(new[] { (Half)1, (Half)1e-4 });

        var calibrated = Invoke(result, "ApplyPlattScalingToOutputProbabilities", probabilities);

        Assert.Equal(1e-4, (double)calibrated[1], 2e-6);
    }

    [Fact]
    public void IsotonicCalibration_Half_ProbabilityOneReachesTheLastBreakpoint()
    {
        // The isotonic map returns y[i] for the first breakpoint x[i] >= p. p = 1 is first reached by x = 1, so
        // the calibrated value is 0.3. Clamping p to 1 - 2^-11 for Half matched the breakpoint 1 - 2^-11 (0.2).
        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1 }));
        var result = CreateResult(model);
        SetProperty(result, "IsotonicCalibrationX", new Vector<Half>(new[] { (Half)0.5, (Half)(1 - (1.0 / 2048)), (Half)1 }));
        SetProperty(result, "IsotonicCalibrationY", new Vector<Half>(new[] { (Half)0.1, (Half)0.2, (Half)0.3 }));
        var probabilities = new Vector<Half>(new[] { (Half)0, (Half)1 });

        var calibrated = Invoke(result, "ApplyIsotonicCalibrationToOutputProbabilities", probabilities);

        Assert.Equal((Half)0.3, calibrated[1]);
    }

    [Fact]
    public void ComputeMeanAndVariance_Half_SumBeyondHalfRange()
    {
        // Samples 30000, 30000, 30032, 30032: mean 30016 and population variance 16^2 = 256. Their sum, 120064,
        // overflows Half (65504), so summing before dividing by the count made the mean infinite.
        var samples = new List<Tensor<Half>>
        {
            new Tensor<Half>(new[] { (Half)30000 }, new[] { 1 }),
            new Tensor<Half>(new[] { (Half)30000 }, new[] { 1 }),
            new Tensor<Half>(new[] { (Half)30032 }, new[] { 1 }),
            new Tensor<Half>(new[] { (Half)30032 }, new[] { 1 })
        };
        var method = typeof(AiModelResult<Half, Matrix<Half>, Vector<Half>>).GetMethod("ComputeMeanAndVariance", Private);
        Assert.NotNull(method);

        var moments = method?.Invoke(null, new object[] { samples });
        var (mean, variance) = Assert.IsType<(Tensor<Half>, Tensor<Half>)>(moments);

        Assert.Equal(30016.0, (double)mean[0], 16.0);
        Assert.Equal(256.0, (double)variance[0], 1.0);
    }

    [Fact]
    public void ComputeMeanAndVariance_Half_SmallCoordinateKeepsItsVarianceNextToALargeOne()
    {
        // Samples [-128, -0.01] and [128, 0.01]: population variances 128^2 = 16384 and 0.01^2 = 1e-4. One scale
        // shared by both coordinates (128) made the second one's normalized squares 6.1e-9, which is 0 in Half, so
        // its variance came back as 0.
        var samples = new List<Tensor<Half>>
        {
            new Tensor<Half>(new[] { (Half)(-128), (Half)(-0.01) }, new[] { 2 }),
            new Tensor<Half>(new[] { (Half)128, (Half)0.01 }, new[] { 2 })
        };
        var method = typeof(AiModelResult<Half, Matrix<Half>, Vector<Half>>).GetMethod("ComputeMeanAndVariance", Private);
        Assert.NotNull(method);

        var moments = method?.Invoke(null, new object[] { samples });
        var (_, variance) = Assert.IsType<(Tensor<Half>, Tensor<Half>)>(moments);

        Assert.Equal(16384.0, (double)variance[0], 16.0);
        Assert.Equal(1e-4, (double)variance[1], 2e-6);
    }

    [Fact]
    public void ComputeR2Score_Half_OppositeSignsNearTheTopOfTheRange()
    {
        // actual = [-60000, 60000], predicted = [60000, -60000]: SS_res = 2 * 120000^2, SS_tot = 2 * 60000^2, so
        // R^2 = 1 - 4 = -3. The residual 120000 is not a Half, so subtracting first gave NaN.
        var actual = new Vector<Half>(new[] { (Half)(-60000), (Half)60000 });
        var predicted = new Vector<Half>(new[] { (Half)60000, (Half)(-60000) });
        var method = typeof(AiModelResult<Half, Matrix<Half>, Vector<Half>>).GetMethod("ComputeR2Score", Private);
        Assert.NotNull(method);

        var score = Assert.IsType<Half>(method?.Invoke(null, new object[] { actual, predicted }));

        Assert.Equal(-3.0, (double)score, 1e-2);
    }

    [Fact]
    public void SampleTemperatureLogits_Half_OppositeSignsNearTheTopOfTheRange()
    {
        // softmax([-60000, 60000] / 60000) = [0.1192, 0.8808]. The shift -60000 - 60000 is not a Half, so the first
        // token used to get probability 0 and was never drawn.
        var logits = new Vector<Half>(new[] { (Half)(-60000), (Half)60000 });
        var method = typeof(AiModelResult<Half, Matrix<Half>, Vector<Half>>).GetMethod("SampleTemperatureLogits", Private);
        Assert.NotNull(method);
        var rng = new Random(7);

        int first = 0;
        const int draws = 1000;
        for (int i = 0; i < draws; i++)
        {
            first += Assert.IsType<int>(method?.Invoke(null, new object[] { logits, 60000.0, rng })) == 0 ? 1 : 0;
        }

        // Binomial(1000, 0.1192): mean 119, standard deviation 10.2.
        Assert.InRange(first, 70, 170);
    }

    [Fact]
    public void CalculateStandardDeviation_Half_OppositeSignsNearTheTopOfTheRange()
    {
        // Predictions -60000, 60000, 60000: mean 20000, population standard deviation sqrt(3.2e9) = 56568.5. The
        // deviation -80000 is not a Half, so subtracting the mean first gave NaN.
        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1 }));
        var result = CreateResult(model);
        var predictions = new List<Vector<Half>>
        {
            new Vector<Half>(new[] { (Half)(-60000) }),
            new Vector<Half>(new[] { (Half)60000 }),
            new Vector<Half>(new[] { (Half)60000 })
        };
        var method = result.GetType().GetMethod("CalculateStandardDeviation", Private);
        Assert.NotNull(method);

        var deviation = Assert.IsType<double>(method?.Invoke(result, new object[] { predictions }));

        Assert.Equal(56568.5, deviation, 150.0);
    }

    private static AiModelResult<Half, Matrix<Half>, Vector<Half>> CreateResult(VectorModel<Half> model)
    {
        return new AiModelResult<Half, Matrix<Half>, Vector<Half>>(
            new AiModelResultOptions<Half, Matrix<Half>, Vector<Half>>
            {
                OptimizationResult = new OptimizationResult<Half, Matrix<Half>, Vector<Half>> { BestSolution = model }
            });
    }

    private static void SetProperty(object target, string name, Vector<Half> value)
    {
        var property = target.GetType().GetProperty(name, Private);
        Assert.NotNull(property);
        property?.SetValue(target, value);
    }

    private static Vector<Half> Invoke(object target, string name, Vector<Half> probabilities)
    {
        var method = target.GetType().GetMethod(name, Private);
        Assert.NotNull(method);
        return Assert.IsType<Vector<Half>>(method?.Invoke(target, new object[] { probabilities }));
    }
}
