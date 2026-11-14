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
