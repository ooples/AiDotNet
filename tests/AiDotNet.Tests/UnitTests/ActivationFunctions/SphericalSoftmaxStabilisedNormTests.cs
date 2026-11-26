using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActivationFunctions;

/// <summary>
/// The vector and tensor overloads of spherical softmax compute the same function,
/// <c>softmax(x / sqrt(sum(x^2) + 1e-12))</c>. The vector overload used to divide by the bare norm, so it
/// returned NaN for the zero vector and disagreed with the tensor overload for very small inputs.
/// </summary>
public class SphericalSoftmaxStabilisedNormTests
{
    private const double Epsilon = 1e-12;

    private static double[] Reference(double[] x, double epsilon)
    {
        double norm = Math.Sqrt(x.Sum(v => v * v) + epsilon);
        double[] exponentials = x.Select(v => Math.Exp(v / norm)).ToArray();
        double sum = exponentials.Sum();
        return exponentials.Select(v => v / sum).ToArray();
    }

    public static TheoryData<double[]> Inputs() => new()
    {
        new[] { 0.3, -1.1, 2.0, 0.0, 0.7 },
        new[] { 5.0 },
        new[] { 1e-8, -2e-8, 3e-8 },
        new[] { 1e-3, 2e-3 },
        new[] { 0.0, 0.0, 0.0, 0.0 },
        new[] { 1e6, -1e6, 3e5 },
    };

    [Theory]
    [MemberData(nameof(Inputs))]
    public void VectorOverload_AgreesWithTheTensorOverloadToRounding(double[] values)
    {
        // The vector overload is computed directly, the tensor overload through the engine: equal up to the
        // rounding of a different evaluation order, a few ulps of a probability.
        var activation = new SphericalSoftmaxActivation<double>();

        double[] fromVector = activation.Activate(new Vector<double>(values)).ToArray();
        double[] fromTensor = activation.Activate(new Tensor<double>(values, new[] { values.Length })).ToArray();

        Assert.Equal(fromTensor.Length, fromVector.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(fromTensor[i] - fromVector[i]) <= 2e-15,
                $"[{i}]: tensor {fromTensor[i]:R} vector {fromVector[i]:R}");
    }

    [Fact]
    public void VectorOverload_Float_AgreesWithTheTensorOverloadToRounding()
    {
        var activation = new SphericalSoftmaxActivation<float>();
        float[] values = { 0.3f, -1.1f, 2.0f, 0.0f, 0.7f, 1e-20f, 3e19f };

        float[] fromVector = activation.Activate(new Vector<float>(values)).ToArray();
        float[] fromTensor = activation.Activate(new Tensor<float>(values, new[] { values.Length })).ToArray();

        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(fromTensor[i] - fromVector[i]) <= 1e-6f, $"[{i}]: tensor {fromTensor[i]:R} vector {fromVector[i]:R}");
    }

    [Theory]
    [MemberData(nameof(Inputs))]
    public void BothOverloads_ComputeSoftmaxOverTheStabilisedNorm(double[] values)
    {
        var activation = new SphericalSoftmaxActivation<double>();
        double[] expected = Reference(values, Epsilon);

        double[] fromVector = activation.Activate(new Vector<double>(values)).ToArray();
        double[] fromTensor = activation.Activate(new Tensor<double>(values, new[] { values.Length })).ToArray();

        for (int i = 0; i < values.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - fromVector[i]) <= 1e-13, $"vector[{i}]: expected {expected[i]:R} but was {fromVector[i]:R}");
            Assert.True(Math.Abs(expected[i] - fromTensor[i]) <= 1e-13, $"tensor[{i}]: expected {expected[i]:R} but was {fromTensor[i]:R}");
        }

        Assert.Equal(1.0, fromVector.Sum(), 12);
    }

    [Fact]
    public void ZeroVector_GivesTheUniformDistribution()
    {
        // 0 / sqrt(0 + 1e-12) = 0 for every element; the bare norm gave 0 / 0 = NaN
        double[] output = new SphericalSoftmaxActivation<double>().Activate(new Vector<double>(new double[5])).ToArray();

        Assert.All(output, p => Assert.Equal(0.2, p, 15));
    }

    [Fact]
    public void ZeroVector_Float_GivesTheUniformDistribution()
    {
        float[] output = new SphericalSoftmaxActivation<float>().Activate(new Vector<float>(new float[4])).ToArray();

        Assert.All(output, p => Assert.Equal(0.25f, p, 6));
    }

    [Fact]
    public void TinyInput_UsesTheEpsilon_NotTheBareNorm()
    {
        // ||x||^2 = 1.4e-15 is far below the epsilon, so the stabilised norm is about 1e-6 and x / norm is about
        // 0.01-0.03: nearly uniform. Dividing by the bare norm (3.7e-8) would give a clearly peaked distribution.
        double[] values = { 1e-8, -2e-8, 3e-8 };
        double[] output = new SphericalSoftmaxActivation<double>().Activate(new Vector<double>(values)).ToArray();
        double[] bareNorm = Reference(values, 0.0);

        Assert.All(output, p => Assert.InRange(p, 0.32, 0.35));
        Assert.True(Math.Abs(bareNorm[2] - output[2]) > 0.1, "the output matches the bare-norm formula");
    }

    [Fact]
    public void OrdinaryInput_IsUnchangedByTheEpsilon()
    {
        // At ordinary magnitudes the epsilon is below the last bit of the squared norm's precision budget here
        double[] values = { 0.3, -1.1, 2.0, 0.0, 0.7 };
        double[] output = new SphericalSoftmaxActivation<double>().Activate(new Vector<double>(values)).ToArray();
        double[] bareNorm = Reference(values, 0.0);

        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(bareNorm[i] - output[i]) <= 1e-12, $"[{i}]: {bareNorm[i]:R} vs {output[i]:R}");
    }

    [Fact]
    public void EmptyVector_StaysEmpty()
        => Assert.Equal(0, new SphericalSoftmaxActivation<double>().Activate(new Vector<double>(0)).Length);
}
