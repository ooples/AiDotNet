using AiDotNet.GaussianProcesses;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Shared likelihood/prior contracts. A fixed zero-mean Student-t GP is
/// neither a linear smoother in its targets nor an interpolating model.
/// Its variance can increase when an observation conflicts with the prior.
/// </summary>
internal static class GaussianProcessPosteriorContract
{
    internal static bool VerifyIndependentAdditionalData(IGaussianProcess<double> model, int features)
    {
        if (model is not StudentTGaussianProcess<double> student) return false;
        Assert.True(VerifyFixedPrior(model, features, 1, 10));
        var first = new Vector<double>(features);
        var second = new Vector<double>(features);
        second[0] = 1000;
        Assert.Equal(0.0, student.Kernel.Calculate(first, second));
        var one = new Matrix<double>(1, features);
        model.Fit(one, new Vector<double>(new[] { 1.0 }));
        var before = model.Predict(first);
        var two = new Matrix<double>(2, features);
        two[1, 0] = second[0];
        model.Fit(two, new Vector<double>(new[] { 1.0, 10.0 }));
        var after = model.Predict(first);
        AssertClose(before.mean, after.mean, 3e-5, "Independent extra data preserves posterior mean");
        AssertClose(before.variance, after.variance, 3e-5, "Independent extra data preserves posterior variance");
        // A conflicting observation increases uncertainty locally; more data
        // is not a general variance-contraction theorem for this likelihood.
        var local = model.Predict(second);
        double self = student.Kernel.Calculate(second, second);
        double prior = self + 1e-6;
        var expected = IntegrateSingleObservation(10, prior, student.DegreesOfFreedom, student.Scale);
        double projection = self / prior;
        double noise = student.DegreesOfFreedom / (student.DegreesOfFreedom - 2) * student.Scale * student.Scale;
        AssertClose(projection * expected.Mean, local.mean, 3e-5, "Independent added observation mean");
        AssertClose(self - self * projection + projection * projection * expected.Variance + noise,
            local.variance, 3e-5, "Independent added observation variance");
        return true;
    }

    internal static bool VerifyFixedPrior(IGaussianProcess<double> model, int features, params double[] targets)
    {
        if (model is not StudentTGaussianProcess<double> student) return false;
        var point = new Vector<double>(features);
        var inputs = new Matrix<double>(1, features);
        double kernelVariance = student.Kernel.Calculate(point, point);
        // The model factories use the constructor's documented default jitter.
        const double jitter = 1e-6;
        double priorVariance = kernelVariance + jitter;
        foreach (double target in targets)
        {
            var expected = IntegrateSingleObservation(target, priorVariance, student.DegreesOfFreedom, student.Scale);
            model.Fit(inputs, new Vector<double>(new[] { target }));
            foreach (double distance in new[] { 0.0, 0.5, 3.0, 100.0 })
            {
                var testPoint = new Vector<double>(features);
                testPoint[0] = distance;
                var actual = model.Predict(testPoint);
                double cross = student.Kernel.Calculate(testPoint, point);
                double projection = cross / priorVariance;
                double expectedMean = projection * expected.Mean;
                double self = student.Kernel.Calculate(testPoint, testPoint);
                // Observation variance differs from scale squared.
                double noiseVariance = student.DegreesOfFreedom / (student.DegreesOfFreedom - 2) * student.Scale * student.Scale;
                double expectedVariance = self - cross * projection +
                    projection * projection * expected.Variance + noiseVariance;
                AssertClose(expectedMean, actual.mean, 3e-5, $"target {target}, distance {distance}: posterior mean");
                AssertClose(expectedVariance, actual.variance, 3e-5, $"target {target}, distance {distance}: predictive variance");
            }
        }
        return true;
    }

    /// <summary>
    /// Independent composite Simpson oracle, directly integrating the scalar
    /// posterior density rather than using EP sites or production quadrature.
    /// The twelve-standard-deviation window is checked against a wider window.
    /// </summary>
    internal static (double Mean, double Variance) IntegrateSingleObservation(
        double target, double priorVariance, double nu, double scale, int intervals = 40000, double radius = 12)
    {
        double std = Math.Sqrt(priorVariance);
        double step = 2 * radius / intervals;
        double logReference = LogDensity(0);
        double mass = 0, first = 0, second = 0;
        for (int i = 0; i <= intervals; i++)
        {
            double z = -radius + step * i;
            double weight = (i == 0 || i == intervals) ? 1 : (i % 2 == 0 ? 2 : 4);
            double density = weight * Math.Exp(LogDensity(z) - logReference);
            mass += density;
            first += density * z;
            second += density * z * z;
        }
        double mean = first / mass;
        return (std * mean, priorVariance * (second / mass - mean * mean));

        double LogDensity(double z)
        {
            double ratio = (target - std * z) / (scale * Math.Sqrt(nu));
            return -0.5 * z * z - 0.5 * (nu + 1) * Math.Log(1 + ratio * ratio);
        }
    }

    internal static void AssertClose(double expected, double actual, double tolerance, string message)
    {
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) &&
            Math.Abs(actual - expected) <= tolerance * (1 + Math.Abs(expected)),
            $"{message}: expected {expected:R}, actual {actual:R}.");
    }
}
