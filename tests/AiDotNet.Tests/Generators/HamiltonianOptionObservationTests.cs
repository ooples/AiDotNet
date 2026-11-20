using System.Reflection;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.PhysicsInformed.ScientificML;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class HamiltonianOptionObservationTests
{
    [Fact]
    public void SimulationObservation_IsDeterministic_AndSolverControlsReduceImplicitResidual()
    {
        // Exercise the actual generated, deterministically seeded nonseparable model.
        var fixtureType = typeof(HamiltonianOptionObservationTests).Assembly.GetType(
            "AiDotNet.Tests.ModelFamilyTests.Generated.HamiltonianNeuralNetworkTests", throwOnError: true)!;
        var fixture = Activator.CreateInstance(fixtureType)!;
        var model = (HamiltonianNeuralNetwork<float>)fixtureType.GetMethod("CreateNetwork",
            BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(fixture, null)!;
        var options = (HamiltonianNeuralNetworkOptions)typeof(HamiltonianNeuralNetwork<float>)
            .GetField("_options", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

        float[] Observe(double tolerance, int iterations)
        {
            options.SymplecticSolverTolerance = tolerance;
            options.SymplecticSolverMaxIterations = iterations;
            return model.Simulate(new Vector<float>(new[] { 1f, 1f }), 4f, 3).ToArray();
        }

        double Residual(float[] trajectory)
        {
            double maximum = 0;
            for (int step = 0; step < 3; step++)
            {
                double oldP = trajectory[2 * step + 1], newP = trajectory[2 * (step + 1) + 1];
                var rate = model.ComputeTimeDerivative(new Vector<float>(new[] { trajectory[2 * step], (float)newP }));
                // Independently evaluate p_next = p_old - dt*dH/dq(q_old,p_next).
                maximum = Math.Max(maximum, Math.Abs(newP - oldP - 4d * rate[1]));
            }
            return maximum;
        }

        var tight = Observe(1e-10, 100);
        Assert.Equal(tight, Observe(1e-10, 100));
        foreach (var approximate in new[] { Observe(1, 100), Observe(1e-10, 1) })
        {
            Assert.All(approximate, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            Assert.True(tight.Zip(approximate, (a, b) => Math.Abs(a - b)).Max() > 1e-6);
            Assert.True(Residual(tight) < 1e-6);
            Assert.True(Residual(approximate) > 10 * Residual(tight));
        }
    }
}
