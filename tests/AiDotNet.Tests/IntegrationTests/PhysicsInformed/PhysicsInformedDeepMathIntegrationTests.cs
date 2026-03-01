using AiDotNet.PhysicsInformed;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.PhysicsInformed;

/// <summary>
/// Deep integration tests for PhysicsInformed:
/// TrainingHistory (loss tracking),
/// DomainDecompositionTrainingHistory (subdomain tracking, validation),
/// MultiFidelityTrainingHistory (multi-fidelity loss tracking),
/// Physics math (PDE residuals, finite differences, domain decomposition theory).
/// </summary>
public class PhysicsInformedDeepMathIntegrationTests
{
    // ============================
    // TrainingHistory: Basics
    // ============================

    [Fact]
    public void TrainingHistory_Defaults_EmptyLosses()
    {
        var history = new TrainingHistory<double>();
        Assert.NotNull(history.Losses);
        Assert.Empty(history.Losses);
    }

    [Fact]
    public void TrainingHistory_AddEpoch_TracksLoss()
    {
        var history = new TrainingHistory<double>();
        history.AddEpoch(1.5);
        history.AddEpoch(1.2);
        history.AddEpoch(0.8);

        Assert.Equal(3, history.Losses.Count);
        Assert.Equal(1.5, history.Losses[0]);
        Assert.Equal(1.2, history.Losses[1]);
        Assert.Equal(0.8, history.Losses[2]);
    }

    // ============================
    // DomainDecompositionTrainingHistory: Construction
    // ============================

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void DomainDecomposition_Construction_ValidSubdomainCount(int count)
    {
        var history = new DomainDecompositionTrainingHistory<double>(count);
        Assert.Equal(count, history.SubdomainCount);
        Assert.Empty(history.SubdomainLosses);
        Assert.Empty(history.InterfaceLosses);
        Assert.Empty(history.PhysicsLosses);
    }

    [Fact]
    public void DomainDecomposition_Construction_ZeroThrows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DomainDecompositionTrainingHistory<double>(0));
    }

    [Fact]
    public void DomainDecomposition_Construction_NegativeThrows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DomainDecompositionTrainingHistory<double>(-1));
    }

    // ============================
    // DomainDecompositionTrainingHistory: AddEpoch
    // ============================

    [Fact]
    public void DomainDecomposition_AddEpoch_TracksAllLosses()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);
        history.AddEpoch(
            totalLoss: 1.5,
            subdomainLosses: new List<double> { 0.8, 0.7 },
            interfaceLoss: 0.3,
            physicsLoss: 0.5);

        Assert.Single(history.SubdomainLosses);
        Assert.Equal(2, history.SubdomainLosses[0].Count);
        Assert.Equal(0.8, history.SubdomainLosses[0][0]);
        Assert.Equal(0.7, history.SubdomainLosses[0][1]);
        Assert.Single(history.InterfaceLosses);
        Assert.Equal(0.3, history.InterfaceLosses[0]);
        Assert.Single(history.PhysicsLosses);
        Assert.Equal(0.5, history.PhysicsLosses[0]);
        Assert.Single(history.Losses); // Base class total loss
        Assert.Equal(1.5, history.Losses[0]);
    }

    [Fact]
    public void DomainDecomposition_AddEpoch_WrongSubdomainCountThrows()
    {
        var history = new DomainDecompositionTrainingHistory<double>(3);
        Assert.Throws<ArgumentException>(() =>
            history.AddEpoch(1.0, new List<double> { 0.5, 0.5 }, 0.1, 0.2)); // Only 2 instead of 3
    }

    [Fact]
    public void DomainDecomposition_AddEpoch_NullSubdomainLossesThrows()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);
        Assert.Throws<ArgumentNullException>(() =>
            history.AddEpoch(1.0, null!, 0.1, 0.2));
    }

    [Fact]
    public void DomainDecomposition_AddEpoch_MultipleEpochs()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);

        for (int epoch = 0; epoch < 5; epoch++)
        {
            double loss = 1.0 - epoch * 0.15;
            history.AddEpoch(loss,
                new List<double> { loss * 0.4, loss * 0.3 },
                loss * 0.2,
                loss * 0.1);
        }

        Assert.Equal(5, history.Losses.Count);
        Assert.Equal(5, history.SubdomainLosses.Count);
        Assert.Equal(5, history.InterfaceLosses.Count);
        Assert.Equal(5, history.PhysicsLosses.Count);
    }

    // ============================
    // MultiFidelityTrainingHistory: Construction
    // ============================

    [Fact]
    public void MultiFidelity_Defaults_EmptyLists()
    {
        var history = new MultiFidelityTrainingHistory<double>();
        Assert.Empty(history.LowFidelityLosses);
        Assert.Empty(history.HighFidelityLosses);
        Assert.Empty(history.CorrelationLosses);
        Assert.Empty(history.PhysicsLosses);
        Assert.Empty(history.Losses);
    }

    [Fact]
    public void MultiFidelity_AddEpoch_TracksAllLosses()
    {
        var history = new MultiFidelityTrainingHistory<double>();
        history.AddEpoch(
            totalLoss: 2.0,
            lowFidelityLoss: 0.5,
            highFidelityLoss: 0.8,
            correlationLoss: 0.3,
            physicsLoss: 0.4);

        Assert.Single(history.Losses);
        Assert.Equal(2.0, history.Losses[0]);
        Assert.Single(history.LowFidelityLosses);
        Assert.Equal(0.5, history.LowFidelityLosses[0]);
        Assert.Single(history.HighFidelityLosses);
        Assert.Equal(0.8, history.HighFidelityLosses[0]);
        Assert.Single(history.CorrelationLosses);
        Assert.Equal(0.3, history.CorrelationLosses[0]);
        Assert.Single(history.PhysicsLosses);
        Assert.Equal(0.4, history.PhysicsLosses[0]);
    }

    // ============================
    // Physics Math: Finite Difference
    // ============================

    [Theory]
    [InlineData(1e-5)]
    [InlineData(1e-4)]
    [InlineData(1e-6)]
    public void PhysicsMath_CentralDifference_DerivativeOfXSquared(double epsilon)
    {
        // f(x) = x^2, f'(x) = 2x
        // Central difference: (f(x+h) - f(x-h)) / (2h)
        double x = 3.0;
        double fPlus = (x + epsilon) * (x + epsilon);
        double fMinus = (x - epsilon) * (x - epsilon);
        double derivative = (fPlus - fMinus) / (2 * epsilon);

        Assert.Equal(6.0, derivative, 1e-4); // f'(3) = 6
    }

    [Theory]
    [InlineData(1e-5)]
    [InlineData(1e-4)]
    public void PhysicsMath_CentralDifference_DerivativeOfSin(double epsilon)
    {
        // f(x) = sin(x), f'(x) = cos(x)
        double x = Math.PI / 4;
        double fPlus = Math.Sin(x + epsilon);
        double fMinus = Math.Sin(x - epsilon);
        double derivative = (fPlus - fMinus) / (2 * epsilon);

        Assert.Equal(Math.Cos(x), derivative, 1e-6);
    }

    // ============================
    // Physics Math: Laplacian
    // ============================

    [Fact]
    public void PhysicsMath_Laplacian_1D()
    {
        // Laplacian of f(x) = x^2 is f''(x) = 2
        double x = 1.0;
        double h = 1e-4;

        Func<double, double> f = (t) => t * t;
        double laplacian = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);

        Assert.Equal(2.0, laplacian, 1e-4);
    }

    [Fact]
    public void PhysicsMath_Laplacian_SinFunction()
    {
        // Laplacian of f(x) = sin(x) is f''(x) = -sin(x)
        double x = Math.PI / 3;
        double h = 1e-4;

        double laplacian = (Math.Sin(x + h) - 2 * Math.Sin(x) + Math.Sin(x - h)) / (h * h);
        Assert.Equal(-Math.Sin(x), laplacian, 1e-4);
    }

    // ============================
    // Physics Math: PDE Residual
    // ============================

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void PhysicsMath_HeatEquation_SteadyState(double x)
    {
        // Steady-state 1D heat equation: d^2T/dx^2 = 0
        // Solution: T(x) = ax + b (linear temperature distribution)
        // For boundary conditions T(0)=0, T(1)=100: T(x) = 100x
        double T = 100.0 * x;
        double h = 1e-3;

        double Tplus = 100.0 * (x + h);
        double Tminus = 100.0 * (x - h);
        double laplacian = (Tplus - 2 * T + Tminus) / (h * h);

        // PDE residual should be ~0 for the exact solution
        Assert.True(Math.Abs(laplacian) < 1e-4,
            $"PDE residual should be near zero, got {laplacian}");
    }

    // ============================
    // Domain Decomposition Math
    // ============================

    [Theory]
    [InlineData(1.0, 2, 0.5)]   // [0,1] split into 2: each size 0.5
    [InlineData(1.0, 4, 0.25)]  // [0,1] split into 4
    [InlineData(10.0, 5, 2.0)]  // [0,10] split into 5
    public void DomainDecomposition_SubdomainSize(double domainSize, int numSubdomains, double expectedSubSize)
    {
        double subSize = domainSize / numSubdomains;
        Assert.Equal(expectedSubSize, subSize, 1e-10);
    }

    [Theory]
    [InlineData(2, 1)]   // 2 subdomains -> 1 interface
    [InlineData(4, 3)]   // 4 subdomains -> 3 interfaces
    [InlineData(8, 7)]   // 8 subdomains -> 7 interfaces
    public void DomainDecomposition_InterfaceCount_1D(int numSubdomains, int expectedInterfaces)
    {
        int interfaces = numSubdomains - 1;
        Assert.Equal(expectedInterfaces, interfaces);
    }
}
