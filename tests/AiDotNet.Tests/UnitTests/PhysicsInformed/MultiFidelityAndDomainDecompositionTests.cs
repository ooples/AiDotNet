using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.PhysicsInformed.PINNs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed;

/// <summary>
/// Unit tests for Sprint 4: Multi-fidelity and Domain Decomposition PINNs.
/// </summary>
public class MultiFidelityAndDomainDecompositionTests
{
    #region Helper Classes

    /// <summary>
    /// Simple boundary condition for testing.
    /// </summary>
    private class SimpleBoundaryCondition : IBoundaryCondition<double>
    {
        private readonly double _boundaryValue;
        private readonly int _dimension;
        private readonly double _location;

        public SimpleBoundaryCondition(double location, double dimension, double boundaryValue)
        {
            _location = location;
            _dimension = (int)dimension;
            _boundaryValue = boundaryValue;
        }

        public string Name => $"Boundary at dim {_dimension} = {_location}";

        public bool IsOnBoundary(double[] inputs)
        {
            return Math.Abs(inputs[_dimension] - _location) < 1e-10;
        }

        public double ComputeBoundaryResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
        {
            return outputs[0] - _boundaryValue;
        }
    }

    #endregion

    #region MultiFidelityTrainingHistory Tests

    [Fact]
    public void MultiFidelityTrainingHistory_AddEpoch_TracksAllMetrics()
    {
        // Arrange
        var history = new MultiFidelityTrainingHistory<double>();

        // Act
        history.AddEpoch(1.0, 0.5, 0.3, 0.1, 0.1);
        history.AddEpoch(0.8, 0.4, 0.2, 0.1, 0.1);

        // Assert
        Assert.Equal(2, history.Losses.Count);
        Assert.Equal(2, history.LowFidelityLosses.Count);
        Assert.Equal(2, history.HighFidelityLosses.Count);
        Assert.Equal(2, history.CorrelationLosses.Count);
        Assert.Equal(2, history.PhysicsLosses.Count);

        Assert.Equal(1.0, history.Losses[0]);
        Assert.Equal(0.5, history.LowFidelityLosses[0]);
        Assert.Equal(0.3, history.HighFidelityLosses[0]);
    }

    [Fact]
    public void MultiFidelityTrainingHistory_ImplementsInterface()
    {
        // Arrange & Act
        var history = new MultiFidelityTrainingHistory<double>();

        // Assert
        Assert.IsAssignableFrom<IMultiFidelityTrainingHistory<double>>(history);
        Assert.IsAssignableFrom<TrainingHistory<double>>(history);
    }

    #endregion

    #region DomainDecompositionTrainingHistory Tests

    [Fact]
    public void DomainDecompositionTrainingHistory_AddEpoch_TracksAllMetrics()
    {
        // Arrange
        var history = new DomainDecompositionTrainingHistory<double>(3);
        var subdomainLosses = new List<double> { 0.3, 0.4, 0.2 };

        // Act
        history.AddEpoch(1.0, subdomainLosses, 0.05, 0.05);

        // Assert
        Assert.Equal(1, history.Losses.Count);
        Assert.Equal(1, history.SubdomainLosses.Count);
        Assert.Equal(1, history.InterfaceLosses.Count);
        Assert.Equal(1, history.PhysicsLosses.Count);
        Assert.Equal(3, history.SubdomainCount);

        Assert.Equal(1.0, history.Losses[0]);
        Assert.Equal(3, history.SubdomainLosses[0].Count);
        Assert.Equal(0.3, history.SubdomainLosses[0][0]);
    }

    [Fact]
    public void DomainDecompositionTrainingHistory_ImplementsInterface()
    {
        // Arrange & Act
        var history = new DomainDecompositionTrainingHistory<double>(2);

        // Assert
        Assert.IsAssignableFrom<IDomainDecompositionTrainingHistory<double>>(history);
        Assert.IsAssignableFrom<TrainingHistory<double>>(history);
    }

    #endregion

    #region MultiFidelityPINN Tests

    [Fact]
    public void MultiFidelityPINN_Constructor_CreatesDefaultLowFidelityNetwork()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        // Act
        var pinn = new MultiFidelityPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        // Assert
        Assert.NotNull(pinn);
        Assert.NotNull(pinn.LowFidelityNetwork);
        Assert.False(pinn.IsLowFidelityFrozen);
    }

    [Fact]
    public void MultiFidelityPINN_SetLowFidelityData_StoresData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new MultiFidelityPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var inputs = new double[10, 2];
        var outputs = new double[10, 1];

        // Act & Assert (no exception)
        pinn.SetLowFidelityData(inputs, outputs);
    }

    [Fact]
    public void MultiFidelityPINN_SetHighFidelityData_StoresData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new MultiFidelityPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var inputs = new double[5, 2];
        var outputs = new double[5, 1];

        // Act & Assert (no exception)
        pinn.SetHighFidelityData(inputs, outputs);
    }

    [Fact]
    public void MultiFidelityPINN_SetLowFidelityFrozen_ChangesState()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new MultiFidelityPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        // Act
        pinn.SetLowFidelityFrozen(true);

        // Assert
        Assert.True(pinn.IsLowFidelityFrozen);
    }

    [Fact]
    public void MultiFidelityPINN_SolveMultiFidelity_ThrowsWithoutData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new MultiFidelityPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            pinn.SolveMultiFidelity(epochs: 10, verbose: false));
    }

    #endregion

    #region DomainDecompositionPINN Tests

    [Fact]
    public void DomainDecompositionPINN_Constructor_CreatesSubdomainNetworks()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var subdomains = new List<SubdomainDefinition<double>>
        {
            new SubdomainDefinition<double>(
                new double[] { 0.0, 0.0 },
                new double[] { 0.5, 1.0 },
                "Left"),
            new SubdomainDefinition<double>(
                new double[] { 0.5, 0.0 },
                new double[] { 1.0, 1.0 },
                "Right")
        };

        // Act
        var pinn = new DomainDecompositionPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            subdomains,
            numCollocationPointsPerSubdomain: 100);

        // Assert
        Assert.NotNull(pinn);
        Assert.Equal(2, pinn.SubdomainCount);
        Assert.NotNull(pinn.GetSubdomainNetwork(0));
        Assert.NotNull(pinn.GetSubdomainNetwork(1));
    }

    [Fact]
    public void DomainDecompositionPINN_Constructor_ThrowsWithNoSubdomains()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var emptySubdomains = new List<SubdomainDefinition<double>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DomainDecompositionPINN<double>(
                architecture,
                pde,
                boundaryConditions,
                emptySubdomains));
    }

    [Fact]
    public void DomainDecompositionPINN_GetSubdomainNetwork_ThrowsOnInvalidIndex()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var subdomains = new List<SubdomainDefinition<double>>
        {
            new SubdomainDefinition<double>(
                new double[] { 0.0, 0.0 },
                new double[] { 1.0, 1.0 })
        };

        var pinn = new DomainDecompositionPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            subdomains,
            numCollocationPointsPerSubdomain: 100);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => pinn.GetSubdomainNetwork(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => pinn.GetSubdomainNetwork(2));
    }

    [Fact]
    public void DomainDecompositionPINN_GetGlobalSolution_ThrowsForPointOutsideDomain()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var subdomains = new List<SubdomainDefinition<double>>
        {
            new SubdomainDefinition<double>(
                new double[] { 0.0, 0.0 },
                new double[] { 1.0, 1.0 })
        };

        var pinn = new DomainDecompositionPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            subdomains,
            numCollocationPointsPerSubdomain: 100);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            pinn.GetGlobalSolution(new double[] { 2.0, 2.0 }));
    }

    [Fact]
    public void DomainDecompositionPINN_GetGlobalSolution_ReturnsForPointInsideDomain()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var subdomains = new List<SubdomainDefinition<double>>
        {
            new SubdomainDefinition<double>(
                new double[] { 0.0, 0.0 },
                new double[] { 1.0, 1.0 })
        };

        var pinn = new DomainDecompositionPINN<double>(
            architecture,
            pde,
            boundaryConditions,
            subdomains,
            numCollocationPointsPerSubdomain: 100);

        // Act
        var solution = pinn.GetGlobalSolution(new double[] { 0.5, 0.5 });

        // Assert
        Assert.NotNull(solution);
        Assert.Single(solution);
    }

    #endregion

    #region SubdomainDefinition Tests

    [Fact]
    public void SubdomainDefinition_Constructor_SetsProperties()
    {
        // Arrange & Act
        var subdomain = new SubdomainDefinition<double>(
            new double[] { 0.0, 0.0 },
            new double[] { 1.0, 1.0 },
            "TestDomain");

        // Assert
        Assert.Equal(0.0, subdomain.LowerBounds[0]);
        Assert.Equal(1.0, subdomain.UpperBounds[1]);
        Assert.Equal("TestDomain", subdomain.Name);
    }

    [Fact]
    public void SubdomainDefinition_Constructor_ThrowsOnDimensionMismatch()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new SubdomainDefinition<double>(
                new double[] { 0.0 },
                new double[] { 1.0, 1.0 }));
    }

    #endregion

    #region InterfaceDefinition Tests

    [Fact]
    public void InterfaceDefinition_DefaultConstructor_InitializesProperties()
    {
        // Arrange & Act
        var iface = new InterfaceDefinition<double>
        {
            Subdomain1Index = 0,
            Subdomain2Index = 1
        };

        // Assert
        Assert.Equal(0, iface.Subdomain1Index);
        Assert.Equal(1, iface.Subdomain2Index);
        Assert.Null(iface.SharedBoundary);
    }

    #endregion
}
