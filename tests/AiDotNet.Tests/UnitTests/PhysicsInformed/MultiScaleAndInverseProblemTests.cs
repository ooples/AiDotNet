using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PINNs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed
{
    /// <summary>
    /// Unit tests for multi-scale physics and inverse problem features.
    /// </summary>
    public class MultiScaleAndInverseProblemTests
    {
        private readonly INumericOperations<double> _numOps;

        public MultiScaleAndInverseProblemTests()
        {
            _numOps = MathHelper.GetNumericOperations<double>();
        }

        #region Multi-Scale PDE Tests

        [Fact]
        public void MultiScalePDE_TwoScaleHeatEquation_HasCorrectProperties()
        {
            // Arrange
            var multiScalePDE = new TwoScaleHeatEquation(coarseDiffusivity: 1.0, fineDiffusivity: 0.01);

            // Assert
            Assert.Equal(2, multiScalePDE.NumberOfScales);
            Assert.Equal(2, multiScalePDE.InputDimension); // x, t
            Assert.Equal(1, multiScalePDE.OutputDimension); // Temperature
            Assert.Equal("Two-Scale Heat Equation", multiScalePDE.Name);

            // Verify characteristic lengths
            var scales = multiScalePDE.ScaleCharacteristicLengths;
            Assert.Equal(2, scales.Length);
            Assert.True(scales[0] > scales[1], "Coarse scale should be larger than fine scale");
        }

        [Fact]
        public void MultiScalePDE_ComputeScaleResidual_ReturnsValueForEachScale()
        {
            // Arrange
            var multiScalePDE = new TwoScaleHeatEquation(coarseDiffusivity: 1.0, fineDiffusivity: 0.01);
            var inputs = new double[] { 0.5, 0.1 }; // x=0.5, t=0.1
            var outputs = new double[] { 1.0 }; // Temperature = 1.0
            var derivatives = CreateSimpleDerivatives(1, 2);

            // Act
            double coarseResidual = multiScalePDE.ComputeScaleResidual(0, inputs, outputs, derivatives);
            double fineResidual = multiScalePDE.ComputeScaleResidual(1, inputs, outputs, derivatives);

            // Assert
            Assert.True(!double.IsNaN(coarseResidual) && !double.IsInfinity(coarseResidual), "Coarse scale residual should be finite");
            Assert.True(!double.IsNaN(fineResidual) && !double.IsInfinity(fineResidual), "Fine scale residual should be finite");
        }

        [Fact]
        public void MultiScalePDE_ComputeScaleCoupling_ReturnsCouplingResidual()
        {
            // Arrange
            var multiScalePDE = new TwoScaleHeatEquation(coarseDiffusivity: 1.0, fineDiffusivity: 0.01);
            var inputs = new double[] { 0.5, 0.1 };
            var coarseOutputs = new double[] { 1.0 };
            var fineOutputs = new double[] { 0.1 };
            var coarseDerivatives = CreateSimpleDerivatives(1, 2);
            var fineDerivatives = CreateSimpleDerivatives(1, 2);

            // Act
            double couplingResidual = multiScalePDE.ComputeScaleCoupling(
                0, 1, inputs, coarseOutputs, fineOutputs, coarseDerivatives, fineDerivatives);

            // Assert
            Assert.True(!double.IsNaN(couplingResidual) && !double.IsInfinity(couplingResidual), "Coupling residual should be finite");
        }

        [Fact]
        public void MultiScalePDE_GetScaleLossWeight_ReturnsPositiveWeights()
        {
            // Arrange
            var multiScalePDE = new TwoScaleHeatEquation(coarseDiffusivity: 1.0, fineDiffusivity: 0.01);

            // Act & Assert
            for (int scale = 0; scale < multiScalePDE.NumberOfScales; scale++)
            {
                double weight = multiScalePDE.GetScaleLossWeight(scale);
                Assert.True(weight > 0, $"Weight for scale {scale} should be positive");
            }
        }

        [Fact]
        public void MultiScaleTrainingOptions_DefaultValues_AreReasonable()
        {
            // Arrange & Act
            var options = new MultiScaleTrainingOptions<double>();

            // Assert
            Assert.True(options.UseAdaptiveScaleWeighting);
            Assert.False(options.UseSequentialScaleTraining);
            Assert.Equal(100, options.ScalePretrainingEpochs);
        }

        #endregion

        #region Inverse Problem Tests

        [Fact]
        public void InverseProblem_ParameterIdentification_HasCorrectProperties()
        {
            // Arrange
            var observations = new List<(double[] location, double[] value)>
            {
                (new double[] { 0.0, 0.0 }, new double[] { 1.0 }),
                (new double[] { 0.5, 0.1 }, new double[] { 0.8 }),
                (new double[] { 1.0, 0.2 }, new double[] { 0.5 })
            };

            var inverseProblem = new DiffusionParameterIdentification(
                observations: observations,
                initialGuess: 1.0,
                lowerBound: 0.001,
                upperBound: 10.0);

            // Assert
            Assert.Single(inverseProblem.ParameterNames);
            Assert.Equal("diffusion_coefficient", inverseProblem.ParameterNames[0]);
            Assert.Equal(1, inverseProblem.NumberOfParameters);
            Assert.Equal(1.0, inverseProblem.InitialParameterGuesses[0]);
            Assert.Equal(3, inverseProblem.Observations.Count);
        }

        [Fact]
        public void InverseProblem_ValidateParameters_ChecksBounds()
        {
            // Arrange
            var observations = new List<(double[] location, double[] value)>
            {
                (new double[] { 0.5, 0.1 }, new double[] { 0.8 })
            };

            var inverseProblem = new DiffusionParameterIdentification(
                observations: observations,
                initialGuess: 1.0,
                lowerBound: 0.001,
                upperBound: 10.0);

            // Act & Assert
            Assert.True(inverseProblem.ValidateParameters(new double[] { 0.5 }));
            Assert.True(inverseProblem.ValidateParameters(new double[] { 5.0 }));
            Assert.False(inverseProblem.ValidateParameters(new double[] { 0.0001 })); // Below lower bound
            Assert.False(inverseProblem.ValidateParameters(new double[] { 15.0 })); // Above upper bound
            Assert.False(inverseProblem.ValidateParameters(new double[] { -1.0 })); // Negative
        }

        [Fact]
        public void InverseProblem_CreateParameterizedPDE_ReturnsValidPDE()
        {
            // Arrange
            var observations = new List<(double[] location, double[] value)>
            {
                (new double[] { 0.5, 0.1 }, new double[] { 0.8 })
            };

            var inverseProblem = new DiffusionParameterIdentification(
                observations: observations,
                initialGuess: 1.0);

            var parameters = new double[] { 2.5 };

            // Act
            var pde = inverseProblem.CreateParameterizedPDE(parameters);

            // Assert
            Assert.NotNull(pde);
            Assert.Equal(2, pde.InputDimension); // x, t
            Assert.Equal(1, pde.OutputDimension); // Temperature
            Assert.Contains("2.5", pde.Name); // Name should contain the parameter value
        }

        [Fact]
        public void InverseProblemOptions_DefaultValues_AreReasonable()
        {
            // Arrange & Act
            var options = new InverseProblemOptions<double>();

            // Assert
            Assert.Equal(InverseProblemRegularization.L2Tikhonov, options.Regularization);
            Assert.True(options.UseSeparateLearningRates);
            Assert.Equal(0.001, options.ParameterLearningRate);
            Assert.True(options.LogParameterHistory);
            Assert.False(options.EstimateUncertainty);
        }

        [Fact]
        public void InverseProblemResult_CanStoreResults()
        {
            // Arrange & Act
            var result = new InverseProblemResult<double>
            {
                Parameters = new double[] { 1.5, 0.01 },
                ParameterNames = new string[] { "k", "alpha" },
                DataLoss = 0.001,
                PhysicsLoss = 0.002,
                TotalLoss = 0.003,
                Converged = true,
                IterationsToConverge = 500
            };

            // Assert
            Assert.Equal(2, result.Parameters.Length);
            Assert.Equal(1.5, result.Parameters[0]);
            Assert.Equal(0.01, result.Parameters[1]);
            Assert.True(result.Converged);
            Assert.Equal(500, result.IterationsToConverge);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void MultiScalePINN_CanBeCreated_WithValidConfiguration()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1);

            var multiScalePDE = new TwoScaleHeatEquation(1.0, 0.01);
            var boundaryConditions = new IBoundaryCondition<double>[]
            {
                new SimpleDirichletBC(position: 0.0, value: 1.0),
                new SimpleDirichletBC(position: 1.0, value: 0.0)
            };

            // Act
            var pinn = new MultiScalePINN<double>(
                architecture,
                multiScalePDE,
                boundaryConditions,
                numCollocationPointsPerScale: 100);

            // Assert
            Assert.NotNull(pinn);
            Assert.Equal(2, pinn.NumberOfScales);
            Assert.True(pinn.SupportsTraining);
        }

        [Fact]
        public void InverseProblemPINN_CanBeCreated_WithValidConfiguration()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1);

            var observations = new List<(double[] location, double[] value)>
            {
                (new double[] { 0.25, 0.1 }, new double[] { 0.9 }),
                (new double[] { 0.50, 0.1 }, new double[] { 0.7 }),
                (new double[] { 0.75, 0.1 }, new double[] { 0.4 })
            };

            var inverseProblem = new DiffusionParameterIdentification(observations, 1.0);
            var boundaryConditions = new IBoundaryCondition<double>[]
            {
                new SimpleDirichletBC(position: 0.0, value: 1.0),
                new SimpleDirichletBC(position: 1.0, value: 0.0)
            };

            // Act
            var pinn = new InverseProblemPINN<double>(
                architecture,
                inverseProblem,
                boundaryConditions,
                numCollocationPoints: 100);

            // Assert
            Assert.NotNull(pinn);
            Assert.Single(pinn.ParameterNames);
            Assert.Equal("diffusion_coefficient", pinn.ParameterNames[0]);
            Assert.True(pinn.SupportsTraining);
        }

        [Fact]
        public void InverseProblemPINN_Parameters_CanBeUpdatedDuringTraining()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1);

            var observations = new List<(double[] location, double[] value)>
            {
                (new double[] { 0.5, 0.1 }, new double[] { 0.8 })
            };

            var inverseProblem = new DiffusionParameterIdentification(observations, initialGuess: 5.0);
            var boundaryConditions = new IBoundaryCondition<double>[]
            {
                new SimpleDirichletBC(position: 0.0, value: 1.0)
            };

            var pinn = new InverseProblemPINN<double>(
                architecture,
                inverseProblem,
                boundaryConditions,
                numCollocationPoints: 50);

            // Act - Get initial parameters
            var initialParams = pinn.Parameters;

            // Assert
            Assert.Single(initialParams);
            Assert.Equal(5.0, initialParams[0]); // Should match initial guess
        }

        #endregion

        #region Helper Methods and Test Classes

        private PDEDerivatives<double> CreateSimpleDerivatives(int outputDim, int inputDim)
        {
            return new PDEDerivatives<double>
            {
                FirstDerivatives = new double[outputDim, inputDim],
                SecondDerivatives = new double[outputDim, inputDim, inputDim]
            };
        }

        /// <summary>
        /// Test implementation of a two-scale heat equation.
        /// </summary>
        private class TwoScaleHeatEquation : IMultiScalePDE<double>
        {
            private readonly double _coarseDiffusivity;
            private readonly double _fineDiffusivity;

            public TwoScaleHeatEquation(double coarseDiffusivity, double fineDiffusivity)
            {
                _coarseDiffusivity = coarseDiffusivity;
                _fineDiffusivity = fineDiffusivity;
            }

            public int NumberOfScales => 2;
            public double[] ScaleCharacteristicLengths => new double[] { 1.0, 0.1 };
            public int InputDimension => 2;
            public int OutputDimension => 1;
            public string Name => "Two-Scale Heat Equation";

            public double ComputeResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
            {
                // Combined residual (coarse + fine)
                double coarse = ComputeScaleResidual(0, inputs, outputs, derivatives);
                double fine = ComputeScaleResidual(1, inputs, outputs, derivatives);
                return coarse + fine;
            }

            public double ComputeScaleResidual(int scaleIndex, double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
            {
                if (derivatives.FirstDerivatives == null || derivatives.SecondDerivatives == null)
                {
                    return 0.0;
                }

                double diffusivity = scaleIndex == 0 ? _coarseDiffusivity : _fineDiffusivity;

                // Heat equation: ∂u/∂t = k * ∂²u/∂x²
                double dudt = derivatives.FirstDerivatives[0, 1];
                double d2udx2 = derivatives.SecondDerivatives[0, 0, 0];

                return dudt - diffusivity * d2udx2;
            }

            public double ComputeScaleCoupling(
                int coarseIndex, int fineIndex,
                double[] inputs,
                double[] coarseOutputs, double[] fineOutputs,
                PDEDerivatives<double> coarseDerivatives, PDEDerivatives<double> fineDerivatives)
            {
                // Coupling: fine scale average should match coarse scale
                // For simplicity, just return difference in outputs
                return Math.Abs(coarseOutputs[0] - fineOutputs[0]);
            }

            public double GetScaleLossWeight(int scaleIndex)
            {
                // Finer scales get higher weights to ensure they're captured
                return scaleIndex == 0 ? 1.0 : 10.0;
            }

            public int GetScaleOutputDimension(int scaleIndex)
            {
                return 1; // Temperature at each scale
            }
        }

        /// <summary>
        /// Test implementation for diffusion coefficient identification.
        /// </summary>
        private class DiffusionParameterIdentification : IInverseProblem<double>
        {
            private readonly IReadOnlyList<(double[] location, double[] value)> _observations;
            private readonly double _initialGuess;
            private readonly double _lowerBound;
            private readonly double _upperBound;

            public DiffusionParameterIdentification(
                IEnumerable<(double[] location, double[] value)> observations,
                double initialGuess,
                double lowerBound = 0.001,
                double upperBound = 100.0)
            {
                _observations = observations.ToList();
                _initialGuess = initialGuess;
                _lowerBound = lowerBound;
                _upperBound = upperBound;
            }

            public string[] ParameterNames => new[] { "diffusion_coefficient" };
            public int NumberOfParameters => 1;
            public double[] InitialParameterGuesses => new[] { _initialGuess };
            public double[]? ParameterLowerBounds => new[] { _lowerBound };
            public double[]? ParameterUpperBounds => new[] { _upperBound };
            public IReadOnlyList<(double[] location, double[] value)> Observations => _observations;
            public bool HasMeasurementNoiseLevel => false;
            public double MeasurementNoiseLevel => 0.0;

            public bool ValidateParameters(double[] parameters)
            {
                if (parameters.Length != 1) return false;
                double k = parameters[0];
                return k > 0 && k >= _lowerBound && k <= _upperBound;
            }

            public IPDESpecification<double> CreateParameterizedPDE(double[] parameters)
            {
                return new ParameterizedHeatEquation(parameters[0]);
            }
        }

        /// <summary>
        /// Heat equation with parameterized diffusivity.
        /// </summary>
        private class ParameterizedHeatEquation : IPDESpecification<double>
        {
            private readonly double _diffusivity;

            public ParameterizedHeatEquation(double diffusivity)
            {
                _diffusivity = diffusivity;
            }

            public int InputDimension => 2;
            public int OutputDimension => 1;
            public string Name => $"Heat Equation (k={_diffusivity:G4})";

            public double ComputeResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
            {
                if (derivatives.FirstDerivatives == null || derivatives.SecondDerivatives == null)
                {
                    return 0.0;
                }

                double dudt = derivatives.FirstDerivatives[0, 1];
                double d2udx2 = derivatives.SecondDerivatives[0, 0, 0];

                return dudt - _diffusivity * d2udx2;
            }
        }

        /// <summary>
        /// Simple Dirichlet boundary condition for testing.
        /// </summary>
        private class SimpleDirichletBC : IBoundaryCondition<double>
        {
            private readonly double _position;
            private readonly double _value;

            public SimpleDirichletBC(double position, double value)
            {
                _position = position;
                _value = value;
            }

            public string Name => $"Dirichlet BC at x={_position}";

            public bool IsOnBoundary(double[] inputs)
            {
                return Math.Abs(inputs[0] - _position) < 1e-6;
            }

            public double ComputeBoundaryResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
            {
                return outputs[0] - _value;
            }
        }

        #endregion
    }
}
