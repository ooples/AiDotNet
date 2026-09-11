using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PINNs;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.PINNs;

/// <summary>
/// Regression tests for the per-epoch loss reported by <see cref="MultiScalePINN{T}"/>. The epoch
/// loss is the mean of the per-batch losses, but it was divided by the integer quotient
/// numPoints / batchSize, which drops the trailing partial batch from the count.
/// </summary>
public class MultiScalePINNPrecisionTests
{
    [Fact]
    public void Solve_WithTrailingPartialBatch_ReportsMeanBatchLoss()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1,
            outputSize: 1);

        // No-op optimizer: this test is about loss bookkeeping, not parameter updates.
        var optimizer = new Mock<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>();

        // 300 collocation points with the fixed batch size of 256 -> two batches (256 + 44).
        var pinn = new MultiScalePINN<double>(
            architecture,
            new ConstantResidualPDE(),
            Array.Empty<IBoundaryCondition<double>>(),
            numCollocationPointsPerScale: 300,
            trainingOptions: new MultiScaleTrainingOptions<double> { UseAdaptiveScaleWeighting = false },
            optimizer: optimizer.Object);

        var history = pinn.Solve(epochs: 1, verbose: false);

        // Every point has residual 1 and the scale weight is 1, so each batch loss is exactly 1.0 and
        // their mean is 1.0. The old divisor 300 / 256 == 1 reported the sum of both batches, 2.0.
        Assert.Single(history.Losses);
        Assert.Equal(1.0, history.Losses[0], 12);
    }

    /// <summary>
    /// Single-scale PDE whose residual is identically 1, making every batch loss exactly 1.
    /// </summary>
    private sealed class ConstantResidualPDE : IMultiScalePDE<double>
    {
        public int NumberOfScales => 1;
        public double[] ScaleCharacteristicLengths => new[] { 1.0 };
        public int InputDimension => 1;
        public int OutputDimension => 1;
        public string Name => "Constant residual";

        public double ComputeResidual(Vector<double> inputs, Vector<double> outputs, PDEDerivatives<double> derivatives) => 1.0;

        public double ComputeScaleResidual(int scaleIndex, double[] inputs, double[] outputs, PDEDerivatives<double> derivatives) => 1.0;

        public double ComputeScaleCoupling(
            int coarseIndex,
            int fineIndex,
            double[] inputs,
            double[] coarseOutputs,
            double[] fineOutputs,
            PDEDerivatives<double> coarseDerivatives,
            PDEDerivatives<double> fineDerivatives) => 0.0;

        public double GetScaleLossWeight(int scaleIndex) => 1.0;

        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }
}
