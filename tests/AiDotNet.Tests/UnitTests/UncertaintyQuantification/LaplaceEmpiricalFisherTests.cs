using AiDotNet;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

/// <summary>
/// Reference test for the diagonal Laplace precision: the empirical Fisher sum_i g_i^2 over per-sample gradients
/// (Daxberger et al. 2021, "Laplace Redux"; laplace-torch's diagonal EF backend), not the square of the batch
/// gradient.
/// </summary>
public class LaplaceEmpiricalFisherTests
{
    /// <summary>Linear model y = w . x with the mean squared-error gradient (1/n) sum_i (w . x_i - y_i) x_i.</summary>
    private sealed class LinearMseModel : IGradientComputable<double, Matrix<double>, Vector<double>>
    {
        private readonly double[] _w = { 0.5, -1.5 };

        public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
        {
            var g = new double[2];
            for (int i = 0; i < input.Rows; i++)
            {
                double residual = _w[0] * input[i, 0] + _w[1] * input[i, 1] - target[i];
                g[0] += residual * input[i, 0] / input.Rows;
                g[1] += residual * input[i, 1] / input.Rows;
            }

            return new Vector<double>(g);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
        }
    }

    [Fact]
    public void EmpiricalFisherDiagonal_SumsSquaredPerSampleGradients()
    {
        var x = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { -1.0, 0.5 }, { 3.0, -2.0 } });
        var y = new Vector<double>(new[] { 1.0, -2.0, 0.25 });

        var fisher = AiModelBuilder<double, Matrix<double>, Vector<double>>.EmpiricalFisherDiagonal(new LinearMseModel(), x, y);

        double f0 = 0, f1 = 0;
        for (int i = 0; i < 3; i++)
        {
            double residual = 0.5 * x[i, 0] - 1.5 * x[i, 1] - y[i];
            f0 += residual * x[i, 0] * residual * x[i, 0];
            f1 += residual * x[i, 1] * residual * x[i, 1];
        }

        Assert.Equal(f0, fisher[0], 1e-12);
        Assert.Equal(f1, fisher[1], 1e-12);
    }
    /// <summary>Same linear model over Tensor batches: input [N, 2], target [N] (rank 1).</summary>
    private sealed class LinearMseTensorModel : IGradientComputable<double, Tensor<double>, Tensor<double>>
    {
        private readonly double[] _w = { 0.5, -1.5 };

        public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
        {
            var g = new double[2];
            int rows = input.Shape[0];
            for (int i = 0; i < rows; i++)
            {
                double residual = _w[0] * input[i, 0] + _w[1] * input[i, 1] - target[i];
                g[0] += residual * input[i, 0] / rows;
                g[1] += residual * input[i, 1] / rows;
            }

            return new Vector<double>(g);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
        }
    }

    [Fact]
    public void EmpiricalFisherDiagonal_RankOneTensorTarget_SumsSquaredPerSampleGradients()
    {
        var x = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 1.0, 2.0, -1.0, 0.5, 3.0, -2.0 }));
        var y = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, -2.0, 0.25 }));

        var fisher = AiModelBuilder<double, Tensor<double>, Tensor<double>>.EmpiricalFisherDiagonal(
            new LinearMseTensorModel(), x, y);

        double f0 = 0, f1 = 0;
        for (int i = 0; i < 3; i++)
        {
            double residual = 0.5 * x[i, 0] - 1.5 * x[i, 1] - y[i];
            f0 += residual * x[i, 0] * residual * x[i, 0];
            f1 += residual * x[i, 1] * residual * x[i, 1];
        }

        Assert.Equal(2, fisher.Length);
        Assert.Equal(f0, fisher[0], 1e-12);
        Assert.Equal(f1, fisher[1], 1e-12);
    }
}
