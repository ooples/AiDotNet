using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.NeuralOperators;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.PhysicsInformed.PINNs;
using AiDotNet.PhysicsInformed.ScientificML;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed;

/// <summary>
/// Reference checks of the physics-informed models against their papers. Each test pins a behavior the old code
/// got wrong; the comment on each says what the old code did.
/// </summary>
public class PhysicsInformedPaperReferenceTests
{
    private const double Tolerance = 1e-9;

    private static NeuralNetworkArchitecture<double> Architecture(int inputSize, int outputSize) =>
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: outputSize);

    private static PDEDerivatives<double> Derivatives(double[] first, int outputs, int inputs, double[]? second = null)
    {
        int n = first.Length / (outputs * inputs);
        return new PDEDerivatives<double>
        {
            FirstDerivatives = new Tensor<double>(first, new[] { n, outputs, inputs }),
            SecondDerivatives = second is null ? null : new Tensor<double>(second, new[] { n, outputs, inputs, inputs })
        };
    }

    [Fact]
    public void Poisson_TapeResidual_IsZeroForTheAnalyticSolution()
    {
        // u = x^2 + y^2 has laplacian 4, so with f = 4 the residual laplacian(u) - f vanishes at every point.
        var pde = new PoissonEquation<double>(points => Tensor<double>.CreateDefault(new[] { points.Shape[0] }, 4.0));
        var inputs = new Tensor<double>(new[] { 0.1, 0.2, 0.7, 0.4 }, new[] { 2, 2 });
        var outputs = new Tensor<double>(new[] { 0.05, 0.65 }, new[] { 2, 1 });
        var second = new double[] { 2, 0, 0, 2, 2, 0, 0, 2 };
        var residual = pde.ComputeTapeResidual(inputs, outputs, Derivatives(new double[] { 0.2, 0.4, 1.4, 0.8 }, 1, 2, second));

        Assert.All(residual.ToArray(), value => Assert.Equal(0.0, value, Tolerance));
    }

    [Fact]
    public void Maxwell_TapeResidual_KeepsEachEquationSeparate()
    {
        // Raissi et al. (2019) take the mean square of each residual. The point-wise residual squared the
        // Faraday and Ampere terms and summed them, so a squared loss was quartic in the field error.
        var pde = new MaxwellEquations<double>(1.0, 1.0);
        var first = new double[9];
        first[2 * 3 + 2] = 3.0; // dBz/dt = 3, everything else 0: Faraday residual 3, Ampere residuals 0.
        var residual = pde.ComputeTapeResidual(
            new Tensor<double>(new double[3], new[] { 1, 3 }),
            new Tensor<double>(new double[3], new[] { 1, 3 }),
            Derivatives(first, 3, 3));

        Assert.Equal(new[] { 1, 3 }, residual.Shape.ToArray());
        Assert.Equal(3.0, residual.ToArray()[0], Tolerance);
        Assert.Equal(9.0, pde.ComputeResidual(new Vector<double>(3), new Vector<double>(3), new PDEDerivatives<double>
        {
            FirstDerivatives = new Tensor<double>(first, new[] { 3, 3 })
        }), Tolerance);
    }

    [Fact]
    public void Pinn_TrainingObjective_IncludesThePdeResidual()
    {
        // Raissi et al. (2019): the loss is MSE_u + MSE_f. The old step minimized the data error alone, so the PDE
        // weight had no effect on the update.
        double[] UpdatedParameters(double pdeWeight)
        {
            var pinn = new PhysicsInformedNeuralNetwork<double>(
                Architecture(2, 1), new PoissonEquation<double>(), Array.Empty<IBoundaryCondition<double>>(),
                options: new PhysicsInformedNeuralNetworkOptions { NumCollocationPoints = 16, PdeWeight = pdeWeight });
            var inputs = new Tensor<double>(new[] { 0.0, 0.0, 1.0, 1.0 }, new[] { 2, 2 });
            var targets = new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2, 1 });
            pinn.Train(inputs, targets);
            return pinn.GetParameters().ToArray();
        }

        var dataOnly = UpdatedParameters(0.0);
        var withPhysics = UpdatedParameters(1.0);
        Assert.True(dataOnly.Zip(withPhysics, (a, b) => Math.Abs(a - b)).Max() > 1e-12);
    }

    [Fact]
    public void DeepOperatorNetwork_EvaluateIncludesTheOutputBias()
    {
        // Lu et al. (2021), Eq. 2: G(u)(y) = sum_k b_k t_k + b0. Evaluate and EvaluateMultiple dropped b0.
        var don = new DeepOperatorNetwork<double>(
            Architecture(6, 1), Architecture(4, 8), Architecture(2, 8),
            options: new DeepOperatorNetworkOptions { LatentDimension = 8, NumSensors = 4 });
        var parameters = don.GetParameters().ToArray();
        parameters[parameters.Length - 1] = 5.0;
        don.SetParameters(new Vector<double>(parameters));

        var sensors = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });
        var query = new Vector<double>(new[] { 0.5, 0.6 });
        double predicted = don.Predict(new Tensor<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, new[] { 1, 6 })).ToArray()[0];

        Assert.Equal(predicted, don.Evaluate(sensors, query), Tolerance);
    }

    [Fact]
    public void FourierLayer_ImaginarySpectralWeightsChangeTheOutput()
    {
        // Li et al. (2021) learn complex weights per kept mode. Keeping both ends of the last axis with shared
        // weights and reading the real part of the inverse cancelled Im(W), so it had no effect.
        var layer = new FourierLayer<double>(2, 3, new[] { 8 });
        var input = new Tensor<double>(Enumerable.Range(0, 16).Select(i => Math.Sin(0.7 * i)).ToArray(), new[] { 1, 2, 8 });
        var parameters = layer.GetParameters().ToArray();
        int spectral = 2 * 2 * 3;

        var withoutImaginary = (double[])parameters.Clone();
        Array.Clear(withoutImaginary, spectral, spectral);
        layer.SetParameters(new Vector<double>(withoutImaginary));
        var realOnly = layer.Forward(input).ToArray();

        var withImaginary = (double[])withoutImaginary.Clone();
        for (int i = 0; i < spectral; i++)
        {
            withImaginary[spectral + i] = 0.3;
        }

        layer.SetParameters(new Vector<double>(withImaginary));
        var complex = layer.Forward(input).ToArray();

        Assert.True(realOnly.Zip(complex, (a, b) => Math.Abs(a - b)).Max() > 1e-6);
    }

    [Fact]
    public void Hamiltonian_TrainingFieldIsTheTapeGradient()
    {
        // Greydanus et al. (2019) differentiate H by autodiff. The training field was a central difference with a
        // 1e-3 step, which differs from the exact symplectic gradient by O(1e-7).
        var hnn = new HamiltonianNeuralNetwork<double>();
        var state = new Vector<double>(new[] { 0.3, -0.2 });
        var exact = hnn.ComputeTimeDerivative(state).ToArray();
        var training = hnn.ForwardForTraining(new Tensor<double>(state.ToArray(), new[] { 1, 2 })).ToArray();

        for (int i = 0; i < exact.Length; i++)
        {
            Assert.Equal(exact[i], training[i], 1e-12);
        }
    }

    [Fact]
    public void Lagrangian_AccelerationSatisfiesEulerLagrange()
    {
        // Cranmer et al. (2020), Eq. 6. The old code requested only first derivatives, found no Hessian and threw.
        var lnn = new LagrangianNeuralNetwork<double>();
        var q = new Vector<double>(new[] { 0.2, -0.1 });
        var qDot = new Vector<double>(new[] { 0.3, 0.4 });
        var acceleration = lnn.ComputeAcceleration(q, qDot).ToArray();

        // Euler-Lagrange: sum_j (d2L/dv_i dv_j) a_j + sum_j (d2L/dv_i dq_j) v_j = dL/dq_i, checked by central
        // differences of L.
        const double h = 1e-4;
        double[] qa = q.ToArray();
        double[] va = qDot.ToArray();
        double L(double[] qq, double[] vv) => lnn.ComputeLagrangian(new Vector<double>(qq), new Vector<double>(vv));
        double[] Shift(double[] v, int i, double d, int j = -1, double d2 = 0)
        {
            var a = (double[])v.Clone();
            a[i] += d;
            if (j >= 0)
            {
                a[j] += d2;
            }

            return a;
        }

        for (int i = 0; i < 2; i++)
        {
            double dLdq = (L(Shift(qa, i, h), va) - L(Shift(qa, i, -h), va)) / (2 * h);
            double total = 0;
            for (int j = 0; j < 2; j++)
            {
                double mass = (L(qa, Shift(va, i, h, j, h)) - L(qa, Shift(va, i, h, j, -h))
                    - L(qa, Shift(va, i, -h, j, h)) + L(qa, Shift(va, i, -h, j, -h))) / (4 * h * h);
                double cross = (L(Shift(qa, j, h), Shift(va, i, h)) - L(Shift(qa, j, -h), Shift(va, i, h))
                    - L(Shift(qa, j, h), Shift(va, i, -h)) + L(Shift(qa, j, -h), Shift(va, i, -h))) / (4 * h * h);
                total += mass * acceleration[j] + cross * va[j];
            }

            Assert.Equal(dLdq, total, 1e-4);
        }
    }

    [Fact]
    public void UniversalDifferentialEquation_NetworkLearnsOnlyTheUnknownPart()
    {
        // Rackauckas et al. (2020), Eq. 2: u' = f_known + NN. With f_known already the whole derivative the network
        // must learn zero; the old step fit NN to the full target, so the prediction drifted to twice the target.
        var ude = new UniversalDifferentialEquation<double>(
            Architecture(3, 2), 2, (state, time) => new[] { -state[0], -state[1] });
        var inputs = new Tensor<double>(new[] { 0.5, -0.5, 0.0, 1.0, 0.2, 0.1 }, new[] { 2, 3 });
        var targets = new Tensor<double>(new[] { -0.5, 0.5, -1.0, -0.2 }, new[] { 2, 2 });

        double Error() => ude.Predict(inputs).ToArray().Zip(targets.ToArray(), (p, t) => (p - t) * (p - t)).Sum();
        double before = Error();
        for (int step = 0; step < 50; step++)
        {
            ude.Train(inputs, targets);
        }

        Assert.True(Error() < before);
    }

    [Fact]
    public void MultiScalePinn_SolveUpdatesTheScaleNetworks()
    {
        // The old step called the optimizer with no backward pass, so the weights never moved.
        var pinn = new MultiScalePINN<double>(
            Architecture(2, 1), new TwoScalePoissonProblem<double>(), Array.Empty<IBoundaryCondition<double>>(),
            options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 });
        var before = pinn.GetParameters().ToArray();
        pinn.Solve(epochs: 1, verbose: false);
        var after = pinn.GetParameters().ToArray();

        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Max() > 0.0);
    }

    [Fact]
    public void GpuPinnTrainer_RefusesToTrainWithoutData()
    {
        // MSE_u runs over initial/boundary data. Without data the trainer fabricated 64 zero inputs with zero
        // targets, which pinned u(0) = 0 whatever the problem.
        var pinn = new PhysicsInformedNeuralNetwork<double>(
            Architecture(2, 1), new PoissonEquation<double>(), Array.Empty<IBoundaryCondition<double>>(),
            options: new PhysicsInformedNeuralNetworkOptions { NumCollocationPoints = 16 });
        var trainer = new GpuPINNTrainer<double>(pinn);

        Assert.Throws<ArgumentNullException>(() => trainer.Train(epochs: 1, verbose: false));
    }
}
