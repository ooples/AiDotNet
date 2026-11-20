using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
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
using AiDotNet.Tensors.Engines;
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
        // weight had no effect on the update. Both networks start from the same seeded weights, so only the objective
        // can make their updates differ.
        (double[] Initial, double[] Final) UpdatedParameters(double pdeWeight)
        {
            var architecture = Architecture(2, 1);
            architecture.RandomSeed = 42;
            var pinn = new PhysicsInformedNeuralNetwork<double>(
                architecture, new PoissonEquation<double>(), Array.Empty<IBoundaryCondition<double>>(),
                options: new PhysicsInformedNeuralNetworkOptions { NumCollocationPoints = 16, PdeWeight = pdeWeight });
            var inputs = new Tensor<double>(new[] { 0.0, 0.0, 1.0, 1.0 }, new[] { 2, 2 });
            var targets = new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2, 1 });
            var initial = pinn.GetParameters().ToArray();
            pinn.Train(inputs, targets);
            return (initial, pinn.GetParameters().ToArray());
        }

        var (dataInitial, dataOnly) = UpdatedParameters(0.0);
        var (physicsInitial, withPhysics) = UpdatedParameters(1.0);
        Assert.Equal(dataInitial, physicsInitial);
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

        // The bias itself: removing b0 = 5 lowers the evaluation by exactly 5.
        parameters[parameters.Length - 1] = 0.0;
        don.SetParameters(new Vector<double>(parameters));
        Assert.Equal(5.0, predicted - don.Evaluate(sensors, query), Tolerance);
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
    public void Lagrangian_SolvesAnInvertibleMassMatrixWithAZeroLeadingPivot()
    {
        // L = v1 v2 - (q1^2 + q2^2) / 2 at q = (1, 2), v = 0: M = [[0, 1], [1, 0]] and dL/dq = (-1, -2), so the
        // acceleration is M^-1 (-1, -2) = (-2, -1). The unpivoted elimination divided by the zero pivot M_00.
        var lnn = new LagrangianNeuralNetwork<double>();
        var mass = new Tensor<double>(new[] { 0.0, 1.0, 1.0, 0.0 }, new[] { 1, 2, 2 });
        var rhs = new Tensor<double>(new[] { -1.0, -2.0 }, new[] { 1, 2 });

        var acceleration = lnn.SolveBatched(mass, rhs).ToArray();

        Assert.Equal(-2.0, acceleration[0], Tolerance);
        Assert.Equal(-1.0, acceleration[1], Tolerance);
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

    [Fact]
    public void InverseProblemPinn_SolveTrainsTheNetwork()
    {
        // Raissi et al. (2019), Sec. 4, train the network and the coefficients together on MSE_u + MSE_f. The old
        // network step ran the optimizer with no backward pass, so Solve never moved the network.
        var observations = new List<(double[] location, double[] value)>
        {
            (new[] { 0.25, 0.1 }, new[] { 0.9 }),
            (new[] { 0.50, 0.1 }, new[] { 0.7 }),
            (new[] { 0.75, 0.1 }, new[] { 0.4 })
        };
        var pinn = new InverseProblemPINN<double>(
            Architecture(2, 1), new DiffusivityIdentification(observations), Array.Empty<IBoundaryCondition<double>>(),
            options: new InverseProblemOptions<double> { NumCollocationPoints = 16 });
        var probe = new Tensor<double>(new[] { 0.3, 0.2, 0.6, 0.8 }, new[] { 2, 2 });

        var before = pinn.Predict(probe).ToArray();
        pinn.Solve(epochs: 1, learningRate: 0.01, verbose: false);
        var after = pinn.Predict(probe).ToArray();

        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Max() > 1e-9);
    }

    [Fact]
    public void VariationalPinn_TestFunctionsAreLegendreAndVanishOnTheBoundary()
    {
        // Kharazmi et al. (2019), Eq. 13: v_k = P_{k+1} - P_{k-1} on xi = 2x - 1, integrated by Gauss-Legendre
        // quadrature, which is exact for these polynomials. The old test functions and integration rule did not
        // give these values.
        var engine = AiDotNetEngine.Current;
        VariationalPINN<double> Vpinn(Func<Tensor<double>, Tensor<double>, Tensor<double>> integrand) =>
            new VariationalPINN<double>(
                Architecture(1, 1),
                (x, u, gradU, v, gradV) => integrand(v, gradV),
                new VariationalPINNOptions { NumTestFunctions = 3, NumQuadraturePoints = 8 });

        // int_0^1 v_k dx = (1/2) int_-1^1 (P_{k+1} - P_{k-1}) dxi: -1 for k = 1 (P_0 integrates to 2), 0 above.
        var mean = Vpinn((v, gradV) => v);
        Assert.Equal(-1.0, mean.ComputeWeakResidual(0), 1e-12);
        Assert.Equal(0.0, mean.ComputeWeakResidual(1), 1e-12);
        Assert.Equal(0.0, mean.ComputeWeakResidual(2), 1e-12);

        // int_0^1 v_k' dx = v_k(1) - v_k(0) = 0: every test function vanishes on the boundary.
        var slope = Vpinn((v, gradV) => engine.Reshape(gradV, new[] { gradV.Shape[0] }));
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(0.0, slope.ComputeWeakResidual(k), 1e-12);
        }

        // int_0^1 v_k^2 dx = (1/2) (2 / (2k + 3) + 2 / (2k - 1)) by Legendre orthogonality: degree 2k + 2, still
        // exact with 8 nodes.
        var square = Vpinn((v, gradV) => engine.TensorMultiply(v, v));
        for (int k = 1; k <= 3; k++)
        {
            double expected = 1.0 / (2 * k + 3) + 1.0 / (2 * k - 1);
            Assert.Equal(expected, square.ComputeWeakResidual(k - 1), 1e-12);
        }
    }

    [Fact]
    public void FourierLayer_SpectralPathIsTheTruncatedRealFft()
    {
        // Li et al. (2021): rfft over the last axis, keep the frequencies 0 .. m - 1, multiply each by its complex
        // weight, irfft, then add the pointwise path W x + b. The old layer kept both ends of the last axis with
        // shared weights instead.
        const int n = 8;
        var layer = new FourierLayer<double>(1, 3, new[] { n }, new IdentityActivation<double>());
        double[] weightRe = { 0.5, -0.3, 0.8 };
        double[] weightIm = { 0.0, 0.4, -0.6 };
        const double pointwise = 0.7;
        const double bias = 0.1;
        layer.SetParameters(new Vector<double>(weightRe.Concat(weightIm).Concat(new[] { pointwise, bias }).ToArray()));

        double[] x = Enumerable.Range(0, n).Select(i => Math.Sin(0.9 * i) + 0.3 * Math.Cos(2.1 * i) + 0.2).ToArray();
        var output = layer.Forward(new Tensor<double>(x, new[] { 1, 1, n })).ToArray();

        for (int i = 0; i < n; i++)
        {
            double spectral = 0.0;
            for (int k = 0; k < 3; k++)
            {
                double xRe = 0.0, xIm = 0.0;
                for (int j = 0; j < n; j++)
                {
                    double angle = -2.0 * Math.PI * k * j / n;
                    xRe += x[j] * Math.Cos(angle);
                    xIm += x[j] * Math.Sin(angle);
                }

                double yRe = weightRe[k] * xRe - weightIm[k] * xIm;
                double yIm = weightRe[k] * xIm + weightIm[k] * xRe;
                double phase = 2.0 * Math.PI * k * i / n;
                double hermitian = k == 0 ? 1.0 : 2.0;
                spectral += hermitian * (yRe * Math.Cos(phase) - yIm * Math.Sin(phase)) / n;
            }

            Assert.Equal(spectral + pointwise * x[i] + bias, output[i], 1e-10);
        }
    }

    [Fact]
    public void FourierNeuralOperator_LiftIsLinearAndProjectionHasAHiddenLayer()
    {
        // Li et al. (2021), Sec. 4: the lift P is a pointwise linear map (fc0) and the projection Q is two pointwise
        // layers, width -> 128 with GELU, then 128 -> output (fc1, fc2). The old lift applied GELU and the
        // projection was a single layer.
        var fno = new FourierNeuralOperator<double>(
            Architecture(1, 1), new[] { 8 }, options: new FourierNeuralOperatorOptions { Modes = 2, Width = 4, NumLayers = 2 });

        Assert.Equal(2 + 3, fno.Layers.Count);
        Assert.Equal(new[] { 128 }, fno.Layers[fno.Layers.Count - 2].GetOutputShape());

        var lift = fno.Layers[0];
        double[] Lift(double scale) =>
            lift.Forward(new Tensor<double>(new[] { 0.4 * scale, -1.3 * scale, 2.2 * scale }, new[] { 3, 1 })).ToArray();
        var origin = Lift(0.0);
        var single = Lift(1.0);
        var twice = Lift(2.0);
        for (int i = 0; i < origin.Length; i++)
        {
            Assert.Equal(2.0 * (single[i] - origin[i]), twice[i] - origin[i], 1e-12);
        }
    }

    [Fact]
    public void GraphNeuralOperator_NormalizedAdjacencyIgnoresTheEdgeScale()
    {
        // Kipf and Welling (2017): D^-1/2 A D^-1/2 is unchanged when every edge weight is scaled. The old code added
        // 1e-10 to every degree, so a graph with small weights (degree 1e-8 here) was normalized about 1% low; an
        // isolated node keeps a zero row either way.
        var gno = new GraphNeuralOperator<double>(
            Architecture(2, 3), options: new GraphNeuralOperatorOptions { HiddenDim = 3, NumLayers = 2, InputDim = 2 });
        var features = new Tensor<double>(new[] { 0.5, -0.2, 0.1, 0.9, -0.7, 0.3 }, new[] { 3, 2 });
        double[] adjacency = { 0, 1, 0, 1, 0, 0, 0, 0, 0 };

        var unit = gno.Forward(features, new Tensor<double>(adjacency, new[] { 3, 3 })).ToArray();
        var small = gno.Forward(features, new Tensor<double>(adjacency.Select(a => a * 1e-8).ToArray(), new[] { 3, 3 })).ToArray();

        for (int i = 0; i < unit.Length; i++)
        {
            Assert.Equal(unit[i], small[i], 1e-12);
        }
    }

    [Fact]
    public void FourierLayer2D_EachModeCornerHasItsOwnWeights()
    {
        // Li et al. (2021), SpectralConv2d in the reference code: rfft2, then out_ft[:, :, :m1, :m2] uses weights1
        // and out_ft[:, :, -m1:, :m2] uses weights2, then irfft2. The old layer used one weight tensor for both
        // corners, which tied R(k1, k2) to R(k1 - m1, k2) and halved the spectral parameters.
        const int n1 = 4, n2 = 6, m = 2;
        var layer = new FourierLayer<double>(1, m, new[] { n1, n2 }, new IdentityActivation<double>());
        double[] lowRe = { 0.5, -0.3, 0.8, 0.2 }, highRe = { -0.7, 0.4, 0.1, 0.9 };
        double[] lowIm = { 0.3, 0.6, -0.2, 0.5 }, highIm = { 0.0, -0.4, 0.7, -0.1 };
        const double pointwise = 0.7, bias = 0.1;
        layer.SetParameters(new Vector<double>(lowRe.Concat(highRe).Concat(lowIm).Concat(highIm)
            .Concat(new[] { pointwise, bias }).ToArray()));

        var x = new double[n1 * n2];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = Math.Sin(0.9 * i) + 0.3 * Math.Cos(2.1 * i);
        }

        var output = layer.Forward(new Tensor<double>(x, new[] { 1, 1, n1, n2 })).ToArray();

        // The kept spectrum Y on the half plane k2 = 0 .. m - 1: rows 0 .. m - 1 with the first weights and rows
        // n1 - m .. n1 - 1 with the second.
        var yRe = new double[n1, m];
        var yIm = new double[n1, m];
        for (int corner = 0; corner < 2; corner++)
        {
            for (int j = 0; j < m; j++)
            {
                int k1 = corner == 0 ? j : n1 - m + j;
                for (int k2 = 0; k2 < m; k2++)
                {
                    double xRe = 0, xIm = 0;
                    for (int a = 0; a < n1; a++)
                    {
                        for (int b = 0; b < n2; b++)
                        {
                            double angle = -2.0 * Math.PI * ((double)k1 * a / n1 + (double)k2 * b / n2);
                            xRe += x[a * n2 + b] * Math.Cos(angle);
                            xIm += x[a * n2 + b] * Math.Sin(angle);
                        }
                    }

                    int w = j * m + k2;
                    double wRe = corner == 0 ? lowRe[w] : highRe[w];
                    double wIm = corner == 0 ? lowIm[w] : highIm[w];
                    yRe[k1, k2] = wRe * xRe - wIm * xIm;
                    yIm[k1, k2] = wRe * xIm + wIm * xRe;
                }
            }
        }

        // irfft2: the k2 = 0 column once, every other kept k2 twice for its Hermitian partner.
        for (int a = 0; a < n1; a++)
        {
            for (int b = 0; b < n2; b++)
            {
                double spectral = 0;
                for (int k1 = 0; k1 < n1; k1++)
                {
                    for (int k2 = 0; k2 < m; k2++)
                    {
                        double phase = 2.0 * Math.PI * ((double)k1 * a / n1 + (double)k2 * b / n2);
                        double hermitian = k2 == 0 ? 1.0 : 2.0;
                        spectral += hermitian * (yRe[k1, k2] * Math.Cos(phase) - yIm[k1, k2] * Math.Sin(phase));
                    }
                }

                double expected = spectral / (n1 * n2) + pointwise * x[a * n2 + b] + bias;
                Assert.Equal(expected, output[a * n2 + b], 1e-10);
            }
        }
    }

    [Fact]
    public void FourierLayer_HermitianWeightsMarkOnlyDcAndNyquist_InHalfPrecision()
    {
        // irfft counts the DC term and the Nyquist term of an even length once and every other kept frequency
        // twice. The weights were found by comparing a frequency ramp built in T with n / 2; in half precision
        // integers above 2048 are not exact, so the comparison flagged a neighbour of the Nyquist index too.
        const int length = 8200;
        const int modes = length / 2 + 1;
        var weights = FourierLayer<Half>.HermitianWeights(length, modes, 1).ToArray();

        Assert.Equal(modes, weights.Length);
        for (int k = 0; k < modes; k++)
        {
            float expected = k == 0 || k == length / 2 ? 1f : 2f;
            Assert.Equal((Half)expected, weights[k]);
        }
    }

    [Fact]
    public void SymbolicPhysicsLearner_MeanSquaredErrorDoesNotOverflowHalfPrecision()
    {
        // The MSE squared the raw errors and divided by n in T. In half precision 300^2 = 90000 is above 65504, so
        // an expression whose mean squared error 22500 is representable scored +infinity and was discarded.
        var learner = new SymbolicPhysicsLearner<Half>();
        var expression = new SymbolicExpression<Half>("x1", SymbolicExpressionType.Variable);
        var inputs = new Tensor<Half>(new[] { (Half)300f, (Half)0f, (Half)0f, (Half)0f }, new[] { 4, 1 });
        var outputs = new Tensor<Half>(new[] { (Half)0f, (Half)0f, (Half)0f, (Half)0f }, new[] { 4 });

        double mse = learner.ComputeMse(expression, inputs, outputs);

        // 22500 rounds to the half value 22496.
        Assert.Equal(22500.0, mse, 32.0);
    }

    [Fact]
    public void DeepOperatorNetwork_GradientsIncludeTheOutputBias()
    {
        // Lu et al. (2021), Eq. 2: G(u)(y) = sum_k b_k t_k + b0, so for the MSE loss dL/db0 = (2 / B) sum (G - y).
        // GetGradients concatenated the branch and trunk gradients only: the vector was one entry shorter than
        // GetParameters and carried no b0 gradient.
        var don = new DeepOperatorNetwork<double>(
            Architecture(6, 1), Architecture(4, 8), Architecture(2, 8),
            options: new DeepOperatorNetworkOptions { LatentDimension = 8, NumSensors = 4 });
        var rows = new Tensor<double>(new[]
        {
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            -0.3, 0.7, 0.2, -0.1, 0.9, 0.1,
            0.4, -0.5, 0.6, 0.8, -0.2, 0.3
        }, new[] { 3, 6 });
        var targets = new Tensor<double>(new[] { 1.5, -0.4, 0.8 }, new[] { 3, 1 });

        var predictions = don.Predict(rows).ToArray();
        double expected = 2.0 / 3.0 * predictions.Zip(targets.ToArray(), (p, t) => p - t).Sum();

        don.Train(rows, targets);
        var gradients = don.GetGradients().ToArray();

        Assert.Equal(don.GetParameters().Length, gradients.Length);
        Assert.Equal(expected, gradients[gradients.Length - 1], 1e-10);
    }

    [Fact]
    public void VariationalPinn_GaussLegendreRuleMatchesLeggaussInHalfPrecision()
    {
        // The rule was computed in T: P'_64 at the outer nodes is about 2000, so P'^2 overflowed half precision and
        // the outer weights became 0. Computed in double and rounded once, every node and weight is the half value
        // nearest numpy.polynomial.legendre.leggauss(64), and the weights sum to 2.
        double[] nodes0 = { -0.99930504173577217, -0.99634011677195522, -0.99101337147674429, -0.98333625388462598 };
        double[] weights0 = { 0.0017832807216941399, 0.0041470332605644993, 0.0065044579689785024, 0.008846759826363397 };
        double[] nodes30 = { -0.072993121787799042, -0.024350292663424429, 0.024350292663424429, 0.072993121787799042 };
        double[] weights30 = { 0.048575467441503511, 0.048690957009139814, 0.048690957009139814, 0.048575467441503511 };

        var nodes = VariationalPINN<Half>.GaussLegendre(64, out var weightTensor).ToArray();
        var weights = weightTensor.ToArray();

        // Round-to-nearest in half precision is within 2^-11 of the value, relatively.
        void Near(double reference, Half actual) =>
            Assert.True(Math.Abs((float)actual - reference) <= Math.Abs(reference) * Math.Pow(2, -11),
                $"{(float)actual} is not the half value nearest {reference}.");

        for (int i = 0; i < 4; i++)
        {
            Near(nodes0[i], nodes[i]);
            Near(weights0[i], weights[i]);
            Near(nodes30[i], nodes[30 + i]);
            Near(weights30[i], weights[30 + i]);
        }

        // 64 half roundings of relative size 2^-11 each.
        Assert.Equal(2.0, weights.Sum(w => (double)(float)w), 2e-3);
    }

    [Fact]
    public void VariationalPinn_GaussLegendreRuleMatchesLeggaussInDouble()
    {
        // Regression guard for the double rule (the old one held P_0 .. P_n for all n nodes): leggauss(64) to
        // round-off, and the rule integrates x^126, the highest degree it is exact for.
        var (nodes, weights) = VariationalPINN<double>.GaussLegendreRule(64);
        Assert.Equal(-0.99930504173577217, nodes[0], 1e-14);
        Assert.Equal(0.0017832807216941399, weights[0], 1e-14);
        Assert.Equal(0.024350292663424429, nodes[32], 1e-14);
        Assert.Equal(0.048690957009139814, weights[32], 1e-14);
        Assert.Equal(2.0, weights.Sum(), 1e-13);
        Assert.Equal(2.0 / 127.0, nodes.Zip(weights, (x, w) => w * Math.Pow(x, 126)).Sum(), 1e-13);
    }

    [Fact]
    public void UniversalDifferentialEquation_TimeGridIsRoundedOnceInHalfPrecision()
    {
        // t_k = t0 + k dt must be the half value nearest the exact time. The ramp was arange(k) in T times dt in T:
        // the engine's arange rounds each k once (exact to 2048), then the product rounds again, so k = 2049 became
        // 2048 and t = 2048 * 3 = 6144 where the exact 6147 rounds to 6148.
        const int steps = 4096;
        var times = UniversalDifferentialEquation<Half>.TimeGrid((Half)0f, (Half)12288f, steps, out var dt);

        Assert.Equal((Half)3f, dt);
        Assert.Equal(steps, times.Length);
        for (int k = 0; k < steps; k++)
        {
            Assert.Equal((Half)(float)(3.0 * k), times[k]);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MultiScalePinn_CompiledTrainToSolveToCompiledTrain_PreservesOptimizerContinuity(bool explicitOptimizer)
    {
        var x1 = new Tensor<float>(new[] { 0.1f, 0.2f, 0.4f, 0.3f, 0.7f, 0.6f, 0.9f, 0.8f }, new[] { 4, 2 });
        var y1 = new Tensor<float>(new[] { 0.5f, -0.3f, 0.8f, 0.2f }, new[] { 4, 1 });
        var x2 = new Tensor<float>(new[] { 0.2f, 0.5f, 0.6f, 0.1f, 0.8f, 0.4f, 0.3f, 0.9f }, new[] { 4, 2 });
        var y2 = new Tensor<float>(new[] { -0.2f, 0.6f, 0.1f, -0.4f }, new[] { 4, 1 });

        MultiScalePINN<float> CreatePinn(Vector<float>? initialParameters, out AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>? optimizer)
        {
            var arch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1)
            {
                RandomSeed = 1337
            };
            optimizer = explicitOptimizer
                ? new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.01,
                        UseAdaptiveLearningRate = false,
                        EnableGradientClipping = false,
                    })
                : null;
            var model = new MultiScalePINN<float>(
                arch,
                new CoupledTwoScalePoissonPde(),
                Array.Empty<IBoundaryCondition<float>>(),
                trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                optimizer: optimizer,
                options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 });
            if (initialParameters is not null)
            {
                model.SetParameters(initialParameters);
            }

            return model;
        }

        static void ResetMoments(MultiScalePINN<float> model, AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>? opt)
        {
            if (opt is not null)
            {
                opt.Reset();
            }
            else
            {
                model.ResetBaseTrainOptimizerState();
            }
        }

        Vector<float> RunTrajectory(bool compiled, Vector<float>? init, bool resetBetweenPhases, out MultiScalePINN<float> model)
        {
            var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
            bool saved = codecOptions.EnableCompilation;
            try
            {
                codecOptions.EnableCompilation = compiled;
                model = CreatePinn(init, out var opt);
                model.Train(x1, y1);
                model.Train(x1, y1);
                if (compiled)
                {
                    Assert.True(model.FusedSession.IsCommitted, "Supervised Train before Solve must commit the fused plan.");
                    Assert.False(model.FusedSession.IsDisabled);
                }

                if (resetBetweenPhases)
                {
                    ResetMoments(model, opt);
                }

                model.Solve(epochs: 1, verbose: false);
                if (compiled)
                {
                    Assert.True(model.FusedSession.IsCommitted, "First Solve must continue the committed fused session without disabling it.");
                    Assert.False(model.FusedSession.IsDisabled);
                }

                if (resetBetweenPhases)
                {
                    ResetMoments(model, opt);
                }

                model.Solve(epochs: 1, verbose: false);
                if (compiled)
                {
                    Assert.True(model.FusedSession.IsCommitted, "Consecutive Solve calls must preserve the committed fused session and Adam moments.");
                    Assert.False(model.FusedSession.IsDisabled);
                }

                if (resetBetweenPhases)
                {
                    ResetMoments(model, opt);
                }

                model.Train(x2, y2);
                model.Train(x2, y2);
                if (compiled)
                {
                    Assert.True(model.FusedSession.IsCommitted, "Supervised Train after Solve must stay on the committed fused session.");
                }

                return model.GetParameters();
            }
            finally
            {
                codecOptions.EnableCompilation = saved;
            }
        }

        var initial = CreatePinn(null, out _).GetParameters();
        var eagerParams = RunTrajectory(compiled: false, initial, resetBetweenPhases: false, out _);
        var resetParams = RunTrajectory(compiled: false, initial, resetBetweenPhases: true, out _);
        var compiledParams = RunTrajectory(compiled: true, initial, resetBetweenPhases: false, out _);

        double maxDiffVsEager = eagerParams.ToArray().Zip(compiledParams.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();
        double maxDiffIfMomentsLost = eagerParams.ToArray().Zip(resetParams.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();

        double minExpectedResetDivergence = explicitOptimizer ? 5e-4 : 1e-4;
        double maxAllowedCompiledDrift = explicitOptimizer ? 1e-4 : 1e-5;
        Assert.True(maxDiffIfMomentsLost > minExpectedResetDivergence, $"Reset-moments control did not diverge enough ({maxDiffIfMomentsLost:E3}).");
        Assert.True(maxDiffVsEager < maxAllowedCompiledDrift, $"Compiled-to-Solve-to-compiled trajectory diverged from eager by {maxDiffVsEager:E3}.");
    }

    [Fact]
    public void MultiScalePinn_SolvePreservesIndividualObjectiveSessions_AndEnforcesCommittedRefusals()
    {
        var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = codecOptions.EnableCompilation;
        try
        {
            codecOptions.EnableCompilation = true;
            var arch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1)
            {
                RandomSeed = 99
            };
            var primaryOptimizer = new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.01,
                    UseAdaptiveLearningRate = false,
                });
            var togglePde = new HostReadToggleMultiScalePde();
            var pinn = new MultiScalePINN<float>(
                arch,
                togglePde,
                Array.Empty<IBoundaryCondition<float>>(),
                trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                optimizer: primaryOptimizer,
                options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 6 });

            var x = new Tensor<float>(new[] { 0.2f, 0.3f, 0.7f, 0.8f }, new[] { 2, 2 });
            var y = new Tensor<float>(new[] { 0.4f, -0.1f }, new[] { 2, 1 });

            // 1. Commit an independent custom-objective session on secondaryOptimizer, and record a declined custom-objective
            // session on tertiaryOptimizer (value clipping has no fused form) BEFORE FusedSession is committed.
            var secondaryOptimizer = new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.005,
                    UseAdaptiveLearningRate = false,
                });
            pinn.TrainWithCustomLoss(
                x,
                pred => AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(pred, pred), new[] { 0, 1 }, keepDims: false),
                secondaryOptimizer);
            Assert.True(pinn.ObjectiveSessionCommitted, "Secondary custom-objective session must commit independently.");

            var tertiaryOptimizer = new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.005,
                    UseAdaptiveLearningRate = false,
                    EnableGradientClipping = true,
                    GradientClippingMethod = AiDotNet.Models.Options.GradientClippingMethod.ByValue,
                });
            pinn.TrainWithCustomLoss(
                x,
                pred => AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(pred, pred), new[] { 0, 1 }, keepDims: false),
                tertiaryOptimizer);

            Assert.False(pinn.FusedSession.IsCommitted, "Primary FusedSession has not run yet.");
            Assert.False(pinn.FusedSession.IsDisabled, "Primary FusedSession must remain enabled when another objective session declines.");
            Assert.True(pinn.ObjectiveSessionCommitted);
            Assert.NotNull(pinn.LastObjectiveMissReason);

            // 2. Run Solve BEFORE Train even though ObjectiveSessionCommitted is true on secondaryOptimizer: Solve must commit
            // primary FusedSession directly (not misroute into _objectiveSessions[primaryOptimizer]), and subsequent Train + Solve must continue it.
            pinn.Solve(epochs: 1, verbose: false);
            Assert.True(pinn.FusedSession.IsCommitted, "Solve must commit primary FusedSession even when another optimizer holds a committed objective session.");
            Assert.False(pinn.FusedSession.IsDisabled);
            pinn.Train(x, y);
            Assert.True(pinn.FusedSession.IsCommitted);
            pinn.Solve(epochs: 1, verbose: false);
            Assert.True(pinn.FusedSession.IsCommitted);
            Assert.False(pinn.FusedSession.IsDisabled);
            Assert.True(pinn.ObjectiveSessionCommitted);
            Assert.NotNull(pinn.LastObjectiveMissReason);

            // Now flip the PDE to perform an undeclared host read after FusedSession has committed: Solve must refuse before mutating weights.
            var beforeRefusal = pinn.GetParameters().ToArray();
            togglePde.ReadOutputOnHost = true;
            Assert.Throws<InvalidOperationException>(() => pinn.Solve(epochs: 1, verbose: false));
            Assert.Equal(beforeRefusal, pinn.GetParameters().ToArray());
            Assert.True(pinn.FusedSession.IsCommitted);
        }
        finally
        {
            codecOptions.EnableCompilation = saved;
        }
    }

    [Fact]
    public void MultiScalePinn_SolveAppliesGradientClipping_AndMatchesEager()
    {
        static (Vector<float> Params, bool Committed) RunClippedSolve(bool compiled, double clipNorm, Vector<float>? init)
        {
            var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
            bool saved = codecOptions.EnableCompilation;
            try
            {
                codecOptions.EnableCompilation = compiled;
                var arch = new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 2,
                    outputSize: 1)
                {
                    RandomSeed = 2026
                };
                var sgd = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.05,
                        UseAdaptiveLearningRate = false,
                        EnableGradientClipping = clipNorm > 0.0,
                        MaxGradientNorm = clipNorm > 0.0 ? clipNorm : 1000.0,
                    });
                var pinn = new MultiScalePINN<float>(
                    arch,
                    new LargeForcingPoissonPde(),
                    Array.Empty<IBoundaryCondition<float>>(),
                    trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                    optimizer: sgd,
                    options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 });
                if (init is not null)
                {
                    pinn.SetParameters(init);
                }

                if (clipNorm >= 0.0)
                {
                    pinn.Solve(epochs: 3, verbose: false);
                }

                return (pinn.GetParameters(), pinn.FusedSession.IsCommitted);
            }
            finally
            {
                codecOptions.EnableCompilation = saved;
            }
        }

        var initVec = RunClippedSolve(compiled: false, clipNorm: -1.0, init: null).Params;
        var initial = initVec.ToArray();
        var (eagerClipped, _) = RunClippedSolve(compiled: false, clipNorm: 0.02, initVec);
        var (compiledClipped, committed) = RunClippedSolve(compiled: true, clipNorm: 0.02, initVec);
        var (compiledDefaultModelClip, unclippedCommitted) = RunClippedSolve(compiled: true, clipNorm: 0.0, initVec);

        Assert.True(committed, "Clipped Solve must commit a fused plan.");
        Assert.True(unclippedCommitted, "Default model-clipped Solve must commit a fused plan.");

        double clippedStepNorm = Math.Sqrt(compiledClipped.ToArray().Zip(initial, (a, b) => ((double)a - b) * ((double)a - b)).Sum());
        double widerStepNorm = Math.Sqrt(compiledDefaultModelClip.ToArray().Zip(initial, (a, b) => ((double)a - b) * ((double)a - b)).Sum());
        double eagerDiff = eagerClipped.ToArray().Zip(compiledClipped.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();

        Assert.True(clippedStepNorm < 0.1 * widerStepNorm,
            $"Gradient clipping did not bound the Solve update (tight={clippedStepNorm:E3}, modelClip={widerStepNorm:E3}).");
        Assert.True(eagerDiff < 1e-4, $"Compiled clipped Solve diverged from eager clipped Solve by {eagerDiff:E3}.");
    }

    [Fact]
    public void MultiScalePinn_SolveReplaysPerStepRandomDraws_AndMatchesEager()
    {
        var x = new Tensor<float>(new[] { 0.15f, 0.25f, 0.65f, 0.75f }, new[] { 2, 2 });
        var y = new Tensor<float>(new[] { 0.3f, -0.2f }, new[] { 2, 1 });

        Vector<float> RunStochasticSolve(bool compiled, bool freezeNoise, Vector<float>? init)
        {
            var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
            bool saved = codecOptions.EnableCompilation;
            try
            {
                codecOptions.EnableCompilation = compiled;
                var arch = new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 2,
                    outputSize: 1)
                {
                    RandomSeed = 314
                };
                var adam = new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.01,
                        UseAdaptiveLearningRate = false,
                    });
                var pde = new StochasticForcingMultiScalePde(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(77), freezeNoise);
                var pinn = new MultiScalePINN<float>(
                    arch,
                    pde,
                    Array.Empty<IBoundaryCondition<float>>(),
                    trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                    optimizer: adam,
                    options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 });
                if (init is not null)
                {
                    pinn.SetParameters(init);
                }

                pinn.Train(x, y);
                pinn.Solve(epochs: 4, verbose: false);
                pinn.Train(x, y);
                if (compiled)
                {
                    Assert.True(pinn.FusedSession.IsCommitted);
                }

                return pinn.GetParameters();
            }
            finally
            {
                codecOptions.EnableCompilation = saved;
            }
        }

        var init = RunStochasticSolve(compiled: false, freezeNoise: false, init: null);
        var eager = RunStochasticSolve(compiled: false, freezeNoise: false, init);
        var compiled = RunStochasticSolve(compiled: true, freezeNoise: false, init);
        var frozenControl = RunStochasticSolve(compiled: false, freezeNoise: true, init);

        double diffVsEager = eager.ToArray().Zip(compiled.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();
        double diffVsFrozen = eager.ToArray().Zip(frozenControl.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();

        Assert.True(diffVsFrozen > 1e-4, $"Frozen-noise control did not differ from per-step draws ({diffVsFrozen:E3}).");
        Assert.True(diffVsEager < 1e-4, $"Compiled Solve with CompiledStepRandom draws diverged from eager by {diffVsEager:E3}.");
    }

    [Fact]
    public void MultiScalePinn_ThirdOrderPdeJetMatchesAutodiff_AndCompiledSolveMatchesEager()
    {
        var points = new Tensor<double>(new[]
        {
            0.0, 0.0,
            0.25, 0.70,
            -0.40, 0.30,
            0.80, -0.50,
            0.10, 0.90,
        }, new[] { 5, 2 });

        static MultiScalePINN<double> CreateThirdOrderPinn(Vector<double>? init)
        {
            var arch = Architecture(2, 1);
            var model = new MultiScalePINN<double>(
                arch,
                new ThirdOrderKdVMultiScalePde(),
                Array.Empty<IBoundaryCondition<double>>(),
                trainingOptions: new MultiScaleTrainingOptions<double> { UseAdaptiveScaleWeighting = false },
                options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 6 });
            if (init is not null)
            {
                model.SetParameters(init);
            }

            return model;
        }

        var probeModel = CreateThirdOrderPinn(null);
        for (int scale = 0; scale < probeModel.ScaleNetworks.Count; scale++)
        {
            var scaleNet = probeModel.ScaleNetworks[scale];
            var jetOut = probeModel.EvaluateScaleNetwork(scaleNet, points, order: 3, out var jetDerivs);
            var autoOut = PinnAutodiff<double>.Evaluate(
                x =>
                {
                    var current = x;
                    foreach (var layer in scaleNet.Layers)
                    {
                        current = layer.Forward(current);
                    }

                    return current;
                },
                points,
                order: 3,
                out var autoDerivs);

            Assert.NotNull(jetDerivs.FirstDerivatives);
            Assert.NotNull(jetDerivs.SecondDerivatives);
            Assert.NotNull(jetDerivs.ThirdDerivatives);
            Assert.NotNull(autoDerivs.FirstDerivatives);
            Assert.NotNull(autoDerivs.SecondDerivatives);
            Assert.NotNull(autoDerivs.ThirdDerivatives);

            double outDiff = jetOut.ToArray().Zip(autoOut.ToArray(), (a, b) => Math.Abs(a - b)).Max();
            double d1Diff = (jetDerivs.FirstDerivatives ?? throw new InvalidOperationException()).ToArray()
                .Zip((autoDerivs.FirstDerivatives ?? throw new InvalidOperationException()).ToArray(), (a, b) => Math.Abs(a - b)).Max();
            double d2Diff = (jetDerivs.SecondDerivatives ?? throw new InvalidOperationException()).ToArray()
                .Zip((autoDerivs.SecondDerivatives ?? throw new InvalidOperationException()).ToArray(), (a, b) => Math.Abs(a - b)).Max();
            double d3Diff = (jetDerivs.ThirdDerivatives ?? throw new InvalidOperationException()).ToArray()
                .Zip((autoDerivs.ThirdDerivatives ?? throw new InvalidOperationException()).ToArray(), (a, b) => Math.Abs(a - b)).Max();

            Assert.True(outDiff < 1e-12, $"Scale {scale} output differed by {outDiff:E3}.");
            Assert.True(d1Diff < 1e-12, $"Scale {scale} first derivatives differed by {d1Diff:E3}.");
            Assert.True(d2Diff < 1e-12, $"Scale {scale} second derivatives differed by {d2Diff:E3}.");
            Assert.True(d3Diff < 1e-12, $"Scale {scale} third derivatives differed by {d3Diff:E3}.");
        }

        var initial = probeModel.GetParameters();
        var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = codecOptions.EnableCompilation;
        Vector<double> eagerParams;
        Vector<double> compiledParams;
        try
        {
            codecOptions.EnableCompilation = false;
            var eagerModel = CreateThirdOrderPinn(initial);
            eagerModel.Solve(epochs: 3, verbose: false);
            eagerParams = eagerModel.GetParameters();

            codecOptions.EnableCompilation = true;
            var compiledModel = CreateThirdOrderPinn(initial);
            compiledModel.Solve(epochs: 3, verbose: false);
            Assert.True(compiledModel.FusedSession.IsCommitted, "Third-order MultiScalePINN Solve must commit a fused plan.");
            Assert.False(compiledModel.FusedSession.IsDisabled);
            compiledParams = compiledModel.GetParameters();
        }
        finally
        {
            codecOptions.EnableCompilation = saved;
        }

        double maxParamDiff = eagerParams.ToArray().Zip(compiledParams.ToArray(), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(maxParamDiff < 1e-11, $"Third-order compiled Solve diverged from eager by {maxParamDiff:E3}.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MultiScalePinn_PureDerivativePde_PreservesAbsentOutputBiasStateAndAges(bool useDecoupledWeightDecay)
    {
        static (MultiScalePINN<float> Model, AiDotNet.Optimizers.GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> Optimizer, int[] BiasSlotIndices)
            CreatePureDerivativePinn(Vector<float>? init, bool adamW)
        {
            var arch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 2,
                outputSize: 1)
            {
                RandomSeed = 4096
            };
            AiDotNet.Optimizers.GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> opt = adamW
                ? new AiDotNet.Optimizers.AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.02,
                        WeightDecay = 0.05,
                        UseAdaptiveLearningRate = false,
                        EnableGradientClipping = false,
                    })
                : new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.02,
                        UseAdaptiveLearningRate = false,
                        EnableGradientClipping = false,
                    });
            var model = new MultiScalePINN<float>(
                arch,
                new PureDerivativePoissonMultiScalePde(),
                Array.Empty<IBoundaryCondition<float>>(),
                trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                optimizer: opt,
                options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 });

            var outDense0 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)model.ScaleNetworks[0].Layers[model.ScaleNetworks[0].Layers.Count - 1];
            var outDense1 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)model.ScaleNetworks[1].Layers[model.ScaleNetworks[1].Layers.Count - 1];
            if (init is not null)
            {
                model.SetParameters(init);
            }
            else
            {
                outDense0.GetBiases()[0] = 1.75f;
                outDense1.GetBiases()[0] = -2.25f;
            }

            var allTensors = AiDotNet.Training.TapeTrainingStep<float>.CollectTrainableLayers(model.Layers)
                .SelectMany(l => l.GetTrainableParameters())
                .ToArray();
            var bias0 = outDense0.GetBiases();
            var bias1 = outDense1.GetBiases();
            int idx0 = Array.FindIndex(allTensors, t => ReferenceEquals(t, bias0));
            int idx1 = Array.FindIndex(allTensors, t => ReferenceEquals(t, bias1));
            Assert.True(idx0 >= 0 && idx1 >= 0 && idx0 != idx1);
            return (model, opt, new[] { idx0, idx1 });
        }

        var (seedModel, _, _) = CreatePureDerivativePinn(null, useDecoupledWeightDecay);
        var initial = seedModel.GetParameters();

        var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = codecOptions.EnableCompilation;
        try
        {
            // 1. Cold-start pure-derivative Solve: output biases are structurally absent from dR_s/db_out and dC/db_out,
            // so neither Adam nor AdamW (WeightDecay = 0.05) may enroll, decay, or update them.
            codecOptions.EnableCompilation = false;
            var (eagerModel, _, _) = CreatePureDerivativePinn(initial, useDecoupledWeightDecay);
            eagerModel.Solve(epochs: 3, verbose: false);
            var eagerAfter = eagerModel.GetParameters().ToArray();

            codecOptions.EnableCompilation = true;
            var (compiledModel, compiledOpt, biasSlots) = CreatePureDerivativePinn(initial, useDecoupledWeightDecay);
            compiledModel.Solve(epochs: 3, verbose: false);
            Assert.True(compiledModel.FusedSession.IsCommitted, "Pure-derivative Solve must commit a fused plan.");
            Assert.False(compiledModel.FusedSession.IsDisabled);
            var compiledAfter = compiledModel.GetParameters().ToArray();

            var compiledDense0 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)compiledModel.ScaleNetworks[0].Layers[compiledModel.ScaleNetworks[0].Layers.Count - 1];
            var compiledDense1 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)compiledModel.ScaleNetworks[1].Layers[compiledModel.ScaleNetworks[1].Layers.Count - 1];
            var eagerDense0 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)eagerModel.ScaleNetworks[0].Layers[eagerModel.ScaleNetworks[0].Layers.Count - 1];
            var eagerDense1 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)eagerModel.ScaleNetworks[1].Layers[eagerModel.ScaleNetworks[1].Layers.Count - 1];

            Assert.Equal(1.75f, eagerDense0.GetBiases()[0]);
            Assert.Equal(-2.25f, eagerDense1.GetBiases()[0]);
            Assert.Equal(1.75f, compiledDense0.GetBiases()[0]);
            Assert.Equal(-2.25f, compiledDense1.GetBiases()[0]);

            double activeMovement = initial.ToArray().Zip(compiledAfter, (a, b) => Math.Abs((double)a - b)).Max();
            double eagerVsCompiled = eagerAfter.Zip(compiledAfter, (a, b) => Math.Abs((double)a - b)).Max();
            Assert.True(activeMovement > 1e-3, $"Derivative-active weights did not move ({activeMovement:E3}).");
            Assert.True(eagerVsCompiled < 1e-4, $"Compiled pure-derivative Solve diverged from eager by {eagerVsCompiled:E3}.");

            var coldCheckpoint = ReadExportedFusedCheckpoint(compiledOpt);
            for (int slot = 0; slot < coldCheckpoint.Parameters.Length; slot++)
            {
                var paramState = coldCheckpoint.Parameters[slot];
                if (biasSlots.Contains(slot))
                {
                    Assert.Equal(0L, paramState.AcceptedUpdates);
                    if (paramState.MFloat is { Length: > 0 })
                    {
                        Assert.All(paramState.MFloat, v => Assert.Equal(0f, v));
                    }

                    if (paramState.VFloat is { Length: > 0 })
                    {
                        Assert.All(paramState.VFloat, v => Assert.Equal(0f, v));
                    }
                }
                else
                {
                    Assert.Equal(3L, paramState.AcceptedUpdates);
                }
            }

            // 2. Supervised Train (2 steps, enrolling output biases at age 2 with nonzero moments) -> pure-derivative
            // Solve (3 epochs, leaving output biases untouched at age 2 and preserving exact moments) -> Supervised Train (1 step, advancing biases to age 3).
            var x1 = new Tensor<float>(new[] { 0.1f, 0.2f, 0.4f, 0.3f, 0.7f, 0.6f, 0.9f, 0.8f }, new[] { 4, 2 });
            var y1 = new Tensor<float>(new[] { 0.6f, -0.4f, 0.9f, 0.3f }, new[] { 4, 1 });
            var x2 = new Tensor<float>(new[] { 0.2f, 0.5f, 0.6f, 0.1f, 0.8f, 0.4f, 0.3f, 0.9f }, new[] { 4, 2 });
            var y2 = new Tensor<float>(new[] { -0.5f, 0.7f, 0.2f, -0.3f }, new[] { 4, 1 });

            codecOptions.EnableCompilation = false;
            var (eagerMixed, _, _) = CreatePureDerivativePinn(initial, useDecoupledWeightDecay);
            eagerMixed.Train(x1, y1);
            eagerMixed.Train(x1, y1);
            var eagerOut0 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)eagerMixed.ScaleNetworks[0].Layers[eagerMixed.ScaleNetworks[0].Layers.Count - 1];
            var eagerOut1 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)eagerMixed.ScaleNetworks[1].Layers[eagerMixed.ScaleNetworks[1].Layers.Count - 1];
            float eagerBias0BeforeSolve = eagerOut0.GetBiases()[0];
            float eagerBias1BeforeSolve = eagerOut1.GetBiases()[0];
            eagerMixed.Solve(epochs: 3, verbose: false);
            Assert.Equal(eagerBias0BeforeSolve, eagerOut0.GetBiases()[0]);
            Assert.Equal(eagerBias1BeforeSolve, eagerOut1.GetBiases()[0]);

            codecOptions.EnableCompilation = true;
            var (compiledMixed, mixedOpt, mixedBiasSlots) = CreatePureDerivativePinn(initial, useDecoupledWeightDecay);
            compiledMixed.Train(x1, y1);
            compiledMixed.Train(x1, y1);
            var mixedOut0 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)compiledMixed.ScaleNetworks[0].Layers[compiledMixed.ScaleNetworks[0].Layers.Count - 1];
            var mixedOut1 = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)compiledMixed.ScaleNetworks[1].Layers[compiledMixed.ScaleNetworks[1].Layers.Count - 1];
            float bias0AfterTrain1 = mixedOut0.GetBiases()[0];
            float bias1AfterTrain1 = mixedOut1.GetBiases()[0];
            Assert.NotEqual(1.75f, bias0AfterTrain1);
            Assert.NotEqual(-2.25f, bias1AfterTrain1);

            var preSolveCheckpoint = ReadExportedFusedCheckpoint(mixedOpt);
            foreach (int bSlot in mixedBiasSlots)
            {
                Assert.Equal(2L, preSolveCheckpoint.Parameters[bSlot].AcceptedUpdates);
            }

            float m0Saved = (preSolveCheckpoint.Parameters[mixedBiasSlots[0]].MFloat ?? throw new InvalidOperationException())[0];
            float v0Saved = (preSolveCheckpoint.Parameters[mixedBiasSlots[0]].VFloat ?? throw new InvalidOperationException())[0];
            float m1Saved = (preSolveCheckpoint.Parameters[mixedBiasSlots[1]].MFloat ?? throw new InvalidOperationException())[0];
            float v1Saved = (preSolveCheckpoint.Parameters[mixedBiasSlots[1]].VFloat ?? throw new InvalidOperationException())[0];
            Assert.NotEqual(0f, m0Saved);
            Assert.NotEqual(0f, m1Saved);

            compiledMixed.Solve(epochs: 3, verbose: false);
            Assert.True(compiledMixed.FusedSession.IsCommitted);
            Assert.False(compiledMixed.FusedSession.IsDisabled);
            Assert.Equal(bias0AfterTrain1, mixedOut0.GetBiases()[0]);
            Assert.Equal(bias1AfterTrain1, mixedOut1.GetBiases()[0]);
            double postSolveParity = eagerMixed.GetParameters().ToArray()
                .Zip(compiledMixed.GetParameters().ToArray(), (a, b) => Math.Abs((double)a - b))
                .Max();
            Assert.True(postSolveParity < 1e-4, $"Train->pure-derivative Solve diverged from eager by {postSolveParity:E3}.");

            var postSolveCheckpoint = ReadExportedFusedCheckpoint(mixedOpt);
            for (int slot = 0; slot < postSolveCheckpoint.Parameters.Length; slot++)
            {
                if (mixedBiasSlots.Contains(slot))
                {
                    Assert.Equal(2L, postSolveCheckpoint.Parameters[slot].AcceptedUpdates);
                }
                else
                {
                    Assert.Equal(5L, postSolveCheckpoint.Parameters[slot].AcceptedUpdates);
                }
            }

            Assert.Equal(m0Saved, (postSolveCheckpoint.Parameters[mixedBiasSlots[0]].MFloat ?? throw new InvalidOperationException())[0]);
            Assert.Equal(v0Saved, (postSolveCheckpoint.Parameters[mixedBiasSlots[0]].VFloat ?? throw new InvalidOperationException())[0]);
            Assert.Equal(m1Saved, (postSolveCheckpoint.Parameters[mixedBiasSlots[1]].MFloat ?? throw new InvalidOperationException())[0]);
            Assert.Equal(v1Saved, (postSolveCheckpoint.Parameters[mixedBiasSlots[1]].VFloat ?? throw new InvalidOperationException())[0]);

            // Resuming supervised Train advances output-bias ages from 2 -> 3 while active parameters advance from 5 -> 6,
            // and applies bias correction at age 3 (not global step 6) to the live output-bias parameters.
            compiledMixed.Train(x2, y2);
            var resumedCheckpoint = ReadExportedFusedCheckpoint(mixedOpt);
            for (int slot = 0; slot < resumedCheckpoint.Parameters.Length; slot++)
            {
                if (mixedBiasSlots.Contains(slot))
                {
                    Assert.Equal(3L, resumedCheckpoint.Parameters[slot].AcceptedUpdates);
                }
                else
                {
                    Assert.Equal(6L, resumedCheckpoint.Parameters[slot].AcceptedUpdates);
                }
            }

            float m0Resumed = (resumedCheckpoint.Parameters[mixedBiasSlots[0]].MFloat ?? throw new InvalidOperationException())[0];
            float v0Resumed = (resumedCheckpoint.Parameters[mixedBiasSlots[0]].VFloat ?? throw new InvalidOperationException())[0];
            double decayedBias0 = useDecoupledWeightDecay ? bias0AfterTrain1 * (1.0 - 0.02 * 0.05) : bias0AfterTrain1;
            double expectedAge3Bias0 = decayedBias0 - 0.02 * (m0Resumed / (1.0 - Math.Pow(0.9, 3))) / (Math.Sqrt(v0Resumed / (1.0 - Math.Pow(0.999, 3))) + 1e-8);
            double wrongAge6Bias0 = decayedBias0 - 0.02 * (m0Resumed / (1.0 - Math.Pow(0.9, 6))) / (Math.Sqrt(v0Resumed / (1.0 - Math.Pow(0.999, 6))) + 1e-8);
            double actualBias0 = mixedOut0.GetBiases()[0];
            Assert.True(Math.Abs(actualBias0 - expectedAge3Bias0) < 1e-6, $"Resumed bias0 ({actualBias0:G9}) did not match age-3 Adam formula ({expectedAge3Bias0:G9}).");
            Assert.True(Math.Abs(actualBias0 - wrongAge6Bias0) > 1e-4, $"Age-3 vs age-6 Adam formula was not distinguishable ({Math.Abs(actualBias0 - wrongAge6Bias0):E3}).");
        }
        finally
        {
            codecOptions.EnableCompilation = saved;
        }
    }

    [Fact]
    public async System.Threading.Tasks.Task MultiScalePinn_ConcurrentAndReentrantSolve_ThrowsAndPreservesState()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1)
        {
            RandomSeed = 777
        };
        var gatePde = new ReentrantAndBarrierMultiScalePde();
        var pinn = new MultiScalePINN<float>(
            arch,
            gatePde,
            Array.Empty<IBoundaryCondition<float>>(),
            trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
            options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 6 });

        var x = new Tensor<float>(new[] { 0.2f, 0.3f, 0.7f, 0.8f }, new[] { 2, 2 });
        var y = new Tensor<float>(new[] { 0.4f, -0.1f }, new[] { 2, 1 });
        var snapshot = pinn.GetParameters().ToArray();

        // 1. Reentrant Solve inside TrainWithCustomLoss callback throws the shared sentinel refusal before mutating weights.
        var fromCustomLossEx = Assert.Throws<InvalidOperationException>(() =>
            pinn.TrainWithCustomLoss(
                x,
                pred =>
                {
                    pinn.Solve(epochs: 1, learningRate: 0.05, verbose: false);
                    return AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(pred, pred), new[] { 0, 1 }, keepDims: false);
                }));
        Assert.Contains("Concurrent TrainWithTape", fromCustomLossEx.Message, StringComparison.Ordinal);
        Assert.Equal(snapshot, pinn.GetParameters().ToArray());
        Assert.False(pinn.IsTrainingMode);

        // 2. Reentrant Solve inside PDE callback throws AcquireTrainSentinel refusal before mutating weights,
        // and releases the sentinel on unwind.
        gatePde.OnResidual = () => pinn.Solve(epochs: 1, learningRate: 0.05, verbose: false);
        try
        {
            var reentrantSolveEx = Assert.Throws<InvalidOperationException>(() => pinn.Solve(epochs: 1, verbose: false));
            Assert.Contains("Concurrent TrainWithTape", reentrantSolveEx.Message, StringComparison.Ordinal);
        }
        finally
        {
            gatePde.OnResidual = null;
        }

        Assert.Equal(snapshot, pinn.GetParameters().ToArray());
        Assert.False(pinn.IsTrainingMode);

        // 3. Reentrant Train inside PDE callback during Solve throws the same shared sentinel refusal.
        gatePde.OnResidual = () => pinn.Train(x, y);
        try
        {
            var reentrantTrainEx = Assert.Throws<InvalidOperationException>(() => pinn.Solve(epochs: 1, verbose: false));
            Assert.Contains("Concurrent TrainWithTape", reentrantTrainEx.Message, StringComparison.Ordinal);
        }
        finally
        {
            gatePde.OnResidual = null;
        }

        Assert.Equal(snapshot, pinn.GetParameters().ToArray());
        Assert.False(pinn.IsTrainingMode);

        // 4. Genuine two-thread concurrency on a live fused session: while Thread A is inside Solve, concurrent Solve and
        // Train on another thread both refuse without clobbering Thread A's training mode, optimizer, or parameter update.
        pinn.FusedSession.Reset(stickyDisable: false);
        using var enteredStep = new System.Threading.ManualResetEventSlim(false);
        using var releaseStep = new System.Threading.ManualResetEventSlim(false);
        int barrierHits = 0;
        gatePde.OnResidual = () =>
        {
            if (System.Threading.Interlocked.Increment(ref barrierHits) == 1)
            {
                enteredStep.Set();
                if (!releaseStep.Wait(TimeSpan.FromSeconds(15)))
                {
                    throw new TimeoutException("Timed out waiting for concurrent Solve refusal check.");
                }
            }
        };

        var worker = System.Threading.Tasks.Task.Run(() => pinn.Solve(epochs: 2, learningRate: 0.002, verbose: false));
        try
        {
            Assert.True(enteredStep.Wait(TimeSpan.FromSeconds(15)), "Worker did not enter Solve in time.");
            var concurrentSolveEx = Assert.Throws<InvalidOperationException>(() => pinn.Solve(epochs: 1, learningRate: 0.25, verbose: false));
            Assert.Contains("Concurrent TrainWithTape", concurrentSolveEx.Message, StringComparison.Ordinal);
            Assert.True(pinn.IsTrainingMode, "Rejected concurrent Solve must not flip IsTrainingMode off while Thread A is still training.");
            var concurrentTrainEx = Assert.Throws<InvalidOperationException>(() => pinn.Train(x, y));
            Assert.Contains("Concurrent TrainWithTape", concurrentTrainEx.Message, StringComparison.Ordinal);
        }
        finally
        {
            releaseStep.Set();
        }

        await worker;
        gatePde.OnResidual = null;
        Assert.False(pinn.IsTrainingMode);
        Assert.True(pinn.FusedSession.IsCommitted);
        Assert.True(snapshot.Zip(pinn.GetParameters().ToArray(), (a, b) => Math.Abs((double)a - b)).Max() > 1e-5);
    }

    [Fact]
    public void MultiScalePinn_SolveHonorsWeightDecayExclusions_AndMatchesEager()
    {
        static (Vector<float> Params, bool Committed) RunL2Solve(bool compiled, bool excludeCoarseScale, Vector<float>? init)
        {
            var codecOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
            bool saved = codecOptions.EnableCompilation;
            try
            {
                codecOptions.EnableCompilation = compiled;
                var arch = new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 2,
                    outputSize: 1)
                {
                    RandomSeed = 8080
                };
                var l2 = new AiDotNet.Regularization.L2Regularization<float, Tensor<float>, Tensor<float>>(
                    new AiDotNet.Models.RegularizationOptions
                    {
                        Type = AiDotNet.Enums.RegularizationType.L2,
                        Strength = 0.25,
                        L1Ratio = 0.0,
                    });
                var sgd = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AiDotNet.Models.Options.StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 0.05,
                        UseAdaptiveLearningRate = false,
                        EnableGradientClipping = false,
                        Regularization = l2,
                    });
                var model = new CoarseDecayExcludedMultiScalePinn(
                    arch,
                    new TwoScalePoissonProblem<float>(),
                    sgd,
                    excludeCoarseScale);
                if (init is not null)
                {
                    model.SetParameters(init);
                }

                if (init is not null)
                {
                    model.Solve(epochs: 2, verbose: false);
                }

                return (model.GetParameters(), model.FusedSession.IsCommitted);
            }
            finally
            {
                codecOptions.EnableCompilation = saved;
            }
        }

        var init = RunL2Solve(compiled: false, excludeCoarseScale: true, init: null).Params;
        var (eagerExcluded, _) = RunL2Solve(compiled: false, excludeCoarseScale: true, init);
        var (compiledExcluded, excludedCommitted) = RunL2Solve(compiled: true, excludeCoarseScale: true, init);
        var (compiledAllDecayed, allDecayedCommitted) = RunL2Solve(compiled: true, excludeCoarseScale: false, init);

        Assert.False(excludedCommitted, "L2 with ExcludesFromWeightDecay layers has no uniform fused form and must run on the eager tape.");
        Assert.True(allDecayedCommitted, "Uniform L2 without excluded layers must commit a fused plan.");

        double eagerParity = eagerExcluded.ToArray().Zip(compiledExcluded.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();
        double exclusionEffect = compiledExcluded.ToArray().Zip(compiledAllDecayed.ToArray(), (a, b) => Math.Abs((double)a - b)).Max();
        Assert.True(eagerParity < 1e-6, $"Excluded-L2 Solve differed from eager by {eagerParity:E3}.");
        Assert.True(exclusionEffect > 1e-3, $"ExcludesFromWeightDecay had no observable effect on Solve ({exclusionEffect:E3}).");
    }

    private static AiDotNet.Tensors.Engines.Compilation.FusedOptimizerCheckpoint ReadExportedFusedCheckpoint(
        AiDotNet.Optimizers.GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> optimizer)
    {
        AiDotNet.Optimizers.GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> probe = optimizer is AiDotNet.Optimizers.AdamWOptimizer<float, Tensor<float>, Tensor<float>>
            ? new AiDotNet.Optimizers.AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null)
            : new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(null);
        probe.Deserialize(optimizer.Serialize());
        byte[] payload = probe.PeekPendingFusedOptimizerState()
            ?? throw new InvalidOperationException("Serialized optimizer did not contain a pending fused optimizer plan state.");
        using var payloadReader = new System.IO.BinaryReader(new System.IO.MemoryStream(payload), System.Text.Encoding.UTF8);
        byte[] magic = payloadReader.ReadBytes(4);
        Assert.Equal("AOPT", System.Text.Encoding.ASCII.GetString(magic));
        int formatVersion = payloadReader.ReadInt32();
        Assert.True(formatVersion > 0, $"Unexpected ExportOptimizerState format version {formatVersion}.");
        return AiDotNet.Tensors.Engines.Compilation.Serialization.FusedOptimizerCheckpointSerializer.Read(payloadReader)
            ?? throw new InvalidOperationException("Deserialized FusedOptimizerCheckpoint was null.");
    }

    private sealed class CoupledTwoScalePoissonPde : IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _poisson = new PoissonEquation<float>();

        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public int InputDimension => _poisson.InputDimension;
        public int OutputDimension => _poisson.OutputDimension;
        public string Name => "Coupled two-scale Poisson";
        public float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeTapeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class PureDerivativePoissonMultiScalePde : PDESpecificationBase<float>, IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _coarse = new PoissonEquation<float>(
            points => Tensor<float>.CreateDefault(new[] { points.Shape[0] }, 2.0f));
        private readonly PoissonEquation<float> _fine = new PoissonEquation<float>(
            points => Tensor<float>.CreateDefault(new[] { points.Shape[0] }, -1.5f));

        public override int DerivativeOrder => 2;
        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public override int InputDimension => 2;
        public override int OutputDimension => 1;
        public override string Name => "Pure-derivative two-scale Poisson";
        public override float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _coarse.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
            => scaleIndex == 0
                ? _coarse.ComputeTapeResidual(inputs, outputs, derivatives)
                : _fine.ComputeTapeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
        {
            var cFirst = coarseDerivatives.FirstDerivatives ?? throw new InvalidOperationException("Missing coarse first derivatives.");
            var fFirst = fineDerivatives.FirstDerivatives ?? throw new InvalidOperationException("Missing fine first derivatives.");
            return AiDotNetEngine.Current.TensorSubtract(cFirst, fFirst);
        }
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class ReentrantAndBarrierMultiScalePde : IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _poisson = new PoissonEquation<float>();
        public Action? OnResidual { get; set; }
        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public int InputDimension => 2;
        public int OutputDimension => 1;
        public string Name => "Reentrant/barrier multi-scale Poisson";
        public float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
        {
            OnResidual?.Invoke();
            return _poisson.ComputeTapeResidual(inputs, outputs, derivatives);
        }
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class CoarseDecayExcludedMultiScalePinn : MultiScalePINN<float>
    {
        private readonly bool _excludeCoarse;

        public CoarseDecayExcludedMultiScalePinn(
            NeuralNetworkArchitecture<float> architecture,
            IMultiScalePDE<float> pde,
            IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer,
            bool excludeCoarse)
            : base(
                architecture,
                pde,
                Array.Empty<IBoundaryCondition<float>>(),
                trainingOptions: new MultiScaleTrainingOptions<float> { UseAdaptiveScaleWeighting = false },
                optimizer: optimizer,
                options: new MultiScalePINNOptions { NumCollocationPointsPerScale = 8 })
        {
            _excludeCoarse = excludeCoarse;
        }

        protected override bool ExcludesFromWeightDecay(ILayer<float> layer)
            => _excludeCoarse && ScaleNetworks.Count > 0 && ScaleNetworks[0].Layers.Contains(layer);
    }

    private sealed class ThirdOrderKdVMultiScalePde : PDESpecificationBase<double>, IMultiScalePDE<double>
    {
        private readonly KortewegDeVriesEquation<double> _coarse = new KortewegDeVriesEquation<double>(6.0, 1.0);
        private readonly KortewegDeVriesEquation<double> _fine = new KortewegDeVriesEquation<double>(6.0, 0.1);

        public override int DerivativeOrder => 3;
        public int NumberOfScales => 2;
        public double[] ScaleCharacteristicLengths => new[] { 1.0, 0.1 };
        public override int InputDimension => 2;
        public override int OutputDimension => 1;
        public override string Name => "Two-scale Korteweg-de Vries";
        public override double ComputeResidual(Vector<double> inputs, Vector<double> outputs, PDEDerivatives<double> derivatives)
            => _coarse.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<double> ComputeScaleResidual(int scaleIndex, Tensor<double> inputs, Tensor<double> outputs, PDEDerivatives<double> derivatives)
            => scaleIndex == 0
                ? _coarse.ComputeTapeResidual(inputs, outputs, derivatives)
                : _fine.ComputeTapeResidual(inputs, outputs, derivatives);
        public Tensor<double> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<double> inputs,
            Tensor<double> coarseOutputs, Tensor<double> fineOutputs,
            PDEDerivatives<double> coarseDerivatives, PDEDerivatives<double> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public double GetScaleLossWeight(int scaleIndex) => 1.0;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class HostReadToggleMultiScalePde : IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _poisson = new PoissonEquation<float>();
        public bool ReadOutputOnHost { get; set; }
        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public int InputDimension => 2;
        public int OutputDimension => 1;
        public string Name => "Host-read toggle multi-scale Poisson";
        public float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
        {
            var residual = _poisson.ComputeTapeResidual(inputs, outputs, derivatives);
            if (!ReadOutputOnHost)
            {
                return residual;
            }

            float first = outputs[0, 0];
            return AiDotNetEngine.Current.TensorMultiplyScalar(residual, first >= 0f ? 1.5f : 0.5f);
        }
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class LargeForcingPoissonPde : IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _poisson = new PoissonEquation<float>(
            points => Tensor<float>.CreateDefault(new[] { points.Shape[0] }, 50.0f));
        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public int InputDimension => 2;
        public int OutputDimension => 1;
        public string Name => "Large-forcing two-scale Poisson";
        public float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeTapeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class StochasticForcingMultiScalePde : IMultiScalePDE<float>
    {
        private readonly PoissonEquation<float> _poisson = new PoissonEquation<float>();
        private readonly Random _random;
        private readonly bool _freezeNoise;
        private Tensor<float>? _frozenDraw;

        public StochasticForcingMultiScalePde(Random random, bool freezeNoise)
        {
            _random = random;
            _freezeNoise = freezeNoise;
        }

        public int NumberOfScales => 2;
        public float[] ScaleCharacteristicLengths => new[] { 1.0f, 0.1f };
        public int InputDimension => 2;
        public int OutputDimension => 1;
        public string Name => "Stochastic-forcing two-scale Poisson";
        public float ComputeResidual(Vector<float> inputs, Vector<float> outputs, PDEDerivatives<float> derivatives)
            => _poisson.ComputeResidual(inputs, outputs, derivatives);
        public Tensor<float> ComputeScaleResidual(int scaleIndex, Tensor<float> inputs, Tensor<float> outputs, PDEDerivatives<float> derivatives)
        {
            var baseResidual = _poisson.ComputeTapeResidual(inputs, outputs, derivatives);
            var u = AiDotNetEngine.Current.TensorSelect(outputs, 1, 0);
            Tensor<float> draw;
            if (_freezeNoise)
            {
                _frozenDraw ??= AiDotNet.Training.CompiledStepRandom<float>.StandardNormal(new[] { inputs.Shape[0] }, _random);
                draw = _frozenDraw;
            }
            else
            {
                draw = AiDotNet.Training.CompiledStepRandom<float>.StandardNormal(new[] { inputs.Shape[0] }, _random);
            }

            return AiDotNetEngine.Current.TensorAdd(baseResidual, AiDotNetEngine.Current.TensorMultiply(u, draw));
        }
        public Tensor<float> ComputeScaleCoupling(
            int coarseIndex, int fineIndex, Tensor<float> inputs,
            Tensor<float> coarseOutputs, Tensor<float> fineOutputs,
            PDEDerivatives<float> coarseDerivatives, PDEDerivatives<float> fineDerivatives)
            => AiDotNetEngine.Current.TensorSubtract(
                AiDotNetEngine.Current.TensorSelect(coarseOutputs, 1, 0),
                AiDotNetEngine.Current.TensorSelect(fineOutputs, 1, 0));
        public float GetScaleLossWeight(int scaleIndex) => 1.0f;
        public int GetScaleOutputDimension(int scaleIndex) => 1;
    }

    private sealed class DiffusivityIdentification : IInverseProblem<double>
    {
        public DiffusivityIdentification(IReadOnlyList<(double[] location, double[] value)> observations)
        {
            Observations = observations;
        }

        public string[] ParameterNames => new[] { "diffusivity" };
        public int NumberOfParameters => 1;
        public double[] InitialParameterGuesses => new[] { 0.5 };
        public double[]? ParameterLowerBounds => new[] { 1e-3 };
        public double[]? ParameterUpperBounds => new[] { 10.0 };
        public IReadOnlyList<(double[] location, double[] value)> Observations { get; }
        public bool HasMeasurementNoiseLevel => false;
        public double MeasurementNoiseLevel => 0.0;
        public bool ValidateParameters(double[] parameters) => parameters.Length == 1 && parameters[0] > 0;
        public IPDESpecification<double> CreateParameterizedPDE(double[] parameters) => new HeatEquation<double>(parameters[0]);
    }
}
