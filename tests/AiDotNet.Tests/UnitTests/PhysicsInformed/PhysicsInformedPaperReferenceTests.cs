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
