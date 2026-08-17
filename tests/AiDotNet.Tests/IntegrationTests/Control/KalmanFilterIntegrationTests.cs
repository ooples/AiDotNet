#nullable disable
using AiDotNet.Control;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for the recursive Kalman filter and the LQG controller.
/// </summary>
/// <remarks>
/// CRITICAL: The expected values come from sources independent of the filter's own arithmetic:
///   1. The exact Bayesian posterior. For a constant scalar state with no process noise the filter
///      reduces to a weighted mean whose value and variance are known in closed form, so those are
///      checked to the last digit rather than approximately.
///   2. The steady-state Riccati solution, computed by a completely different algorithm (the
///      doubling solver). The recursive gain must converge to it — two code paths that share no
///      lines agreeing on a number.
///   3. Simulation against a known ground truth with seeded noise, where the filter must beat the
///      raw measurements it is given.
/// If a test fails, FIX THE FILTER — do not relax the assertion.
/// </remarks>
public class KalmanFilterIntegrationTests
{
    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    #region Exact closed-form behaviour

    /// <summary>
    /// A constant scalar state with no process noise, measured repeatedly. This is the textbook
    /// Bayesian update, and it has an exact answer: starting from an estimate of 0 with variance
    /// equal to the measurement variance r, the estimate after k measurements is
    /// (Σy) / (k + 1) and its variance is r / (k + 1). The prior counts as exactly one extra
    /// pseudo-measurement of zero, which is what makes those denominators k + 1 rather than k.
    /// </summary>
    [Fact]
    public void Filter_ConstantStateNoProcessNoise_ReducesToTheExactPosterior()
    {
        var filter = new KalmanFilter<double>(
            transition: M(new[,] { { 1.0 } }),
            observation: M(new[,] { { 1.0 } }),
            processNoise: M(new[,] { { 0.0 } }),
            measurementNoise: M(new[,] { { 4.0 } }));

        filter.Initialize(V(0.0), M(new[,] { { 4.0 } }));

        double[] measurements = { 10.0, 12.0, 8.0, 14.0, 6.0 };
        double running = 0.0;

        for (int k = 0; k < measurements.Length; k++)
        {
            filter.Predict();
            filter.Update(V(measurements[k]));

            running += measurements[k];

            Assert.Equal(running / (k + 2), filter.State[0], 10);
            Assert.Equal(4.0 / (k + 2), filter.Covariance[0, 0], 10);
        }
    }

    /// <summary>
    /// Measurements with no noise at all must be believed completely: the estimate becomes the
    /// measurement and the remaining uncertainty is zero.
    /// </summary>
    [Fact]
    public void Filter_NoiselessMeasurement_AdoptsItExactly()
    {
        var filter = new KalmanFilter<double>(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 1.0 } }), M(new[,] { { 0.0 } }));

        filter.Initialize(V(-50.0), M(new[,] { { 1.0 } }));

        filter.Predict();
        filter.Update(V(7.0));

        Assert.Equal(7.0, filter.State[0], 10);
        Assert.Equal(0.0, filter.Covariance[0, 0], 10);
    }

    /// <summary>
    /// A measurement so noisy as to be worthless must be ignored: the estimate stays where the model
    /// put it. This is the opposite extreme from the previous test, and between them they pin the
    /// gain's whole range.
    /// </summary>
    [Fact]
    public void Filter_WorthlessMeasurement_IsAlmostIgnored()
    {
        var filter = new KalmanFilter<double>(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 0.0 } }), M(new[,] { { 1e12 } }));

        filter.Initialize(V(3.0), M(new[,] { { 1.0 } }));

        filter.Predict();
        filter.Update(V(1000.0));

        Assert.Equal(3.0, filter.State[0], 6);
    }

    /// <summary>
    /// With no measurement to fold in, uncertainty must grow by exactly the process noise each step:
    /// P ← A P Aᵀ + Q, which for A = 1 is P + Q.
    /// </summary>
    [Fact]
    public void Filter_PredictWithoutUpdate_AccumulatesProcessNoise()
    {
        var filter = new KalmanFilter<double>(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 0.5 } }), M(new[,] { { 1.0 } }));

        filter.Initialize(V(0.0), M(new[,] { { 2.0 } }));

        for (int step = 1; step <= 4; step++)
        {
            filter.Predict();
            Assert.Equal(2.0 + 0.5 * step, filter.Covariance[0, 0], 10);
        }
    }

    /// <summary>
    /// A known control input must move the estimate by exactly its modelled effect. An estimator
    /// that ignored its own commands would attribute their consequences to noise.
    /// </summary>
    [Fact]
    public void Filter_KnownInput_MovesTheEstimateByItsModelledEffect()
    {
        var filter = new KalmanFilter<double>(
            transition: M(new[,] { { 1.0 } }),
            observation: M(new[,] { { 1.0 } }),
            processNoise: M(new[,] { { 0.0 } }),
            measurementNoise: M(new[,] { { 1.0 } }),
            control: M(new[,] { { 2.0 } }));

        filter.Initialize(V(5.0), M(new[,] { { 1.0 } }));

        filter.Predict(V(3.0));

        Assert.Equal(5.0 + 2.0 * 3.0, filter.State[0], 10);
    }

    #endregion

    #region Agreement with the Riccati solution

    /// <summary>
    /// The recursive gain must converge to the steady-state gain computed from the Riccati equation
    /// by the doubling solver — two implementations sharing no code agreeing on a number.
    ///
    /// The relationship is not equality but a factor of A: the recursion's gain is the filter form
    /// P⁻Cᵀ(CP⁻Cᵀ + R)⁻¹, applied to the already-predicted covariance, while the steady-state value
    /// is the predictor form A·P·Cᵀ(CPCᵀ + R)⁻¹ that maps a measurement straight to the next state.
    /// </summary>
    [Fact]
    public void Filter_RecursiveGain_ConvergesToTheSteadyStateRiccatiGain()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 0.9 } });
        var c = M(new[,] { { 1.0, 0.0 } });
        var q = M(new[,] { { 0.05, 0.0 }, { 0.0, 0.05 } });
        var r = M(new[,] { { 2.0 } });

        var filter = new KalmanFilter<double>(a, c, q, r);
        filter.Initialize(V(0.0, 0.0), Matrix<double>.CreateIdentity(2));

        for (int step = 0; step < 500; step++)
        {
            filter.Predict();
            filter.Update(V(0.0));
        }

        var steadyState = KalmanFilter<double>.SteadyStateGain(a, c, q, r);

        // steadyState = A * convergedFilterGain
        double expectedFirst = a[0, 0] * filter.Gain[0, 0] + a[0, 1] * filter.Gain[1, 0];
        double expectedSecond = a[1, 0] * filter.Gain[0, 0] + a[1, 1] * filter.Gain[1, 0];

        Assert.Equal(expectedFirst, steadyState[0, 0], 8);
        Assert.Equal(expectedSecond, steadyState[1, 0], 8);
    }

    /// <summary>
    /// The scalar steady state has a closed form. With A = C = 1, process noise q and measurement
    /// noise r, the predicted covariance solves p = p − p²/(p + r) + q, so p² − qp − qr = 0 and
    /// p = (q + √(q² + 4qr)) / 2. The predictor gain is then p / (p + r).
    /// </summary>
    [Fact]
    public void Filter_ScalarSteadyStateGain_MatchesTheClosedForm()
    {
        const double Q = 1.0;
        const double R = 3.0;

        var gain = KalmanFilter<double>.SteadyStateGain(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { Q } }), M(new[,] { { R } }));

        double covariance = (Q + Math.Sqrt(Q * Q + 4 * Q * R)) / 2.0;

        Assert.Equal(covariance / (covariance + R), gain[0, 0], 9);
    }

    #endregion

    #region Verification by simulation

    /// <summary>
    /// The point of a filter: with a known ground truth and seeded noise, the estimate must be
    /// substantially closer to the truth than the raw measurements are. This tracks position and
    /// velocity while measuring position only, so the velocity estimate is inferred entirely — there
    /// is no sensor for it.
    /// </summary>
    [Fact]
    public void Filter_TrackingWithNoisyMeasurements_BeatsTheRawSensor()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var c = M(new[,] { { 1.0, 0.0 } });
        var q = M(new[,] { { 1e-6, 0.0 }, { 0.0, 1e-6 } });
        var r = M(new[,] { { 4.0 } });

        var filter = new KalmanFilter<double>(a, c, q, r);
        filter.Initialize(V(0.0, 0.0), M(new[,] { { 100.0, 0.0 }, { 0.0, 100.0 } }));

        var noise = new DeterministicGaussian(seed: 20260816);

        double truePosition = 0.0;
        const double TrueVelocity = 2.0;

        double filterSquaredError = 0.0;
        double sensorSquaredError = 0.0;
        const int Steps = 300;

        for (int step = 0; step < Steps; step++)
        {
            truePosition += TrueVelocity;

            double measurement = truePosition + 2.0 * noise.Next();

            filter.Predict();
            filter.Update(V(measurement));

            double filterError = filter.State[0] - truePosition;
            double sensorError = measurement - truePosition;

            filterSquaredError += filterError * filterError;
            sensorSquaredError += sensorError * sensorError;
        }

        Assert.True(
            filterSquaredError < sensorSquaredError / 5.0,
            $"The filter's squared error was {filterSquaredError} against the raw sensor's " +
            $"{sensorSquaredError}; filtering should cut it by far more than that.");

        // The velocity is never measured, only inferred, so recovering it is the stronger claim.
        Assert.Equal(TrueVelocity, filter.State[1], 1);
    }

    /// <summary>
    /// The covariance must stay symmetric and positive-semidefinite for the whole run. This is what
    /// Joseph's form exists to guarantee, and a filter that loses it diverges without warning.
    /// </summary>
    [Fact]
    public void Filter_CovarianceOverALongRun_StaysSymmetricAndPositiveSemidefinite()
    {
        var filter = new KalmanFilter<double>(
            M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } }),
            M(new[,] { { 1.0, 0.0 } }),
            M(new[,] { { 1e-8, 0.0 }, { 0.0, 1e-8 } }),
            M(new[,] { { 1e-4 } }));

        filter.Initialize(V(0.0, 0.0), M(new[,] { { 1e6, 0.0 }, { 0.0, 1e6 } }));

        var noise = new DeterministicGaussian(seed: 7);

        for (int step = 0; step < 1000; step++)
        {
            filter.Predict();
            filter.Update(V(step + 0.01 * noise.Next()));

            var p = filter.Covariance;

            Assert.Equal(p[0, 1], p[1, 0], 12);

            // Positive semidefinite for a 2x2 symmetric matrix: both diagonal entries non-negative
            // and the determinant non-negative.
            Assert.True(p[0, 0] >= -1e-12, $"Variance went negative at step {step}: {p[0, 0]}.");
            Assert.True(p[1, 1] >= -1e-12, $"Variance went negative at step {step}: {p[1, 1]}.");
            Assert.True(
                p[0, 0] * p[1, 1] - p[0, 1] * p[1, 0] >= -1e-12,
                $"The covariance went indefinite at step {step}.");
        }
    }

    /// <summary>
    /// The innovation sequence must have near-zero mean when the model is correct. A biased
    /// innovation means the model is wrong, so this is the check that the filter is consistent with
    /// the system it is filtering rather than merely producing numbers.
    /// </summary>
    [Fact]
    public void Filter_InnovationSequence_HasNearZeroMean()
    {
        var filter = new KalmanFilter<double>(
            M(new[,] { { 0.95 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 0.1 } }), M(new[,] { { 1.0 } }));

        filter.Initialize(V(0.0), M(new[,] { { 1.0 } }));

        var noise = new DeterministicGaussian(seed: 99);
        double trueState = 0.0;
        double innovationTotal = 0.0;
        const int Steps = 4000;

        for (int step = 0; step < Steps; step++)
        {
            trueState = 0.95 * trueState + Math.Sqrt(0.1) * noise.Next();

            filter.Predict();
            filter.Update(V(trueState + noise.Next()));

            innovationTotal += filter.Innovation[0];
        }

        double mean = innovationTotal / Steps;
        Assert.True(
            Math.Abs(mean) < 0.1,
            $"The mean innovation was {mean}, which is too far from zero for a correctly specified " +
            "filter.");
    }

    #endregion

    #region LQG

    /// <summary>
    /// The separation principle in action: an unstable system, measured through one noisy sensor
    /// that sees only position, must still be driven to rest. Neither half could do this alone — the
    /// regulator cannot see the state and the filter cannot act.
    /// </summary>
    [Fact]
    public void Lqg_UnstableSystemWithPartialNoisyMeasurements_IsStillRegulated()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var c = M(new[,] { { 1.0, 0.0 } });

        var controller = new LinearQuadraticGaussian<double>(
            stateMatrix: a,
            inputMatrix: b,
            observationMatrix: c,
            stateCost: Matrix<double>.CreateIdentity(2),
            inputCost: Matrix<double>.CreateIdentity(1),
            processNoise: M(new[,] { { 1e-6, 0.0 }, { 0.0, 1e-6 } }),
            measurementNoise: M(new[,] { { 1e-4 } }));

        controller.Initialize(V(0.0, 0.0), M(new[,] { { 10.0, 0.0 }, { 0.0, 10.0 } }));

        var noise = new DeterministicGaussian(seed: 4242);
        var trueState = V(8.0, -3.0);

        for (int step = 0; step < 400; step++)
        {
            var measurement = V(trueState[0] + 0.01 * noise.Next());
            var input = controller.Step(measurement);

            trueState = V(
                a[0, 0] * trueState[0] + a[0, 1] * trueState[1] + b[0, 0] * input[0],
                a[1, 0] * trueState[0] + a[1, 1] * trueState[1] + b[1, 0] * input[0]);
        }

        Assert.True(
            Math.Abs(trueState[0]) < 0.1 && Math.Abs(trueState[1]) < 0.1,
            $"LQG failed to regulate the system: ({trueState[0]}, {trueState[1]}).");
    }

    /// <summary>
    /// The regulator half of an LQG must be exactly the LQR designed from the same Q and R — the
    /// separation principle says the noise does not change the control law at all.
    /// </summary>
    [Fact]
    public void Lqg_ControlLaw_IsExactlyTheLqrGain()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);

        var controller = new LinearQuadraticGaussian<double>(
            a, b, M(new[,] { { 1.0, 0.0 } }), q, r,
            M(new[,] { { 0.3, 0.0 }, { 0.0, 0.3 } }), M(new[,] { { 5.0 } }));

        Assert.Equal(regulator.Gain[0, 0], controller.Regulator.Gain[0, 0], 12);
        Assert.Equal(regulator.Gain[0, 1], controller.Regulator.Gain[0, 1], 12);
    }

    #endregion

    #region Validation

    [Fact(Timeout = 120000)]
    public async Task Filter_NullRequiredMatrices_ThrowNamedArguments()
    {
        await Task.Yield();

        var identity = Matrix<double>.CreateIdentity(1);

        Assert.Equal("transition", Assert.Throws<ArgumentNullException>(
            () => new KalmanFilter<double>(null, identity, identity, identity)).ParamName);
        Assert.Equal("observation", Assert.Throws<ArgumentNullException>(
            () => new KalmanFilter<double>(identity, null, identity, identity)).ParamName);
        Assert.Equal("processNoise", Assert.Throws<ArgumentNullException>(
            () => new KalmanFilter<double>(identity, identity, null, identity)).ParamName);
        Assert.Equal("measurementNoise", Assert.Throws<ArgumentNullException>(
            () => new KalmanFilter<double>(identity, identity, identity, null)).ParamName);
    }

    [Fact(Timeout = 120000)]
    public async Task Filter_SingularInnovationCovariance_ThrowsNamedDiagnostic()
    {
        await Task.Yield();

        var filter = new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1),
            Matrix<double>.CreateIdentity(1), M(new[,] { { 0.0 } }));
        filter.Initialize(V(0.0), M(new[,] { { 0.0 } }));

        var exception = Assert.Throws<InvalidOperationException>(() => filter.Update(V(0.0)));

        Assert.Contains("innovation covariance", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task SteadyStateGain_ValidatesAllMatrixDimensions()
    {
        await Task.Yield();

        var exception = Assert.Throws<ArgumentException>(() =>
            KalmanFilter<double>.SteadyStateGain(
                Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0, 0.0 } }),
                Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(2)));

        Assert.Equal("measurementNoise", exception.ParamName);
    }

    [Fact]
    public void Filter_NonSquareTransition_Throws()
    {
        Assert.Throws<ArgumentException>(() => new KalmanFilter<double>(
            M(new[,] { { 1.0, 0.0 } }), M(new[,] { { 1.0 } }),
            Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)));
    }

    [Fact]
    public void Filter_ObservationWithWrongColumnCount_Throws()
    {
        Assert.Throws<ArgumentException>(() => new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1)));
    }

    [Fact]
    public void Filter_MeasurementNoiseWithWrongSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0, 0.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(2)));
    }

    [Fact]
    public void Filter_MeasurementOfWrongLength_Throws()
    {
        var filter = new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0, 0.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        filter.Initialize(V(0.0, 0.0), Matrix<double>.CreateIdentity(2));

        Assert.Throws<ArgumentException>(() => filter.Update(V(1.0, 2.0)));
    }

    [Fact]
    public void Filter_InputWithoutAControlMatrix_Throws()
    {
        var filter = new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1),
            Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1));

        Assert.Throws<InvalidOperationException>(() => filter.Predict(V(1.0)));
    }

    [Fact]
    public void Filter_InitialStateOfWrongLength_Throws()
    {
        var filter = new KalmanFilter<double>(
            Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0, 0.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        Assert.Throws<ArgumentException>(
            () => filter.Initialize(V(1.0), Matrix<double>.CreateIdentity(2)));
    }

    #endregion

    /// <summary>
    /// A seeded Gaussian source, so every run of these tests sees the same noise.
    /// </summary>
    /// <remarks>
    /// Uses the Box-Muller transform over a seeded uniform generator. Tests that depend on noise
    /// must be reproducible or a failure cannot be investigated — an unseeded generator turns a real
    /// defect into something that shows up once and then cannot be reproduced.
    /// </remarks>
    private sealed class DeterministicGaussian
    {
        private readonly Random _uniform;
        private double _spare;
        private bool _hasSpare;

        public DeterministicGaussian(int seed) => _uniform = new Random(seed);

        public double Next()
        {
            if (_hasSpare)
            {
                _hasSpare = false;
                return _spare;
            }

            double first = 1.0 - _uniform.NextDouble();
            double second = 1.0 - _uniform.NextDouble();

            double magnitude = Math.Sqrt(-2.0 * Math.Log(first));
            double angle = 2.0 * Math.PI * second;

            _spare = magnitude * Math.Sin(angle);
            _hasSpare = true;

            return magnitude * Math.Cos(angle);
        }
    }
}
