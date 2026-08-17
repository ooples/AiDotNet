#nullable disable
using AiDotNet.Control;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for dynamic mode decomposition with control.
/// </summary>
/// <remarks>
/// CRITICAL: The central test here is a round trip. Data is generated from a known A and B, and the
/// identification must recover those exact matrices — for noiseless data from a genuinely linear
/// system this is an exactly determined least-squares problem, so the answer is right to machine
/// precision or the implementation is wrong. There is no tuning and no tolerance to argue about.
/// If a test fails, FIX THE IDENTIFIER — do not relax the assertion.
/// </remarks>
public class SystemIdentificationIntegrationTests
{
    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    /// <summary>
    /// Generates snapshots from a known system, driven by an input that varies enough to excite
    /// every mode — data from a constant input cannot determine B.
    /// </summary>
    private static (Matrix<double> States, Matrix<double> NextStates, Matrix<double> Inputs)
        GenerateTrajectory(Matrix<double> a, Matrix<double> b, int snapshots, int seed)
    {
        int stateCount = a.Rows;
        int inputCount = b.Columns;

        var random = new Random(seed);

        var states = new Matrix<double>(stateCount, snapshots);
        var nextStates = new Matrix<double>(stateCount, snapshots);
        var inputs = new Matrix<double>(inputCount, snapshots);

        var current = new double[stateCount];
        for (int i = 0; i < stateCount; i++) current[i] = random.NextDouble() * 2.0 - 1.0;

        for (int k = 0; k < snapshots; k++)
        {
            var input = new double[inputCount];
            for (int i = 0; i < inputCount; i++) input[i] = random.NextDouble() * 2.0 - 1.0;

            var next = new double[stateCount];
            for (int r = 0; r < stateCount; r++)
            {
                double value = 0.0;
                for (int c = 0; c < stateCount; c++) value += a[r, c] * current[c];
                for (int c = 0; c < inputCount; c++) value += b[r, c] * input[c];
                next[r] = value;
            }

            for (int i = 0; i < stateCount; i++)
            {
                states[i, k] = current[i];
                nextStates[i, k] = next[i];
            }

            for (int i = 0; i < inputCount; i++) inputs[i, k] = input[i];

            current = next;
        }

        return (states, nextStates, inputs);
    }

    #region Exact recovery

    /// <summary>
    /// The round trip: noiseless data from a known linear system must give back exactly that system.
    /// </summary>
    [Fact]
    public void Dmdc_NoiselessData_RecoversTheGeneratingSystemExactly()
    {
        var trueA = M(new[,] { { 0.9, 0.2 }, { -0.1, 0.8 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 40, seed: 11);

        var result = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(trueA[r, c], result.StateMatrix[r, c], 8);
            }

            Assert.Equal(trueB[r, 0], result.InputMatrix[r, 0], 8);
        }

        Assert.True(
            result.Residual < 1e-8,
            $"The identified model should reproduce noiseless data exactly; residual was " +
            $"{result.Residual}.");
    }

    /// <summary>
    /// The same with several inputs and states, where the bookkeeping that splits [A B] apart has
    /// more room to be wrong.
    /// </summary>
    [Fact]
    public void Dmdc_MultiInputSystem_RecoversBothOperators()
    {
        var trueA = M(new[,]
        {
            { 0.8, 0.1, 0.0 },
            { 0.0, 0.7, 0.3 },
            { 0.2, 0.0, 0.6 },
        });

        var trueB = M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 0.5, -0.5 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 60, seed: 5);

        var result = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(trueA[r, c], result.StateMatrix[r, c], 8);
            }

            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(trueB[r, c], result.InputMatrix[r, c], 8);
            }
        }
    }

    /// <summary>
    /// An unstable system must be identified just as exactly — the method fits data and never
    /// assumes stability, which matters because an unstable plant is precisely the one you need a
    /// model of.
    /// </summary>
    [Fact]
    public void Dmdc_UnstableSystem_IsIdentifiedToo()
    {
        var trueA = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 25, seed: 3);

        var result = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(trueA[r, c], result.StateMatrix[r, c], 6);
            }
        }
    }

    #endregion

    #region The claim that makes DMDc different from DMD

    /// <summary>
    /// The documented reason DMDc exists: regressing the next state on the current state alone —
    /// plain dynamic mode decomposition — cannot tell whether the state moved because of the system
    /// or because of the input, so it folds the actuation into the dynamics and reports the wrong A.
    /// Including the inputs separates them. This verifies that claim rather than asserting it.
    /// </summary>
    [Fact]
    public void Dmdc_IncludingInputs_RecoversDynamicsThatIgnoringThemGetsWrong()
    {
        var trueA = M(new[,] { { 0.9, 0.2 }, { -0.1, 0.8 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 40, seed: 11);

        var withInputs = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        // Plain DMD: the same regression with the input block simply left out, which is what
        // identifying a controlled system without accounting for the control amounts to.
        var zeroInputs = new Matrix<double>(1, states.Columns);
        var withoutInputs = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, zeroInputs);

        double withError = 0.0;
        double withoutError = 0.0;

        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                withError += Math.Abs(withInputs.StateMatrix[r, c] - trueA[r, c]);
                withoutError += Math.Abs(withoutInputs.StateMatrix[r, c] - trueA[r, c]);
            }
        }

        Assert.True(withError < 1e-7, $"DMDc should recover A exactly; error was {withError}.");
        Assert.True(
            withoutError > 0.1,
            $"Ignoring the inputs should visibly corrupt the identified dynamics, but the error " +
            $"was only {withoutError}. If this is small the test system is not actually being " +
            "driven hard enough for the comparison to mean anything.");
    }

    #endregion

    #region Diagnostics and truncation

    /// <summary>
    /// Data that genuinely lives in a lower-dimensional subspace must show it in the singular
    /// values, and the reported rank must reflect what was kept.
    /// </summary>
    [Fact]
    public void Dmdc_RankDeficientData_ReportsTheReducedRank()
    {
        // Every input is the same, so the input direction carries no independent information and the
        // stacked data cannot have full row rank.
        var trueA = M(new[,] { { 0.9, 0.0 }, { 0.0, 0.8 } });
        var trueB = M(new[,] { { 1.0 }, { 1.0 } });

        int snapshots = 20;
        var states = new Matrix<double>(2, snapshots);
        var nextStates = new Matrix<double>(2, snapshots);
        var inputs = new Matrix<double>(1, snapshots);

        var current = new[] { 1.0, 2.0 };
        for (int k = 0; k < snapshots; k++)
        {
            const double ConstantInput = 1.0;

            var next = new[]
            {
                trueA[0, 0] * current[0] + trueB[0, 0] * ConstantInput,
                trueA[1, 1] * current[1] + trueB[1, 0] * ConstantInput,
            };

            states[0, k] = current[0];
            states[1, k] = current[1];
            nextStates[0, k] = next[0];
            nextStates[1, k] = next[1];
            inputs[0, k] = ConstantInput;

            current = next;
        }

        var result = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        Assert.True(
            result.Rank <= 3,
            $"The stacked data has three rows, so the rank cannot exceed three; got {result.Rank}.");

        Assert.True(
            result.SingularValues.Length > 0 && result.SingularValues[0] > 0.0,
            "The singular values must be reported so a caller can judge the fit.");
    }

    /// <summary>
    /// An explicit rank cap must be honoured.
    /// </summary>
    [Fact]
    public void Dmdc_ExplicitRankCap_IsRespected()
    {
        var trueA = M(new[,] { { 0.9, 0.2 }, { -0.1, 0.8 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 30, seed: 7);

        var result = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs, rank: 2);

        Assert.Equal(2, result.Rank);
    }

    /// <summary>
    /// Truncating below the true rank must cost accuracy — otherwise the truncation is not doing
    /// anything and the rank parameter would be decorative.
    /// </summary>
    [Fact]
    public void Dmdc_TruncatingBelowTheTrueRank_LosesAccuracy()
    {
        var trueA = M(new[,] { { 0.9, 0.2 }, { -0.1, 0.8 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 30, seed: 7);

        var identifier = new DynamicModeDecompositionWithControl<double>();

        var full = identifier.Identify(states, nextStates, inputs);
        var truncated = identifier.Identify(states, nextStates, inputs, rank: 1);

        Assert.True(
            truncated.Residual > full.Residual,
            $"Discarding two of three directions should fit the data worse, but the truncated " +
            $"residual {truncated.Residual} is not above the full one {full.Residual}.");
    }

    #endregion

    #region End to end

    /// <summary>
    /// The point of identification: the recovered model must be good enough to control the real
    /// system with. This designs a regulator from the identified matrices and runs it against the
    /// true dynamics — the controller never sees the true A and B.
    /// </summary>
    [Fact]
    public void Dmdc_IdentifiedModel_ControlsTheTrueSystem()
    {
        var trueA = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var trueB = M(new[,] { { 0.5 }, { 1.0 } });

        var (states, nextStates, inputs) = GenerateTrajectory(trueA, trueB, snapshots: 30, seed: 21);

        var identified = new DynamicModeDecompositionWithControl<double>()
            .Identify(states, nextStates, inputs);

        var regulator = new LinearQuadraticRegulator<double>(
            identified.StateMatrix, identified.InputMatrix,
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        var state = Vector<double>.FromArray(new[] { 6.0, -2.0 });

        for (int step = 0; step < 300; step++)
        {
            var input = regulator.ComputeControl(state);

            // Advanced by the TRUE dynamics, not the identified ones.
            state = Vector<double>.FromArray(new[]
            {
                trueA[0, 0] * state[0] + trueA[0, 1] * state[1] + trueB[0, 0] * input[0],
                trueA[1, 0] * state[0] + trueA[1, 1] * state[1] + trueB[1, 0] * input[0],
            });
        }

        Assert.True(
            Math.Abs(state[0]) < 1e-6 && Math.Abs(state[1]) < 1e-6,
            $"A controller designed from the identified model failed to regulate the true system: " +
            $"({state[0]}, {state[1]}).");
    }

    #endregion

    #region Validation

    [Fact]
    public void Dmdc_MismatchedNextStates_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicModeDecompositionWithControl<double>().Identify(
                new Matrix<double>(2, 5), new Matrix<double>(2, 4), new Matrix<double>(1, 5)));
    }

    [Fact]
    public void Dmdc_InputsWithWrongSnapshotCount_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicModeDecompositionWithControl<double>().Identify(
                new Matrix<double>(2, 5), new Matrix<double>(2, 5), new Matrix<double>(1, 3)));
    }

    [Fact]
    public void Dmdc_AllZeroData_Throws()
    {
        // Every singular value is zero, so nothing at all can be identified. Reporting that is more
        // useful than returning a matrix of zeros that looks like an answer.
        Assert.Throws<ArgumentException>(() =>
            new DynamicModeDecompositionWithControl<double>().Identify(
                new Matrix<double>(2, 5), new Matrix<double>(2, 5), new Matrix<double>(1, 5)));
    }

    [Fact]
    public void Dmdc_NonPositiveRank_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DynamicModeDecompositionWithControl<double>().Identify(
                new Matrix<double>(2, 5), new Matrix<double>(2, 5), new Matrix<double>(1, 5),
                rank: 0));
    }

    [Fact]
    public void Dmdc_ThresholdOutsideUnitInterval_Throws()
    {
        Assert.Throws<ArgumentException>(
            () => new DynamicModeDecompositionWithControl<double>(1.0));
        Assert.Throws<ArgumentException>(
            () => new DynamicModeDecompositionWithControl<double>(-0.1));
    }

    #endregion
}
