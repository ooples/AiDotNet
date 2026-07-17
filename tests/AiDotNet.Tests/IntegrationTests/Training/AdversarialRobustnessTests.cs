using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureAdversarialAttack runs the attack against the trained model and surfaces an
/// empirical robustness report (accuracy-vs-budget curve + margin) on the result.
/// </summary>
public class AdversarialRobustnessTests
{
    /// <summary>A deterministic attack that shifts every feature by the current epsilon.</summary>
    private sealed class EpsilonShiftAttack : AdversarialAttackBase<double, Matrix<double>, Vector<double>>
    {
        public EpsilonShiftAttack(AdversarialAttackOptions<double> options) : base(options) { }

        public override Matrix<double> GenerateAdversarialExample(
            Matrix<double> input, Vector<double> trueLabel, IFullModel<double, Matrix<double>, Vector<double>> targetModel)
        {
            double eps = Options.Epsilon;
            var adv = new Matrix<double>(input.Rows, input.Columns);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Columns; j++) adv[i, j] = input[i, j] + eps;
            return adv;
        }

        public override Matrix<double> CalculatePerturbation(Matrix<double> original, Matrix<double> adversarial)
        {
            var p = new Matrix<double>(original.Rows, original.Columns);
            for (int i = 0; i < original.Rows; i++)
                for (int j = 0; j < original.Columns; j++) p[i, j] = adversarial[i, j] - original[i, j];
            return p;
        }
    }

    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 80, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.02);
            y[i] = 2.0 * x[i, 0] - x[i, 2];
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureAdversarialAttack_SurfacesRobustnessCurve()
    {
        var (x, y) = BuildData();
        var attack = new EpsilonShiftAttack(new AdversarialAttackOptions<double> { Epsilon = 0.5 });

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureAdversarialAttack(attack)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        var report = result.AdversarialRobustness;
        Assert.NotNull(report);
        Assert.True((report?.RobustnessCurve.Count ?? 0) >= 1, "no robustness curve produced");
        // Attacking should not improve accuracy; a shift attack should cost some accuracy.
        Assert.True(report.RobustAccuracy <= report.CleanAccuracy + 1e-9);
        // Higher budgets should be at least as damaging as lower ones (monotone non-increasing accuracy).
        for (int k = 1; k < report.RobustnessCurve.Count; k++)
            Assert.True(report.RobustnessCurve[k].RobustAccuracy <= report.RobustnessCurve[k - 1].RobustAccuracy + 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task NoAttack_LeavesRobustnessNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.AdversarialRobustness);
    }
}
