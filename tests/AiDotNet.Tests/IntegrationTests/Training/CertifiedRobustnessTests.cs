using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureCertifiedDefense surfaces certified accuracy, and that with an adversarial attack
/// also configured the certified-vs-empirical robustness sandwich is reported.
/// </summary>
public class CertifiedRobustnessTests
{
    /// <summary>A defense returning fixed certified metrics, for the wiring test.</summary>
    private sealed class FixedCertifiedDefense : ICertifiedDefense<double, Matrix<double>, Vector<double>>
    {
        public const double Certified = 0.6;
        public CertifiedPrediction<double> CertifyPrediction(Matrix<double> input, IFullModel<double, Matrix<double>, Vector<double>> model)
            => new() { CertifiedRadius = 0.05, IsCertified = true };
        public CertifiedPrediction<double>[] CertifyBatch(Matrix<double>[] inputs, IFullModel<double, Matrix<double>, Vector<double>> model)
            => Array.Empty<CertifiedPrediction<double>>();
        public double ComputeCertifiedRadius(Matrix<double> input, IFullModel<double, Matrix<double>, Vector<double>> model) => 0.05;
        public CertifiedAccuracyMetrics<double> EvaluateCertifiedAccuracy(
            Matrix<double>[] testData, Vector<double>[] labels, IFullModel<double, Matrix<double>, Vector<double>> model, double radius)
            => new()
            {
                CleanAccuracy = 0.9,
                CertifiedAccuracy = Certified,
                CertificationRadius = radius,
                AverageCertifiedRadius = 0.05,
                CertificationRate = 0.6,
                MedianCertifiedRadius = 0.05,
            };
        public CertifiedDefenseOptions<double> GetOptions() => new();
        public void Reset() { }
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    private sealed class EpsilonShiftAttack : AdversarialAttackBase<double, Matrix<double>, Vector<double>>
    {
        public EpsilonShiftAttack(AdversarialAttackOptions<double> options) : base(options) { }
        public override Matrix<double> GenerateAdversarialExample(
            Matrix<double> input, Vector<double> trueLabel, IFullModel<double, Matrix<double>, Vector<double>> targetModel)
        {
            var adv = new Matrix<double>(input.Rows, input.Columns);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Columns; j++) adv[i, j] = input[i, j] + Options.Epsilon;
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
    public async Task ConfigureCertifiedDefense_SurfacesCertifiedMetrics()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureCertifiedDefense(new FixedCertifiedDefense())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.CertifiedRobustness);
        Assert.Equal(FixedCertifiedDefense.Certified, result.CertifiedRobustness?.Metrics.CertifiedAccuracy ?? double.NaN);
        Assert.False(result.CertifiedRobustness?.SandwichAvailable ?? true); // no attack configured
    }

    [Fact(Timeout = 120000)]
    public async Task CertifiedPlusAttack_ReportsSandwich()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureCertifiedDefense(new FixedCertifiedDefense())
            .ConfigureAdversarialAttack(new EpsilonShiftAttack(new AdversarialAttackOptions<double> { Epsilon = 0.3 }))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        var report = result.CertifiedRobustness;
        Assert.NotNull(report);
        // Both bounds present, so the certified-vs-empirical sandwich is available.
        Assert.True(report?.SandwichAvailable == true);
        Assert.NotNull(report?.EmpiricalRobustAccuracy);
        // The gap is empirical (upper bound) minus certified (lower bound), computed from both bounds.
        var empirical = report?.EmpiricalRobustAccuracy ?? 0.0;
        var certified = report?.Metrics.CertifiedAccuracy ?? 0.0;
        Assert.Equal(empirical - certified, report?.CertifiedEmpiricalGap ?? double.NaN, 9);
        // Certification is evaluated at the empirical attack's budget so the two bounds are comparable.
        Assert.Equal(0.3, report?.RadiusUsed ?? 0.0, 3);
    }
}
