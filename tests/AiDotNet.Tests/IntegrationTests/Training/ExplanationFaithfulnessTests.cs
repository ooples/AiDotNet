using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers the explanation-faithfulness audit: a faithful attribution scores higher than a misleading one,
/// and ConfigureModelExplainer surfaces the audit and the explainer on the built result.
/// </summary>
public class ExplanationFaithfulnessTests
{
    [Fact]
    public void FaithfulnessAuditor_FaithfulAttributionOutscoresMisleadingOne()
    {
        // A linear model: feature 0 matters most, feature 1 is irrelevant, feature 2 matters somewhat.
        double[] weights = { 2.0, 0.0, -1.0 };
        double Score(Vector<double> x)
        {
            double s = 0;
            for (int j = 0; j < x.Length; j++) s += weights[j] * x[j];
            return s;
        }

        var rng = new Random(1);
        var data = new Matrix<double>(60, 3);
        for (int i = 0; i < 60; i++)
            for (int j = 0; j < 3; j++) data[i, j] = rng.NextDouble() * 2 - 1;

        var auditor = new FaithfulnessAuditor<double>();
        var faithful = auditor.Audit(Score, data, new Vector<double>(new[] { 2.0, 0.0, 1.0 })); // |weights|
        var misleading = auditor.Audit(Score, data, new Vector<double>(new[] { 0.0, 2.0, 0.0 })); // points at the irrelevant feature

        Assert.True(faithful.FaithfulnessScore > misleading.FaithfulnessScore,
            $"faithful attribution should score higher (faithful={faithful.FaithfulnessScore:F4}, misleading={misleading.FaithfulnessScore:F4})");
        // A faithful attribution makes deletion collapse the output faster than insertion restores it late.
        Assert.True(faithful.Comprehensiveness >= misleading.Comprehensiveness);
    }

    /// <summary>A minimal explainer that reports fixed global attributions, for the wiring test.</summary>
    private sealed class FixedGlobalExplainer : IModelExplainer<double>, IGlobalAttributionExplainer<double>
    {
        private readonly Vector<double> _attributions;
        public FixedGlobalExplainer(Vector<double> attributions) => _attributions = attributions;
        public string MethodName => "fixed-global";
        public bool SupportsLocalExplanations => false;
        public bool SupportsGlobalExplanations => true;
        public Vector<double> ComputeGlobalAttributions(Matrix<double> data) => _attributions;
    }

    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 80, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.01);
            y[i] = 2.0 * x[i, 0] - x[i, 2]; // depends on features 0 and 2, not 1
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureModelExplainer_SurfacesExplainerAndFaithfulnessAudit()
    {
        var (x, y) = BuildData();
        var explainer = new FixedGlobalExplainer(new Vector<double>(new[] { 2.0, 0.0, 1.0 }));

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureModelExplainer(explainer)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Same(explainer, result.ModelExplainer);
        Assert.NotNull(result.ExplanationFaithfulness);
        Assert.NotNull(result.ExplanationFaithfulness!.ExplainerFaithfulness);
        Assert.Equal("FixedGlobalExplainer", result.ExplanationFaithfulness.ExplainerName);
    }

    [Fact(Timeout = 120000)]
    public async Task NoModelExplainer_LeavesResultNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.ModelExplainer);
        Assert.Null(result.ExplanationFaithfulness);
    }
}
