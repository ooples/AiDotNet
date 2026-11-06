using AiDotNet.Interfaces;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class RandomForestClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new RandomForestClassifier<double>();

    /// <summary>
    /// Bootstrap samples and tree seeds are drawn before any tree trains, so training trees in parallel grows the
    /// same forest as training them one at a time.
    /// </summary>
    [Fact]
    public void Train_ParallelJobsGrowTheSameForest()
    {
        const int rows = 300, features = 5;
        var x = new Matrix<double>(rows, features);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++)
                x[i, j] = ((i * 31 + j * 17 + i * j * 7) % 89) / 89.0;
            y[i] = (x[i, 0] + x[i, 3] > 1.0 ? 1 : 0) + (x[i, 1] > 0.7 ? 1 : 0);
        }

        RandomForestClassifier<double> Fit(int jobs)
        {
            var forest = new RandomForestClassifier<double>(new RandomForestClassifierOptions<double>
            {
                NEstimators = 16, Seed = 9, NJobs = jobs
            });
            forest.Train(x, y);
            return forest;
        }

        var serial = Fit(1).PredictProbabilities(x);
        foreach (int jobs in new[] { 4, -1 })
        {
            var parallel = Fit(jobs).PredictProbabilities(x);
            for (int i = 0; i < rows; i++)
                for (int c = 0; c < serial.Columns; c++)
                    Assert.Equal(serial[i, c], parallel[i, c]);
        }
    }

    [Fact]
    public void Train_RejectsZeroJobs()
    {
        var forest = new RandomForestClassifier<double>(new RandomForestClassifierOptions<double> { NJobs = 0 });
        var x = new Matrix<double>(4, 1);
        for (int i = 0; i < 4; i++) x[i, 0] = i;
        Assert.Throws<InvalidOperationException>(() => forest.Train(x, new Vector<double>(new double[] { 0, 0, 1, 1 })));
    }
}
