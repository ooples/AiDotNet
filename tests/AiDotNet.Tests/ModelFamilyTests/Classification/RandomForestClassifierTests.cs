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

    /// <summary>
    /// The forest predicts by walking flattened copies of its trees; its probabilities must be exactly each tree's own
    /// probabilities averaged in tree order, including for rows on a threshold, beyond the training range, or NaN.
    /// </summary>
    [Fact]
    public void PredictProbabilities_EqualTheTreesAveraged()
    {
        const int rows = 400, features = 4;
        var x = new Matrix<double>(rows, features);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++)
                x[i, j] = ((i * 37 + j * 11 + i * j * 5) % 23) / 4.0;
            y[i] = (x[i, 0] + x[i, 2] > 5 ? 1 : 0) + (x[i, 1] > 3 ? 1 : 0);
        }

        var forest = new ExposedForest();
        forest.Train(x, y);

        // The training rows; then a grid at every eighth, which holds every threshold (midpoints of the quarter-step
        // values) in every feature; then NaN and values far outside the training range.
        const int grid = 23 * 8;
        var probe = new Matrix<double>(rows + grid + 3, features);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < features; j++)
                probe[i, j] = x[i, j];
        for (int g = 0; g < grid; g++)
            for (int j = 0; j < features; j++)
                probe[rows + g, j] = ((g + j * 41) % grid) / 8.0;
        for (int j = 0; j < features; j++)
        {
            probe[rows + grid, j] = double.NaN;
            probe[rows + grid + 1, j] = 1e9;
            probe[rows + grid + 2, j] = -1e9;
        }

        var actual = forest.PredictProbabilities(probe);
        var expected = forest.TreesAveraged(probe);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < probe.Rows; i++)
            for (int c = 0; c < expected.Columns; c++)
                Assert.Equal(expected[i, c], actual[i, c]);
    }

    /// <summary>A forest that also averages its trees' probabilities through each tree's own prediction.</summary>
    private sealed class ExposedForest : RandomForestClassifier<double>
    {
        public ExposedForest()
            : base(new RandomForestClassifierOptions<double> { NEstimators = 12, Seed = 4, NJobs = 3 })
        {
        }

        public Matrix<double> TreesAveraged(Matrix<double> input)
        {
            var result = new Matrix<double>(input.Rows, NumClasses);
            var perTree = new List<Matrix<double>>();
            foreach (var tree in Estimators)
                perTree.Add(((IProbabilisticClassifier<double>)tree).PredictProbabilities(input));
            for (int i = 0; i < input.Rows; i++)
                for (int c = 0; c < NumClasses; c++)
                {
                    double sum = 0;
                    foreach (var probabilities in perTree) sum += probabilities[i, c];
                    result[i, c] = sum / perTree.Count;
                }

            return result;
        }
    }
}
