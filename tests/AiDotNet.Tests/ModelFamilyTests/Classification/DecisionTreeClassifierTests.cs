using AiDotNet.Interfaces;
using AiDotNet.Classification.Trees;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class DecisionTreeClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DecisionTreeClassifier<double>();

    /// <summary>
    /// A depth-3 tree on fixed data must grow exactly as scikit-learn 1.x's <c>DecisionTreeClassifier</c> grows it:
    /// the same node count and, at one row from every leaf, the same leaf class fractions. scikit-learn's result does
    /// not depend on its random_state here (checked over 20 seeds), so no split ties are involved.
    /// </summary>
    [Theory]
    [InlineData(ClassificationSplitCriterion.Gini)]
    [InlineData(ClassificationSplitCriterion.Entropy)]
    public void Train_DepthLimitedTreeMatchesScikitLearn(ClassificationSplitCriterion criterion)
    {
        const int rows = 120, features = 3;
        var x = new Matrix<double>(rows, features);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++)
                x[i, j] = ((i * 37 + j * 101 + i * j * 13) % 97) / 97.0;

            y[i] = (i * 7) % 11 == 0
                ? i % 3
                : x[i, 0] + 0.5 * x[i, 1] < 0.6 ? 0 : x[i, 2] < 0.55 ? 1 : 2;
        }

        var tree = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Criterion = criterion, MaxDepth = 3
        });
        tree.Train(x, y);

        // (row, class fractions of its leaf) from sklearn.tree.DecisionTreeClassifier(max_depth=3).
        var (nodes, leaves) = criterion == ClassificationSplitCriterion.Gini
            ? (13, new (int Row, double[] Fractions)[]
            {
                (0, new[] { 1.0, 0, 0 }), (1, new[] { 0, 1.0 / 31, 30.0 / 31 }),
                (2, new[] { 1.0 / 18, 11.0 / 12, 1.0 / 36 }), (4, new[] { 1.0, 0, 0 }),
                (11, new[] { 2.0 / 9, 1.0 / 3, 4.0 / 9 }), (21, new[] { 1.0, 0, 0 }),
                (22, new[] { 2.0 / 5, 2.0 / 5, 1.0 / 5 })
            })
            : (15, new (int Row, double[] Fractions)[]
            {
                (0, new[] { 22.0 / 23, 1.0 / 23, 0 }), (1, new[] { 5.0 / 14, 0, 9.0 / 14 }),
                (2, new[] { 2.0 / 5, 3.0 / 5, 0 }), (4, new[] { 1.0, 0, 0 }),
                (5, new[] { 0, 30.0 / 31, 1.0 / 31 }), (7, new[] { 0, 0, 1.0 }),
                (11, new[] { 1.0 / 3, 4.0 / 9, 2.0 / 9 }), (55, new[] { 0, 1.0, 0 })
            });

        Assert.Equal(nodes, tree.NodeCount);
        var probabilities = tree.PredictProbabilities(x);
        foreach (var (row, fractions) in leaves)
            for (int c = 0; c < fractions.Length; c++)
                Assert.Equal(fractions[c], probabilities[row, c], 12);
    }
}
