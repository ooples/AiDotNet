using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Classification;

/// <summary>
/// Reference checks for the decision tree against scikit-learn's DecisionTreeClassifier (_tree.pyx).
/// </summary>
public class DecisionTreeReferenceTests
{
    private const double Tolerance = 1e-12;

    [Fact]
    [Trait("category", "unit")]
    public void Xor_is_learned_although_no_first_split_lowers_the_impurity()
    {
        // DecisionTreeClassifier(random_state=0).fit(X, y): get_n_leaves() == 4, predict(X) == y.
        var x = new Matrix<double>(new[,] { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } });
        var y = new Vector<double>(new[] { 0.0, 1, 1, 0 });

        var tree = new DecisionTreeClassifier<double>();
        tree.Train(x, y);

        Assert.Equal(4, tree.LeafCount);
        var predictions = tree.Predict(x);
        for (int i = 0; i < y.Length; i++)
        {
            Assert.Equal(y[i], predictions[i]);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Min_impurity_decrease_is_compared_with_the_sample_weighted_decrease_in_bits()
    {
        // DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.2): the root splits at 4.5; the right
        // child (5 of 10 rows, [1, 1, 1, 0, 1]) would gain 0.3219 bits at 7.5, weighted 0.161 < 0.2, so it is a leaf.
        // get_n_leaves() == 2; predict_proba(X)[:, 1] == [0, 0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8, 0.8].
        var x = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++) x[i, 0] = i;
        var y = new Vector<double>(new[] { 0.0, 0, 0, 0, 0, 1, 1, 1, 0, 1 });

        var tree = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Criterion = ClassificationSplitCriterion.Entropy,
            MinImpurityDecrease = 0.2
        });
        tree.Train(x, y);

        Assert.Equal(2, tree.LeafCount);
        double[] expected = { 0, 0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8, 0.8 };
        var probabilities = tree.PredictProbabilities(x);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], probabilities[i, 1], Tolerance);
            Assert.Equal(1 - expected[i], probabilities[i, 0], Tolerance);
        }

        // Without the threshold the tree grows until pure: 4 leaves (thresholds 4.5, 7.5, 8.5).
        var full = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Criterion = ClassificationSplitCriterion.Entropy
        });
        full.Train(x, y);
        Assert.Equal(4, full.LeafCount);
    }
}
