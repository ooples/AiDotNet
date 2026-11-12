using System;
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

    [Fact]
    [Trait("category", "unit")]
    public void Float_and_half_trees_match_the_scikit_learn_references()
    {
        // The same references in narrower T: the split search runs in T, counts stay integers.
        var xorX = new Matrix<float>(new[,] { { 0f, 0f }, { 0f, 1f }, { 1f, 0f }, { 1f, 1f } });
        var xorTree = new DecisionTreeClassifier<float>();
        xorTree.Train(xorX, new Vector<float>(new[] { 0f, 1, 1, 0 }));
        Assert.Equal(4, xorTree.LeafCount);

        var x = new Matrix<Half>(10, 1);
        for (int i = 0; i < 10; i++) x[i, 0] = (Half)i;
        var y = new Vector<Half>(new[] { (Half)0, (Half)0, (Half)0, (Half)0, (Half)0, (Half)1, (Half)1, (Half)1, (Half)0, (Half)1 });
        var tree = new DecisionTreeClassifier<Half>(new DecisionTreeClassifierOptions<Half>
        {
            Criterion = ClassificationSplitCriterion.Entropy,
            MinImpurityDecrease = 0.2
        });
        tree.Train(x, y);

        Assert.Equal(2, tree.LeafCount);
        var probabilities = tree.PredictProbabilities(x);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i < 5 ? 0.0 : 0.8, (double)probabilities[i, 1], 1e-3);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_tree_on_more_rows_than_half_can_count_does_not_saturate()
    {
        // 70,000 rows (> Half's 65,504): one split at 499.5 separates the classes; every leaf is pure.
        const int rows = 70000;
        var x = new Matrix<Half>(rows, 1);
        var y = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            int v = i % 1000;
            x[i, 0] = (Half)v;
            y[i] = v < 500 ? (Half)0 : (Half)1;
        }

        var tree = new DecisionTreeClassifier<Half>();
        tree.Train(x, y);

        Assert.Equal(2, tree.LeafCount);
        var probabilities = tree.PredictProbabilities(x);
        for (int i = 0; i < rows; i += 997)
        {
            double expected = i % 1000 < 500 ? 0.0 : 1.0;
            Assert.Equal(expected, (double)probabilities[i, 1], 1e-6);
            Assert.Equal(1.0 - expected, (double)probabilities[i, 0], 1e-6);
        }
    }
}
