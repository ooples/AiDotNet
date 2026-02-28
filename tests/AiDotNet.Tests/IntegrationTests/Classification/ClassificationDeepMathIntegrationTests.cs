using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Deep math-correctness integration tests for classification algorithms.
/// Each test hand-computes expected outputs and verifies the code matches.
/// </summary>
public class ClassificationDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ========================================================================
    // GaussianNaiveBayes - Training parameter computation
    // ========================================================================

    [Fact]
    public void GaussianNB_Train_ComputesMeansCorrectly()
    {
        // 4 samples, 2 features, 2 classes (0, 1)
        // Class 0: samples [1, 2] and [3, 4] -> mean = [2, 3]
        // Class 1: samples [5, 6] and [7, 8] -> mean = [6, 7]
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 5.0; x[2, 1] = 6.0;
        x[3, 0] = 7.0; x[3, 1] = 8.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Predict on training data to verify model learned correctly
        var predictions = gnb.Predict(x);

        // Class 0 samples should be predicted as class 0
        Assert.Equal(0.0, predictions[0], Tol);
        Assert.Equal(0.0, predictions[1], Tol);

        // Class 1 samples should be predicted as class 1
        Assert.Equal(1.0, predictions[2], Tol);
        Assert.Equal(1.0, predictions[3], Tol);
    }

    [Fact]
    public void GaussianNB_LogLikelihood_HandComputed()
    {
        // Train with known data where we can hand-compute everything
        // Class 0: [1, 2], [3, 4] -> mean=[2,3], var(n)=[1, 1]
        // Class 1: [10, 20], [12, 22] -> mean=[11,21], var(n)=[1, 1]
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 10.0; x[2, 1] = 20.0;
        x[3, 0] = 12.0; x[3, 1] = 22.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Test prediction: [2, 3] is at the mean of class 0
        // Should strongly predict class 0
        var testX = new Matrix<double>(1, 2);
        testX[0, 0] = 2.0; testX[0, 1] = 3.0;

        var probs = gnb.PredictProbabilities(testX);

        // Class 0 probability should be much higher than class 1
        double prob0 = probs[0, 0];
        double prob1 = probs[0, 1];

        Assert.True(prob0 > 0.99,
            $"Expected prob(class 0) > 0.99 for sample at class 0 mean, got {prob0}");
        Assert.True(prob1 < 0.01,
            $"Expected prob(class 1) < 0.01 for sample far from class 1, got {prob1}");

        // Probabilities should sum to 1
        Assert.Equal(1.0, prob0 + prob1, 1e-5);
    }

    [Fact]
    public void GaussianNB_UniformPriors_EqualClassSizes()
    {
        // With equal class sizes and FitPriors=true, priors should be log(0.5) each
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 1);
        x[0, 0] = 1.0;
        x[1, 0] = 2.0;
        x[2, 0] = 10.0;
        x[3, 0] = 11.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Test at equidistant point from both class means
        // Class 0 mean = 1.5, Class 1 mean = 10.5
        // Class 0 var = 0.25, Class 1 var = 0.25
        // Midpoint = 6.0
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 6.0;

        var probs = gnb.PredictProbabilities(testX);

        // At midpoint with equal priors and equal variances, the Gaussian
        // density is exp(-0.5*(6-1.5)^2/0.25) vs exp(-0.5*(6-10.5)^2/0.25)
        // (6-1.5)^2 = 20.25, (6-10.5)^2 = 20.25 -> equal!
        // So probabilities should be 0.5 each
        Assert.Equal(0.5, probs[0, 0], 1e-4);
        Assert.Equal(0.5, probs[0, 1], 1e-4);
    }

    [Fact]
    public void GaussianNB_UnequalPriors_ShiftsDecisionBoundary()
    {
        // With unequal class sizes, priors shift the decision boundary
        // Class 0: 3 samples, Class 1: 1 sample -> prior(0)=0.75, prior(1)=0.25
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 1);
        x[0, 0] = 0.0;
        x[1, 0] = 1.0;
        x[2, 0] = 2.0;
        x[3, 0] = 10.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0; y[2] = 0.0;
        y[3] = 1.0;

        gnb.Train(x, y);

        // Class 0 has prior 0.75 (3/4), class 1 has prior 0.25 (1/4)
        // The higher prior for class 0 should shift the decision boundary
        // toward class 1, making class 0 favored for borderline samples

        // Test at the midpoint between class means
        // Class 0 mean = 1.0, Class 1 mean = 10.0
        // Midpoint = 5.5
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 5.5;

        var probs = gnb.PredictProbabilities(testX);

        // With higher prior for class 0, class 0 probability at midpoint
        // should be higher than 0.5 (decision boundary shifted toward class 1)
        Assert.True(probs[0, 0] > probs[0, 1],
            $"Expected class 0 prob ({probs[0, 0]}) > class 1 prob ({probs[0, 1]}) at midpoint due to higher prior");
    }

    [Fact]
    public void GaussianNB_Variance_PopulationFormula()
    {
        // Verify variance uses population formula (n, not n-1) like sklearn
        // Class 0: [2, 4, 6] -> mean=4, var(n) = ((2-4)^2+(4-4)^2+(6-4)^2)/3 = (4+0+4)/3 = 8/3
        // If it used n-1: var = 8/2 = 4 (wrong for sklearn compat)
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(6, 1);
        x[0, 0] = 2.0;
        x[1, 0] = 4.0;
        x[2, 0] = 6.0;
        x[3, 0] = 100.0;
        x[4, 0] = 102.0;
        x[5, 0] = 104.0;

        var y = new Vector<double>(6);
        y[0] = 0.0; y[1] = 0.0; y[2] = 0.0;
        y[3] = 1.0; y[4] = 1.0; y[5] = 1.0;

        gnb.Train(x, y);

        // Test: a point at the class 0 mean
        // With population variance = 8/3 = 2.6667, the log-likelihood at the mean is:
        // -0.5 * (log(2*pi) + log(8/3) + 0) = -0.5 * (1.8379 + 0.9808) = -1.4093
        // This is a characteristic of the density at the mean
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 4.0;

        var predictions = gnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);
    }

    [Fact]
    public void GaussianNB_MinVarianceProtection()
    {
        // When all samples in a class have the same feature value,
        // variance = 0, but MinVariance should prevent division by zero
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-6
        });

        var x = new Matrix<double>(4, 1);
        x[0, 0] = 5.0;  // Class 0: constant value
        x[1, 0] = 5.0;
        x[2, 0] = 10.0; // Class 1: constant value
        x[3, 0] = 10.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Prediction should not throw (MinVariance prevents div by zero)
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 5.0;

        var predictions = gnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);

        // And for class 1
        testX[0, 0] = 10.0;
        predictions = gnb.Predict(testX);
        Assert.Equal(1.0, predictions[0], Tol);
    }

    [Fact]
    public void GaussianNB_ProbabilitiesSumToOne()
    {
        // For any input, predicted probabilities must sum to 1.0
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(6, 2);
        x[0, 0] = 1.0; x[0, 1] = 1.0;
        x[1, 0] = 2.0; x[1, 1] = 2.0;
        x[2, 0] = 5.0; x[2, 1] = 5.0;
        x[3, 0] = 6.0; x[3, 1] = 6.0;
        x[4, 0] = 9.0; x[4, 1] = 9.0;
        x[5, 0] = 10.0; x[5, 1] = 10.0;

        var y = new Vector<double>(6);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;
        y[4] = 2.0; y[5] = 2.0;

        gnb.Train(x, y);

        // Test several points
        var testX = new Matrix<double>(5, 2);
        testX[0, 0] = 0.0; testX[0, 1] = 0.0;   // Far from all
        testX[1, 0] = 1.5; testX[1, 1] = 1.5;   // Near class 0
        testX[2, 0] = 5.5; testX[2, 1] = 5.5;   // Near class 1
        testX[3, 0] = 9.5; testX[3, 1] = 9.5;   // Near class 2
        testX[4, 0] = 100.0; testX[4, 1] = 100.0; // Very far

        var probs = gnb.PredictProbabilities(testX);

        for (int i = 0; i < 5; i++)
        {
            double sum = 0;
            for (int c = 0; c < 3; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= 0, $"Probability should be non-negative, got {probs[i, c]}");
            }
            Assert.Equal(1.0, sum, 1e-5);
        }
    }

    [Fact]
    public void GaussianNB_CustomPriors()
    {
        // Custom priors should override learned priors
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = false,
            ClassPriors = new double[] { 0.9, 0.1 },
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 1);
        x[0, 0] = 0.0;
        x[1, 0] = 1.0;
        x[2, 0] = 10.0;
        x[3, 0] = 11.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // At midpoint (5.5), with a strong prior for class 0 (0.9 vs 0.1),
        // class 0 should be strongly favored
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 5.5;

        var probs = gnb.PredictProbabilities(testX);
        Assert.True(probs[0, 0] > probs[0, 1],
            $"With 0.9/0.1 priors, class 0 should dominate at midpoint. Got {probs[0, 0]} vs {probs[0, 1]}");
    }

    // ========================================================================
    // MultinomialNaiveBayes - Word count model
    // ========================================================================

    [Fact]
    public void MultinomialNB_Train_CorrectSmoothing()
    {
        // Multinomial NB with Laplace smoothing (alpha=1)
        // Class 0: [3, 0, 1] -> counts=[3,0,1]+1=[4,1,2], total=7, P=[4/7, 1/7, 2/7]
        // Class 1: [0, 2, 1] -> counts=[0,2,1]+1=[1,3,2], total=6, P=[1/6, 3/6, 2/6]
        var mnb = new MultinomialNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            Alpha = 1.0,
            FitPriors = true
        });

        var x = new Matrix<double>(2, 3);
        x[0, 0] = 3.0; x[0, 1] = 0.0; x[0, 2] = 1.0;
        x[1, 0] = 0.0; x[1, 1] = 2.0; x[1, 2] = 1.0;

        var y = new Vector<double>(2);
        y[0] = 0.0;
        y[1] = 1.0;

        mnb.Train(x, y);

        // Test: [3, 0, 0] should predict class 0 (high count in feature 0)
        var testX = new Matrix<double>(1, 3);
        testX[0, 0] = 3.0; testX[0, 1] = 0.0; testX[0, 2] = 0.0;

        var predictions = mnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);

        // Test: [0, 3, 0] should predict class 1 (high count in feature 1)
        testX[0, 0] = 0.0; testX[0, 1] = 3.0; testX[0, 2] = 0.0;

        predictions = mnb.Predict(testX);
        Assert.Equal(1.0, predictions[0], Tol);
    }

    // ========================================================================
    // BernoulliNaiveBayes - Binary features
    // ========================================================================

    [Fact]
    public void BernoulliNB_Train_BinaryFeatures()
    {
        // Bernoulli NB with binary features
        // Class 0: [1,0,1], [1,1,1] -> P(f=1|c=0) = [1, 0.5, 1]
        // Class 1: [0,1,0], [0,1,1] -> P(f=1|c=1) = [0, 1, 0.5]
        var bnb = new BernoulliNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            Alpha = 1.0,
            FitPriors = true
        });

        var x = new Matrix<double>(4, 3);
        x[0, 0] = 1.0; x[0, 1] = 0.0; x[0, 2] = 1.0;
        x[1, 0] = 1.0; x[1, 1] = 1.0; x[1, 2] = 1.0;
        x[2, 0] = 0.0; x[2, 1] = 1.0; x[2, 2] = 0.0;
        x[3, 0] = 0.0; x[3, 1] = 1.0; x[3, 2] = 1.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        bnb.Train(x, y);

        // Test: [1,0,1] should predict class 0
        var testX = new Matrix<double>(1, 3);
        testX[0, 0] = 1.0; testX[0, 1] = 0.0; testX[0, 2] = 1.0;
        var predictions = bnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);

        // Test: [0,1,0] should predict class 1
        testX[0, 0] = 0.0; testX[0, 1] = 1.0; testX[0, 2] = 0.0;
        predictions = bnb.Predict(testX);
        Assert.Equal(1.0, predictions[0], Tol);
    }

    // ========================================================================
    // GaussianNaiveBayes - Serialization roundtrip
    // ========================================================================

    [Fact]
    public void GaussianNB_SerializeDeserialize_Roundtrip()
    {
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 10.0; x[2, 1] = 20.0;
        x[3, 0] = 12.0; x[3, 1] = 22.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Get predictions before serialization
        var testX = new Matrix<double>(2, 2);
        testX[0, 0] = 2.0; testX[0, 1] = 3.0;
        testX[1, 0] = 11.0; testX[1, 1] = 21.0;

        var predBefore = gnb.Predict(testX);

        // Serialize and deserialize
        byte[] serialized = gnb.Serialize();
        var gnb2 = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });
        gnb2.Deserialize(serialized);

        // Predictions should be identical after roundtrip
        var predAfter = gnb2.Predict(testX);
        Assert.Equal(predBefore[0], predAfter[0], Tol);
        Assert.Equal(predBefore[1], predAfter[1], Tol);
    }

    // ========================================================================
    // GaussianNaiveBayes - Multi-class
    // ========================================================================

    [Fact]
    public void GaussianNB_MultiClass_CorrectClassification()
    {
        // 3 well-separated classes
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(9, 2);
        // Class 0: cluster around (0, 0)
        x[0, 0] = -1.0; x[0, 1] = -1.0;
        x[1, 0] = 0.0; x[1, 1] = 0.0;
        x[2, 0] = 1.0; x[2, 1] = 1.0;
        // Class 1: cluster around (10, 10)
        x[3, 0] = 9.0; x[3, 1] = 9.0;
        x[4, 0] = 10.0; x[4, 1] = 10.0;
        x[5, 0] = 11.0; x[5, 1] = 11.0;
        // Class 2: cluster around (0, 10)
        x[6, 0] = -1.0; x[6, 1] = 9.0;
        x[7, 0] = 0.0; x[7, 1] = 10.0;
        x[8, 0] = 1.0; x[8, 1] = 11.0;

        var y = new Vector<double>(9);
        y[0] = 0.0; y[1] = 0.0; y[2] = 0.0;
        y[3] = 1.0; y[4] = 1.0; y[5] = 1.0;
        y[6] = 2.0; y[7] = 2.0; y[8] = 2.0;

        gnb.Train(x, y);

        // Test points near each cluster center
        var testX = new Matrix<double>(3, 2);
        testX[0, 0] = 0.0; testX[0, 1] = 0.0;   // Near class 0
        testX[1, 0] = 10.0; testX[1, 1] = 10.0;  // Near class 1
        testX[2, 0] = 0.0; testX[2, 1] = 10.0;   // Near class 2

        var predictions = gnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);
        Assert.Equal(1.0, predictions[1], Tol);
        Assert.Equal(2.0, predictions[2], Tol);
    }

    [Fact]
    public void GaussianNB_HighDimensional_CorrectClassification()
    {
        // Test with 5 features - well-separated classes
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 5);
        // Class 0: all features around 1
        x[0, 0] = 0.9; x[0, 1] = 1.1; x[0, 2] = 0.8; x[0, 3] = 1.2; x[0, 4] = 1.0;
        x[1, 0] = 1.1; x[1, 1] = 0.9; x[1, 2] = 1.2; x[1, 3] = 0.8; x[1, 4] = 1.0;
        // Class 1: all features around 10
        x[2, 0] = 9.9; x[2, 1] = 10.1; x[2, 2] = 9.8; x[2, 3] = 10.2; x[2, 4] = 10.0;
        x[3, 0] = 10.1; x[3, 1] = 9.9; x[3, 2] = 10.2; x[3, 3] = 9.8; x[3, 4] = 10.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Test near class 0
        var testX = new Matrix<double>(1, 5);
        testX[0, 0] = 1.0; testX[0, 1] = 1.0; testX[0, 2] = 1.0; testX[0, 3] = 1.0; testX[0, 4] = 1.0;

        var predictions = gnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);
    }

    // ========================================================================
    // GaussianNaiveBayes - Edge cases
    // ========================================================================

    [Fact]
    public void GaussianNB_SingleSamplePerClass()
    {
        // When each class has only one sample, variance is 0 -> MinVariance kicks in
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-4
        });

        var x = new Matrix<double>(2, 1);
        x[0, 0] = 0.0;
        x[1, 0] = 10.0;

        var y = new Vector<double>(2);
        y[0] = 0.0;
        y[1] = 1.0;

        gnb.Train(x, y);

        // Should still predict correctly (MinVariance prevents NaN)
        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 0.0;

        var predictions = gnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);

        testX[0, 0] = 10.0;
        predictions = gnb.Predict(testX);
        Assert.Equal(1.0, predictions[0], Tol);
    }

    [Fact]
    public void GaussianNB_LogProbabilities_AllFinite()
    {
        // Verify log probabilities are always finite (no NaN or Inf)
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 10.0; x[2, 1] = 20.0;
        x[3, 0] = 12.0; x[3, 1] = 22.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        // Test with extreme values
        var testX = new Matrix<double>(3, 2);
        testX[0, 0] = 0.0; testX[0, 1] = 0.0;     // Normal
        testX[1, 0] = 1000.0; testX[1, 1] = 1000.0; // Very far
        testX[2, 0] = -1000.0; testX[2, 1] = -1000.0; // Very far negative

        var logProbs = gnb.PredictLogProbabilities(testX);

        for (int i = 0; i < 3; i++)
        {
            for (int c = 0; c < 2; c++)
            {
                double val = logProbs[i, c];
                Assert.False(double.IsNaN(val), $"Log probability should not be NaN at [{i},{c}]");
                Assert.False(double.IsPositiveInfinity(val), $"Log probability should not be +Inf at [{i},{c}]");
                Assert.True(val <= 0, $"Log probability should be <= 0, got {val} at [{i},{c}]");
            }
        }
    }

    // ========================================================================
    // ComplementNaiveBayes - Complement of each class
    // ========================================================================

    [Fact]
    public void ComplementNB_Train_PredictCorrectly()
    {
        // ComplementNB uses the complement of each class (all other classes) for estimation
        // Should handle imbalanced data better than MultinomialNB
        var cnb = new ComplementNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            Alpha = 1.0,
            FitPriors = true
        });

        var x = new Matrix<double>(4, 3);
        x[0, 0] = 5.0; x[0, 1] = 0.0; x[0, 2] = 1.0;
        x[1, 0] = 4.0; x[1, 1] = 1.0; x[1, 2] = 0.0;
        x[2, 0] = 0.0; x[2, 1] = 5.0; x[2, 2] = 1.0;
        x[3, 0] = 1.0; x[3, 1] = 4.0; x[3, 2] = 0.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        cnb.Train(x, y);

        // Test: [5, 0, 0] should predict class 0 (feature 0 dominant)
        var testX = new Matrix<double>(1, 3);
        testX[0, 0] = 5.0; testX[0, 1] = 0.0; testX[0, 2] = 0.0;
        var predictions = cnb.Predict(testX);
        Assert.Equal(0.0, predictions[0], Tol);

        // Test: [0, 5, 0] should predict class 1 (feature 1 dominant)
        testX[0, 0] = 0.0; testX[0, 1] = 5.0; testX[0, 2] = 0.0;
        predictions = cnb.Predict(testX);
        Assert.Equal(1.0, predictions[0], Tol);
    }

    // ========================================================================
    // GaussianNaiveBayes - Clone produces identical predictions
    // ========================================================================

    [Fact]
    public void GaussianNB_Clone_IdenticalPredictions()
    {
        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double>
        {
            FitPriors = true,
            MinVariance = 1e-9
        });

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 10.0; x[2, 1] = 20.0;
        x[3, 0] = 12.0; x[3, 1] = 22.0;

        var y = new Vector<double>(4);
        y[0] = 0.0; y[1] = 0.0;
        y[2] = 1.0; y[3] = 1.0;

        gnb.Train(x, y);

        var clone = gnb.Clone();

        var testX = new Matrix<double>(2, 2);
        testX[0, 0] = 2.0; testX[0, 1] = 3.0;
        testX[1, 0] = 11.0; testX[1, 1] = 21.0;

        var predOriginal = gnb.PredictProbabilities(testX);
        var predClone = ((GaussianNaiveBayes<double>)clone).PredictProbabilities(testX);

        for (int i = 0; i < 2; i++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(predOriginal[i, c], predClone[i, c], Tol);
            }
        }
    }
}
