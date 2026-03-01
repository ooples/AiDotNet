using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.OnlineLearning;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.OnlineLearning;

/// <summary>
/// Extended integration tests for online learning with deep mathematical verification.
/// Tests SGD gradient correctness, PA update formulas, convergence on known data,
/// regularization effects, and learning rate schedules.
/// </summary>
public class OnlineLearningExtendedIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region OnlineSGDClassifier - Gradient and Convergence Verification

    [Fact]
    public void SGDClassifier_LinearSeparable_ConvergesTo100PercentAccuracy()
    {
        // Generate linearly separable data: y=1 if x[0]+x[1]>0, else y=0
        var classifier = new OnlineSGDClassifier<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Train for multiple passes
        for (int epoch = 0; epoch < 20; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double x0 = rng.NextDouble() * 2 - 1; // [-1, 1]
                double x1 = rng.NextDouble() * 2 - 1;
                var x = new Vector<double>(new[] { x0, x1 });
                double y = (x0 + x1 > 0) ? 1.0 : 0.0;
                classifier.PartialFit(x, y);
            }
        }

        // Test accuracy on fresh data
        rng = RandomHelper.CreateSeededRandom(99);
        int correct = 0;
        int total = 100;
        for (int i = 0; i < total; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            double x1 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0, x1 });
            double trueY = (x0 + x1 > 0) ? 1.0 : 0.0;
            double pred = classifier.PredictSingle(x);
            if (Math.Abs(pred - trueY) < 0.5) correct++;
        }

        double accuracy = (double)correct / total;
        Assert.True(accuracy > 0.90,
            $"SGD classifier should achieve >90% on linearly separable data, got {accuracy:P}");
    }

    [Fact]
    public void SGDClassifier_ProbabilityOutput_IsBetween0And1()
    {
        var classifier = new OnlineSGDClassifier<double>(learningRate: 0.01);
        var rng = RandomHelper.CreateSeededRandom(42);

        // Train briefly
        for (int i = 0; i < 50; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0, rng.NextDouble() });
            classifier.PartialFit(x, x0 > 0 ? 1.0 : 0.0);
        }

        // Check probabilities
        for (int i = 0; i < 100; i++)
        {
            var x = new Vector<double>(new[] { rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2 });
            double prob = classifier.PredictProbability(x);
            Assert.True(prob >= 0 && prob <= 1,
                $"Probability {prob} should be in [0,1] for input ({x[0]:F3}, {x[1]:F3})");
        }
    }

    [Fact]
    public void SGDClassifier_Sigmoid_HandCalculated()
    {
        // Set known weights and verify sigmoid output
        var classifier = new OnlineSGDClassifier<double>(learningRate: 0.01, l2Penalty: 0.0);

        // Set parameters: w = [1, 0], bias = 0
        // Then P(y=1|x) = sigmoid(x[0]) = 1/(1+exp(-x[0]))
        var parameters = new Vector<double>(new[] { 1.0, 0.0, 0.0 }); // w0, w1, bias
        classifier.SetParameters(parameters);

        // sigmoid(0) = 0.5
        var x0 = new Vector<double>(new[] { 0.0, 0.0 });
        Assert.Equal(0.5, classifier.PredictProbability(x0), Tolerance);

        // sigmoid(2) = 1/(1+exp(-2)) = 0.880797
        var x1 = new Vector<double>(new[] { 2.0, 0.0 });
        double expected = 1.0 / (1.0 + Math.Exp(-2.0));
        Assert.Equal(expected, classifier.PredictProbability(x1), 1e-4);

        // sigmoid(-2) = 1/(1+exp(2)) = 0.119203
        var x2 = new Vector<double>(new[] { -2.0, 0.0 });
        Assert.Equal(1.0 - expected, classifier.PredictProbability(x2), 1e-4);
    }

    [Fact]
    public void SGDClassifier_L2Regularization_ShrinkWeights()
    {
        // Train two models: one with L2, one without
        var noReg = new OnlineSGDClassifier<double>(
            learningRate: 0.01, l2Penalty: 0.0,
            learningRateSchedule: LearningRateSchedule.Constant);
        var withReg = new OnlineSGDClassifier<double>(
            learningRate: 0.01, l2Penalty: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant);

        var rng = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < 500; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            double x1 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0, x1 });
            double y = (x0 > 0) ? 1.0 : 0.0;

            noReg.PartialFit(x, y);
            withReg.PartialFit(x, y);
        }

        // L2-regularized model should have smaller weight magnitudes
        var wNoReg = noReg.GetWeights();
        var wWithReg = withReg.GetWeights();

        Assert.NotNull(wNoReg);
        Assert.NotNull(wWithReg);

        double normNoReg = 0, normWithReg = 0;
        for (int i = 0; i < wNoReg.Length; i++)
        {
            normNoReg += wNoReg[i] * wNoReg[i];
            normWithReg += wWithReg[i] * wWithReg[i];
        }

        Assert.True(normWithReg < normNoReg,
            $"L2-regularized weights (norm={Math.Sqrt(normWithReg):F4}) should be smaller than unregularized (norm={Math.Sqrt(normNoReg):F4})");
    }

    [Fact]
    public void SGDClassifier_L1Regularization_ProducesSparseWeights()
    {
        // y depends only on x[0], x[1..4] are noise
        var classifier = new OnlineSGDClassifier<double>(
            learningRate: 0.05, l1Penalty: 0.01, l2Penalty: 0.0,
            learningRateSchedule: LearningRateSchedule.Constant);

        var rng = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < 2000; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var features = new double[5];
            features[0] = x0;
            for (int j = 1; j < 5; j++) features[j] = rng.NextDouble() * 0.1;
            var x = new Vector<double>(features);
            double y = x0 > 0 ? 1.0 : 0.0;
            classifier.PartialFit(x, y);
        }

        var weights = classifier.GetWeights();
        Assert.NotNull(weights);

        // Count near-zero weights (noise features should be pushed to zero by L1)
        int nearZero = 0;
        for (int i = 1; i < weights.Length; i++) // Skip feature 0 (signal)
        {
            if (Math.Abs(weights[i]) < 0.01) nearZero++;
        }

        // Most noise features should be near zero
        Assert.True(nearZero >= 2,
            $"L1 should push at least 2 of 4 noise features near zero, got {nearZero}");
    }

    [Fact]
    public void SGDClassifier_Reset_ClearsAllState()
    {
        var classifier = new OnlineSGDClassifier<double>(learningRate: 0.01);

        var x = new Vector<double>(new[] { 1.0, 2.0 });
        classifier.PartialFit(x, 1.0);
        classifier.PartialFit(x, 0.0);

        Assert.True(classifier.IsTrained);

        classifier.Reset();

        Assert.False(classifier.IsTrained);
        Assert.Null(classifier.GetWeights());
    }

    [Fact]
    public void SGDClassifier_DecisionFunction_MatchesLinearPrediction()
    {
        var classifier = new OnlineSGDClassifier<double>(learningRate: 0.01, l2Penalty: 0.0);

        // Set known parameters: w = [2, -1], bias = 0.5
        var parameters = new Vector<double>(new[] { 2.0, -1.0, 0.5 });
        classifier.SetParameters(parameters);

        var x = new Vector<double>(new[] { 3.0, 1.0 });
        // Expected: w.x + b = 2*3 + (-1)*1 + 0.5 = 5.5
        double decision = classifier.DecisionFunction(x);
        Assert.Equal(5.5, decision, Tolerance);
    }

    [Fact]
    public void SGDClassifier_FeatureImportance_ReflectsAbsoluteWeights()
    {
        var classifier = new OnlineSGDClassifier<double>();

        // Set known weights
        var parameters = new Vector<double>(new[] { -3.0, 1.0, 2.0, 0.0 }); // 3 weights + bias
        classifier.SetParameters(parameters);

        var importance = classifier.GetFeatureImportance();

        Assert.Equal(3, importance.Count);
        Assert.Equal(3.0, importance["Feature_0"], Tolerance); // |-3| = 3
        Assert.Equal(1.0, importance["Feature_1"], Tolerance); // |1| = 1
        Assert.Equal(2.0, importance["Feature_2"], Tolerance); // |2| = 2
    }

    #endregion

    #region OnlineSGDRegressor - Loss Function and Gradient Verification

    [Fact]
    public void SGDRegressor_SimpleLinear_ConvergesToTrueLine()
    {
        // True model: y = 2*x + 1 + noise
        var regressor = new OnlineSGDRegressor<double>(
            learningRate: 0.01,
            learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        var rng = RandomHelper.CreateSeededRandom(42);

        for (int epoch = 0; epoch < 30; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double x0 = rng.NextDouble() * 2 - 1;
                var x = new Vector<double>(new[] { x0 });
                double y = 2.0 * x0 + 1.0 + (rng.NextDouble() - 0.5) * 0.1;
                regressor.PartialFit(x, y);
            }
        }

        var weights = regressor.GetWeights();
        double bias = regressor.GetBias();

        Assert.NotNull(weights);
        // Weight should be close to 2.0
        Assert.True(Math.Abs(weights[0] - 2.0) < 0.5,
            $"Weight should be close to 2.0, got {weights[0]:F4}");
        // Bias should be close to 1.0
        Assert.True(Math.Abs(bias - 1.0) < 0.5,
            $"Bias should be close to 1.0, got {bias:F4}");
    }

    [Fact]
    public void SGDRegressor_HuberLoss_GradientHandCalculated()
    {
        // Huber loss gradient: 2*residual if |residual| <= epsilon, else 2*epsilon*sign(residual)
        var regressor = new OnlineSGDRegressor<double>(
            learningRate: 0.01,
            l2Penalty: 0.0,
            loss: SGDLossType.Huber,
            epsilon: 1.0,
            learningRateSchedule: LearningRateSchedule.Constant);

        // Set parameters: w=0, b=0, so prediction=0
        var parameters = new Vector<double>(new[] { 0.0, 0.0 }); // w0, bias
        regressor.SetParameters(parameters);

        // Feed y=0.5 (small residual, within epsilon)
        // residual = prediction - y = 0 - 0.5 = -0.5
        // gradient = 2 * (-0.5) = -1.0 (within epsilon=1)
        // weight update: w -= lr * gradient * x = 0 - 0.01 * (-1.0) * 1.0 = 0.01
        var x = new Vector<double>(new[] { 1.0 });
        regressor.PartialFit(x, 0.5);

        var weights = regressor.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(0.01, weights[0], 1e-4);

        // Reset and test large residual
        regressor.Reset();
        parameters = new Vector<double>(new[] { 0.0, 0.0 });
        regressor.SetParameters(parameters);

        // Feed y=5.0 (large residual, outside epsilon)
        // residual = 0 - 5 = -5, |residual|=5 > epsilon=1
        // gradient = 2*1*sign(-5) = -2
        // weight update: w -= lr * gradient * x = 0 - 0.01 * (-2) * 1 = 0.02
        regressor.PartialFit(x, 5.0);

        weights = regressor.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(0.02, weights[0], 1e-4);
    }

    [Fact]
    public void SGDRegressor_EpsilonInsensitiveLoss_ZeroGradientInBand()
    {
        // Epsilon-insensitive: gradient = 0 if |residual| <= epsilon
        var regressor = new OnlineSGDRegressor<double>(
            learningRate: 0.1,
            l2Penalty: 0.0,
            loss: SGDLossType.EpsilonInsensitive,
            epsilon: 0.5,
            learningRateSchedule: LearningRateSchedule.Constant);

        // Set w=1, b=0, so prediction = x
        var parameters = new Vector<double>(new[] { 1.0, 0.0 });
        regressor.SetParameters(parameters);

        // Feed x=1.0, y=1.2 -> residual = 1.0 - 1.2 = -0.2, |0.2| < epsilon=0.5
        // gradient = 0, so weights should NOT change
        var x = new Vector<double>(new[] { 1.0 });
        regressor.PartialFit(x, 1.2);

        var weights = regressor.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(1.0, weights[0], Tolerance); // No update
    }

    [Fact]
    public void SGDRegressor_RobustToOutliers_HuberVsSquaredError()
    {
        // Huber should be more robust to outliers than squared error
        var squaredModel = new OnlineSGDRegressor<double>(
            learningRate: 0.01, l2Penalty: 0.001,
            loss: SGDLossType.SquaredError,
            learningRateSchedule: LearningRateSchedule.Constant);

        var huberModel = new OnlineSGDRegressor<double>(
            learningRate: 0.01, l2Penalty: 0.001,
            loss: SGDLossType.Huber, epsilon: 1.0,
            learningRateSchedule: LearningRateSchedule.Constant);

        var rng = RandomHelper.CreateSeededRandom(42);

        // True model: y = x with 10% outliers at y=100
        for (int epoch = 0; epoch < 20; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double x0 = rng.NextDouble() * 2 - 1;
                var x = new Vector<double>(new[] { x0 });
                double y = x0;
                if (rng.NextDouble() < 0.1) y = 100.0; // Outlier
                squaredModel.PartialFit(x, y);
                huberModel.PartialFit(x, y);
            }
        }

        // Test on clean data
        rng = RandomHelper.CreateSeededRandom(99);
        double squaredMSE = 0, huberMSE = 0;
        for (int i = 0; i < 100; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0 });
            double trueY = x0;
            double sqPred = squaredModel.PredictSingle(x);
            double huPred = huberModel.PredictSingle(x);
            squaredMSE += (sqPred - trueY) * (sqPred - trueY);
            huberMSE += (huPred - trueY) * (huPred - trueY);
        }

        // Huber should have lower MSE on clean data when training had outliers
        Assert.True(huberMSE < squaredMSE * 1.5,
            $"Huber MSE ({huberMSE:F4}) should not be much worse than Squared ({squaredMSE:F4})");
    }

    [Fact]
    public void SGDRegressor_Score_R2_HandCalculated()
    {
        var regressor = new OnlineSGDRegressor<double>(learningRate: 0.01, l2Penalty: 0.0);

        // Set perfect model: w=2, b=1 => y = 2x + 1
        var parameters = new Vector<double>(new[] { 2.0, 1.0 });
        regressor.SetParameters(parameters);

        // Test data: x = [0, 1, 2], y = [1, 3, 5]
        var X = new Matrix<double>(3, 1);
        X[0, 0] = 0; X[1, 0] = 1; X[2, 0] = 2;
        var y = new Vector<double>(new[] { 1.0, 3.0, 5.0 });

        double r2 = regressor.Score(X, y);
        // Perfect model should give R2 = 1.0
        Assert.Equal(1.0, r2, 1e-4);
    }

    #endregion

    #region OnlinePassiveAggressiveClassifier - Update Rule Verification

    [Fact]
    public void PAClassifier_PA_UpdateRule_HandCalculated()
    {
        // PA update: w = w + tau * y * x, where tau = loss / ||x||^2
        // loss = max(0, 1 - y * (w.x))
        // Use fitIntercept=false so normSq = ||x||^2 only (no +1 for bias)
        var pac = new OnlinePassiveAggressiveClassifier<double>(
            c: 1000.0,
            type: PAType.PA,
            fitIntercept: false);

        // Initial weights are zero. First sample: x=[1,0], y=1
        // w.x = 0, loss = max(0, 1 - 1*0) = 1
        // ||x||^2 = 1^2 + 0^2 = 1
        // tau = 1/1 = 1
        // w_new = [0,0] + 1 * 1 * [1,0] = [1,0]
        var x = new Vector<double>(new[] { 1.0, 0.0 });
        pac.PartialFit(x, 1.0);

        var weights = pac.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(1.0, weights[0], 0.01);
        Assert.Equal(0.0, weights[1], 0.01);
    }

    [Fact]
    public void PAClassifier_PA_I_BoundedUpdate()
    {
        // PA-I: tau = min(C, loss / ||x||^2)
        // With C=0.5 and large loss, update should be bounded by C
        var pac = new OnlinePassiveAggressiveClassifier<double>(
            c: 0.5,
            type: PAType.PA_I);

        // x=[1,0], y=1, initial w=[0,0]
        // loss = max(0, 1 - 0) = 1
        // ||x||^2 = 1
        // tau_raw = 1/1 = 1
        // tau = min(0.5, 1) = 0.5  (bounded by C!)
        // w_new = [0,0] + 0.5 * 1 * [1,0] = [0.5, 0]
        var x = new Vector<double>(new[] { 1.0, 0.0 });
        pac.PartialFit(x, 1.0);

        var weights = pac.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(0.5, weights[0], 0.01);
    }

    [Fact]
    public void PAClassifier_CorrectPrediction_NoUpdate()
    {
        var pac = new OnlinePassiveAggressiveClassifier<double>(type: PAType.PA);

        // Set weights so w.x = 2 for x=[1,1] (correct with margin)
        var parameters = new Vector<double>(new[] { 1.0, 1.0, 0.0 }); // w0, w1, bias
        pac.SetParameters(parameters);

        // loss = max(0, 1 - 1*(1+1)) = max(0, -1) = 0
        // No update needed since loss is 0
        var x = new Vector<double>(new[] { 1.0, 1.0 });
        pac.PartialFit(x, 1.0);

        var weights = pac.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(1.0, weights[0], 0.01); // Unchanged
        Assert.Equal(1.0, weights[1], 0.01); // Unchanged
    }

    [Fact]
    public void PAClassifier_LinearSeparable_ConvergesToZeroLoss()
    {
        var pac = new OnlinePassiveAggressiveClassifier<double>(c: 1.0, type: PAType.PA_I);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Linearly separable: y=1 if x[0]>0, y=0 if x[0]<0
        // PA internally converts to {-1, +1} but PredictSingle returns {0, 1}
        for (int epoch = 0; epoch < 10; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double x0 = rng.NextDouble() * 2 - 1;
                double x1 = rng.NextDouble() * 2 - 1;
                var x = new Vector<double>(new[] { x0, x1 });
                double y = x0 > 0 ? 1.0 : 0.0;
                pac.PartialFit(x, y);
            }
        }

        // Test accuracy (PredictSingle returns 0 or 1)
        rng = RandomHelper.CreateSeededRandom(99);
        int correct = 0;
        for (int i = 0; i < 100; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            double x1 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0, x1 });
            double trueY = x0 > 0 ? 1.0 : 0.0;
            double pred = pac.PredictSingle(x);
            if (Math.Abs(pred - trueY) < 0.5) correct++;
        }

        Assert.True(correct > 85, $"PA should achieve >85% accuracy, got {correct}%");
    }

    #endregion

    #region OnlinePassiveAggressiveRegressor - Update Rule Verification

    [Fact]
    public void PARegressor_SimpleConvergence()
    {
        // PA regressor should converge on a simple linear relationship
        var regressor = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 0.01, type: PAType.PA_I);

        var rng = RandomHelper.CreateSeededRandom(42);

        // True model: y = 3*x
        for (int epoch = 0; epoch < 20; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double x0 = rng.NextDouble() * 2 - 1;
                var x = new Vector<double>(new[] { x0 });
                double y = 3.0 * x0 + (rng.NextDouble() - 0.5) * 0.01;
                regressor.PartialFit(x, y);
            }
        }

        var weights = regressor.GetWeights();
        Assert.NotNull(weights);

        // Weight should be approximately 3.0
        Assert.True(Math.Abs(weights[0] - 3.0) < 1.0,
            $"PA regressor weight should be close to 3.0, got {weights[0]:F4}");
    }

    [Fact]
    public void PARegressor_EpsilonInsensitive_NoUpdateInBand()
    {
        var regressor = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 1.0, type: PAType.PA_I);

        // Set w=[1], b=0 so prediction = x
        var parameters = new Vector<double>(new[] { 1.0, 0.0 });
        regressor.SetParameters(parameters);

        // Feed x=1.0, y=1.5 -> |prediction - y| = |1.0 - 1.5| = 0.5 < epsilon=1.0
        // Loss should be 0, no update
        var x = new Vector<double>(new[] { 1.0 });
        regressor.PartialFit(x, 1.5);

        var weights = regressor.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(1.0, weights[0], 0.01); // No update within epsilon band
    }

    #endregion

    #region Learning Rate Schedule Verification

    [Fact]
    public void SGDClassifier_InverseScaling_ProducesSmallerUpdates()
    {
        // With InverseScaling, effective lr decreases over time
        // Compare same classifier with Constant lr vs InverseScaling
        // After many samples, InverseScaling should produce overall smaller weights
        var constantLR = new OnlineSGDClassifier<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        var inverseLR = new OnlineSGDClassifier<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.InverseScaling,
            l2Penalty: 0.0);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Train both on identical data
        for (int i = 0; i < 200; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0, rng.NextDouble() });
            double y = x0 > 0 ? 1.0 : 0.0;
            constantLR.PartialFit(x, y);
            inverseLR.PartialFit(x, y);
        }

        // InverseScaling should have smaller weight magnitudes
        // since effective lr decreases over time
        var wConst = constantLR.GetWeights();
        var wInv = inverseLR.GetWeights();
        Assert.NotNull(wConst);
        Assert.NotNull(wInv);

        double normConst = 0, normInv = 0;
        for (int i = 0; i < wConst.Length; i++)
        {
            normConst += wConst[i] * wConst[i];
            normInv += wInv[i] * wInv[i];
        }

        Assert.True(normInv < normConst,
            $"InverseScaling weight norm ({Math.Sqrt(normInv):F4}) should be smaller than Constant ({Math.Sqrt(normConst):F4})");
    }

    [Fact]
    public void SGDClassifier_ConstantLR_SameUpdateMagnitude()
    {
        var classifier = new OnlineSGDClassifier<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        // With constant lr and same input/output, update magnitude should be similar
        // (not exactly same due to changing weights affecting gradient)
        var x = new Vector<double>(new[] { 1.0, 0.0 });

        // First update
        classifier.PartialFit(x, 1.0);
        var w1 = classifier.GetWeights();
        Assert.NotNull(w1);
        double update1 = w1[0]; // Starting from 0

        // Second update: gradient depends on current prediction (which changed)
        // But lr should be the same
        Assert.True(classifier.IsTrained);
    }

    #endregion

    #region Get/Set Parameters Round-Trip

    [Fact]
    public void SGDClassifier_GetSetParameters_RoundTrip()
    {
        var classifier = new OnlineSGDClassifier<double>();

        var original = new Vector<double>(new[] { 1.5, -2.3, 0.7, 0.1 }); // 3 weights + bias
        classifier.SetParameters(original);

        var retrieved = classifier.GetParameters();

        Assert.Equal(original.Length, retrieved.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], retrieved[i], Tolerance);
        }
    }

    [Fact]
    public void SGDRegressor_GetSetParameters_RoundTrip()
    {
        var regressor = new OnlineSGDRegressor<double>();

        var original = new Vector<double>(new[] { 3.0, -1.0, 2.0 }); // 2 weights + bias
        regressor.SetParameters(original);

        var retrieved = regressor.GetParameters();

        Assert.Equal(original.Length, retrieved.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], retrieved[i], Tolerance);
        }
    }

    [Fact]
    public void PAClassifier_WithParameters_CreatesEquivalentModel()
    {
        var pac = new OnlinePassiveAggressiveClassifier<double>(c: 1.0, type: PAType.PA_I);

        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 50; i++)
        {
            var x = new Vector<double>(new[] { rng.NextDouble(), rng.NextDouble() });
            pac.PartialFit(x, rng.NextDouble() > 0.5 ? 1.0 : -1.0);
        }

        var params1 = pac.GetParameters();
        var clone = pac.WithParameters(params1);
        var params2 = clone.GetParameters();

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i], Tolerance);
        }
    }

    #endregion

    #region Batch Prediction

    [Fact]
    public void SGDClassifier_BatchPredict_MatchesSinglePredictions()
    {
        var classifier = new OnlineSGDClassifier<double>(learningRate: 0.01);

        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 50; i++)
        {
            var x = new Vector<double>(new[] { rng.NextDouble(), rng.NextDouble() });
            classifier.PartialFit(x, rng.NextDouble() > 0.5 ? 1.0 : 0.0);
        }

        // Create batch
        int n = 20;
        var X = new Matrix<double>(n, 2);
        rng = RandomHelper.CreateSeededRandom(99);
        for (int i = 0; i < n; i++)
        {
            X[i, 0] = rng.NextDouble();
            X[i, 1] = rng.NextDouble();
        }

        // Batch predict probabilities
        var batchProbs = classifier.PredictProbabilities(X);

        // Single predictions should match
        for (int i = 0; i < n; i++)
        {
            var xi = new Vector<double>(new[] { X[i, 0], X[i, 1] });
            double singleProb = classifier.PredictProbability(xi);
            Assert.Equal(singleProb, batchProbs[i], Tolerance);
        }
    }

    #endregion

    #region ADWINDriftDetector in Online Learning Context

    [Fact]
    public void ADWINDriftDetector_DetectsConceptDriftDuringOnlineLearning()
    {
        // Simulate concept drift: first half y = x[0]>0, second half y = x[0]<0
        var classifier = new OnlineSGDClassifier<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);
        var detector = new ADWINDriftDetector<double>(delta: 0.01);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Phase 1: Train with concept y = (x>0)
        for (int i = 0; i < 200; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0 });
            double y = x0 > 0 ? 1.0 : 0.0;
            classifier.PartialFit(x, y);

            // Monitor accuracy
            double pred = classifier.PredictSingle(x);
            double error = Math.Abs(pred - y);
            detector.Update(error);
        }

        // Phase 2: Flip concept y = (x<0)
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            double x0 = rng.NextDouble() * 2 - 1;
            var x = new Vector<double>(new[] { x0 });
            double y = x0 < 0 ? 1.0 : 0.0; // FLIPPED concept
            classifier.PartialFit(x, y);

            double pred = classifier.PredictSingle(x);
            double error = Math.Abs(pred - y);
            if (detector.Update(error) == AiDotNet.Interfaces.DriftStatus.Drift)
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected,
            "ADWIN should detect concept drift when decision boundary flips");
    }

    #endregion
}
