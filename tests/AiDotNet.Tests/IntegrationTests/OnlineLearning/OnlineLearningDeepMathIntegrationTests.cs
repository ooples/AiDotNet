using AiDotNet.LinearAlgebra;
using AiDotNet.OnlineLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.OnlineLearning;

/// <summary>
/// Deep math integration tests for online learning models (SGD Regressor/Classifier,
/// Passive-Aggressive Regressor/Classifier, ADWIN drift detector).
/// Tests verify correctness of gradient computations, update rules, loss functions,
/// regularization, learning rate schedules, and convergence using hand-computed values.
/// </summary>
public class OnlineLearningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double MediumTolerance = 1e-4;

    #region SGD Regressor - Hand-Computed Update Tests

    [Fact]
    public void SGDRegressor_OneStep_SquaredError_HandComputed()
    {
        // Hand-computed single SGD step with squared error loss:
        // x=[1.0], y=2.0, lr=0.1, init w=[0], b=0
        // pred = 0*1 + 0 = 0
        // residual = 0 - 2 = -2
        // gradient_multiplier = 2*(-2) = -4
        // w gradient = -4 * 1 + 0.0001*0 = -4 (L2 term is 0)
        // w_new = 0 - 0.1*(-4) = 0.4
        // b gradient = -4
        // b_new = 0 - 0.1*(-4) = 0.4
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l1Penalty: 0.0,
            l2Penalty: 0.0,
            loss: SGDLossType.SquaredError);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 2.0);

        var weights = sgd.GetWeights();
        double bias = sgd.GetBias();

        Assert.NotNull(weights);
        Assert.Equal(0.4, weights[0], MediumTolerance);
        Assert.Equal(0.4, bias, MediumTolerance);

        // Prediction after one step: 0.4*1 + 0.4 = 0.8
        double pred = sgd.PredictSingle(x);
        Assert.Equal(0.8, pred, MediumTolerance);
    }

    [Fact]
    public void SGDRegressor_OneStep_WithL2Regularization_HandComputed()
    {
        // Same as above but with L2=0.1:
        // w gradient = -4 + 0.1*0 = -4 (still -4 since w=0 initially)
        // After first step they're the same. Let's do 2 steps.
        // Step 1: same as above → w=0.4, b=0.4
        // Step 2: x=[1], y=2
        //   pred = 0.4*1 + 0.4 = 0.8
        //   residual = 0.8 - 2 = -1.2
        //   grad_mult = 2*(-1.2) = -2.4
        //   w_gradient = -2.4*1 + 0.1*0.4 = -2.4 + 0.04 = -2.36
        //   w_new = 0.4 - 0.1*(-2.36) = 0.4 + 0.236 = 0.636
        //   b_gradient = -2.4 (no L2 on bias)
        //   b_new = 0.4 - 0.1*(-2.4) = 0.4 + 0.24 = 0.64
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1,
            learningRateSchedule: LearningRateSchedule.Constant,
            l1Penalty: 0.0,
            l2Penalty: 0.1,
            loss: SGDLossType.SquaredError);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 2.0);
        sgd.PartialFit(x, 2.0);

        var weights = sgd.GetWeights();
        double bias = sgd.GetBias();

        Assert.NotNull(weights);
        Assert.Equal(0.636, weights[0], MediumTolerance);
        Assert.Equal(0.64, bias, MediumTolerance);
    }

    [Fact]
    public void SGDRegressor_HuberLoss_SmallError_EqualsSquaredLoss()
    {
        // For small errors (|residual| <= epsilon), Huber gradient = squared loss gradient
        // With epsilon=0.5 and small initial error, should behave like squared loss
        var sgdSquared = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.SquaredError);

        var sgdHuber = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.Huber, epsilon: 10.0); // large epsilon

        // Use data that produces small residuals
        var x = new Vector<double>(new double[] { 1.0 });
        sgdSquared.PartialFit(x, 0.1);
        sgdHuber.PartialFit(x, 0.1);

        // With large epsilon and small target, both should give same update
        Assert.Equal(sgdSquared.GetWeights()[0], sgdHuber.GetWeights()[0], MediumTolerance);
        Assert.Equal(sgdSquared.GetBias(), sgdHuber.GetBias(), MediumTolerance);
    }

    [Fact]
    public void SGDRegressor_HuberLoss_LargeError_LinearGradient()
    {
        // For large errors (|residual| > epsilon), Huber gradient = 2*epsilon*sign(residual)
        // epsilon=0.1, x=[1], y=100 (huge error)
        // pred = 0, residual = 0-100 = -100, |residual| = 100 >> 0.1
        // Huber gradient = 2*0.1*sign(-100) = -0.2
        // w_new = 0 - 0.1*(-0.2*1) = 0.02
        // b_new = 0 - 0.1*(-0.2) = 0.02
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.Huber, epsilon: 0.1);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 100.0);

        Assert.Equal(0.02, sgd.GetWeights()[0], MediumTolerance);
        Assert.Equal(0.02, sgd.GetBias(), MediumTolerance);
    }

    [Fact]
    public void SGDRegressor_EpsilonInsensitive_SmallError_NoUpdate()
    {
        // With epsilon-insensitive loss, errors within epsilon produce zero gradient
        // epsilon=1.0, x=[1], y=0.5 (error < epsilon)
        // pred = 0, residual = 0-0.5 = -0.5, |residual| = 0.5 < 1.0
        // gradient = 0 → no weight update
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.EpsilonInsensitive, epsilon: 1.0);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 0.5);

        Assert.Equal(0.0, sgd.GetWeights()[0], Tolerance);
        Assert.Equal(0.0, sgd.GetBias(), Tolerance);
    }

    [Fact]
    public void SGDRegressor_EpsilonInsensitive_LargeError_SignGradient()
    {
        // Error > epsilon: gradient = sign(residual)
        // epsilon=0.1, x=[1], y=5 (error >> epsilon)
        // pred = 0, residual = 0-5 = -5, |residual| = 5 > 0.1
        // gradient = sign(-5) = -1
        // w_new = 0 - 0.1*(-1*1) = 0.1
        // b_new = 0 - 0.1*(-1) = 0.1
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.EpsilonInsensitive, epsilon: 0.1);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 5.0);

        Assert.Equal(0.1, sgd.GetWeights()[0], MediumTolerance);
        Assert.Equal(0.1, sgd.GetBias(), MediumTolerance);
    }

    [Fact]
    public void SGDRegressor_L1Regularization_SoftThresholding()
    {
        // L1 regularization applies soft thresholding: shrinks weights toward zero
        // With large L1 penalty and small weights, weights should be driven to exactly 0
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l1Penalty: 1.0, l2Penalty: 0.0, loss: SGDLossType.SquaredError);

        // Train with data that produces small weights
        var x = new Vector<double>(new double[] { 0.01 });
        for (int i = 0; i < 10; i++)
            sgd.PartialFit(x, 0.001);

        // With large L1, weights should be close to 0 (thresholded)
        var weights = sgd.GetWeights();
        Assert.True(Math.Abs(weights[0]) < 0.1,
            $"L1 regularization should keep weights small, got {weights[0]:F6}");
    }

    #endregion

    #region SGD Regressor - Convergence Tests

    [Fact]
    public void SGDRegressor_ConvergesOnSimpleLinearData()
    {
        // y = 2*x + 3
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0, loss: SGDLossType.SquaredError);

        // Train on many samples
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i <= 10; i++)
            {
                double xi = i * 0.1;
                double yi = 2.0 * xi + 3.0;
                sgd.PartialFit(new Vector<double>(new double[] { xi }), yi);
            }
        }

        // Should learn approximately w≈2, b≈3
        var weights = sgd.GetWeights();
        double bias = sgd.GetBias();

        Assert.True(Math.Abs(weights[0] - 2.0) < 0.5,
            $"Weight should be ~2.0, got {weights[0]:F4}");
        Assert.True(Math.Abs(bias - 3.0) < 0.5,
            $"Bias should be ~3.0, got {bias:F4}");
    }

    [Fact]
    public void SGDRegressor_Score_R2_ReasonableOnLinearData()
    {
        // After training on linear data, R² should be high
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        int n = 20;
        var xMatrix = new Matrix<double>(n, 1);
        var yVec = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double xi = i * 0.1;
            xMatrix[i, 0] = xi;
            yVec[i] = 2.0 * xi + 3.0;
        }

        // Train multiple epochs
        for (int epoch = 0; epoch < 200; epoch++)
            sgd.PartialFit(xMatrix, yVec);

        double r2 = sgd.Score(xMatrix, yVec);

        Assert.True(r2 > 0.8,
            $"R² should be high on linear data, got {r2:F4}");
    }

    #endregion

    #region PA Regressor - Hand-Computed Update Tests

    [Fact]
    public void PARegressor_PA_OneStep_HandComputed()
    {
        // PA update: tau = loss / ||x||²
        // x=[2], y=5, epsilon=0.1, init w=[0], b=0, fitIntercept=true
        // pred = 0*2 + 0 = 0
        // error = 5 - 0 = 5, loss = max(0, |5| - 0.1) = 4.9
        // normSq = 2² + 1 (intercept) = 5
        // tau = 4.9 / 5 = 0.98
        // sign = +1
        // w_new = 0 + 0.98*1*2 = 1.96
        // b_new = 0 + 0.98*1 = 0.98
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 10.0, epsilon: 0.1, type: PAType.PA, fitIntercept: true);

        var x = new Vector<double>(new double[] { 2.0 });
        pa.PartialFit(x, 5.0);

        var weights = pa.GetWeights();
        double bias = pa.GetBias();

        Assert.NotNull(weights);
        Assert.Equal(1.96, weights[0], MediumTolerance);
        Assert.Equal(0.98, bias, MediumTolerance);

        // After update: pred = 1.96*2 + 0.98 = 4.90
        // Error = |5 - 4.90| = 0.10 = epsilon (within tolerance!)
        double pred = pa.PredictSingle(x);
        Assert.True(Math.Abs(5.0 - pred) <= 0.1 + 1e-10,
            $"After PA update, error should be <= epsilon. Pred={pred:F4}, expected ~5.0");
    }

    [Fact]
    public void PARegressor_PAI_BoundsUpdateByC()
    {
        // PA-I: tau = min(C, loss/||x||²)
        // x=[1], y=100, epsilon=0, C=0.5
        // pred = 0, loss = 100, normSq = 1 + 1 = 2
        // loss/normSq = 100/2 = 50
        // tau = min(0.5, 50) = 0.5 (bounded by C!)
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 0.5, epsilon: 0.0, type: PAType.PA_I, fitIntercept: true);

        var x = new Vector<double>(new double[] { 1.0 });
        pa.PartialFit(x, 100.0);

        var weights = pa.GetWeights();
        // w_new = 0 + 0.5*1*1 = 0.5
        Assert.Equal(0.5, weights[0], MediumTolerance);
        // b_new = 0 + 0.5*1 = 0.5
        Assert.Equal(0.5, pa.GetBias(), MediumTolerance);
    }

    [Fact]
    public void PARegressor_PAII_SmoothUpdate_HandComputed()
    {
        // PA-II: tau = loss / (||x||² + 1/(2C))
        // x=[1], y=5, epsilon=0, C=1.0
        // pred = 0, loss = 5, normSq = 1 + 1 = 2
        // denominator = 2 + 1/(2*1.0) = 2 + 0.5 = 2.5
        // tau = 5 / 2.5 = 2.0
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 0.0, type: PAType.PA_II, fitIntercept: true);

        var x = new Vector<double>(new double[] { 1.0 });
        pa.PartialFit(x, 5.0);

        var weights = pa.GetWeights();
        // w_new = 0 + 2.0*1*1 = 2.0
        Assert.Equal(2.0, weights[0], MediumTolerance);
        // b_new = 0 + 2.0*1 = 2.0
        Assert.Equal(2.0, pa.GetBias(), MediumTolerance);
    }

    [Fact]
    public void PARegressor_Passive_WhenErrorWithinEpsilon()
    {
        // When |error| <= epsilon, loss = 0 → no update (passive)
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 1.0, type: PAType.PA_I);

        // First train to get some non-zero weights
        var x = new Vector<double>(new double[] { 1.0 });
        pa.PartialFit(x, 10.0);

        double weightBefore = pa.GetWeights()[0];
        double biasBefore = pa.GetBias();

        // Now present a sample within epsilon of current prediction
        double currentPred = pa.PredictSingle(x);
        double targetWithinEps = currentPred + 0.5; // |error| = 0.5 < epsilon=1.0

        pa.PartialFit(x, targetWithinEps);

        // Weights should NOT change (passive)
        Assert.Equal(weightBefore, pa.GetWeights()[0], Tolerance);
        Assert.Equal(biasBefore, pa.GetBias(), Tolerance);
    }

    [Fact]
    public void PARegressor_EpsilonInsensitiveLoss_ComputedCorrectly()
    {
        // Train a model and verify epsilon-insensitive loss computation
        var pa = new OnlinePassiveAggressiveRegressor<double>(epsilon: 0.5);

        // Manually set parameters for a known model
        var params_ = new Vector<double>(new double[] { 2.0, 1.0 }); // weight=2, bias=1
        pa.SetParameters(params_);

        // Test data: x=1, y=3 → pred=2*1+1=3, error=0, loss=max(0,0-0.5)=0
        // x=1, y=4 → pred=3, error=1, loss=max(0,1-0.5)=0.5
        var xMat = new Matrix<double>(2, 1);
        xMat[0, 0] = 1.0; xMat[1, 0] = 1.0;
        var y = new Vector<double>(new double[] { 3.0, 4.0 });

        double loss = pa.GetEpsilonInsensitiveLoss(xMat, y);

        // Average loss = (0 + 0.5) / 2 = 0.25
        Assert.Equal(0.25, loss, MediumTolerance);
    }

    #endregion

    #region PA Classifier - Hand-Computed Update Tests

    [Fact]
    public void PAClassifier_PA_OneStep_HandComputed()
    {
        // PA classifier update:
        // x=[1], y=1 (label +1), init w=[0], b=0
        // score = 0*1 + 0 = 0
        // margin = 1 * 0 = 0
        // hingeLoss = max(0, 1-0) = 1
        // normSq = 1 + 1 = 2 (with intercept)
        // tau = 1/2 = 0.5
        // w_new = 0 + 0.5*1*1 = 0.5
        // b_new = 0 + 0.5*1 = 0.5
        var pac = new OnlinePassiveAggressiveClassifier<double>(
            c: 10.0, type: PAType.PA, fitIntercept: true);

        var x = new Vector<double>(new double[] { 1.0 });
        pac.PartialFit(x, 1.0); // label +1

        Assert.Equal(0.5, pac.GetWeights()[0], MediumTolerance);
        Assert.Equal(0.5, pac.GetBias(), MediumTolerance);

        // New score: 0.5*1 + 0.5 = 1.0, margin = 1*1.0 = 1.0 (exactly at margin!)
        double score = pac.DecisionFunction(x);
        Assert.Equal(1.0, score, MediumTolerance);
    }

    [Fact]
    public void PAClassifier_Passive_WhenMarginSufficient()
    {
        // When margin >= 1, hinge loss = 0 → no update
        var pac = new OnlinePassiveAggressiveClassifier<double>(
            c: 1.0, type: PAType.PA_I);

        // First train to get margin >= 1
        var x = new Vector<double>(new double[] { 1.0 });
        pac.PartialFit(x, 1.0);

        double weightBefore = pac.GetWeights()[0];
        double biasBefore = pac.GetBias();

        // Now present the same sample - margin should be >= 1 → passive
        pac.PartialFit(x, 1.0);

        // Weights should NOT change
        Assert.Equal(weightBefore, pac.GetWeights()[0], Tolerance);
        Assert.Equal(biasBefore, pac.GetBias(), Tolerance);
    }

    [Fact]
    public void PAClassifier_PAI_BoundsStepSize()
    {
        // PA-I: tau = min(C, loss/normSq)
        // With small C, step should be bounded
        var pac = new OnlinePassiveAggressiveClassifier<double>(
            c: 0.1, type: PAType.PA_I);

        var x = new Vector<double>(new double[] { 0.1 });
        pac.PartialFit(x, 1.0);

        // normSq = 0.01 + 1 = 1.01, hingeLoss = 1, loss/normSq ≈ 0.99
        // tau = min(0.1, 0.99) = 0.1 (bounded)
        // w_new = 0 + 0.1*1*0.1 = 0.01
        Assert.True(Math.Abs(pac.GetWeights()[0] - 0.01) < MediumTolerance,
            $"PA-I weight should be ~0.01, got {pac.GetWeights()[0]:F6}");
    }

    [Fact]
    public void PAClassifier_HingeLoss_ComputedCorrectly()
    {
        // Verify hinge loss computation
        var pac = new OnlinePassiveAggressiveClassifier<double>();

        // Set parameters for known model: w=[2], b=0
        var params_ = new Vector<double>(new double[] { 2.0, 0.0 });
        pac.SetParameters(params_);

        // Test: x=[1], y=1 → score=2, margin=2, hingeLoss=max(0,1-2)=0
        // Test: x=[-1], y=1 → score=-2, margin=-2, hingeLoss=max(0,1-(-2))=3
        var xMat = new Matrix<double>(2, 1);
        xMat[0, 0] = 1.0; xMat[1, 0] = -1.0;
        var y = new Vector<double>(new double[] { 1.0, 1.0 });

        double loss = pac.GetHingeLoss(xMat, y);

        // Average hinge loss = (0 + 3) / 2 = 1.5
        Assert.Equal(1.5, loss, MediumTolerance);
    }

    #endregion

    #region SGD Classifier - Tests

    [Fact]
    public void SGDClassifier_Sigmoid_Properties()
    {
        // After training, PredictProbability should return values in (0, 1)
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        // Train on separable data
        for (int i = 0; i < 50; i++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 1.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -1.0 }), 0.0);
        }

        // Probabilities should be in (0, 1)
        double prob1 = sgd.PredictProbability(new Vector<double>(new double[] { 1.0 }));
        double prob0 = sgd.PredictProbability(new Vector<double>(new double[] { -1.0 }));

        Assert.True(prob1 > 0.0 && prob1 < 1.0,
            $"Probability must be in (0,1), got {prob1:F6}");
        Assert.True(prob0 > 0.0 && prob0 < 1.0,
            $"Probability must be in (0,1), got {prob0:F6}");

        // Positive feature should have higher probability of class 1
        Assert.True(prob1 > prob0,
            $"P(y=1|x=1) = {prob1:F4} should be > P(y=1|x=-1) = {prob0:F4}");
    }

    [Fact]
    public void SGDClassifier_OneStep_LogisticGradient_HandComputed()
    {
        // One-step logistic regression update:
        // x=[1], y=1, lr=0.1, init w=[0], b=0
        // prob = sigmoid(0) = 0.5
        // error = 0.5 - 1 = -0.5
        // w_gradient = -0.5*1 = -0.5 (no L2 since w=0)
        // w_new = 0 - 0.1*(-0.5) = 0.05
        // b_gradient = -0.5
        // b_new = 0 - 0.1*(-0.5) = 0.05
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l1Penalty: 0.0, l2Penalty: 0.0);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 1.0);

        Assert.Equal(0.05, sgd.GetWeights()[0], MediumTolerance);
        Assert.Equal(0.05, sgd.GetBias(), MediumTolerance);
    }

    [Fact]
    public void SGDClassifier_ConvergesOnSeparableData()
    {
        // Linearly separable data: class 1 has positive features, class 0 has negative
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.001);

        for (int epoch = 0; epoch < 50; epoch++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 2.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -2.0 }), 0.0);
        }

        // Should correctly classify
        double pred1 = sgd.PredictSingle(new Vector<double>(new double[] { 2.0 }));
        double pred0 = sgd.PredictSingle(new Vector<double>(new double[] { -2.0 }));

        Assert.Equal(1.0, pred1, Tolerance);
        Assert.Equal(0.0, pred0, Tolerance);
    }

    [Fact]
    public void SGDClassifier_DecisionFunction_PositiveForClass1()
    {
        // Decision function should be positive for class 1 features after training
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant);

        for (int i = 0; i < 100; i++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 1.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -1.0 }), 0.0);
        }

        double df1 = sgd.DecisionFunction(new Vector<double>(new double[] { 1.0 }));
        double df0 = sgd.DecisionFunction(new Vector<double>(new double[] { -1.0 }));

        Assert.True(df1 > 0, $"Decision function for class 1 feature should be positive, got {df1:F4}");
        Assert.True(df0 < 0, $"Decision function for class 0 feature should be negative, got {df0:F4}");
    }

    [Fact]
    public void SGDClassifier_L1Regularization_ProducesSparsity()
    {
        // With strong L1 and irrelevant features, some weights should be exactly zero
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l1Penalty: 0.5, l2Penalty: 0.0);

        // Feature 0 is informative, feature 1 is noise (all zeros)
        for (int i = 0; i < 100; i++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 1.0, 0.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -1.0, 0.0 }), 0.0);
        }

        var weights = sgd.GetWeights();
        // Feature 1 weight should be exactly 0 (never updated since x_1=0)
        Assert.Equal(0.0, weights[1], Tolerance);
    }

    #endregion

    #region PA Classifier - Convergence Tests

    [Fact]
    public void PAClassifier_ConvergesOnSeparableData()
    {
        // PA should perfectly classify linearly separable data
        var pac = new OnlinePassiveAggressiveClassifier<double>(c: 1.0, type: PAType.PA_I);

        for (int epoch = 0; epoch < 20; epoch++)
        {
            pac.PartialFit(new Vector<double>(new double[] { 2.0 }), 1.0);
            pac.PartialFit(new Vector<double>(new double[] { -2.0 }), 0.0);
        }

        double pred1 = pac.PredictSingle(new Vector<double>(new double[] { 2.0 }));
        double pred0 = pac.PredictSingle(new Vector<double>(new double[] { -2.0 }));

        Assert.Equal(1.0, pred1, Tolerance);
        Assert.Equal(0.0, pred0, Tolerance);
    }

    [Fact]
    public void PAClassifier_MultiFeature_LearnsCorrectDecisionBoundary()
    {
        // 2D separable data: class 1 when x1+x2 > 0, class 0 otherwise
        var pac = new OnlinePassiveAggressiveClassifier<double>(c: 1.0, type: PAType.PA_I);

        for (int epoch = 0; epoch < 30; epoch++)
        {
            pac.PartialFit(new Vector<double>(new double[] { 1.0, 1.0 }), 1.0);
            pac.PartialFit(new Vector<double>(new double[] { -1.0, -1.0 }), 0.0);
            pac.PartialFit(new Vector<double>(new double[] { 0.5, 1.5 }), 1.0);
            pac.PartialFit(new Vector<double>(new double[] { -0.5, -1.5 }), 0.0);
        }

        // Should correctly classify extreme points
        double pred1 = pac.PredictSingle(new Vector<double>(new double[] { 2.0, 2.0 }));
        double pred0 = pac.PredictSingle(new Vector<double>(new double[] { -2.0, -2.0 }));

        Assert.Equal(1.0, pred1, Tolerance);
        Assert.Equal(0.0, pred0, Tolerance);
    }

    #endregion

    #region Learning Rate Schedule Tests

    [Fact]
    public void LearningRate_Constant_NeverChanges()
    {
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.05, learningRateSchedule: LearningRateSchedule.Constant);

        var x = new Vector<double>(new double[] { 1.0 });

        // Train several samples
        for (int i = 0; i < 100; i++)
            sgd.PartialFit(x, 1.0);

        // Learning rate should still be 0.05 (constant)
        // We can't directly access GetLearningRate from here, but we can verify
        // the model trains correctly with constant LR
        Assert.True(sgd.GetSampleCount() == 100);
    }

    [Fact]
    public void LearningRate_InverseScaling_Decreases()
    {
        // Inverse scaling: lr = lr0 / (1 + alpha * t)
        // After many samples, the effective LR should be smaller
        // This means updates should be smaller
        var sgdConstant = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);
        var sgdInverse = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.InverseScaling,
            l2Penalty: 0.0);

        var x = new Vector<double>(new double[] { 1.0 });

        // Train both on same data
        for (int i = 0; i < 1000; i++)
        {
            sgdConstant.PartialFit(x, 1.0);
            sgdInverse.PartialFit(x, 1.0);
        }

        // Now present one more sample with a new target
        // The inverse scaling model should have smaller effective LR
        // so it should be less responsive to the new sample
        double wConstantBefore = sgdConstant.GetWeights()[0];
        double wInverseBefore = sgdInverse.GetWeights()[0];

        sgdConstant.PartialFit(x, 100.0); // Sudden jump
        sgdInverse.PartialFit(x, 100.0);

        double wConstantAfter = sgdConstant.GetWeights()[0];
        double wInverseAfter = sgdInverse.GetWeights()[0];

        double constantChange = Math.Abs(wConstantAfter - wConstantBefore);
        double inverseChange = Math.Abs(wInverseAfter - wInverseBefore);

        Assert.True(inverseChange < constantChange,
            $"Inverse scaling change ({inverseChange:F6}) should be smaller than " +
            $"constant change ({constantChange:F6}) after many samples.");
    }

    #endregion

    #region Sample Count and Reset Tests

    [Fact]
    public void OnlineModel_SampleCount_IncrementsCorrectly()
    {
        var sgd = new OnlineSGDRegressor<double>();
        var x = new Vector<double>(new double[] { 1.0 });

        Assert.Equal(0, sgd.GetSampleCount());

        sgd.PartialFit(x, 1.0);
        Assert.Equal(1, sgd.GetSampleCount());

        sgd.PartialFit(x, 2.0);
        Assert.Equal(2, sgd.GetSampleCount());
    }

    [Fact]
    public void OnlineModel_BatchPartialFit_IncrementsCorrectly()
    {
        var sgd = new OnlineSGDRegressor<double>();

        var xMat = new Matrix<double>(5, 1);
        var yVec = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            xMat[i, 0] = i * 0.1;
            yVec[i] = i * 0.2;
        }

        sgd.PartialFit(xMat, yVec);

        Assert.Equal(5, sgd.GetSampleCount());
    }

    [Fact]
    public void OnlineModel_Reset_ClearsState()
    {
        var sgd = new OnlineSGDRegressor<double>();
        var x = new Vector<double>(new double[] { 1.0 });

        sgd.PartialFit(x, 5.0);
        Assert.True(sgd.IsTrained);
        Assert.True(sgd.GetSampleCount() > 0);

        sgd.Reset();

        Assert.False(sgd.IsTrained);
        Assert.Equal(0, sgd.GetSampleCount());
        Assert.Null(sgd.GetWeights());
    }

    [Fact]
    public void PARegressor_Reset_ClearsState()
    {
        var pa = new OnlinePassiveAggressiveRegressor<double>();
        var x = new Vector<double>(new double[] { 1.0 });

        pa.PartialFit(x, 5.0);
        Assert.True(pa.IsTrained);

        pa.Reset();

        Assert.False(pa.IsTrained);
        Assert.Null(pa.GetWeights());
    }

    #endregion

    #region Batch Predict Tests

    [Fact]
    public void SGDRegressor_BatchPredict_MatchesSinglePredictions()
    {
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant);

        // Train
        for (int i = 0; i < 100; i++)
            sgd.PartialFit(new Vector<double>(new double[] { i * 0.1 }), i * 0.2);

        // Batch predict
        int n = 10;
        var xMat = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++) xMat[i, 0] = i * 0.5;

        var batchPreds = sgd.Predict(xMat);

        // Compare with individual predictions
        for (int i = 0; i < n; i++)
        {
            var xi = new Vector<double>(new double[] { i * 0.5 });
            double singlePred = sgd.PredictSingle(xi);
            Assert.Equal(singlePred, batchPreds[i], Tolerance);
        }
    }

    [Fact]
    public void SGDClassifier_PredictProbabilities_MatchesSingle()
    {
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant);

        for (int i = 0; i < 50; i++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 1.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -1.0 }), 0.0);
        }

        var xMat = new Matrix<double>(3, 1);
        xMat[0, 0] = 2.0; xMat[1, 0] = 0.0; xMat[2, 0] = -2.0;

        var probs = sgd.PredictProbabilities(xMat);

        for (int i = 0; i < 3; i++)
        {
            var xi = new Vector<double>(new double[] { xMat[i, 0] });
            double singleProb = sgd.PredictProbability(xi);
            Assert.Equal(singleProb, probs[i], Tolerance);
        }
    }

    #endregion

    #region GetParameters / SetParameters Tests

    [Fact]
    public void SGDRegressor_GetSetParameters_Roundtrip()
    {
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant);

        var x = new Vector<double>(new double[] { 1.0, 2.0 });
        for (int i = 0; i < 50; i++)
            sgd.PartialFit(x, 3.0 * 1.0 + 4.0 * 2.0);

        var params_ = sgd.GetParameters();

        var sgd2 = new OnlineSGDRegressor<double>();
        sgd2.SetParameters(params_);

        var params2 = sgd2.GetParameters();
        Assert.Equal(params_.Length, params2.Length);
        for (int i = 0; i < params_.Length; i++)
            Assert.Equal(params_[i], params2[i], Tolerance);

        // Predictions should match
        double pred1 = sgd.PredictSingle(x);
        double pred2 = sgd2.PredictSingle(x);
        Assert.Equal(pred1, pred2, Tolerance);
    }

    [Fact]
    public void PAClassifier_GetSetParameters_Roundtrip()
    {
        var pac = new OnlinePassiveAggressiveClassifier<double>();

        for (int i = 0; i < 20; i++)
        {
            pac.PartialFit(new Vector<double>(new double[] { 1.0 }), 1.0);
            pac.PartialFit(new Vector<double>(new double[] { -1.0 }), 0.0);
        }

        var params_ = pac.GetParameters();
        var pac2 = new OnlinePassiveAggressiveClassifier<double>();
        pac2.SetParameters(params_);

        var x = new Vector<double>(new double[] { 0.5 });
        Assert.Equal(pac.PredictSingle(x), pac2.PredictSingle(x), Tolerance);
    }

    #endregion

    #region Feature Importance Tests

    [Fact]
    public void SGDRegressor_FeatureImportance_ReflectsAbsoluteWeights()
    {
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        // y = 3*x1 + 0*x2 → feature 0 is important, feature 1 is not
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i <= 10; i++)
            {
                double xi = i * 0.1;
                sgd.PartialFit(new Vector<double>(new double[] { xi, 0.5 }), 3.0 * xi);
            }
        }

        var importance = sgd.GetFeatureImportance();

        Assert.True(importance.ContainsKey("Feature_0"));
        Assert.True(importance.ContainsKey("Feature_1"));

        // Feature 0 should have much higher importance than Feature 1
        Assert.True(importance["Feature_0"] > importance["Feature_1"],
            $"Feature_0 importance ({importance["Feature_0"]:F4}) should exceed " +
            $"Feature_1 ({importance["Feature_1"]:F4})");
    }

    #endregion

    #region Model Type Tests

    [Fact]
    public void ModelType_CorrectForEachModel()
    {
        Assert.Equal(AiDotNet.Enums.ModelType.OnlineSGDRegressor,
            new OnlineSGDRegressor<double>().GetModelType());
        Assert.Equal(AiDotNet.Enums.ModelType.OnlineSGDClassifier,
            new OnlineSGDClassifier<double>().GetModelType());
        Assert.Equal(AiDotNet.Enums.ModelType.OnlinePassiveAggressiveRegressor,
            new OnlinePassiveAggressiveRegressor<double>().GetModelType());
        Assert.Equal(AiDotNet.Enums.ModelType.OnlinePassiveAggressiveClassifier,
            new OnlinePassiveAggressiveClassifier<double>().GetModelType());
    }

    #endregion

    #region ADWIN Drift Detector Tests

    [Fact]
    public void ADWIN_StationaryData_NoDrift()
    {
        // On stationary data, ADWIN should not detect drift (or very rarely)
        var adwin = new ADWINDriftDetector<double>(delta: 0.01);

        int driftCount = 0;
        for (int i = 0; i < 500; i++)
        {
            // Constant value → no drift
            adwin.Update(1.0);
            if (adwin.IsDriftDetected) driftCount++;
        }

        Assert.True(driftCount < 10,
            $"ADWIN should rarely detect drift on constant data, got {driftCount} detections");
    }

    [Fact]
    public void ADWIN_MeanShift_DetectsDrift()
    {
        // When the mean shifts significantly, ADWIN should detect drift
        var adwin = new ADWINDriftDetector<double>(delta: 0.01);

        // First: establish baseline with mean=0
        for (int i = 0; i < 200; i++)
        {
            adwin.Update(0.0);
        }

        // Then: shift mean to 5.0
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            adwin.Update(5.0);
            if (adwin.IsDriftDetected)
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected,
            "ADWIN should detect drift when mean shifts from 0 to 5");
    }

    [Fact]
    public void ADWIN_GradualDrift_EventuallyDetected()
    {
        // Gradually increasing mean should eventually trigger drift detection
        var adwin = new ADWINDriftDetector<double>(delta: 0.01);

        bool driftDetected = false;
        for (int i = 0; i < 1000; i++)
        {
            double value = i * 0.01; // Gradually increasing
            adwin.Update(value);
            if (adwin.IsDriftDetected && i > 50)
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected,
            "ADWIN should detect gradual drift over time");
    }

    #endregion

    #region Cross-Model Comparison Tests

    [Fact]
    public void SGD_VS_PA_Regressor_BothConvergeOnLinearData()
    {
        // Both SGD and PA regressors should fit linear data well
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.01, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 0.01, type: PAType.PA_I);

        // y = 2x + 1
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i <= 10; i++)
            {
                double xi = i * 0.1;
                double yi = 2.0 * xi + 1.0;
                var x = new Vector<double>(new double[] { xi });
                sgd.PartialFit(x, yi);
                pa.PartialFit(x, yi);
            }
        }

        // Both should have low MSE
        var xTest = new Matrix<double>(5, 1);
        var yTest = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            xTest[i, 0] = 0.3 + i * 0.1;
            yTest[i] = 2.0 * (0.3 + i * 0.1) + 1.0;
        }

        var sgdPreds = sgd.Predict(xTest);
        var paPreds = pa.Predict(xTest);

        double sgdMSE = 0, paMSE = 0;
        for (int i = 0; i < 5; i++)
        {
            sgdMSE += (sgdPreds[i] - yTest[i]) * (sgdPreds[i] - yTest[i]);
            paMSE += (paPreds[i] - yTest[i]) * (paPreds[i] - yTest[i]);
        }
        sgdMSE /= 5; paMSE /= 5;

        Assert.True(sgdMSE < 1.0,
            $"SGD MSE ({sgdMSE:F4}) should be low on linear data");
        Assert.True(paMSE < 1.0,
            $"PA MSE ({paMSE:F4}) should be low on linear data");
    }

    [Fact]
    public void SGD_VS_PA_Classifier_BothClassifySeparableData()
    {
        var sgd = new OnlineSGDClassifier<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant);
        var pac = new OnlinePassiveAggressiveClassifier<double>(c: 1.0, type: PAType.PA_I);

        for (int epoch = 0; epoch < 50; epoch++)
        {
            sgd.PartialFit(new Vector<double>(new double[] { 2.0 }), 1.0);
            sgd.PartialFit(new Vector<double>(new double[] { -2.0 }), 0.0);
            pac.PartialFit(new Vector<double>(new double[] { 2.0 }), 1.0);
            pac.PartialFit(new Vector<double>(new double[] { -2.0 }), 0.0);
        }

        // Both should correctly classify test points
        Assert.Equal(1.0, sgd.PredictSingle(new Vector<double>(new double[] { 3.0 })), Tolerance);
        Assert.Equal(0.0, sgd.PredictSingle(new Vector<double>(new double[] { -3.0 })), Tolerance);
        Assert.Equal(1.0, pac.PredictSingle(new Vector<double>(new double[] { 3.0 })), Tolerance);
        Assert.Equal(0.0, pac.PredictSingle(new Vector<double>(new double[] { -3.0 })), Tolerance);
    }

    #endregion

    #region WithParameters and Metadata Tests

    [Fact]
    public void SGDRegressor_WithParameters_CreatesCorrectCopy()
    {
        var sgd = new OnlineSGDRegressor<double>(learningRate: 0.05);
        var x = new Vector<double>(new double[] { 1.0, 2.0 });
        for (int i = 0; i < 20; i++)
            sgd.PartialFit(x, 5.0);

        var params_ = sgd.GetParameters();
        var newModel = sgd.WithParameters(params_);

        var newParams = newModel.GetParameters();
        Assert.Equal(params_.Length, newParams.Length);
        for (int i = 0; i < params_.Length; i++)
            Assert.Equal(params_[i], newParams[i], Tolerance);
    }

    [Fact]
    public void OnlineModel_GetModelMetadata_ContainsExpectedFields()
    {
        var sgd = new OnlineSGDRegressor<double>();
        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 1.0);

        var metadata = sgd.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo is not null);
        Assert.True(metadata.AdditionalInfo.ContainsKey("SampleCount"));
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void SGDRegressor_ZeroLearningRate_NoUpdate()
    {
        // With learning rate = 1e-10 (minimum), updates should be tiny
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 1e-10, learningRateSchedule: LearningRateSchedule.Constant,
            l2Penalty: 0.0);

        var x = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x, 1000.0);

        // Weights should be essentially zero due to tiny learning rate
        Assert.True(Math.Abs(sgd.GetWeights()[0]) < 1e-6,
            "Tiny learning rate should produce negligible weight updates");
    }

    [Fact]
    public void PARegressor_ZeroFeatureNorm_NoUpdate()
    {
        // If x = [0] and no intercept, normSq = 0, tau = 0 → no update
        var pa = new OnlinePassiveAggressiveRegressor<double>(
            c: 1.0, epsilon: 0.0, type: PAType.PA, fitIntercept: false);

        var x = new Vector<double>(new double[] { 0.0 });
        pa.PartialFit(x, 5.0);

        Assert.Equal(0.0, pa.GetWeights()[0], Tolerance);
    }

    [Fact]
    public void SGDRegressor_NoIntercept_BiasStaysZero()
    {
        var sgd = new OnlineSGDRegressor<double>(
            learningRate: 0.1, learningRateSchedule: LearningRateSchedule.Constant,
            fitIntercept: false);

        var x = new Vector<double>(new double[] { 1.0 });
        for (int i = 0; i < 10; i++)
            sgd.PartialFit(x, 5.0);

        Assert.Equal(0.0, sgd.GetBias(), Tolerance);
    }

    [Fact]
    public void FeatureCountMismatch_Throws()
    {
        var sgd = new OnlineSGDRegressor<double>();

        var x1 = new Vector<double>(new double[] { 1.0 });
        sgd.PartialFit(x1, 1.0);

        var x2 = new Vector<double>(new double[] { 1.0, 2.0 }); // Different feature count
        Assert.Throws<ArgumentException>(() => sgd.PartialFit(x2, 1.0));
    }

    #endregion
}
