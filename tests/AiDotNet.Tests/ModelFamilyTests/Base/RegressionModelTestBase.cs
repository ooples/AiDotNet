using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for regression models implementing IFullModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;.
/// Tests deep mathematical invariants that any correctly implemented regression model must satisfy.
/// </summary>
public abstract class RegressionModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 100;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 3;

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance
    // Shifting all targets by constant C must shift predictions by C.
    // Any regression model violating this has a bias bug.
    // =====================================================

    [Fact]
    public void TranslationEquivariance_ShiftingTargets_ShiftsPredictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double shift = 1000.0;
        var shiftedY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            shiftedY[i] = trainY2[i] + shift;

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, shiftedY);

        var testX = ModelTestHelpers.GenerateLinearData(10, Features, ModelTestHelpers.CreateSeededRandom(99), noise: 0.0).X;
        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            for (int i = 0; i < pred1.Length; i++)
            {
                double actualShift = pred2[i] - pred1[i];
                Assert.True(Math.Abs(actualShift - shift) < shift * 0.3,
                    $"Translation equivariance violated: predicted shift = {actualShift:F2}, expected ~{shift}. " +
                    $"pred_original={pred1[i]:F4}, pred_shifted={pred2[i]:F4}");
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Equivariance
    // Scaling all targets by factor K must scale predictions by K.
    // =====================================================

    [Fact]
    public void ScalingEquivariance_ScalingTargets_ScalesPredictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double scale = 100.0;
        var scaledY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            scaledY[i] = trainY2[i] * scale;

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, scaledY);

        var testX = ModelTestHelpers.GenerateLinearData(10, Features, ModelTestHelpers.CreateSeededRandom(99), noise: 0.0).X;
        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            for (int i = 0; i < pred1.Length; i++)
            {
                if (Math.Abs(pred1[i]) > 0.01)
                {
                    double ratio = pred2[i] / pred1[i];
                    Assert.True(ratio > scale * 0.5 && ratio < scale * 2.0,
                        $"Scaling equivariance violated at sample {i}: ratio = {ratio:F2}, expected ~{scale}. " +
                        $"pred_original={pred1[i]:F4}, pred_scaled={pred2[i]:F4}");
                }
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Error ≤ Test Error
    // On average, the model should fit training data at least as well as unseen test data.
    // Violation indicates a bug in Train or Predict.
    // =====================================================

    [Fact]
    public void TrainingError_ShouldNotExceedTestError_OnAverage()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng, noise: 0.5);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng, noise: 0.5);

        model.Train(trainX, trainY);
        var trainPred = model.Predict(trainX);
        var testPred = model.Predict(testX);

        if (ModelTestHelpers.AllFinite(trainPred) && ModelTestHelpers.AllFinite(testPred))
        {
            double trainMSE = ModelTestHelpers.CalculateMSE(trainY, trainPred);
            double testMSE = ModelTestHelpers.CalculateMSE(testY, testPred);

            // Training MSE should generally be ≤ test MSE (allow 2x slack for variance)
            Assert.True(trainMSE <= testMSE * 2.0 + 1e-10,
                $"Training MSE ({trainMSE:F4}) is much higher than test MSE ({testMSE:F4}). " +
                "This suggests the model is not actually fitting the training data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data → Better or Equal Fit
    // Doubling training data should not make R² worse by more than noise.
    // =====================================================

    [Fact]
    public void MoreData_ShouldNotDegrade_R2()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(30, Features, rng1, noise: 0.1);

        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model2 = CreateModel();
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(120, Features, rng2, noise: 0.1);

        // Fixed test set
        var rngTest = ModelTestHelpers.CreateSeededRandom(99);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(50, Features, rngTest, noise: 0.1);

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, trainY2);

        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            double r2Small = ModelTestHelpers.CalculateR2(testY, pred1);
            double r2Large = ModelTestHelpers.CalculateR2(testY, pred2);

            // Model with 4x data should be at least as good (allow 0.15 margin for stochasticity)
            Assert.True(r2Large >= r2Small - 0.15,
                $"4x more data made R² worse: R²(30)={r2Small:F4}, R²(120)={r2Large:F4}. " +
                "Model may not be correctly learning from additional data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Irrelevant Feature Should Not Help
    // Adding a random noise feature should not improve predictions.
    // =====================================================

    [Fact]
    public void IrrelevantFeature_ShouldNotImprove_Predictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        // Generate data with 2 real features: y = 2*x1 + 4*x2 + 1
        var (trainX_real, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, 2, rng1, noise: 0.1);
        var (testX_real, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, 2, rng2, noise: 0.1);

        // Create version with 1 added noise feature
        var rngNoise = ModelTestHelpers.CreateSeededRandom(77);
        var trainX_noisy = new Matrix<double>(TrainSamples, 3);
        var testX_noisy = new Matrix<double>(TestSamples, 3);
        for (int i = 0; i < TrainSamples; i++)
        {
            trainX_noisy[i, 0] = trainX_real[i, 0];
            trainX_noisy[i, 1] = trainX_real[i, 1];
            trainX_noisy[i, 2] = rngNoise.NextDouble() * 100.0; // pure noise
        }
        for (int i = 0; i < TestSamples; i++)
        {
            testX_noisy[i, 0] = testX_real[i, 0];
            testX_noisy[i, 1] = testX_real[i, 1];
            testX_noisy[i, 2] = rngNoise.NextDouble() * 100.0;
        }

        model1.Train(trainX_real, trainY);
        model2.Train(trainX_noisy, trainY);

        var pred1 = model1.Predict(testX_real);
        var pred2 = model2.Predict(testX_noisy);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            double r2Real = ModelTestHelpers.CalculateR2(testY, pred1);
            double r2Noisy = ModelTestHelpers.CalculateR2(testY, pred2);

            // Adding noise feature should not improve R² substantially
            Assert.True(r2Noisy <= r2Real + 0.15,
                $"Adding irrelevant noise feature improved R²: clean={r2Real:F4}, noisy={r2Noisy:F4}. " +
                "Model may be overfitting to noise.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Monotonic Response
    // For data y = 2*x1 + 4*x2 + 1, increasing x1 while holding x2 constant
    // must increase prediction. Tests the model learned correct sign/direction.
    // =====================================================

    [Fact]
    public void MonotonicResponse_IncreasingFeature_IncreasesPrediction()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, 2, rng, noise: 0.01);

        model.Train(trainX, trainY);

        // Create probe with x1 varying from -5 to 15, x2 fixed at 5
        var probe = new Matrix<double>(5, 2);
        for (int i = 0; i < 5; i++)
        {
            probe[i, 0] = i * 5.0 - 5.0; // -5, 0, 5, 10, 15
            probe[i, 1] = 5.0;
        }

        var predictions = model.Predict(probe);
        if (ModelTestHelpers.AllFinite(predictions))
        {
            int monotoneViolations = 0;
            for (int i = 1; i < predictions.Length; i++)
            {
                if (predictions[i] < predictions[i - 1])
                    monotoneViolations++;
            }
            Assert.True(monotoneViolations <= 1,
                $"Monotonicity violated {monotoneViolations}/4 times. " +
                $"Predictions: [{string.Join(", ", Enumerable.Range(0, predictions.Length).Select(i => predictions[i].ToString("F2")))}]. " +
                "Model failed to learn positive coefficient direction.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Residual Mean ≈ 0
    // For unbiased estimators, the mean of residuals should be near zero.
    // Large residual mean indicates systematic bias in the model.
    // =====================================================

    [Fact]
    public void ResidualMean_ShouldBeNearZero()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, Features, rng, noise: 0.5);

        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double residualSum = 0;
            double targetRange = 0;
            double minY = double.MaxValue, maxY = double.MinValue;
            for (int i = 0; i < trainY.Length; i++)
            {
                residualSum += trainY[i] - predictions[i];
                if (trainY[i] < minY) minY = trainY[i];
                if (trainY[i] > maxY) maxY = trainY[i];
            }
            targetRange = maxY - minY;
            double meanResidual = residualSum / trainY.Length;

            // Mean residual should be small relative to the target range
            Assert.True(Math.Abs(meanResidual) < targetRange * 0.1,
                $"Mean residual = {meanResidual:F4} is large relative to target range {targetRange:F4}. " +
                "Model has systematic prediction bias.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Coefficient Sign Recovery
    // For y = 2*x1 + 4*x2 + 1 with low noise, probing must show
    // both features have positive effect on prediction.
    // =====================================================

    [Fact]
    public void CoefficientSigns_ShouldMatchDataGeneratingProcess()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, 2, rng, noise: 0.01);

        model.Train(trainX, trainY);

        // Probe: baseline at origin, then increase each feature independently
        var probe = new Matrix<double>(3, 2);
        probe[0, 0] = 0; probe[0, 1] = 0;   // baseline
        probe[1, 0] = 10; probe[1, 1] = 0;   // x1 effect
        probe[2, 0] = 0; probe[2, 1] = 10;   // x2 effect

        var predictions = model.Predict(probe);
        if (ModelTestHelpers.AllFinite(predictions))
        {
            double effectX1 = predictions[1] - predictions[0];
            double effectX2 = predictions[2] - predictions[0];

            Assert.True(effectX1 > 0,
                $"Feature x1 effect = {effectX1:F4}, expected positive (true coeff = 2.0). " +
                "Model learned wrong sign for x1.");
            Assert.True(effectX2 > 0,
                $"Feature x2 effect = {effectX2:F4}, expected positive (true coeff = 4.0). " +
                "Model learned wrong sign for x2.");
            Assert.True(effectX2 > effectX1,
                $"Feature x2 effect ({effectX2:F4}) should be larger than x1 effect ({effectX1:F4}) " +
                "since true coefficients are 4.0 vs 2.0.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Permutation Consistency
    // Permuting feature columns and correspondingly permuting any learned
    // structure should give equivalent predictions.
    // =====================================================

    [Fact]
    public void FeaturePermutation_ShouldGiveConsistentPredictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, 2, rng1, noise: 0.01);

        // Create permuted version: swap columns 0 and 1
        var permutedX = new Matrix<double>(TrainSamples, 2);
        for (int i = 0; i < TrainSamples; i++)
        {
            permutedX[i, 0] = trainX[i, 1]; // swap
            permutedX[i, 1] = trainX[i, 0];
        }

        model1.Train(trainX, trainY);
        model2.Train(permutedX, trainY);

        // Test with a specific point
        var testOrig = new Matrix<double>(1, 2);
        var testPerm = new Matrix<double>(1, 2);
        testOrig[0, 0] = 3.0; testOrig[0, 1] = 7.0;
        testPerm[0, 0] = 7.0; testPerm[0, 1] = 3.0; // swapped

        var pred1 = model1.Predict(testOrig);
        var pred2 = model2.Predict(testPerm);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            Assert.True(Math.Abs(pred1[0] - pred2[0]) < Math.Abs(pred1[0]) * 0.2 + 1.0,
                $"Feature permutation inconsistency: pred_orig={pred1[0]:F4}, pred_permuted={pred2[0]:F4}. " +
                "Swapping feature columns and correspondingly swapping test inputs should give ~same prediction.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: R² > 0 on Linear Data
    // Any regression model should outperform the mean baseline on data
    // that is actually linear. R²≤0 means the model is worse than guessing the mean.
    // =====================================================

    [Fact]
    public void R2_ShouldBePositive_OnLinearData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng, noise: 0.1);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng, noise: 0.1);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double r2 = ModelTestHelpers.CalculateR2(testY, predictions);
            Assert.True(r2 > 0.0,
                $"R² = {r2:F4} on linear data — model is worse than predicting the mean. " +
                "Either the model is not learning, or Train/Predict has a bug.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Predictions Are Finite
    // No NaN, no Infinity. Violations indicate numerical instability.
    // =====================================================

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        Assert.Equal(TestSamples, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN — numerical instability in model.");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction[{i}] is Infinity — overflow in model computation.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Deterministic Prediction
    // Same trained model + same input = same output. Always.
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(trainX, trainY);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Dimension
    // Predict(N×F matrix) must return length-N vector.
    // =====================================================

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(trainX, trainY);

        // Test with various sample counts
        foreach (int n in new[] { 1, 5, 50 })
        {
            var testX = ModelTestHelpers.GenerateLinearData(n, Features, ModelTestHelpers.CreateSeededRandom(n), noise: 0.0).X;
            var pred = model.Predict(testX);
            Assert.Equal(n, pred.Length);
        }
    }

    // =====================================================
    // CONTRACT: Clone Produces Identical Predictions
    // =====================================================

    [Fact]
    public void Clone_ShouldProduceIdenticalPredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();

        var pred1 = model.Predict(testX);
        var pred2 = cloned.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    // =====================================================
    // CONTRACT: Metadata Should Exist After Training
    // =====================================================

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    // =====================================================
    // CONTRACT: Parameters Should Be Non-Empty After Training
    // =====================================================

    [Fact]
    public void Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(trainX, trainY);
        var parameters = model.GetParameters();
        Assert.True(parameters.Length > 0, "Trained model should have learnable parameters.");
    }

    // =====================================================
    // CONTRACT: Active Feature Indices Should Be Valid
    // =====================================================

    [Fact]
    public void ActiveFeatureIndices_ShouldBeValid()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(trainX, trainY);
        var activeFeatures = model.GetActiveFeatureIndices().ToList();

        Assert.True(activeFeatures.Count > 0, "Trained model should have at least one active feature.");
        foreach (var idx in activeFeatures)
        {
            Assert.True(idx >= 0 && idx < Features,
                $"Active feature index {idx} is out of bounds [0, {Features}).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Intercept Recovery
    // On constant data y = C, all predictions should equal C.
    // If not, the bias/intercept term is broken.
    // =====================================================

    [Fact]
    public void InterceptRecovery_ConstantTarget_ShouldPredictConstant()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, Features);
        var y = new Vector<double>(n);
        const double constant = 7.5;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 10.0;
            y[i] = constant;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double meanPred = 0;
            for (int i = 0; i < predictions.Length; i++) meanPred += predictions[i];
            meanPred /= predictions.Length;

            Assert.True(Math.Abs(meanPred - constant) < constant * 0.3,
                $"Mean prediction = {meanPred:F4} on constant data (y={constant}). " +
                "Intercept/bias term may be broken.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Collinear Features Should Not Crash
    // Perfectly correlated features should not cause NaN/Infinity.
    // =====================================================

    [Fact]
    public void CollinearFeatures_ShouldNotCrash()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, Features);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double val = rng.NextDouble() * 10.0;
            for (int j = 0; j < Features; j++)
                x[i, j] = val + j * 0.001; // nearly perfectly collinear
            y[i] = val * 2.0 + 1.0;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN with collinear features — numerical instability.");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction[{i}] is Infinity with collinear features.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Single Feature Should Work
    // Regression model should handle 1-dimensional input.
    // =====================================================

    [Fact]
    public void SingleFeature_ShouldWork()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = rng.NextDouble() * 10.0;
            y[i] = 3.0 * x[i, 0] + 1.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);
        Assert.Equal(n, predictions.Length);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN for 1-feature input.");
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline Produces Valid Result
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact]
    public void Builder_R2ShouldBePositive()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = result.Predict(testX);
        double r2 = ModelTestHelpers.CalculateR2(testY, predictions);
        Assert.True(r2 > 0.0,
            $"Builder pipeline R² = {r2:F4} — should be positive on linear data.");
    }
}
