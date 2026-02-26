using AiDotNet.Prototypes;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Prototypes;

/// <summary>
/// Deep integration tests for Prototypes:
/// PrototypeVector (factory methods, arithmetic, indexing, ToString),
/// SimpleLinearRegression (training, prediction, MSE, R2, gradient descent).
/// </summary>
public class PrototypesDeepMathIntegrationTests
{
    // ============================
    // PrototypeVector: Construction
    // ============================

    [Fact]
    public void PrototypeVector_FromLength_AllZeros()
    {
        var vec = new PrototypeVector<double>(5);
        Assert.Equal(5, vec.Length);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(0.0, vec[i]);
        }
    }

    [Fact]
    public void PrototypeVector_FromArray_CopiesValues()
    {
        var data = new double[] { 1.0, 2.0, 3.0 };
        var vec = new PrototypeVector<double>(data);
        Assert.Equal(3, vec.Length);
        Assert.Equal(1.0, vec[0]);
        Assert.Equal(2.0, vec[1]);
        Assert.Equal(3.0, vec[2]);
    }

    [Fact]
    public void PrototypeVector_FromArray_DoesNotMutateSource()
    {
        var data = new double[] { 1.0, 2.0, 3.0 };
        var vec = new PrototypeVector<double>(data);
        vec[0] = 99.0;
        Assert.Equal(1.0, data[0]);
    }

    // ============================
    // PrototypeVector: Factory Methods
    // ============================

    [Fact]
    public void PrototypeVector_Zeros_AllZero()
    {
        var vec = PrototypeVector<double>.Zeros(10);
        Assert.Equal(10, vec.Length);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(0.0, vec[i]);
        }
    }

    [Fact]
    public void PrototypeVector_Ones_AllOne()
    {
        var vec = PrototypeVector<double>.Ones(5);
        Assert.Equal(5, vec.Length);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(1.0, vec[i]);
        }
    }

    [Fact]
    public void PrototypeVector_FromArray_Factory()
    {
        var vec = PrototypeVector<double>.FromArray(new double[] { 4.0, 5.0, 6.0 });
        Assert.Equal(3, vec.Length);
        Assert.Equal(4.0, vec[0]);
        Assert.Equal(5.0, vec[1]);
        Assert.Equal(6.0, vec[2]);
    }

    // ============================
    // PrototypeVector: Indexer
    // ============================

    [Fact]
    public void PrototypeVector_Indexer_SetAndGet()
    {
        var vec = new PrototypeVector<double>(3);
        vec[0] = 10.0;
        vec[1] = 20.0;
        vec[2] = 30.0;

        Assert.Equal(10.0, vec[0]);
        Assert.Equal(20.0, vec[1]);
        Assert.Equal(30.0, vec[2]);
    }

    // ============================
    // PrototypeVector: Arithmetic Operations
    // ============================

    [Fact]
    public void PrototypeVector_Add_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 4.0, 5.0, 6.0 });

        var result = a.Add(b);
        Assert.Equal(3, result.Length);
        Assert.Equal(5.0, result[0], 1e-10);
        Assert.Equal(7.0, result[1], 1e-10);
        Assert.Equal(9.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Subtract_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 10.0, 20.0, 30.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 5.0, 10.0 });

        var result = a.Subtract(b);
        Assert.Equal(9.0, result[0], 1e-10);
        Assert.Equal(15.0, result[1], 1e-10);
        Assert.Equal(20.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Multiply_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 2.0, 3.0, 4.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 5.0, 6.0, 7.0 });

        var result = a.Multiply(b);
        Assert.Equal(10.0, result[0], 1e-10);
        Assert.Equal(18.0, result[1], 1e-10);
        Assert.Equal(28.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Multiply_Scalar()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var result = a.Multiply(3.0);
        Assert.Equal(3.0, result[0], 1e-10);
        Assert.Equal(6.0, result[1], 1e-10);
        Assert.Equal(9.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Divide_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 10.0, 20.0, 30.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 2.0, 5.0, 10.0 });

        var result = a.Divide(b);
        Assert.Equal(5.0, result[0], 1e-10);
        Assert.Equal(4.0, result[1], 1e-10);
        Assert.Equal(3.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Divide_Scalar()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 6.0, 12.0, 18.0 });
        var result = a.Divide(3.0);
        Assert.Equal(2.0, result[0], 1e-10);
        Assert.Equal(4.0, result[1], 1e-10);
        Assert.Equal(6.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Sqrt_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 4.0, 9.0, 16.0 });
        var result = a.Sqrt();
        Assert.Equal(2.0, result[0], 1e-10);
        Assert.Equal(3.0, result[1], 1e-10);
        Assert.Equal(4.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_Power_ElementWise()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 2.0, 3.0, 4.0 });
        var result = a.Power(2.0);
        Assert.Equal(4.0, result[0], 1e-10);
        Assert.Equal(9.0, result[1], 1e-10);
        Assert.Equal(16.0, result[2], 1e-10);
    }

    // ============================
    // PrototypeVector: Arithmetic Identities
    // ============================

    [Fact]
    public void PrototypeVector_AddZero_Identity()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var zero = PrototypeVector<double>.Zeros(3);
        var result = a.Add(zero);
        Assert.Equal(1.0, result[0], 1e-10);
        Assert.Equal(2.0, result[1], 1e-10);
        Assert.Equal(3.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_MultiplyByOne_Identity()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 5.0, 10.0, 15.0 });
        var result = a.Multiply(1.0);
        Assert.Equal(5.0, result[0], 1e-10);
        Assert.Equal(10.0, result[1], 1e-10);
        Assert.Equal(15.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_MultiplyByZero_AllZero()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 5.0, 10.0, 15.0 });
        var result = a.Multiply(0.0);
        Assert.Equal(0.0, result[0], 1e-10);
        Assert.Equal(0.0, result[1], 1e-10);
        Assert.Equal(0.0, result[2], 1e-10);
    }

    [Fact]
    public void PrototypeVector_SubtractSelf_AllZero()
    {
        var a = PrototypeVector<double>.FromArray(new double[] { 7.0, 14.0, 21.0 });
        var result = a.Subtract(a);
        Assert.Equal(0.0, result[0], 1e-10);
        Assert.Equal(0.0, result[1], 1e-10);
        Assert.Equal(0.0, result[2], 1e-10);
    }

    // ============================
    // PrototypeVector: ToArray and ToVector
    // ============================

    [Fact]
    public void PrototypeVector_ToArray_ReturnsCorrectValues()
    {
        var vec = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var arr = vec.ToArray();
        Assert.Equal(new double[] { 1.0, 2.0, 3.0 }, arr);
    }

    [Fact]
    public void PrototypeVector_ToVector_ReturnsVector()
    {
        var vec = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });
        var vector = vec.ToVector();
        Assert.Equal(2, vector.Length);
        Assert.Equal(1.0, vector[0]);
        Assert.Equal(2.0, vector[1]);
    }

    // ============================
    // PrototypeVector: ToString
    // ============================

    [Fact]
    public void PrototypeVector_ToString_Short()
    {
        var vec = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var str = vec.ToString();
        Assert.Contains("PrototypeVector<Double>", str);
        Assert.Contains("1", str);
        Assert.Contains("2", str);
        Assert.Contains("3", str);
    }

    [Fact]
    public void PrototypeVector_ToString_Long_ShowsEllipsis()
    {
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = i;
        var vec = PrototypeVector<double>.FromArray(data);
        var str = vec.ToString();
        Assert.Contains("...", str);
        Assert.Contains("length=20", str);
    }

    // ============================
    // PrototypeVector: Float Type
    // ============================

    [Fact]
    public void PrototypeVector_Float_BasicArithmetic()
    {
        var a = PrototypeVector<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var b = PrototypeVector<float>.FromArray(new float[] { 4.0f, 5.0f, 6.0f });
        var result = a.Add(b);
        Assert.Equal(5.0f, result[0], 1e-5f);
        Assert.Equal(7.0f, result[1], 1e-5f);
        Assert.Equal(9.0f, result[2], 1e-5f);
    }

    // ============================
    // SimpleLinearRegression: Construction
    // ============================

    [Fact]
    public void SimpleLinearRegression_Construction_Properties()
    {
        var model = new SimpleLinearRegression<double>(5);
        Assert.Equal(5, model.NumFeatures);
        Assert.True(model.IsTrained); // Initialized with zeros, so technically "trained"
    }

    [Fact]
    public void SimpleLinearRegression_ZeroFeatures_Throws()
    {
        Assert.Throws<ArgumentException>(() => new SimpleLinearRegression<double>(0));
    }

    [Fact]
    public void SimpleLinearRegression_NegativeFeatures_Throws()
    {
        Assert.Throws<ArgumentException>(() => new SimpleLinearRegression<double>(-1));
    }

    // ============================
    // SimpleLinearRegression: Initial Weights and Bias
    // ============================

    [Fact]
    public void SimpleLinearRegression_InitialWeights_AllZero()
    {
        var model = new SimpleLinearRegression<double>(3);
        var weights = model.GetWeights();
        Assert.NotNull(weights);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(0.0, weights[i]);
        }
    }

    [Fact]
    public void SimpleLinearRegression_InitialBias_Zero()
    {
        var model = new SimpleLinearRegression<double>(3);
        var bias = model.GetBias();
        Assert.NotNull(bias);
        Assert.Equal(0.0, bias);
    }

    // ============================
    // SimpleLinearRegression: Prediction (untrained = zeros)
    // ============================

    [Fact]
    public void SimpleLinearRegression_PredictUntrained_ReturnsZero()
    {
        var model = new SimpleLinearRegression<double>(2);
        var features = PrototypeVector<double>.FromArray(new double[] { 5.0, 10.0 });
        double prediction = model.Predict(features);
        // With zero weights and zero bias: 0*5 + 0*10 + 0 = 0
        Assert.Equal(0.0, prediction, 1e-10);
    }

    [Fact]
    public void SimpleLinearRegression_Predict_WrongFeatureCount_Throws()
    {
        var model = new SimpleLinearRegression<double>(3);
        var features = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });
        Assert.Throws<ArgumentException>(() => model.Predict(features));
    }

    // ============================
    // SimpleLinearRegression: ComputeMSE
    // ============================

    [Fact]
    public void SimpleLinearRegression_ComputeMSE_PerfectPrediction_Zero()
    {
        var model = new SimpleLinearRegression<double>(1);
        var predictions = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        double mse = model.ComputeMSE(predictions, targets);
        Assert.Equal(0.0, mse, 1e-10);
    }

    [Fact]
    public void SimpleLinearRegression_ComputeMSE_KnownValues()
    {
        var model = new SimpleLinearRegression<double>(1);
        // MSE = mean((pred - target)^2)
        // (1-2)^2 + (3-4)^2 + (5-6)^2 = 1+1+1 = 3, mean = 1.0
        var predictions = PrototypeVector<double>.FromArray(new double[] { 1.0, 3.0, 5.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 2.0, 4.0, 6.0 });

        double mse = model.ComputeMSE(predictions, targets);
        Assert.Equal(1.0, mse, 1e-10);
    }

    [Fact]
    public void SimpleLinearRegression_ComputeMSE_NonNegative()
    {
        var model = new SimpleLinearRegression<double>(1);
        var predictions = PrototypeVector<double>.FromArray(new double[] { 10.0, -5.0, 3.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { -2.0, 8.0, 0.0 });

        double mse = model.ComputeMSE(predictions, targets);
        Assert.True(mse >= 0, $"MSE should be non-negative, got {mse}");
    }

    // ============================
    // SimpleLinearRegression: ComputeR2Score
    // ============================

    [Fact]
    public void SimpleLinearRegression_ComputeR2_PerfectPrediction_IsOne()
    {
        var model = new SimpleLinearRegression<double>(1);
        var predictions = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });

        double r2 = model.ComputeR2Score(predictions, targets);
        Assert.Equal(1.0, r2, 1e-10);
    }

    [Fact]
    public void SimpleLinearRegression_ComputeR2_MeanPrediction_IsZero()
    {
        var model = new SimpleLinearRegression<double>(1);
        // If predictions = mean of targets, R2 = 0
        // targets: 1, 2, 3, 4 -> mean = 2.5
        var predictions = PrototypeVector<double>.FromArray(new double[] { 2.5, 2.5, 2.5, 2.5 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0 });

        double r2 = model.ComputeR2Score(predictions, targets);
        Assert.Equal(0.0, r2, 1e-10);
    }

    // ============================
    // SimpleLinearRegression: Training
    // ============================

    [Fact]
    public void SimpleLinearRegression_Train_SimpleLine()
    {
        // y = 2*x + 1
        var model = new SimpleLinearRegression<double>(1);
        int numSamples = 100;

        var xData = new double[numSamples];
        var yData = new double[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            xData[i] = i * 0.1;
            yData[i] = 2.0 * xData[i] + 1.0;
        }

        var X = PrototypeVector<double>.FromArray(xData);
        var y = PrototypeVector<double>.FromArray(yData);

        model.Train(X, y, numSamples, learningRate: 0.01, numEpochs: 500);

        // After training, weight should be ~2.0 and bias ~1.0
        var weights = model.GetWeights();
        var bias = model.GetBias();

        Assert.NotNull(weights);
        Assert.NotNull(bias);

        // Check with reasonable tolerance for gradient descent convergence
        Assert.True(Math.Abs(weights[0] - 2.0) < 0.5,
            $"Weight should be near 2.0, got {weights[0]}");
        Assert.True(Math.Abs((double)bias - 1.0) < 0.5,
            $"Bias should be near 1.0, got {bias}");
    }

    [Fact]
    public void SimpleLinearRegression_Train_ReducesLoss()
    {
        var model = new SimpleLinearRegression<double>(1);
        int numSamples = 50;

        var xData = new double[numSamples];
        var yData = new double[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            xData[i] = i * 0.1;
            yData[i] = 3.0 * xData[i] + 2.0;
        }

        var X = PrototypeVector<double>.FromArray(xData);
        var y = PrototypeVector<double>.FromArray(yData);

        // Get initial loss (untrained predictions = 0)
        var initialPredictions = model.PredictBatch(X, numSamples);
        double initialLoss = model.ComputeMSE(initialPredictions, y);

        // Train
        model.Train(X, y, numSamples, learningRate: 0.01, numEpochs: 200);

        // Get final loss
        var finalPredictions = model.PredictBatch(X, numSamples);
        double finalLoss = model.ComputeMSE(finalPredictions, y);

        Assert.True(finalLoss < initialLoss,
            $"Training should reduce loss: initial={initialLoss}, final={finalLoss}");
    }

    [Fact]
    public void SimpleLinearRegression_Train_WrongXLength_Throws()
    {
        var model = new SimpleLinearRegression<double>(2);
        var X = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 }); // Wrong: 3 != 2*2
        var y = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        Assert.Throws<ArgumentException>(() => model.Train(X, y, 2, learningRate: 0.01, numEpochs: 1));
    }

    [Fact]
    public void SimpleLinearRegression_Train_WrongYLength_Throws()
    {
        var model = new SimpleLinearRegression<double>(1);
        var X = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var y = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 }); // Wrong: 2 != 3

        Assert.Throws<ArgumentException>(() => model.Train(X, y, 3, learningRate: 0.01, numEpochs: 1));
    }

    // ============================
    // SimpleLinearRegression: PredictBatch
    // ============================

    [Fact]
    public void SimpleLinearRegression_PredictBatch_WrongLength_Throws()
    {
        var model = new SimpleLinearRegression<double>(2);
        var X = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 }); // 3 != 2*2

        Assert.Throws<ArgumentException>(() => model.PredictBatch(X, 2));
    }

    // ============================
    // SimpleLinearRegression: ToString
    // ============================

    [Fact]
    public void SimpleLinearRegression_ToString_ContainsFeatureCount()
    {
        var model = new SimpleLinearRegression<double>(3);
        var str = model.ToString();
        Assert.Contains("features=3", str);
        Assert.Contains("SimpleLinearRegression<Double>", str);
    }

    // ============================
    // SimpleLinearRegression: Float Type
    // ============================

    [Fact]
    public void SimpleLinearRegression_Float_ConstructAndPredict()
    {
        var model = new SimpleLinearRegression<float>(2);
        var features = PrototypeVector<float>.FromArray(new float[] { 1.0f, 2.0f });
        float prediction = model.Predict(features);
        Assert.Equal(0.0f, prediction, 1e-5f);
    }
}
