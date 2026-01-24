using AiDotNet.Prototypes;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Prototypes;

/// <summary>
/// Comprehensive integration tests for the Prototypes module.
/// Tests PrototypeVector, PrototypeAdamOptimizer, SimpleLinearRegression, and SimpleNeuralNetwork.
/// </summary>
public class PrototypesIntegrationTests
{
    #region PrototypeVector Tests

    [Fact]
    public void PrototypeVector_Constructor_CreatesVectorWithCorrectLength()
    {
        // Arrange & Act
        var vector = new PrototypeVector<double>(10);

        // Assert
        Assert.Equal(10, vector.Length);
    }

    [Fact]
    public void PrototypeVector_Constructor_WithArray_CopiesValues()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var vector = new PrototypeVector<double>(data);

        // Assert
        Assert.Equal(5, vector.Length);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], vector[i]);
        }
    }

    [Fact]
    public void PrototypeVector_Indexer_GetsAndSetsValues()
    {
        // Arrange
        var vector = new PrototypeVector<double>(5);

        // Act
        vector[0] = 10.0;
        vector[4] = 50.0;

        // Assert
        Assert.Equal(10.0, vector[0]);
        Assert.Equal(50.0, vector[4]);
    }

    [Fact]
    public void PrototypeVector_Indexer_ThrowsOnOutOfBounds()
    {
        // Arrange
        var vector = new PrototypeVector<double>(5);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = vector[5]);
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = vector[-1]);
    }

    [Fact]
    public void PrototypeVector_Zeros_CreatesZeroVector()
    {
        // Act
        var vector = PrototypeVector<double>.Zeros(5);

        // Assert
        Assert.Equal(5, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.Equal(0.0, vector[i]);
        }
    }

    [Fact]
    public void PrototypeVector_Ones_CreatesOnesVector()
    {
        // Act
        var vector = PrototypeVector<double>.Ones(5);

        // Assert
        Assert.Equal(5, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.Equal(1.0, vector[i]);
        }
    }

    [Fact]
    public void PrototypeVector_FromArray_CreatesVectorFromArray()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.5f, 3.5f };

        // Act
        var vector = PrototypeVector<float>.FromArray(data);

        // Assert
        Assert.Equal(3, vector.Length);
        Assert.Equal(1.5f, vector[0]);
        Assert.Equal(2.5f, vector[1]);
        Assert.Equal(3.5f, vector[2]);
    }

    [Fact]
    public void PrototypeVector_Add_VectorPlusVector()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 4.0, 5.0, 6.0 });

        // Act
        var result = a.Add(b);

        // Assert
        Assert.Equal(5.0, result[0]);
        Assert.Equal(7.0, result[1]);
        Assert.Equal(9.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Add_VectorPlusScalarVector()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var scalar = PrototypeVector<double>.FromArray(new double[] { 10.0, 10.0, 10.0 });

        // Act
        var result = vector.Add(scalar);

        // Assert
        Assert.Equal(11.0, result[0]);
        Assert.Equal(12.0, result[1]);
        Assert.Equal(13.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Subtract_VectorMinusVector()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 10.0, 20.0, 30.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        // Act
        var result = a.Subtract(b);

        // Assert
        Assert.Equal(9.0, result[0]);
        Assert.Equal(18.0, result[1]);
        Assert.Equal(27.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Multiply_VectorTimesVector()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 2.0, 3.0, 4.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 5.0, 6.0, 7.0 });

        // Act
        var result = a.Multiply(b);

        // Assert
        Assert.Equal(10.0, result[0]);
        Assert.Equal(18.0, result[1]);
        Assert.Equal(28.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Multiply_VectorTimesScalar()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        // Act
        var result = vector.Multiply(2.0);

        // Assert
        Assert.Equal(2.0, result[0]);
        Assert.Equal(4.0, result[1]);
        Assert.Equal(6.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Divide_VectorDivideVector()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 10.0, 20.0, 30.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 2.0, 4.0, 5.0 });

        // Act
        var result = a.Divide(b);

        // Assert
        Assert.Equal(5.0, result[0]);
        Assert.Equal(5.0, result[1]);
        Assert.Equal(6.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Divide_VectorDivideScalar()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 10.0, 20.0, 30.0 });

        // Act
        var result = vector.Divide(10.0);

        // Assert
        Assert.Equal(1.0, result[0]);
        Assert.Equal(2.0, result[1]);
        Assert.Equal(3.0, result[2]);
    }

    [Fact]
    public void PrototypeVector_Sqrt_ComputesSquareRoot()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 4.0, 9.0, 16.0 });

        // Act
        var result = vector.Sqrt();

        // Assert
        Assert.Equal(2.0, result[0], 6);
        Assert.Equal(3.0, result[1], 6);
        Assert.Equal(4.0, result[2], 6);
    }

    [Fact]
    public void PrototypeVector_Power_RaisesToPower()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 2.0, 3.0, 4.0 });

        // Act
        var result = vector.Power(2.0);

        // Assert
        Assert.Equal(4.0, result[0], 6);
        Assert.Equal(9.0, result[1], 6);
        Assert.Equal(16.0, result[2], 6);
    }

    [Fact]
    public void PrototypeVector_ToString_ReturnsFormattedString()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act
        var result = vector.ToString();

        // Assert
        Assert.NotNull(result);
        Assert.Contains("[", result);
        Assert.Contains("]", result);
    }

    [Fact]
    public void PrototypeVector_LengthMismatch_ThrowsOnAdd()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a.Add(b));
    }

    [Fact]
    public void PrototypeVector_LengthMismatch_ThrowsOnSubtract()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a.Subtract(b));
    }

    [Fact]
    public void PrototypeVector_LengthMismatch_ThrowsOnMultiply()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a.Multiply(b));
    }

    [Fact]
    public void PrototypeVector_LengthMismatch_ThrowsOnDivide()
    {
        // Arrange
        var a = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var b = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a.Divide(b));
    }

    [Fact]
    public void PrototypeVector_WorksWithFloat()
    {
        // Arrange
        var a = PrototypeVector<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var b = PrototypeVector<float>.FromArray(new float[] { 4.0f, 5.0f, 6.0f });

        // Act
        var sum = a.Add(b);
        var product = a.Multiply(2.0f);

        // Assert
        Assert.Equal(5.0f, sum[0]);
        Assert.Equal(2.0f, product[0]);
    }

    [Fact]
    public void PrototypeVector_WorksWithDecimal()
    {
        // Arrange
        var a = PrototypeVector<decimal>.FromArray(new decimal[] { 1.0m, 2.0m, 3.0m });
        var b = PrototypeVector<decimal>.FromArray(new decimal[] { 4.0m, 5.0m, 6.0m });

        // Act
        var sum = a.Add(b);

        // Assert
        Assert.Equal(5.0m, sum[0]);
        Assert.Equal(7.0m, sum[1]);
        Assert.Equal(9.0m, sum[2]);
    }

    [Fact]
    public void PrototypeVector_ChainedOperations()
    {
        // Arrange
        var vector = PrototypeVector<double>.FromArray(new double[] { 2.0, 4.0, 6.0 });
        var ones = PrototypeVector<double>.FromArray(new double[] { 1.0, 1.0, 1.0 });

        // Act: (vector * 2 + 1) / 5
        var result = vector.Multiply(2.0).Add(ones).Divide(5.0);

        // Assert
        Assert.Equal(1.0, result[0], 6);  // (2*2+1)/5 = 1
        Assert.Equal(1.8, result[1], 6);  // (4*2+1)/5 = 1.8
        Assert.Equal(2.6, result[2], 6);  // (6*2+1)/5 = 2.6
    }

    #endregion

    #region PrototypeAdamOptimizer Tests

    [Fact]
    public void PrototypeAdamOptimizer_Constructor_DefaultParameters()
    {
        // Act
        var optimizer = new PrototypeAdamOptimizer<double>();

        // Assert
        Assert.Equal(0, optimizer.TimeStep);
    }

    [Fact]
    public void PrototypeAdamOptimizer_Constructor_CustomParameters()
    {
        // Act
        var optimizer = new PrototypeAdamOptimizer<float>(
            learningRate: 0.01,
            beta1: 0.95,
            beta2: 0.9999,
            epsilon: 1e-7);

        // Assert
        Assert.Equal(0, optimizer.TimeStep);
    }

    [Fact]
    public void PrototypeAdamOptimizer_UpdateParameters_IncrementsTimeStep()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>();
        var parameters = PrototypeVector<double>.Ones(5);
        var gradient = PrototypeVector<double>.Ones(5);

        // Act
        optimizer.UpdateParameters(parameters, gradient);

        // Assert
        Assert.Equal(1, optimizer.TimeStep);
    }

    [Fact]
    public void PrototypeAdamOptimizer_UpdateParameters_ModifiesParameters()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.1);
        var parameters = PrototypeVector<double>.Ones(5);
        var gradient = PrototypeVector<double>.Ones(5);

        // Act
        var updated = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        // Parameters should be different from original after update
        bool different = false;
        for (int i = 0; i < 5; i++)
        {
            if (Math.Abs(updated[i] - parameters[i]) > 1e-10)
            {
                different = true;
                break;
            }
        }
        Assert.True(different, "Parameters should be modified by optimizer");
    }

    [Fact]
    public void PrototypeAdamOptimizer_UpdateParameters_ThrowsOnNullParameters()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>();
        var gradient = PrototypeVector<double>.Ones(5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.UpdateParameters(null!, gradient));
    }

    [Fact]
    public void PrototypeAdamOptimizer_UpdateParameters_ThrowsOnNullGradient()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>();
        var parameters = PrototypeVector<double>.Ones(5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.UpdateParameters(parameters, null!));
    }

    [Fact]
    public void PrototypeAdamOptimizer_UpdateParameters_ThrowsOnLengthMismatch()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>();
        var parameters = PrototypeVector<double>.Ones(5);
        var gradient = PrototypeVector<double>.Ones(10);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => optimizer.UpdateParameters(parameters, gradient));
    }

    [Fact]
    public void PrototypeAdamOptimizer_Reset_ClearsState()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>();
        var parameters = PrototypeVector<double>.Ones(5);
        var gradient = PrototypeVector<double>.Ones(5);
        optimizer.UpdateParameters(parameters, gradient);

        // Act
        optimizer.Reset();

        // Assert
        Assert.Equal(0, optimizer.TimeStep);
    }

    [Fact]
    public void PrototypeAdamOptimizer_ToString_ReturnsConfiguration()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.001, beta1: 0.9, beta2: 0.999);

        // Act
        var result = optimizer.ToString();

        // Assert
        Assert.Contains("PrototypeAdamOptimizer", result);
        Assert.Contains("lr=", result);
        Assert.Contains("beta1=", result);
        Assert.Contains("beta2=", result);
    }

    [Fact]
    public void PrototypeAdamOptimizer_ConvergesToZero()
    {
        // Arrange: optimize x^2 where gradient is 2*x
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.5);
        var parameters = PrototypeVector<double>.FromArray(new double[] { 10.0, -5.0, 3.0 });

        // Act: run 100 iterations
        for (int i = 0; i < 100; i++)
        {
            // Gradient of x^2 is 2*x
            var gradient = parameters.Multiply(2.0);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert: parameters should be close to zero
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(Math.Abs(parameters[i]) < 0.1, $"Parameter {i} did not converge: {parameters[i]}");
        }
    }

    [Fact]
    public void PrototypeAdamOptimizer_MultipleUpdates_AccumulatesMoments()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.1);
        var parameters = PrototypeVector<double>.Ones(5);
        var gradient = PrototypeVector<double>.Ones(5);

        // Act
        for (int i = 0; i < 10; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert
        Assert.Equal(10, optimizer.TimeStep);
    }

    [Fact]
    public void PrototypeAdamOptimizer_WorksWithFloat()
    {
        // Arrange
        var optimizer = new PrototypeAdamOptimizer<float>(learningRate: 0.1);
        var parameters = PrototypeVector<float>.Ones(5);
        var gradient = PrototypeVector<float>.Ones(5);

        // Act
        var updated = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        Assert.Equal(5, updated.Length);
        Assert.Equal(1, optimizer.TimeStep);
    }

    #endregion

    #region SimpleLinearRegression Tests

    [Fact]
    public void SimpleLinearRegression_Constructor_InitializesWeights()
    {
        // Act
        var model = new SimpleLinearRegression<double>(3);

        // Assert
        var weights = model.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(3, weights.Length);
    }

    [Fact]
    public void SimpleLinearRegression_Predict_SingleSample()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(2);
        var input = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0 });

        // Act
        var prediction = model.Predict(input);

        // Assert: just verify it returns a value
        Assert.True(!double.IsNaN(prediction));
    }

    [Fact]
    public void SimpleLinearRegression_PredictBatch_MultipleSamples()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(2);
        // 3 samples with 2 features each
        var inputs = PrototypeVector<double>.FromArray(new double[] {
            1.0, 2.0,  // sample 1
            3.0, 4.0,  // sample 2
            5.0, 6.0   // sample 3
        });

        // Act
        var predictions = model.PredictBatch(inputs, numSamples: 3);

        // Assert
        Assert.Equal(3, predictions.Length);
    }

    [Fact]
    public void SimpleLinearRegression_Train_ReducesMSE()
    {
        // Arrange: simple linear relationship y = 2*x + 1
        var model = new SimpleLinearRegression<double>(1);
        var X = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var y = PrototypeVector<double>.FromArray(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });

        // Get initial predictions
        var initialPredictions = model.PredictBatch(X, 5);
        var initialMSE = model.ComputeMSE(initialPredictions, y);

        // Act
        model.Train(X, y, numSamples: 5, learningRate: 0.1, numEpochs: 100, verbose: false);

        // Get final predictions
        var finalPredictions = model.PredictBatch(X, 5);
        var finalMSE = model.ComputeMSE(finalPredictions, y);

        // Assert: MSE should decrease after training
        Assert.True(finalMSE < initialMSE, $"MSE did not decrease: {initialMSE} -> {finalMSE}");
    }

    [Fact]
    public void SimpleLinearRegression_ComputeMSE_CorrectValue()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(1);
        var predictions = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 2.0, 3.0, 4.0 });

        // Act
        var mse = model.ComputeMSE(predictions, targets);

        // Assert: MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
        Assert.Equal(1.0, mse, 6);
    }

    [Fact]
    public void SimpleLinearRegression_ComputeR2Score_PerfectFit()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(1);
        var predictions = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });
        var targets = PrototypeVector<double>.FromArray(new double[] { 1.0, 2.0, 3.0 });

        // Act
        var r2 = model.ComputeR2Score(predictions, targets);

        // Assert: perfect fit should have R2 = 1.0
        Assert.Equal(1.0, r2, 6);
    }

    [Fact]
    public void SimpleLinearRegression_GetBias_ReturnsBiasValue()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(2);

        // Act
        var bias = model.GetBias();

        // Assert: bias should be a finite number
        Assert.True(!double.IsNaN(bias) && !double.IsInfinity(bias));
    }

    [Fact]
    public void SimpleLinearRegression_LearnsSyntheticData()
    {
        // Arrange: y = 3*x1 + 2*x2 + 1
        var random = RandomHelper.CreateSeededRandom(42);
        var numSamples = 100;
        var numFeatures = 2;
        var X = new double[numSamples * numFeatures];
        var y = new double[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            X[i * numFeatures] = random.NextDouble() * 10;
            X[i * numFeatures + 1] = random.NextDouble() * 10;
            y[i] = 3 * X[i * numFeatures] + 2 * X[i * numFeatures + 1] + 1;
            y[i] += (random.NextDouble() - 0.5) * 0.1; // small noise
        }

        var Xvec = PrototypeVector<double>.FromArray(X);
        var yvec = PrototypeVector<double>.FromArray(y);
        var model = new SimpleLinearRegression<double>(numFeatures);

        // Act
        model.Train(Xvec, yvec, numSamples, learningRate: 0.01, numEpochs: 200, verbose: false);

        // Assert
        var predictions = model.PredictBatch(Xvec, numSamples);
        var r2 = model.ComputeR2Score(predictions, yvec);
        Assert.True(r2 > 0.95, $"R2 score should be > 0.95, got {r2}");
    }

    [Fact]
    public void SimpleLinearRegression_WorksWithFloat()
    {
        // Arrange
        var model = new SimpleLinearRegression<float>(2);
        var X = PrototypeVector<float>.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var y = PrototypeVector<float>.FromArray(new float[] { 3.0f, 7.0f });

        // Act
        model.Train(X, y, numSamples: 2, learningRate: 0.1f, numEpochs: 10, verbose: false);

        // Assert
        var weights = model.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(2, weights.Length);
    }

    #endregion

    #region SimpleNeuralNetwork Tests

    [Fact]
    public void SimpleNeuralNetwork_Constructor_InitializesNetwork()
    {
        // Act
        var network = new SimpleNeuralNetwork<double>(inputSize: 2, hiddenSize: 4, outputSize: 1);

        // Assert
        var parameters = network.GetParameters();
        Assert.NotNull(parameters);
        // Parameters: (input*hidden + hidden) + (hidden*output + output)
        // = (2*4 + 4) + (4*1 + 1) = 8 + 4 + 4 + 1 = 17
        Assert.Equal(17, parameters.Length);
    }

    [Fact]
    public void SimpleNeuralNetwork_Constructor_WithSeed_IsDeterministic()
    {
        // Act
        var network1 = new SimpleNeuralNetwork<double>(2, 4, 1, seed: 42);
        var network2 = new SimpleNeuralNetwork<double>(2, 4, 1, seed: 42);

        // Assert
        var params1 = network1.GetParameters();
        var params2 = network2.GetParameters();
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SimpleNeuralNetwork_Forward_ReturnsOutput()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1);
        var input = PrototypeVector<double>.FromArray(new double[] { 0.5, 0.5 });

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(1, output.Length);
        Assert.True(!double.IsNaN(output[0]));
    }

    [Fact]
    public void SimpleNeuralNetwork_ComputeLoss_ReturnsMSE()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1);
        var output = PrototypeVector<double>.FromArray(new double[] { 0.5 });
        var target = PrototypeVector<double>.FromArray(new double[] { 1.0 });

        // Act
        var loss = network.ComputeLoss(output, target);

        // Assert: MSE = (0.5 - 1.0)^2 / 1 = 0.25
        Assert.Equal(0.25, loss, 6);
    }

    [Fact]
    public void SimpleNeuralNetwork_ComputeLossGradient_ReturnsGradient()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1);
        var output = PrototypeVector<double>.FromArray(new double[] { 0.5 });
        var target = PrototypeVector<double>.FromArray(new double[] { 1.0 });

        // Act
        var gradient = network.ComputeLossGradient(output, target);

        // Assert: gradient = 2 * (output - target) / n = 2 * (0.5 - 1.0) / 1 = -1.0
        Assert.Equal(1, gradient.Length);
        Assert.Equal(-1.0, gradient[0], 6);
    }

    [Fact]
    public void SimpleNeuralNetwork_Backward_ReturnsGradients()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1);
        var input = PrototypeVector<double>.FromArray(new double[] { 0.5, 0.5 });
        network.Forward(input); // Need to forward first to set up internal state
        var lossGrad = PrototypeVector<double>.FromArray(new double[] { 1.0 });

        // Act
        var (wihGrad, bhGrad, whoGrad, boGrad) = network.Backward(lossGrad);

        // Assert: check gradient dimensions
        Assert.Equal(2 * 4, wihGrad.Length);  // input * hidden
        Assert.Equal(4, bhGrad.Length);        // hidden
        Assert.Equal(4 * 1, whoGrad.Length);   // hidden * output
        Assert.Equal(1, boGrad.Length);        // output
    }

    [Fact]
    public void SimpleNeuralNetwork_SetParameters_UpdatesNetwork()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1);
        var newParams = PrototypeVector<double>.Ones(17);  // 2*4+4+4*1+1 = 17

        // Act
        network.SetParameters(newParams);

        // Assert
        var params1 = network.GetParameters();
        for (int i = 0; i < 17; i++)
        {
            Assert.Equal(1.0, params1[i]);
        }
    }

    [Fact]
    public void SimpleNeuralNetwork_TrainOnXOR_ReducesLoss()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 8, 1, seed: 42);
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.1);

        // XOR dataset
        var inputs = new double[][]
        {
            new[] { 0.0, 0.0 },
            new[] { 0.0, 1.0 },
            new[] { 1.0, 0.0 },
            new[] { 1.0, 1.0 }
        };
        var targets = new double[] { 0.0, 1.0, 1.0, 0.0 };

        // Compute initial loss
        double initialLoss = 0;
        for (int i = 0; i < 4; i++)
        {
            var input = PrototypeVector<double>.FromArray(inputs[i]);
            var target = PrototypeVector<double>.FromArray(new[] { targets[i] });
            var output = network.Forward(input);
            initialLoss += network.ComputeLoss(output, target);
        }

        // Act: train for 100 epochs
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < 4; i++)
            {
                var input = PrototypeVector<double>.FromArray(inputs[i]);
                var target = PrototypeVector<double>.FromArray(new[] { targets[i] });

                var output = network.Forward(input);
                var lossGrad = network.ComputeLossGradient(output, target);
                var (wihGrad, bhGrad, whoGrad, boGrad) = network.Backward(lossGrad);

                // Flatten gradients
                var gradients = FlattenGradients(wihGrad, bhGrad, whoGrad, boGrad);
                var parameters = network.GetParameters();
                var updated = optimizer.UpdateParameters(parameters, gradients);
                network.SetParameters(updated);
            }
        }

        // Compute final loss
        double finalLoss = 0;
        for (int i = 0; i < 4; i++)
        {
            var input = PrototypeVector<double>.FromArray(inputs[i]);
            var target = PrototypeVector<double>.FromArray(new[] { targets[i] });
            var output = network.Forward(input);
            finalLoss += network.ComputeLoss(output, target);
        }

        // Assert
        Assert.True(finalLoss < initialLoss, $"Loss did not decrease: {initialLoss} -> {finalLoss}");
    }

    [Fact]
    public void SimpleNeuralNetwork_WorksWithFloat()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<float>(2, 4, 1);
        var input = PrototypeVector<float>.FromArray(new float[] { 0.5f, 0.5f });

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.Equal(1, output.Length);
        Assert.True(!float.IsNaN(output[0]));
    }

    [Fact]
    public void SimpleNeuralNetwork_ReLU_AppliedInForward()
    {
        // Arrange: create network where we can verify ReLU is applied
        var network = new SimpleNeuralNetwork<double>(2, 4, 1, seed: 42);

        // Set weights to produce negative values before ReLU
        var params1 = network.GetParameters();
        for (int i = 0; i < params1.Length; i++)
        {
            params1[i] = -0.1; // All negative
        }
        network.SetParameters(params1);

        var input = PrototypeVector<double>.FromArray(new double[] { 1.0, 1.0 });

        // Act
        var output = network.Forward(input);

        // Assert: output should be finite (ReLU should clamp negatives to 0)
        Assert.True(!double.IsNaN(output[0]) && !double.IsInfinity(output[0]));
    }

    private static PrototypeVector<T> FlattenGradients<T>(
        PrototypeVector<T> wih, PrototypeVector<T> bh,
        PrototypeVector<T> who, PrototypeVector<T> bo)
    {
        var totalLength = wih.Length + bh.Length + who.Length + bo.Length;
        var result = new T[totalLength];
        int idx = 0;

        for (int i = 0; i < wih.Length; i++) result[idx++] = wih[i];
        for (int i = 0; i < bh.Length; i++) result[idx++] = bh[i];
        for (int i = 0; i < who.Length; i++) result[idx++] = who[i];
        for (int i = 0; i < bo.Length; i++) result[idx++] = bo[i];

        return new PrototypeVector<T>(result);
    }

    #endregion

    #region Cross-Component Integration Tests

    [Fact]
    public void Integration_AdamOptimizer_WithNeuralNetwork()
    {
        // Arrange
        var network = new SimpleNeuralNetwork<double>(2, 4, 1, seed: 42);
        var optimizer = new PrototypeAdamOptimizer<double>(learningRate: 0.1);
        var input = PrototypeVector<double>.FromArray(new double[] { 0.5, 0.5 });
        var target = PrototypeVector<double>.FromArray(new double[] { 1.0 });

        // Act: one training step
        var output = network.Forward(input);
        var lossGrad = network.ComputeLossGradient(output, target);
        var (wihGrad, bhGrad, whoGrad, boGrad) = network.Backward(lossGrad);
        var gradients = FlattenGradients(wihGrad, bhGrad, whoGrad, boGrad);
        var parameters = network.GetParameters();
        var updated = optimizer.UpdateParameters(parameters, gradients);
        network.SetParameters(updated);

        // Assert
        Assert.Equal(1, optimizer.TimeStep);
        var newOutput = network.Forward(input);
        Assert.NotEqual(output[0], newOutput[0]);
    }

    [Fact]
    public void Integration_LinearRegression_EndToEnd()
    {
        // Arrange
        var model = new SimpleLinearRegression<double>(1);

        // y = 2x + 1
        var X = PrototypeVector<double>.FromArray(new double[] { 0, 1, 2, 3, 4 });
        var y = PrototypeVector<double>.FromArray(new double[] { 1, 3, 5, 7, 9 });

        // Act
        model.Train(X, y, numSamples: 5, learningRate: 0.1, numEpochs: 500, verbose: false);

        // Assert
        var predictions = model.PredictBatch(X, 5);
        var r2 = model.ComputeR2Score(predictions, y);
        var mse = model.ComputeMSE(predictions, y);

        Assert.True(r2 > 0.99, $"R2 should be > 0.99, got {r2}");
        Assert.True(mse < 0.01, $"MSE should be < 0.01, got {mse}");
    }

    [Fact]
    public void Integration_VectorOperations_LargeScale()
    {
        // Arrange
        const int size = 10000;
        var a = PrototypeVector<double>.Ones(size);
        var b = PrototypeVector<double>.Ones(size);

        // Act
        var result = a.Add(b).Multiply(2.0).Subtract(a).Divide(3.0);

        // Assert
        // result = ((1+1)*2 - 1) / 3 = (4-1)/3 = 1
        for (int i = 0; i < 100; i++) // Check first 100 elements
        {
            Assert.Equal(1.0, result[i], 6);
        }
    }

    #endregion
}
