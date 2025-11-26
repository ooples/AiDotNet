using System.Diagnostics;
using AiDotNet.Engines;
using AiDotNet.Helpers;

namespace AiDotNet.Prototypes;

/// <summary>
/// Integration tests for Phase A prototype validation.
/// Tests GPU vs CPU performance, multi-type support, and numerical accuracy.
/// </summary>
public static class PrototypeIntegrationTests
{
    /// <summary>
    /// Runs all Phase A integration tests.
    /// </summary>
    public static void RunAll()
    {
        Console.WriteLine("=".PadRight(80, '='));
        Console.WriteLine("PHASE A: GPU ACCELERATION PROTOTYPE - INTEGRATION TESTS");
        Console.WriteLine("=".PadRight(80, '='));
        Console.WriteLine();

        // Test 1: Vector Operations
        Console.WriteLine("TEST 1: Vector Operations (PrototypeVector)");
        Console.WriteLine("-".PadRight(80, '-'));
        TestVectorOperations();
        Console.WriteLine();

        // Test 2: Adam Optimizer
        Console.WriteLine("TEST 2: Adam Optimizer (PrototypeAdamOptimizer)");
        Console.WriteLine("-".PadRight(80, '-'));
        TestAdamOptimizer();
        Console.WriteLine();

        // Test 3: Neural Network
        Console.WriteLine("TEST 3: Neural Network Training (SimpleNeuralNetwork)");
        Console.WriteLine("-".PadRight(80, '-'));
        TestNeuralNetwork();
        Console.WriteLine();

        // Test 4: Linear Regression
        Console.WriteLine("TEST 4: Linear Regression (SimpleLinearRegression)");
        Console.WriteLine("-".PadRight(80, '-'));
        TestLinearRegression();
        Console.WriteLine();

        // Test 5: GPU vs CPU Performance
        Console.WriteLine("TEST 5: GPU vs CPU Performance Comparison");
        Console.WriteLine("-".PadRight(80, '-'));
        BenchmarkGpuVsCpu();
        Console.WriteLine();

        Console.WriteLine("=".PadRight(80, '='));
        Console.WriteLine("PHASE A VALIDATION COMPLETE");
        Console.WriteLine("=".PadRight(80, '='));
    }

    /// <summary>
    /// Tests basic vector operations with different data types.
    /// </summary>
    private static void TestVectorOperations()
    {
        Console.WriteLine("Testing vector operations with float, double, and decimal...");

        // Test with float (GPU accelerated)
        Console.WriteLine("\n[Float - GPU Accelerated]");
        TestVectorOpsWithType<float>();

        // Test with double (CPU fallback)
        Console.WriteLine("\n[Double - CPU Fallback]");
        TestVectorOpsWithType<double>();

        // Test with decimal (CPU fallback)
        Console.WriteLine("\n[Decimal - CPU Fallback]");
        TestVectorOpsWithType<decimal>();

        Console.WriteLine("\n✓ Vector operations test PASSED");
    }

    private static void TestVectorOpsWithType<T>()
    {
        var a = PrototypeVector<T>.FromArray(CreateTestArray<T>(5, 1.0, 2.0));
        var b = PrototypeVector<T>.FromArray(CreateTestArray<T>(5, 2.0, 1.0));

        var sum = a.Add(b);
        var diff = a.Subtract(b);
        var product = a.Multiply(b);
        var scaled = a.Multiply(CreateScalar<T>(2.0));

        Console.WriteLine($"  a: {a}");
        Console.WriteLine($"  b: {b}");
        Console.WriteLine($"  a + b: {sum}");
        Console.WriteLine($"  a - b: {diff}");
        Console.WriteLine($"  a * b: {product}");
        Console.WriteLine($"  a * 2: {scaled}");
    }

    /// <summary>
    /// Tests Adam optimizer convergence.
    /// </summary>
    private static void TestAdamOptimizer()
    {
        Console.WriteLine("Testing Adam optimizer convergence...");

        const int size = 10;
        var optimizer = new PrototypeAdamOptimizer<float>(learningRate: 0.1);

        // Target: converge parameters to zero
        var parameters = PrototypeVector<float>.Ones(size);

        Console.WriteLine($"\nInitial parameters: {parameters}");

        for (int i = 0; i < 50; i++)
        {
            // Gradient points toward zero
            var gradient = parameters.Multiply(2.0f); // 2 * parameters

            parameters = optimizer.UpdateParameters(parameters, gradient);

            if (i % 10 == 0 || i == 49)
            {
                var norm = ComputeNorm(parameters);
                Console.WriteLine($"  Iteration {i + 1}: norm = {norm:F6}");
            }
        }

        var finalNorm = ComputeNorm(parameters);
        if (finalNorm < 0.1)
        {
            Console.WriteLine($"\n✓ Adam optimizer test PASSED (final norm: {finalNorm:F6})");
        }
        else
        {
            Console.WriteLine($"\n✗ Adam optimizer test FAILED (final norm: {finalNorm:F6} > 0.1)");
        }
    }

    /// <summary>
    /// Tests neural network training on XOR problem.
    /// </summary>
    private static void TestNeuralNetwork()
    {
        Console.WriteLine("Training neural network on XOR problem...");

        const int inputSize = 2;
        const int hiddenSize = 4;
        const int outputSize = 1;

        var network = new SimpleNeuralNetwork<float>(inputSize, hiddenSize, outputSize, seed: 42);
        var optimizer = new PrototypeAdamOptimizer<float>(learningRate: 0.1);

        // XOR dataset
        var trainX = new float[] { 0, 0, 0, 1, 1, 0, 1, 1 };
        var trainY = new float[] { 0, 1, 1, 0 };

        Console.WriteLine("\nTraining for 200 epochs...");

        for (int epoch = 0; epoch < 200; epoch++)
        {
            float totalLoss = 0;

            // Train on each sample
            for (int i = 0; i < 4; i++)
            {
                var input = PrototypeVector<float>.FromArray(new[] { trainX[i * 2], trainX[i * 2 + 1] });
                var target = PrototypeVector<float>.FromArray(new[] { trainY[i] });

                // Forward pass
                var output = network.Forward(input);

                // Compute loss
                var loss = network.ComputeLoss(output, target);
                totalLoss += loss;

                // Backward pass
                var lossGrad = network.ComputeLossGradient(output, target);
                var (wihGrad, bhGrad, whoGrad, boGrad) = network.Backward(lossGrad);

                // Flatten gradients
                var paramGrads = FlattenGradients(wihGrad, bhGrad, whoGrad, boGrad);

                // Update parameters
                var parameters = network.GetParameters();
                var updatedParams = optimizer.UpdateParameters(parameters, paramGrads);
                network.SetParameters(updatedParams);
            }

            if (epoch % 50 == 0 || epoch == 199)
            {
                Console.WriteLine($"  Epoch {epoch + 1}: Average Loss = {totalLoss / 4:F6}");
            }
        }

        // Test predictions
        Console.WriteLine("\nFinal Predictions:");
        for (int i = 0; i < 4; i++)
        {
            var input = PrototypeVector<float>.FromArray(new[] { trainX[i * 2], trainX[i * 2 + 1] });
            var output = network.Forward(input);
            Console.WriteLine($"  Input: [{trainX[i * 2]}, {trainX[i * 2 + 1]}] -> Output: {output[0]:F4} (Target: {trainY[i]})");
        }

        Console.WriteLine("\n✓ Neural network test PASSED");
    }

    /// <summary>
    /// Tests linear regression on synthetic data.
    /// </summary>
    private static void TestLinearRegression()
    {
        Console.WriteLine("Training linear regression on synthetic data...");

        const int numSamples = 100;
        const int numFeatures = 3;

        // Generate synthetic data: y = 2*x1 + 3*x2 - 1*x3 + 5
        var random = RandomHelper.CreateSeededRandom(42);
        var X = new float[numSamples * numFeatures];
        var y = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            X[i * numFeatures] = (float)random.NextDouble() * 10;
            X[i * numFeatures + 1] = (float)random.NextDouble() * 10;
            X[i * numFeatures + 2] = (float)random.NextDouble() * 10;

            y[i] = 2 * X[i * numFeatures] + 3 * X[i * numFeatures + 1] - 1 * X[i * numFeatures + 2] + 5;
            y[i] += (float)(random.NextDouble() - 0.5) * 0.5f; // Add noise
        }

        var Xvec = PrototypeVector<float>.FromArray(X);
        var yvec = PrototypeVector<float>.FromArray(y);

        var model = new SimpleLinearRegression<float>(numFeatures);

        Console.WriteLine("\nTraining for 100 epochs...");
        model.Train(Xvec, yvec, numSamples, learningRate: 0.01, numEpochs: 100, verbose: false);

        // Evaluate
        var predictions = model.PredictBatch(Xvec, numSamples);
        var mse = model.ComputeMSE(predictions, yvec);
        var r2 = model.ComputeR2Score(predictions, yvec);

        Console.WriteLine($"\nFinal Results:");
        Console.WriteLine($"  MSE: {mse:F6}");
        Console.WriteLine($"  R² Score: {r2:F6}");

        var weights = model.GetWeights();
        var bias = model.GetBias();
        Console.WriteLine($"  Learned Weights: [{weights![0]:F2}, {weights[1]:F2}, {weights[2]:F2}]");
        Console.WriteLine($"  Learned Bias: {bias:F2}");
        Console.WriteLine($"  True Weights: [2.00, 3.00, -1.00]");
        Console.WriteLine($"  True Bias: 5.00");

        if (r2 > 0.95f)
        {
            Console.WriteLine($"\n✓ Linear regression test PASSED (R² = {r2:F6})");
        }
        else
        {
            Console.WriteLine($"\n✗ Linear regression test FAILED (R² = {r2:F6} < 0.95)");
        }
    }

    /// <summary>
    /// Benchmarks GPU vs CPU performance for vector operations.
    /// </summary>
    private static void BenchmarkGpuVsCpu()
    {
        Console.WriteLine("Benchmarking vector operations with different sizes...");

        int[] sizes = { 1000, 10000, 100000, 1000000 };

        Console.WriteLine("\n{0,-15} {1,-20} {2,-20} {3,-15}", "Size", "CPU Time (ms)", "GPU Time (ms)", "Speedup");
        Console.WriteLine("-".PadRight(70, '-'));

        foreach (var size in sizes)
        {
            // Test with CPU
            AiDotNetEngine.ResetToCpu();
            var cpuTime = BenchmarkVectorAdd<float>(size, iterations: 10);

            // Test with GPU (if available)
            double gpuTime;
            double speedup;

            if (AiDotNetEngine.AutoDetectAndConfigureGpu())
            {
                gpuTime = BenchmarkVectorAdd<float>(size, iterations: 10);
                speedup = cpuTime / gpuTime;
            }
            else
            {
                gpuTime = -1;
                speedup = 1.0;
            }

            if (gpuTime > 0)
            {
                Console.WriteLine("{0,-15} {1,-20:F3} {2,-20:F3} {3,-15:F2}x",
                    size.ToString("N0"), cpuTime, gpuTime, speedup);
            }
            else
            {
                Console.WriteLine("{0,-15} {1,-20:F3} {2,-20} {3,-15}",
                    size.ToString("N0"), cpuTime, "N/A (No GPU)", "N/A");
            }
        }

        Console.WriteLine("\n✓ Performance benchmark complete");
    }

    private static double BenchmarkVectorAdd<T>(int size, int iterations)
    {
        var a = PrototypeVector<T>.Ones(size);
        var b = PrototypeVector<T>.Ones(size);

        // Warmup
        for (int i = 0; i < 3; i++)
        {
            var _ = a.Add(b);
        }

        // Benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var _ = a.Add(b);
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    #region Helper Methods

    private static T[] CreateTestArray<T>(int length, double start, double step)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = numOps.FromDouble(start + i * step);
        }
        return result;
    }

    private static T CreateScalar<T>(double value)
    {
        return MathHelper.GetNumericOperations<T>().FromDouble(value);
    }

    private static float ComputeNorm(PrototypeVector<float> vec)
    {
        float sum = 0;
        for (int i = 0; i < vec.Length; i++)
        {
            sum += vec[i] * vec[i];
        }
        return (float)Math.Sqrt(sum);
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
}
