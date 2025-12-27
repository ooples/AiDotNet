using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Benchmark example for Self-Supervised Learning on CIFAR-10.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This example shows how to pretrain a neural network using
/// self-supervised learning (SSL) on image data without labels, then evaluate the learned
/// representations using k-NN classification.</para>
///
/// <para><b>What this benchmark demonstrates:</b></para>
/// <list type="bullet">
/// <item>SimCLR contrastive learning on CIFAR-10 images</item>
/// <item>Large batch training with LARS optimizer</item>
/// <item>k-NN evaluation of representation quality</item>
/// <item>Comparison between SSL methods (SimCLR, MoCo, BYOL)</item>
/// </list>
///
/// <para><b>Expected Results (industry baselines):</b></para>
/// <list type="bullet">
/// <item>SimCLR (100 epochs): ~85% k-NN accuracy on CIFAR-10</item>
/// <item>MoCo v2 (200 epochs): ~86% k-NN accuracy</item>
/// <item>BYOL (300 epochs): ~89% k-NN accuracy</item>
/// </list>
/// </remarks>
public class SSLBenchmarkExample
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    // CIFAR-10 parameters
    private const int ImageSize = 32;
    private const int Channels = 3;
    private const int NumClasses = 10;
    private const int TrainSamples = 50000;
    private const int TestSamples = 10000;

    /// <summary>
    /// Runs the complete SSL benchmark suite.
    /// </summary>
    public static async Task RunBenchmarkAsync()
    {
        Console.WriteLine("=================================================");
        Console.WriteLine("Self-Supervised Learning Benchmark on CIFAR-10");
        Console.WriteLine("=================================================\n");

        // Load CIFAR-10 data (simulated for this example)
        Console.WriteLine("Loading CIFAR-10 dataset...");
        var (trainData, trainLabels, testData, testLabels) = GenerateSyntheticCIFAR10();
        Console.WriteLine($"  Training samples: {trainData.Shape[0]}");
        Console.WriteLine($"  Test samples: {testData.Shape[0]}");
        Console.WriteLine();

        // Run benchmarks for each SSL method
        await BenchmarkSimCLR(trainData, trainLabels, testData, testLabels);
        await BenchmarkMoCo(trainData, trainLabels, testData, testLabels);
        await BenchmarkBYOL(trainData, trainLabels, testData, testLabels);

        Console.WriteLine("\n=================================================");
        Console.WriteLine("Benchmark Complete");
        Console.WriteLine("=================================================");
    }

    /// <summary>
    /// Benchmark SimCLR training.
    /// </summary>
    private static async Task BenchmarkSimCLR(
        Tensor<double> trainData, int[] trainLabels,
        Tensor<double> testData, int[] testLabels)
    {
        Console.WriteLine("=== SimCLR Benchmark ===");
        Console.WriteLine("Configuration:");
        Console.WriteLine("  - Batch size: 256 (effective: 256 for single GPU)");
        Console.WriteLine("  - Epochs: 100");
        Console.WriteLine("  - Optimizer: LARS");
        Console.WriteLine("  - Learning rate: 0.3 * batch_size/256 = 0.3");
        Console.WriteLine("  - Temperature: 0.1");
        Console.WriteLine();

        // Create encoder (ResNet-like for CIFAR-10)
        var encoder = CreateResNetEncoder();

        // Create SimCLR method
        var config = new SSLConfig
        {
            Method = SSLMethodType.SimCLR,
            PretrainingEpochs = 100,
            BatchSize = 256,
            LearningRate = 0.3,
            Temperature = 0.1,
            ProjectorOutputDimension = 128,
            ProjectorHiddenDimension = 512,
            WarmupEpochs = 10,
            UseCosineDecay = true,
            OptimizerType = SSLOptimizerType.LARS,
            EnableKNNEvaluation = true,
            KNNNeighbors = 20
        };

        var simclr = SimCLR<double>.Create(
            encoder,
            encoderOutputDim: 512,
            projectionDim: 128,
            hiddenDim: 512);

        // Create session and train
        var session = new SSLSession<double>(simclr, config);

        // Hook up progress monitoring
        session.OnEpochEnd += (epoch, loss) =>
        {
            if (epoch % 10 == 0 || epoch == 0)
            {
                Console.WriteLine($"  Epoch {epoch + 1}: Loss = {NumOps.ToDouble(loss):F4}");
            }
        };

        session.OnCollapseDetected += (epoch) =>
        {
            Console.WriteLine($"  [WARNING] Representation collapse detected at epoch {epoch + 1}");
        };

        // Cache training features for k-NN evaluation
        var trainBatches = CreateBatches(trainData, config.BatchSize ?? 256);
        session.CacheTrainingFeaturesForKNN(trainBatches, trainLabels);

        Console.WriteLine("Training...");
        var startTime = DateTime.Now;
        var result = session.Train(() => trainBatches, testData, testLabels);
        var trainingTime = DateTime.Now - startTime;

        // Report results
        Console.WriteLine($"\nResults:");
        Console.WriteLine($"  Training time: {trainingTime.TotalMinutes:F1} minutes");
        Console.WriteLine($"  Final loss: {NumOps.ToDouble(result.FinalLoss):F4}");
        Console.WriteLine($"  Epochs trained: {result.EpochsTrained}");

        // Final k-NN evaluation
        var knnAccuracy = EvaluateKNN(session.Method, trainData, trainLabels, testData, testLabels);
        Console.WriteLine($"  k-NN Accuracy: {knnAccuracy * 100:F2}%");
        Console.WriteLine();

        await Task.CompletedTask;
    }

    /// <summary>
    /// Benchmark MoCo v2 training.
    /// </summary>
    private static async Task BenchmarkMoCo(
        Tensor<double> trainData, int[] trainLabels,
        Tensor<double> testData, int[] testLabels)
    {
        Console.WriteLine("=== MoCo v2 Benchmark ===");
        Console.WriteLine("Configuration:");
        Console.WriteLine("  - Batch size: 256");
        Console.WriteLine("  - Epochs: 200");
        Console.WriteLine("  - Queue size: 65536");
        Console.WriteLine("  - Momentum: 0.999");
        Console.WriteLine("  - Temperature: 0.07");
        Console.WriteLine();

        // Create encoder
        var encoder = CreateResNetEncoder();

        // Create MoCo v2 method
        var config = new SSLConfig
        {
            Method = SSLMethodType.MoCoV2,
            PretrainingEpochs = 200,
            BatchSize = 256,
            LearningRate = 0.03,
            Temperature = 0.07,
            WarmupEpochs = 10,
            OptimizerType = SSLOptimizerType.LARS,
            MoCo = new MoCoConfig
            {
                QueueSize = 65536,
                MomentumCoefficient = 0.999,
                UseMLPProjector = true
            }
        };

        var moco = MoCoV2<double>.Create(
            encoder,
            createEncoderCopy: e => CreateResNetEncoder(), // Create copy for momentum encoder
            encoderOutputDim: 512,
            projectionDim: 128,
            hiddenDim: 512,
            queueSize: 65536);

        var session = new SSLSession<double>(moco, config);

        session.OnEpochEnd += (epoch, loss) =>
        {
            if (epoch % 20 == 0 || epoch == 0)
            {
                Console.WriteLine($"  Epoch {epoch + 1}: Loss = {NumOps.ToDouble(loss):F4}");
            }
        };

        var trainBatches = CreateBatches(trainData, config.BatchSize ?? 256);
        session.CacheTrainingFeaturesForKNN(trainBatches, trainLabels);

        Console.WriteLine("Training...");
        var startTime = DateTime.Now;
        var result = session.Train(() => trainBatches, testData, testLabels);
        var trainingTime = DateTime.Now - startTime;

        Console.WriteLine($"\nResults:");
        Console.WriteLine($"  Training time: {trainingTime.TotalMinutes:F1} minutes");
        Console.WriteLine($"  Final loss: {NumOps.ToDouble(result.FinalLoss):F4}");

        var knnAccuracy = EvaluateKNN(session.Method, trainData, trainLabels, testData, testLabels);
        Console.WriteLine($"  k-NN Accuracy: {knnAccuracy * 100:F2}%");
        Console.WriteLine();

        await Task.CompletedTask;
    }

    /// <summary>
    /// Benchmark BYOL training.
    /// </summary>
    private static async Task BenchmarkBYOL(
        Tensor<double> trainData, int[] trainLabels,
        Tensor<double> testData, int[] testLabels)
    {
        Console.WriteLine("=== BYOL Benchmark ===");
        Console.WriteLine("Configuration:");
        Console.WriteLine("  - Batch size: 256");
        Console.WriteLine("  - Epochs: 300");
        Console.WriteLine("  - Target momentum: 0.996 -> 1.0 (cosine)");
        Console.WriteLine("  - No negative samples needed");
        Console.WriteLine();

        var encoder = CreateResNetEncoder();

        var config = new SSLConfig
        {
            Method = SSLMethodType.BYOL,
            PretrainingEpochs = 300,
            BatchSize = 256,
            LearningRate = 0.2,
            WarmupEpochs = 10,
            OptimizerType = SSLOptimizerType.LARS,
            BYOL = new BYOLConfig
            {
                BaseMomentum = 0.996,
                FinalMomentum = 1.0,
                UseMomentumSchedule = true
            }
        };

        var byol = BYOL<double>.Create(
            encoder,
            createEncoderCopy: e => CreateResNetEncoder(),
            encoderOutputDim: 512,
            projectionDim: 256,
            hiddenDim: 4096);

        var session = new SSLSession<double>(byol, config);

        session.OnEpochEnd += (epoch, loss) =>
        {
            if (epoch % 30 == 0 || epoch == 0)
            {
                Console.WriteLine($"  Epoch {epoch + 1}: Loss = {NumOps.ToDouble(loss):F4}");
            }
        };

        var trainBatches = CreateBatches(trainData, config.BatchSize ?? 256);
        session.CacheTrainingFeaturesForKNN(trainBatches, trainLabels);

        Console.WriteLine("Training...");
        var startTime = DateTime.Now;
        var result = session.Train(() => trainBatches, testData, testLabels);
        var trainingTime = DateTime.Now - startTime;

        Console.WriteLine($"\nResults:");
        Console.WriteLine($"  Training time: {trainingTime.TotalMinutes:F1} minutes");
        Console.WriteLine($"  Final loss: {NumOps.ToDouble(result.FinalLoss):F4}");

        var knnAccuracy = EvaluateKNN(session.Method, trainData, trainLabels, testData, testLabels);
        Console.WriteLine($"  k-NN Accuracy: {knnAccuracy * 100:F2}%");
        Console.WriteLine();

        await Task.CompletedTask;
    }

    /// <summary>
    /// Creates a ResNet-like encoder for CIFAR-10.
    /// </summary>
    private static INeuralNetwork<double> CreateResNetEncoder()
    {
        // Simplified ResNet-like architecture for CIFAR-10
        // Input: 32x32x3 -> Output: 512-dim features
        var layers = new List<ILayer<double>>
        {
            // Initial conv: 32x32x3 -> 32x32x64
            new ConvolutionalLayer<double>(
                inputChannels: Channels,
                outputChannels: 64,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationType: ActivationType.ReLU),
            new BatchNormalizationLayer<double>(64),

            // Conv block 1: 32x32x64 -> 16x16x128
            new ConvolutionalLayer<double>(64, 128, 3, 2, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(128),
            new ConvolutionalLayer<double>(128, 128, 3, 1, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(128),

            // Conv block 2: 16x16x128 -> 8x8x256
            new ConvolutionalLayer<double>(128, 256, 3, 2, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(256),
            new ConvolutionalLayer<double>(256, 256, 3, 1, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(256),

            // Conv block 3: 8x8x256 -> 4x4x512
            new ConvolutionalLayer<double>(256, 512, 3, 2, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(512),
            new ConvolutionalLayer<double>(512, 512, 3, 1, 1, ActivationType.ReLU),
            new BatchNormalizationLayer<double>(512),

            // Global average pooling: 4x4x512 -> 512
            new GlobalAveragePoolingLayer<double>()
        };

        return new NeuralNetwork<double>(layers);
    }

    /// <summary>
    /// Generates synthetic CIFAR-10-like data for testing.
    /// </summary>
    private static (Tensor<double> trainData, int[] trainLabels,
                    Tensor<double> testData, int[] testLabels) GenerateSyntheticCIFAR10()
    {
        // For actual benchmarking, load real CIFAR-10 data
        // This synthetic data is for demonstration purposes
        var random = new Random(42);

        // Generate synthetic training data
        int trainN = 1000; // Reduced for testing
        int testN = 200;
        int dim = ImageSize * ImageSize * Channels;

        var trainFlat = new double[trainN * dim];
        var trainLabels = new int[trainN];
        for (int i = 0; i < trainN; i++)
        {
            trainLabels[i] = i % NumClasses;
            for (int j = 0; j < dim; j++)
            {
                // Generate class-dependent patterns
                trainFlat[i * dim + j] = (trainLabels[i] * 0.1 + random.NextDouble()) / NumClasses;
            }
        }
        var trainData = new Tensor<double>(trainFlat, [trainN, Channels, ImageSize, ImageSize]);

        // Generate synthetic test data
        var testFlat = new double[testN * dim];
        var testLabels = new int[testN];
        for (int i = 0; i < testN; i++)
        {
            testLabels[i] = i % NumClasses;
            for (int j = 0; j < dim; j++)
            {
                testFlat[i * dim + j] = (testLabels[i] * 0.1 + random.NextDouble()) / NumClasses;
            }
        }
        var testData = new Tensor<double>(testFlat, [testN, Channels, ImageSize, ImageSize]);

        return (trainData, trainLabels, testData, testLabels);
    }

    /// <summary>
    /// Creates batches from the full dataset.
    /// </summary>
    private static IEnumerable<Tensor<double>> CreateBatches(Tensor<double> data, int batchSize)
    {
        int numSamples = data.Shape[0];
        int numBatches = (numSamples + batchSize - 1) / batchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            // Extract batch
            var batchData = new double[currentBatchSize * data.Shape[1] * data.Shape[2] * data.Shape[3]];
            int stride = data.Shape[1] * data.Shape[2] * data.Shape[3];

            for (int i = 0; i < currentBatchSize; i++)
            {
                for (int j = 0; j < stride; j++)
                {
                    batchData[i * stride + j] = data[(start + i) * stride + j];
                }
            }

            yield return new Tensor<double>(batchData, [currentBatchSize, data.Shape[1], data.Shape[2], data.Shape[3]]);
        }
    }

    /// <summary>
    /// Evaluates k-NN accuracy on the learned representations.
    /// </summary>
    private static double EvaluateKNN(
        ISSLMethod<double> method,
        Tensor<double> trainData, int[] trainLabels,
        Tensor<double> testData, int[] testLabels,
        int k = 20)
    {
        // Encode all samples
        var trainFeatures = method.Encode(trainData);
        var testFeatures = method.Encode(testData);

        int numTest = testFeatures.Shape[0];
        int numTrain = trainFeatures.Shape[0];
        int dim = trainFeatures.Shape[1];

        int correct = 0;

        for (int t = 0; t < numTest; t++)
        {
            // Compute distances to all training samples
            var distances = new (double dist, int label)[numTrain];

            for (int i = 0; i < numTrain; i++)
            {
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = NumOps.ToDouble(testFeatures[t, d]) - NumOps.ToDouble(trainFeatures[i, d]);
                    dist += diff * diff;
                }
                distances[i] = (dist, trainLabels[i]);
            }

            // Sort by distance
            Array.Sort(distances, (a, b) => a.dist.CompareTo(b.dist));

            // Majority vote among k nearest
            var votes = new Dictionary<int, int>();
            for (int i = 0; i < Math.Min(k, numTrain); i++)
            {
                var label = distances[i].label;
                votes[label] = votes.GetValueOrDefault(label, 0) + 1;
            }

            // Find predicted label (manual for .NET 4.7.1 compatibility)
            int predictedLabel = 0;
            int maxVotes = -1;
            foreach (var kv in votes)
            {
                if (kv.Value > maxVotes)
                {
                    maxVotes = kv.Value;
                    predictedLabel = kv.Key;
                }
            }

            if (predictedLabel == testLabels[t])
            {
                correct++;
            }
        }

        return (double)correct / numTest;
    }
}
