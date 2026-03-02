using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;

namespace AiDotNetTestConsole.Examples.MetaLearning;

/// <summary>
/// Demonstrates the AiDotNet meta-learning suite with runnable examples covering
/// MAML, ProtoNets, Neural Processes, data infrastructure, and algorithm comparison.
/// </summary>
public class MetaLearningExample
{
    public void RunExample()
    {
        Console.WriteLine("Meta-Learning Examples");
        Console.WriteLine("=====================\n");
        Console.WriteLine("1. MAML (Model-Agnostic Meta-Learning) — Few-shot adaptation");
        Console.WriteLine("2. ProtoNets — Prototypical Networks for metric-based learning");
        Console.WriteLine("3. CNP (Conditional Neural Process) — Function regression");
        Console.WriteLine("4. Episodic Data Infrastructure — Datasets and samplers");
        Console.WriteLine("5. Algorithm Comparison — Benchmark multiple algorithms");
        Console.WriteLine("0. Back to main menu");
        Console.WriteLine();
        Console.Write("Select an example (0-5): ");

        if (!int.TryParse(Console.ReadLine(), out int choice))
        {
            Console.WriteLine("Invalid input.");
            return;
        }

        Console.WriteLine();

        switch (choice)
        {
            case 0: return;
            case 1: RunMAMLExample(); break;
            case 2: RunProtoNetsExample(); break;
            case 3: RunCNPExample(); break;
            case 4: RunDataInfrastructureExample(); break;
            case 5: RunAlgorithmComparison(); break;
            default:
                Console.WriteLine("Invalid choice.");
                break;
        }
    }

    /// <summary>
    /// Demonstrates MAML: learns an initialization that adapts quickly to new tasks.
    /// Creates synthetic classification tasks and shows loss reduction over meta-training.
    /// </summary>
    private static void RunMAMLExample()
    {
        Console.WriteLine("MAML (Model-Agnostic Meta-Learning)");
        Console.WriteLine("===================================\n");
        Console.WriteLine("MAML learns model parameters that can be quickly adapted to new tasks.");
        Console.WriteLine("It optimizes for the best 'starting point' across many tasks.\n");

        const int inputDim = 4;
        const int numMetaEpochs = 10;
        const int tasksPerEpoch = 4;
        var rng = new Random(42);

        // Create model and MAML algorithm
        var model = new SimpleMetaModel(inputDim);
        var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.005,
            AdaptationSteps = 3,
            UseFirstOrder = true // Reptile-style for speed
        };
        var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

        Console.WriteLine($"Model: Linear model with {model.ParameterCount} parameters");
        Console.WriteLine($"Inner LR: {options.InnerLearningRate}, Outer LR: {options.OuterLearningRate}");
        Console.WriteLine($"Adaptation steps: {options.AdaptationSteps}");
        Console.WriteLine($"\nMeta-training over {numMetaEpochs} epochs, {tasksPerEpoch} tasks each...\n");

        var initialParams = model.GetParameters();

        // Meta-training loop
        for (int epoch = 0; epoch < numMetaEpochs; epoch++)
        {
            var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[tasksPerEpoch];
            for (int t = 0; t < tasksPerEpoch; t++)
            {
                tasks[t] = CreateClassificationTask(inputDim, numWays: 2, numShots: 3,
                    numQuery: 4, rng);
            }

            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            var loss = maml.MetaTrain(batch);

            if (epoch % 2 == 0 || epoch == numMetaEpochs - 1)
            {
                Console.WriteLine($"  Epoch {epoch + 1,2}/{numMetaEpochs}: Meta-loss = {loss:F6}");
            }
        }

        // Show parameter change
        var finalParams = model.GetParameters();
        double paramDelta = ComputeParamDelta(initialParams, finalParams);
        Console.WriteLine($"\nParameter change (L2 norm): {paramDelta:F6}");

        // Demonstrate adaptation on a new unseen task
        Console.WriteLine("\n--- Adapting to a new task ---");
        var newTask = CreateClassificationTask(inputDim, 2, 5, 5, rng);
        var adapted = maml.Adapt(newTask);

        var predictions = adapted.Predict(newTask.QuerySetX);
        Console.WriteLine($"Query predictions (first 5): [{FormatVector(predictions, 5)}]");
        Console.WriteLine($"Query labels     (first 5): [{FormatVector(newTask.QuerySetY, 5)}]");

        double accuracy = ComputeAccuracy(predictions, newTask.QuerySetY);
        Console.WriteLine($"Adaptation accuracy: {accuracy:P1}");
        Console.WriteLine("\nMAML example completed.");
    }

    /// <summary>
    /// Demonstrates Prototypical Networks: classifies by distance to class prototypes.
    /// </summary>
    private static void RunProtoNetsExample()
    {
        Console.WriteLine("Prototypical Networks (ProtoNets)");
        Console.WriteLine("================================\n");
        Console.WriteLine("ProtoNets classifies by computing distances to class prototypes.");
        Console.WriteLine("Each prototype is the mean embedding of support examples for a class.\n");

        const int inputDim = 4;
        const int numEpochs = 10;
        var rng = new Random(123);

        var model = new SimpleMetaModel(inputDim);
        var options = new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.005,
            DistanceFunction = ProtoNetsDistanceFunction.Euclidean,
            Temperature = 1.0
        };
        var protonets = new ProtoNetsAlgorithm<double, Matrix<double>, Vector<double>>(options);

        Console.WriteLine($"Distance function: {options.DistanceFunction}");
        Console.WriteLine($"Temperature: {options.Temperature}");
        Console.WriteLine($"\nMeta-training over {numEpochs} epochs...\n");

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[3];
            for (int t = 0; t < 3; t++)
            {
                tasks[t] = CreateClassificationTask(inputDim, 3, 3, 4, rng);
            }

            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            var loss = protonets.MetaTrain(batch);

            Console.WriteLine($"  Epoch {epoch + 1,2}/{numEpochs}: Loss = {loss:F6}");
        }

        // Adapt and predict
        Console.WriteLine("\n--- Few-shot classification on new task ---");
        var evalTask = CreateClassificationTask(inputDim, 3, 5, 5, rng);
        var adapted = protonets.Adapt(evalTask);
        var preds = adapted.Predict(evalTask.QuerySetX);

        Console.WriteLine($"Query predictions: [{FormatVector(preds, 8)}]");
        Console.WriteLine($"Query labels:      [{FormatVector(evalTask.QuerySetY, 8)}]");

        double acc = ComputeAccuracy(preds, evalTask.QuerySetY);
        Console.WriteLine($"Classification accuracy: {acc:P1}");
        Console.WriteLine("\nProtoNets example completed.");
    }

    /// <summary>
    /// Demonstrates CNP: learns to predict function values from context points.
    /// Uses sine-wave-like regression tasks.
    /// </summary>
    private static void RunCNPExample()
    {
        Console.WriteLine("Conditional Neural Process (CNP)");
        Console.WriteLine("===============================\n");
        Console.WriteLine("CNP encodes context points, aggregates into a representation,");
        Console.WriteLine("and decodes predictions at target locations.\n");

        const int inputDim = 3;
        const int numEpochs = 10;
        var rng = new Random(456);

        var model = new SimpleMetaModel(inputDim);
        var options = new CNPOptions<double, Matrix<double>, Vector<double>>(model)
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.005,
            RepresentationDim = 32
        };
        var cnp = new CNPAlgorithm<double, Matrix<double>, Vector<double>>(options);

        Console.WriteLine($"Representation dim: {options.RepresentationDim}");
        Console.WriteLine($"\nMeta-training on regression tasks...\n");

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[4];
            for (int t = 0; t < 4; t++)
            {
                tasks[t] = CreateRegressionTask(inputDim, numContext: 5, numTarget: 5, rng);
            }

            var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
            var loss = cnp.MetaTrain(batch);

            Console.WriteLine($"  Epoch {epoch + 1,2}/{numEpochs}: Loss = {loss:F6}");
        }

        // Evaluate
        Console.WriteLine("\n--- Predicting on new regression task ---");
        var evalTask = CreateRegressionTask(inputDim, 5, 5, rng);
        var adapted = cnp.Adapt(evalTask);
        var preds = adapted.Predict(evalTask.QuerySetX);

        Console.WriteLine("Target predictions vs actual (first 5):");
        int count = Math.Min(5, preds.Length);
        for (int i = 0; i < count; i++)
        {
            Console.WriteLine($"  Predicted: {preds[i],8:F4}  |  Actual: {evalTask.QuerySetY[i],8:F4}");
        }

        double mse = ComputeMSE(preds, evalTask.QuerySetY);
        Console.WriteLine($"\nMean Squared Error: {mse:F6}");
        Console.WriteLine("\nCNP example completed.");
    }

    /// <summary>
    /// Demonstrates the episodic data infrastructure: datasets, samplers, and episodes.
    /// </summary>
    private static void RunDataInfrastructureExample()
    {
        Console.WriteLine("Episodic Data Infrastructure");
        Console.WriteLine("============================\n");
        Console.WriteLine("AiDotNet provides datasets and samplers for episodic meta-learning.\n");

        // 1. SineWaveMetaDataset
        Console.WriteLine("--- SineWaveMetaDataset ---");
        var sineDataset = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, seed: 42);
        Console.WriteLine($"Name: {sineDataset.Name}");
        Console.WriteLine($"Total classes: {sineDataset.TotalClasses}");
        Console.WriteLine($"Total examples: {sineDataset.TotalExamples}");

        var sineEpisode = sineDataset.SampleEpisode(numWays: 2, numShots: 3, numQueryPerClass: 4);
        Console.WriteLine($"Sampled episode: {sineEpisode.Task.NumWays}-way {sineEpisode.Task.NumShots}-shot");
        Console.WriteLine($"Episode domain: {sineEpisode.Domain ?? "N/A"}");
        Console.WriteLine($"Episode difficulty: {sineEpisode.Difficulty:F3}\n");

        // 2. GaussianClassificationMetaDataset
        Console.WriteLine("--- GaussianClassificationMetaDataset ---");
        var gaussianDataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 10, examplesPerClass: 20, featureDim: 4, classSeparation: 2.0, seed: 42);
        Console.WriteLine($"Name: {gaussianDataset.Name}");
        Console.WriteLine($"Total classes: {gaussianDataset.TotalClasses}");
        Console.WriteLine($"Total examples: {gaussianDataset.TotalExamples}");
        Console.WriteLine($"Supports 5-way 1-shot: {gaussianDataset.SupportsConfiguration(5, 1, 5)}\n");

        // 3. RotatedDigitsMetaDataset
        Console.WriteLine("--- RotatedDigitsMetaDataset ---");
        var rotatedDataset = new RotatedDigitsMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 15, featureDim: 6, seed: 42);
        Console.WriteLine($"Name: {rotatedDataset.Name}");
        Console.WriteLine($"Total classes: {rotatedDataset.TotalClasses}");
        Console.WriteLine($"Total examples: {rotatedDataset.TotalExamples}\n");

        // 4. Task Samplers
        Console.WriteLine("--- Task Samplers ---");

        var uniformSampler = new UniformTaskSampler<double, Matrix<double>, Vector<double>>(
            gaussianDataset, numWays: 3, numShots: 2, numQueryPerClass: 3);
        Console.WriteLine($"UniformTaskSampler: {uniformSampler.NumWays}-way {uniformSampler.NumShots}-shot");
        var uniformBatch = uniformSampler.SampleBatch(4);
        Console.WriteLine($"  Sampled batch of {uniformBatch.BatchSize} tasks");

        var balancedSampler = new BalancedTaskSampler<double, Matrix<double>, Vector<double>>(
            gaussianDataset, numWays: 3, numShots: 2, numQueryPerClass: 3, seed: 42);
        Console.WriteLine($"BalancedTaskSampler: {balancedSampler.NumWays}-way {balancedSampler.NumShots}-shot");
        var balancedBatch = balancedSampler.SampleBatch(4);
        Console.WriteLine($"  Sampled batch of {balancedBatch.BatchSize} tasks");

        var dynamicSampler = new DynamicTaskSampler<double, Matrix<double>, Vector<double>>(
            gaussianDataset, numWays: 3, numShots: 2, numQueryPerClass: 3, seed: 42);
        Console.WriteLine($"DynamicTaskSampler: {dynamicSampler.NumWays}-way {dynamicSampler.NumShots}-shot");
        var episode = dynamicSampler.SampleOne();
        Console.WriteLine($"  Single episode difficulty: {episode.Difficulty:F3}\n");

        // 5. MetaDatasetFormat (multi-domain)
        Console.WriteLine("--- MetaDatasetFormat (Multi-Domain) ---");
        var multiDomain = new MetaDatasetFormat<double, Matrix<double>, Vector<double>>(
            new IMetaDataset<double, Matrix<double>, Vector<double>>[] { sineDataset, gaussianDataset },
            new[] { "SineWave", "Gaussian" },
            seed: 42);
        Console.WriteLine($"Domains: {multiDomain.DomainCount} ({string.Join(", ", multiDomain.DomainNames)})");
        Console.WriteLine($"Total classes: {multiDomain.TotalClasses}");

        var domainEpisode = multiDomain.SampleEpisode(2, 3, 4);
        Console.WriteLine($"Random domain episode: domain={domainEpisode.Domain}");

        var specificEpisode = multiDomain.SampleFromDomain(0, 2, 3, 4);
        Console.WriteLine($"Domain 0 episode: domain={specificEpisode.Domain}");

        // 6. EpisodeCache
        Console.WriteLine("\n--- EpisodeCache ---");
        var cache = new EpisodeCache<double, Matrix<double>, Vector<double>>(capacity: 50);
        for (int i = 0; i < 10; i++)
        {
            cache.Put(gaussianDataset.SampleEpisode(3, 2, 3));
        }
        Console.WriteLine($"Cached {cache.Count} episodes (capacity: {cache.Capacity})");
        Console.WriteLine($"Hit rate: {cache.HitRate:P0}");

        Console.WriteLine("\nData infrastructure example completed.");
    }

    /// <summary>
    /// Compares multiple meta-learning algorithms on the same task distribution.
    /// </summary>
    private static void RunAlgorithmComparison()
    {
        Console.WriteLine("Meta-Learning Algorithm Comparison");
        Console.WriteLine("==================================\n");
        Console.WriteLine("Comparing MAML, ProtoNets, and CNP on synthetic classification tasks.\n");

        const int inputDim = 4;
        const int numEpochs = 8;
        const int tasksPerEpoch = 4;
        var rng = new Random(789);

        // Pre-generate tasks so all algorithms see the same data
        var trainBatches = new List<TaskBatch<double, Matrix<double>, Vector<double>>>();
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            var tasks = new IMetaLearningTask<double, Matrix<double>, Vector<double>>[tasksPerEpoch];
            for (int t = 0; t < tasksPerEpoch; t++)
            {
                tasks[t] = CreateClassificationTask(inputDim, 2, 3, 4, rng);
            }
            trainBatches.Add(new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
        }

        var evalTask = CreateClassificationTask(inputDim, 2, 5, 10, new Random(999));

        // Table header
        Console.WriteLine($"{"Algorithm",-12} | {"Final Loss",12} | {"Time (ms)",10} | {"Eval Acc",10}");
        Console.WriteLine(new string('-', 52));

        // MAML
        RunBenchmark("MAML", () =>
        {
            var m = new SimpleMetaModel(inputDim);
            return new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(
                new MAMLOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.005, AdaptationSteps = 3, UseFirstOrder = true });
        }, trainBatches, evalTask);

        // ProtoNets
        RunBenchmark("ProtoNets", () =>
        {
            var m = new SimpleMetaModel(inputDim);
            return new ProtoNetsAlgorithm<double, Matrix<double>, Vector<double>>(
                new ProtoNetsOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.005, DistanceFunction = ProtoNetsDistanceFunction.Euclidean });
        }, trainBatches, evalTask);

        // CNP
        RunBenchmark("CNP", () =>
        {
            var m = new SimpleMetaModel(inputDim);
            return new CNPAlgorithm<double, Matrix<double>, Vector<double>>(
                new CNPOptions<double, Matrix<double>, Vector<double>>(m)
                { InnerLearningRate = 0.01, OuterLearningRate = 0.005, RepresentationDim = 32 });
        }, trainBatches, evalTask);

        Console.WriteLine("\nAlgorithm comparison completed.");
    }

    #region Task Generation Helpers

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateClassificationTask(
        int inputDim, int numWays, int numShots, int numQuery, Random rng)
    {
        // Generate class centroids
        var centroids = new double[numWays][];
        for (int c = 0; c < numWays; c++)
        {
            centroids[c] = new double[inputDim];
            for (int d = 0; d < inputDim; d++)
            {
                centroids[c][d] = (rng.NextDouble() - 0.5) * 4.0; // spread out
            }
        }

        int supportCount = numWays * numShots;
        int queryCount = numWays * numQuery;

        var supportX = new Matrix<double>(supportCount, inputDim);
        var supportY = new Vector<double>(supportCount);
        var queryX = new Matrix<double>(queryCount, inputDim);
        var queryY = new Vector<double>(queryCount);

        int sIdx = 0, qIdx = 0;
        for (int c = 0; c < numWays; c++)
        {
            for (int i = 0; i < numShots; i++)
            {
                for (int d = 0; d < inputDim; d++)
                    supportX[sIdx, d] = centroids[c][d] + (rng.NextDouble() - 0.5) * 0.5;
                supportY[sIdx] = c;
                sIdx++;
            }
            for (int i = 0; i < numQuery; i++)
            {
                for (int d = 0; d < inputDim; d++)
                    queryX[qIdx, d] = centroids[c][d] + (rng.NextDouble() - 0.5) * 0.5;
                queryY[qIdx] = c;
                qIdx++;
            }
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX, SupportSetY = supportY,
            QuerySetX = queryX, QuerySetY = queryY,
            NumWays = numWays, NumShots = numShots, NumQueryPerClass = numQuery
        };
    }

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateRegressionTask(
        int inputDim, int numContext, int numTarget, Random rng)
    {
        // Random linear function with noise: y = w^T x + b + noise
        var weights = new double[inputDim];
        double bias = (rng.NextDouble() - 0.5) * 2.0;
        for (int d = 0; d < inputDim; d++)
            weights[d] = (rng.NextDouble() - 0.5) * 2.0;

        var contextX = new Matrix<double>(numContext, inputDim);
        var contextY = new Vector<double>(numContext);
        var targetX = new Matrix<double>(numTarget, inputDim);
        var targetY = new Vector<double>(numTarget);

        for (int i = 0; i < numContext; i++)
        {
            double y = bias;
            for (int d = 0; d < inputDim; d++)
            {
                contextX[i, d] = (rng.NextDouble() - 0.5) * 2.0;
                y += weights[d] * contextX[i, d];
            }
            contextY[i] = y + (rng.NextDouble() - 0.5) * 0.1;
        }

        for (int i = 0; i < numTarget; i++)
        {
            double y = bias;
            for (int d = 0; d < inputDim; d++)
            {
                targetX[i, d] = (rng.NextDouble() - 0.5) * 2.0;
                y += weights[d] * targetX[i, d];
            }
            targetY[i] = y + (rng.NextDouble() - 0.5) * 0.1;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = contextX, SupportSetY = contextY,
            QuerySetX = targetX, QuerySetY = targetY,
            NumWays = 1, NumShots = numContext, NumQueryPerClass = numTarget
        };
    }

    #endregion

    #region Benchmark Helper

    private static void RunBenchmark<TAlgo>(
        string name,
        Func<TAlgo> factory,
        List<TaskBatch<double, Matrix<double>, Vector<double>>> trainBatches,
        MetaLearningTask<double, Matrix<double>, Vector<double>> evalTask)
        where TAlgo : MetaLearnerBase<double, Matrix<double>, Vector<double>>
    {
        var algo = factory();
        var sw = Stopwatch.StartNew();

        double lastLoss = 0;
        foreach (var batch in trainBatches)
        {
            lastLoss = algo.MetaTrain(batch);
        }

        sw.Stop();

        var adapted = algo.Adapt(evalTask);
        var preds = adapted.Predict(evalTask.QuerySetX);
        double acc = ComputeAccuracy(preds, evalTask.QuerySetY);

        Console.WriteLine($"{name,-12} | {lastLoss,12:F6} | {sw.ElapsedMilliseconds,10} | {acc,10:P1}");
    }

    #endregion

    #region Utility Helpers

    private static double ComputeParamDelta(Vector<double> a, Vector<double> b)
    {
        double sum = 0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double ComputeAccuracy(Vector<double> predictions, Vector<double> labels)
    {
        int correct = 0;
        int total = Math.Min(predictions.Length, labels.Length);
        for (int i = 0; i < total; i++)
        {
            if (Math.Round(predictions[i]) == Math.Round(labels[i]))
                correct++;
        }
        return total > 0 ? (double)correct / total : 0;
    }

    private static double ComputeMSE(Vector<double> predictions, Vector<double> labels)
    {
        double sum = 0;
        int count = Math.Min(predictions.Length, labels.Length);
        for (int i = 0; i < count; i++)
        {
            double diff = predictions[i] - labels[i];
            sum += diff * diff;
        }
        return count > 0 ? sum / count : 0;
    }

    private static string FormatVector(Vector<double> v, int maxElements)
    {
        int count = Math.Min(v.Length, maxElements);
        var parts = new string[count];
        for (int i = 0; i < count; i++)
            parts[i] = v[i].ToString("F2");
        string result = string.Join(", ", parts);
        if (v.Length > maxElements)
            result += ", ...";
        return result;
    }

    #endregion
}
