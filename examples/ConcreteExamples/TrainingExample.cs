using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Training;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.Benchmarks.Data;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Concrete example: Training with reinforcement learning.
/// </summary>
public class TrainingExample
{
    public static async Task RunAsync(IChatModel chatModel)
    {
        Console.WriteLine("=== RL Training Example ===\n");

        // Configure training
        var config = new RLConfig
        {
            Epochs = 3,  // Small number for demo
            BatchSize = 10,
            LearningRate = 0.0001,
            ValidationFrequency = 1,
            EarlyStoppingPatience = 2,
            SaveCheckpoints = true
        };

        // Setup reward models
        var prm = new ProcessRewardModel<double>(chatModel);
        var orm = new OutcomeRewardModel<double>(chatModel);
        var rewardModel = HybridRewardModel<double>.CreateBalanced(prm, orm);

        // Create learner with config
        var learner = new ReinforcementLearner<double>(chatModel, rewardModel, config);

        // Setup event handlers
        learner.OnEpochComplete += (sender, metrics) =>
        {
            Console.WriteLine($"\n--- Epoch {metrics.Epoch} Complete ---");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P1}");
            Console.WriteLine($"Avg Reward: {Convert.ToDouble(metrics.AverageReward):F3}");
            Console.WriteLine($"Avg Loss: {Convert.ToDouble(metrics.AverageLoss):F4}");
        };

        learner.OnBatchComplete += (sender, progress) =>
        {
            if (progress.BatchNumber % 5 == 0)
            {
                Console.WriteLine($"  Batch {progress.BatchNumber}/{progress.TotalBatches}");
            }
        };

        // Load training data
        Console.WriteLine("Loading training data from GSM8K...");
        var problems = GSM8KDataLoader.GetSampleProblems();

        var trainingData = problems
            .Take(30)
            .Select(p => (p.Question, p.FinalAnswer))
            .ToList();

        var validationData = problems
            .Skip(30)
            .Take(10)
            .Select(p => (p.Question, p.FinalAnswer))
            .ToList();

        Console.WriteLine($"Training samples: {trainingData.Count}");
        Console.WriteLine($"Validation samples: {validationData.Count}");

        Console.WriteLine("\nStarting training...\n");

        try
        {
            var results = await learner.TrainAsync(trainingData, validationData);

            Console.WriteLine("\n" + new string('=', 80));
            Console.WriteLine("=== Training Complete ===");
            Console.WriteLine(new string('=', 80));
            Console.WriteLine($"Total Epochs: {results.EpochsTrained}");
            Console.WriteLine($"Best Accuracy: {results.BestAccuracy:P1}");
            Console.WriteLine($"Best Epoch: {results.BestEpoch}");

            // Show training progression
            if (results.EpochMetrics.Count > 0)
            {
                Console.WriteLine("\nTraining Progression:");
                foreach (var metric in results.EpochMetrics)
                {
                    Console.WriteLine($"  Epoch {metric.Epoch}: Accuracy={metric.Accuracy:P1}, " +
                                    $"Reward={Convert.ToDouble(metric.AverageReward):F3}");
                }
            }

            // Show collected data statistics
            if (results.DataCollector != null)
            {
                var stats = results.DataCollector.Statistics;
                Console.WriteLine("\nCollected Training Data:");
                Console.WriteLine(stats.ToString());
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Training failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }

    public static async Task RunSTaRTrainingAsync(IChatModel chatModel)
    {
        Console.WriteLine("=== Self-Taught Reasoner (STaR) Training Example ===\n");

        var prm = new ProcessRewardModel<double>(chatModel);
        var orm = new OutcomeRewardModel<double>(chatModel);
        var rewardModel = HybridRewardModel<double>.CreateProcessFocused(prm, orm);

        var learner = new ReinforcementLearner<double>(chatModel, rewardModel);

        // Load data
        var problems = GSM8KDataLoader.GetSampleProblems();

        var trainingData = problems
            .Take(20)
            .Select(p => (p.Question, p.FinalAnswer))
            .ToList();

        var validationData = problems
            .Skip(20)
            .Take(10)
            .Select(p => (p.Question, p.FinalAnswer))
            .ToList();

        Console.WriteLine($"STaR Training with {trainingData.Count} problems");
        Console.WriteLine("Generating 5 attempts per problem...\n");

        try
        {
            var results = await learner.TrainSTaRAsync(
                trainingData,
                validationData,
                samplesPerProblem: 5
            );

            Console.WriteLine("\n=== STaR Training Complete ===");
            Console.WriteLine($"Best Accuracy: {results.BestAccuracy:P1}");
            Console.WriteLine($"Best Epoch: {results.BestEpoch}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"STaR training failed: {ex.Message}");
        }
    }
}
