using AiDotNet.Data.Text.Benchmarks;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Training;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.HarmonicEngine;

/// <summary>
/// Integration tests for training an <see cref="HRELanguageModel{T}"/> on
/// Tiny Shakespeare using the paper's no-backpropagation training strategies.
/// These tests produce the loss curves and perplexity numbers that feed the
/// paper's empirical results.
/// </summary>
public class HRELanguageModelTrainingTests
{
    private readonly ITestOutputHelper _output;

    public HRELanguageModelTrainingTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Training hyperparameters, all configurable — nothing hardcoded.
    /// </summary>
    public class TrainingConfig
    {
        /// <summary>Sequence length (context window). Must be ≤ Sidon-carrier capacity of fftSize.</summary>
        public int SeqLen { get; set; } = 16;

        /// <summary>Embedding dimension. Must be a power of 2.</summary>
        public int EmbedDim { get; set; } = 32;

        /// <summary>Number of stacked HRE blocks.</summary>
        public int NumLayers { get; set; } = 2;

        /// <summary>FFT size used by sequence-axis IMD attention.</summary>
        public int FftSize { get; set; } = 2048;

        /// <summary>Batch size for training.</summary>
        public int BatchSize { get; set; } = 4;

        /// <summary>Number of training epochs (full passes over the training set).</summary>
        public int Epochs { get; set; } = 3;

        /// <summary>Hebbian learning rate.</summary>
        public double HebbianLearningRate { get; set; } = 0.01;

        /// <summary>Anti-Hebbian decorrelation strength α.</summary>
        public double AntiHebbianAlpha { get; set; } = 0.5;

        /// <summary>Number of warm-up batches before switching to target propagation.</summary>
        public int WarmupSteps { get; set; } = 20;

        /// <summary>Fraction of corpus used for training (rest is validation).</summary>
        public double TrainFraction { get; set; } = 0.9;

        /// <summary>Random seed for model initialization.</summary>
        public int Seed { get; set; } = 42;
    }

    /// <summary>
    /// Phase 3 moment-of-truth: train an HRELanguageModel on Tiny Shakespeare
    /// using Spectral Target Propagation (the paper's novel no-backprop rule)
    /// and verify that the validation loss decreases meaningfully from the
    /// random-initialization baseline (log(256) ≈ 5.545 nats).
    /// </summary>
    [Fact]
    public async Task SpectralTargetPropagation_TinyShakespeare_ReducesValLoss()
    {
        await RunTrainingExperiment(new TrainingConfig());
    }

    /// <summary>
    /// Parameterized training experiment. Extracted so tests can sweep
    /// hyperparameters without duplicating the training loop.
    /// </summary>
    private async Task RunTrainingExperiment(TrainingConfig config)
    {
        // --- Load dataset ---
        var trainOptions = new TinyShakespeareDataLoaderOptions
        {
            Split = AiDotNet.Data.Geometry.DatasetSplit.Train,
            SequenceLength = config.SeqLen,
            TrainFraction = config.TrainFraction,
        };
        var trainLoader = new TinyShakespeareDataLoader<double>(trainOptions);
        await trainLoader.LoadAsync();

        var valOptions = new TinyShakespeareDataLoaderOptions
        {
            Split = AiDotNet.Data.Geometry.DatasetSplit.Validation,
            SequenceLength = config.SeqLen,
            TrainFraction = config.TrainFraction,
        };
        var valLoader = new TinyShakespeareDataLoader<double>(valOptions);
        await valLoader.LoadAsync();

        _output.WriteLine($"Train samples: {trainLoader.TotalCount}");
        _output.WriteLine($"Val samples:   {valLoader.TotalCount}");
        _output.WriteLine($"Config: seqLen={config.SeqLen}, embed={config.EmbedDim}, " +
                          $"layers={config.NumLayers}, batch={config.BatchSize}, epochs={config.Epochs}");

        Assert.True(trainLoader.TotalCount > 0, "Training set should have non-zero samples");
        Assert.True(valLoader.TotalCount > 0, "Validation set should have non-zero samples");

        // --- Build model ---
        var model = new HRELanguageModel<double>(
            vocabSize: 256,
            seqLen: config.SeqLen,
            embedDim: config.EmbedDim,
            numLayers: config.NumLayers,
            fftSize: config.FftSize,
            hebbianLearningRate: config.HebbianLearningRate,
            antiHebbianAlpha: config.AntiHebbianAlpha,
            seed: config.Seed);

        _output.WriteLine($"Model params: {model.ParameterCount}");
        _output.WriteLine($"Random-init baseline loss: {Math.Log(256):F3} nats");

        // --- Measure initial validation loss ---
        double initialValLoss = ComputeValLoss(model, valLoader, config.BatchSize);
        _output.WriteLine($"Initial val loss: {initialValLoss:F4}");

        // --- Train with SpectralTargetPropagation ---
        var strategy = new SpectralTargetPropagation<double>(
            hebbianLearningRate: config.HebbianLearningRate,
            warmupSteps: config.WarmupSteps);
        var trainer = new HRETrainer<double>(model, strategy);

        int batchesPerEpoch = Math.Max(1, trainLoader.TotalCount / config.BatchSize);
        int totalBatches = config.Epochs * batchesPerEpoch;
        _output.WriteLine($"Training: {config.Epochs} epochs × {batchesPerEpoch} batches = {totalBatches} total steps");

        int step = 0;
        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            trainLoader.Reset();
            foreach (var (features, labels) in
                trainLoader.GetBatches(batchSize: config.BatchSize, shuffle: true))
            {
                trainer.Step(new TrainingBatch<double>(features, labels));

                if (step % Math.Max(1, totalBatches / 10) == 0 || step == totalBatches - 1)
                {
                    var metrics = trainer.GetMetrics();
                    _output.WriteLine(
                        $"Step {step:D3}  epoch={epoch}  " +
                        $"train_loss={metrics.GetValueOrDefault("last_train_loss"):F4}  " +
                        $"in_warmup={metrics.GetValueOrDefault("in_warmup"):F0}");
                }
                step++;
            }
        }

        // --- Measure final validation loss ---
        double finalValLoss = ComputeValLoss(model, valLoader, config.BatchSize);
        _output.WriteLine($"Final val loss:   {finalValLoss:F4}");
        _output.WriteLine($"Improvement:      {initialValLoss - finalValLoss:F4} nats");

        // --- Assertions ---
        Assert.True(double.IsFinite(initialValLoss), "Initial val loss must be finite");
        Assert.True(double.IsFinite(finalValLoss), "Final val loss must be finite");

        // The core claim: target propagation must reduce val loss. Any
        // meaningful decrease confirms the training rule is working.
        Assert.True(finalValLoss < initialValLoss,
            $"Val loss should decrease after training. Initial: {initialValLoss:F4}, Final: {finalValLoss:F4}");
    }

    /// <summary>
    /// Computes mean cross-entropy loss per token across all validation batches.
    /// Fully configurable — no hardcoded dimensions.
    /// </summary>
    private static double ComputeValLoss(
        HRELanguageModel<double> model,
        TinyShakespeareDataLoader<double> loader,
        int batchSize)
    {
        model.SetTrainingMode(false);
        double totalLoss = 0;
        int tokenCount = 0;

        loader.Reset();
        foreach (var (features, labels) in loader.GetBatches(batchSize: batchSize, shuffle: false))
        {
            int actualBatch = features.Shape[0];
            int seqLen = features.Shape[1];

            for (int b = 0; b < actualBatch; b++)
            {
                var inputSeq = new Tensor<double>([seqLen]);
                for (int s = 0; s < seqLen; s++) inputSeq[s] = features[b, s];

                var logits = model.Forward(inputSeq);
                int vocabSize = logits.Shape[1];

                for (int s = 0; s < seqLen; s++)
                {
                    double maxLogit = double.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                        if (logits[s, v] > maxLogit) maxLogit = logits[s, v];
                    double sumExp = 0;
                    for (int v = 0; v < vocabSize; v++)
                        sumExp += Math.Exp(logits[s, v] - maxLogit);
                    double logSumExp = maxLogit + Math.Log(sumExp);

                    int target = (int)labels[b, s];
                    double targetLogit = logits[s, target];
                    totalLoss += logSumExp - targetLogit;
                    tokenCount++;
                }
            }
        }

        return tokenCount > 0 ? totalLoss / tokenCount : 0;
    }
}
