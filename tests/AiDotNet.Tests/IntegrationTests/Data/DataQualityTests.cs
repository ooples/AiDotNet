using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using AiDotNet.Data.Quality;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class DataQualityTests
{
    // ==================== Deduplication Tests ====================

    [Fact]
    public void ExactHashDeduplicator_DefaultOptions()
    {
        var options = new ExactHashDeduplicatorOptions();
        Assert.True(options.NormalizeWhitespace);
        Assert.True(options.CaseInsensitive);
    }

    [Fact]
    public void ExactHashDeduplicator_FindsDuplicates()
    {
        var dedup = new ExactHashDeduplicator();
        var documents = new List<string>
        {
            "The quick brown fox jumps over the lazy dog.",
            "A completely different document about cats.",
            "the quick brown fox jumps over the lazy dog.",  // Case-insensitive duplicate of 0
            "Yet another unique document here.",
            "A completely different document about cats."     // Exact duplicate of 1
        };

        var duplicates = dedup.FindDuplicates(documents);

        Assert.Contains(2, duplicates);
        Assert.Contains(4, duplicates);
        Assert.DoesNotContain(0, duplicates);
        Assert.DoesNotContain(1, duplicates);
        Assert.DoesNotContain(3, duplicates);
    }

    [Fact]
    public void ExactHashDeduplicator_WhitespaceNormalization()
    {
        var dedup = new ExactHashDeduplicator(new ExactHashDeduplicatorOptions
        {
            NormalizeWhitespace = true,
            CaseInsensitive = false
        });

        string hash1 = dedup.ComputeHash("hello   world");
        string hash2 = dedup.ComputeHash("hello world");

        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void MinHashDeduplicator_DefaultOptions()
    {
        var options = new MinHashDeduplicatorOptions();
        Assert.Equal(128, options.NumHashFunctions);
        Assert.Equal(0.8, options.SimilarityThreshold);
        Assert.Equal(16, options.NumBands);
        Assert.Equal(5, options.ShingleSize);
    }

    [Fact]
    public void MinHashDeduplicator_FindsNearDuplicates()
    {
        var dedup = new MinHashDeduplicator(new MinHashDeduplicatorOptions
        {
            Seed = 42,
            SimilarityThreshold = 0.5
        });

        var documents = new List<string>
        {
            "The quick brown fox jumps over the lazy dog in the park.",
            "A completely different and unrelated text about quantum physics and black holes.",
            "The quick brown fox jumped over the lazy dog in the park.",  // Near-duplicate of 0
            "Machine learning is a subset of artificial intelligence research."
        };

        var sig0 = dedup.ComputeSignature(documents[0]);
        var sig2 = dedup.ComputeSignature(documents[2]);
        double similarity = dedup.EstimateSimilarity(sig0, sig2);

        // Near-duplicates should have higher similarity than random pairs
        var sig1 = dedup.ComputeSignature(documents[1]);
        double dissimilarity = dedup.EstimateSimilarity(sig0, sig1);

        Assert.True(similarity > dissimilarity,
            $"Near-duplicate similarity {similarity} should exceed unrelated {dissimilarity}");
    }

    [Fact]
    public void SemanticDeduplicator_DefaultOptions()
    {
        var options = new SemanticDeduplicatorOptions();
        Assert.Equal(0.95, options.SimilarityThreshold);
        Assert.Equal(768, options.EmbeddingDimension);
        Assert.Equal(64, options.BatchSize);
    }

    [Fact]
    public void SemanticDeduplicator_FindsSimilarEmbeddings()
    {
        var dedup = new SemanticDeduplicator(new SemanticDeduplicatorOptions
        {
            SimilarityThreshold = 0.99
        });

        var embeddings = new double[][]
        {
            new double[] { 1.0, 0.0, 0.0 },
            new double[] { 0.0, 1.0, 0.0 },
            new double[] { 1.0, 0.001, 0.0 },  // Very similar to [0]
            new double[] { 0.0, 0.0, 1.0 }
        };

        var duplicates = dedup.FindDuplicates(embeddings);

        Assert.Contains(2, duplicates);
        Assert.DoesNotContain(0, duplicates);
        Assert.DoesNotContain(1, duplicates);
        Assert.DoesNotContain(3, duplicates);
    }

    [Fact]
    public void SemanticDeduplicator_CosineSimilarity()
    {
        var dedup = new SemanticDeduplicator();

        double sim = dedup.CosineSimilarity(
            new double[] { 1, 0, 0 },
            new double[] { 0, 1, 0 });
        Assert.Equal(0.0, sim, 5);

        double simSame = dedup.CosineSimilarity(
            new double[] { 1, 2, 3 },
            new double[] { 1, 2, 3 });
        Assert.Equal(1.0, simSame, 5);
    }

    // ==================== Quality Filtering Tests ====================

    [Fact]
    public void PerplexityFilter_DefaultOptions()
    {
        var options = new PerplexityFilterOptions();
        Assert.Equal(1000.0, options.MaxPerplexity);
        Assert.Equal(0.0, options.MinPerplexity);
        Assert.Equal(3, options.NGramOrder);
        Assert.Equal(1.0, options.SmoothingFactor);
    }

    [Fact]
    public void PerplexityFilter_TrainsAndScores()
    {
        var filter = new PerplexityFilter(new PerplexityFilterOptions
        {
            NGramOrder = 2,
            MaxPerplexity = 50
        });

        var reference = new List<string>
        {
            "the cat sat on the mat and the cat was happy",
            "the dog ran in the park and the dog was tired",
            "the bird flew over the tree and the bird sang"
        };

        filter.Train(reference);

        // Text similar to reference should have lower perplexity
        double perpSimilar = filter.ComputePerplexity("the cat sat on the mat");
        double perpRandom = filter.ComputePerplexity("xyzzy flurb quux blargh");

        Assert.True(perpSimilar < perpRandom,
            $"Similar text perplexity {perpSimilar} should be less than random {perpRandom}");
    }

    [Fact]
    public void PerplexityFilter_ThrowsWithoutTraining()
    {
        var filter = new PerplexityFilter();
        Assert.Throws<InvalidOperationException>(() => filter.ComputePerplexity("test text"));
    }

    [Fact]
    public void HeuristicTextFilter_DefaultOptions()
    {
        var options = new HeuristicTextFilterOptions();
        Assert.Equal(50, options.MinWordCount);
        Assert.Equal(100000, options.MaxWordCount);
        Assert.Equal(0.3, options.MaxSpecialCharRatio);
        Assert.True(options.FilterBoilerplate);
    }

    [Fact]
    public void HeuristicTextFilter_PassesGoodText()
    {
        var filter = new HeuristicTextFilter(new HeuristicTextFilterOptions
        {
            MinWordCount = 5,
            FilterBoilerplate = true
        });

        string goodText = "This is a well-written paragraph with proper punctuation. " +
                          "It contains multiple sentences of reasonable length. " +
                          "The content is meaningful and not boilerplate.";

        Assert.True(filter.PassesFilter(goodText));
    }

    [Fact]
    public void HeuristicTextFilter_FiltersBoilerplate()
    {
        var filter = new HeuristicTextFilter(new HeuristicTextFilterOptions
        {
            MinWordCount = 3,
            FilterBoilerplate = true
        });

        string boilerplate = "Please subscribe to our newsletter for the latest updates and news content.";

        Assert.False(filter.PassesFilter(boilerplate));
    }

    [Fact]
    public void HeuristicTextFilter_FiltersTooShort()
    {
        var filter = new HeuristicTextFilter();
        Assert.False(filter.PassesFilter("Too short."));
    }

    [Fact]
    public void LanguageIdFilter_DefaultOptions()
    {
        var options = new LanguageIdFilterOptions();
        Assert.Single(options.TargetLanguages);
        Assert.Equal("en", options.TargetLanguages[0]);
        Assert.Equal(0.8, options.MinConfidence);
    }

    [Fact]
    public void LanguageIdFilter_DetectsLanguage()
    {
        var filter = new LanguageIdFilter(new LanguageIdFilterOptions
        {
            MinConfidence = 0.0,
            MinTextLength = 10
        });

        filter.AddLanguageProfile("en", new List<string>
        {
            "The quick brown fox jumps over the lazy dog. This is English text with common patterns and words."
        });
        filter.AddLanguageProfile("fr", new List<string>
        {
            "Le rapide renard brun saute par dessus le chien paresseux. Ceci est du texte francais avec des mots."
        });

        var (lang, _) = filter.DetectLanguage("This is a test of the English language detection system working properly.");
        Assert.Equal("en", lang);
    }

    [Fact]
    public void ImageQualityFilter_DefaultOptions()
    {
        var options = new ImageQualityFilterOptions();
        Assert.Equal(64, options.MinWidth);
        Assert.Equal(64, options.MinHeight);
        Assert.Equal(5.0, options.MaxAspectRatio);
        Assert.Equal(5.0, options.MinPixelStdDev);
    }

    [Fact]
    public void ImageQualityFilter_PassesGoodImage()
    {
        var filter = new ImageQualityFilter();

        Assert.True(filter.PassesDimensionCheck(256, 256));
        Assert.True(filter.PassesFileSizeCheck(10000));

        // Create diverse pixel data
        var rng = new Random(42);
        var pixels = new double[256 * 256];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = rng.NextDouble() * 255;

        Assert.True(filter.PassesPixelCheck(pixels));
    }

    [Fact]
    public void ImageQualityFilter_RejectsTinyImage()
    {
        var filter = new ImageQualityFilter();
        Assert.False(filter.PassesDimensionCheck(32, 32));
    }

    [Fact]
    public void ImageQualityFilter_RejectsSolidImage()
    {
        var filter = new ImageQualityFilter();
        var solidPixels = new double[100];
        for (int i = 0; i < solidPixels.Length; i++)
            solidPixels[i] = 128.0;

        Assert.False(filter.PassesPixelCheck(solidPixels));
    }

    // ==================== Dataset Efficiency Tests ====================

    [Fact]
    public void CoresetSelector_DefaultOptions()
    {
        var options = new CoresetSelectorOptions();
        Assert.Equal(0.1, options.SelectionRatio);
        Assert.Equal(CoresetStrategy.Greedy, options.Strategy);
        Assert.Null(options.Seed);
    }

    [Fact]
    public void CoresetSelector_SelectsSubset()
    {
        var selector = new CoresetSelector(new CoresetSelectorOptions
        {
            SelectionRatio = 0.5,
            Strategy = CoresetStrategy.Random,
            Seed = 42
        });

        var embeddings = new double[20][];
        var rng = new Random(42);
        for (int i = 0; i < 20; i++)
        {
            embeddings[i] = new double[] { rng.NextDouble(), rng.NextDouble(), rng.NextDouble() };
        }

        var selected = selector.Select(embeddings);

        Assert.Equal(10, selected.Count);
        Assert.True(selected.All(idx => idx >= 0 && idx < 20));
        Assert.Equal(selected.Count, selected.Distinct().Count()); // No duplicates
    }

    [Fact]
    public void CoresetSelector_GreedySelectsRepresentative()
    {
        var selector = new CoresetSelector(new CoresetSelectorOptions
        {
            SelectionRatio = 0.3,
            Strategy = CoresetStrategy.Greedy
        });

        // Two clusters
        var embeddings = new double[][]
        {
            new double[] { 0, 0 }, new double[] { 0.1, 0.1 }, new double[] { 0.2, 0 },
            new double[] { 10, 10 }, new double[] { 10.1, 10.1 }, new double[] { 10.2, 10 },
            new double[] { 5, 5 }, new double[] { 5.1, 5 }
        };

        var selected = selector.Select(embeddings);

        Assert.True(selected.Count >= 2);
    }

    [Fact]
    public void DataPruner_DefaultOptions()
    {
        var options = new DataPrunerOptions();
        Assert.Equal(0.3, options.PruneRatio);
        Assert.Equal(PruneStrategy.HighConfidence, options.Strategy);
        Assert.Equal(5, options.MinEpochsForScoring);
    }

    [Fact]
    public void DataPruner_PrunesHighConfidence()
    {
        var pruner = new DataPruner(new DataPrunerOptions { PruneRatio = 0.4 });

        var confidences = new double[] { 0.99, 0.5, 0.95, 0.3, 0.98 };
        var pruned = pruner.PruneByConfidence(confidences);

        Assert.Equal(2, pruned.Count); // 40% of 5 = 2
        // Should prune highest confidence: indices 0 (0.99) and 4 (0.98)
        Assert.Contains(0, pruned);
        Assert.Contains(4, pruned);
    }

    [Fact]
    public void DataPruner_PrunesByForgetting()
    {
        var pruner = new DataPruner(new DataPrunerOptions { PruneRatio = 0.4 });

        var forgettingCounts = new int[] { 5, 0, 3, 0, 2 };
        var pruned = pruner.PruneByForgetting(forgettingCounts);

        Assert.Equal(2, pruned.Count);
        // Should prune lowest forgetting: indices 1 and 3 (both 0)
        Assert.Contains(1, pruned);
        Assert.Contains(3, pruned);
    }

    [Fact]
    public void DatasetDistiller_DefaultOptions()
    {
        var options = new DatasetDistillerOptions();
        Assert.Equal(10, options.SamplesPerClass);
        Assert.Equal(0.01, options.DistillLearningRate);
        Assert.Equal(1000, options.NumSteps);
    }

    [Fact]
    public void DatasetDistiller_DistillsDataset()
    {
        var distiller = new DatasetDistiller(new DatasetDistillerOptions
        {
            SamplesPerClass = 2,
            NumSteps = 50,
            Seed = 42
        });

        var features = new double[][]
        {
            new double[] { 1, 0 }, new double[] { 1.1, 0.1 }, new double[] { 0.9, -0.1 },
            new double[] { 10, 10 }, new double[] { 10.1, 10.2 }, new double[] { 9.9, 9.8 }
        };
        var labels = new int[] { 0, 0, 0, 1, 1, 1 };

        var (distFeatures, distLabels) = distiller.Distill(features, labels);

        Assert.Equal(4, distFeatures.Length); // 2 per class * 2 classes
        Assert.Equal(4, distLabels.Length);
        Assert.Equal(2, distLabels.Count(l => l == 0));
        Assert.Equal(2, distLabels.Count(l => l == 1));
    }

    [Fact]
    public void CurriculumDataScheduler_DefaultOptions()
    {
        var options = new CurriculumDataSchedulerOptions();
        Assert.Equal(CurriculumOrder.EasyToHard, options.Order);
        Assert.Equal(CurriculumPacing.Linear, options.Pacing);
        Assert.Equal(0.2, options.InitialFraction);
        Assert.Equal(10, options.FullDataEpoch);
    }

    [Fact]
    public void CurriculumDataScheduler_LinearPacing()
    {
        var scheduler = new CurriculumDataScheduler(new CurriculumDataSchedulerOptions
        {
            InitialFraction = 0.2,
            FullDataEpoch = 10,
            Pacing = CurriculumPacing.Linear
        });

        double frac0 = scheduler.ComputeFraction(0);
        double frac5 = scheduler.ComputeFraction(5);
        double frac10 = scheduler.ComputeFraction(10);

        Assert.Equal(0.2, frac0, 2);
        Assert.Equal(0.6, frac5, 2);
        Assert.Equal(1.0, frac10, 2);
    }

    [Fact]
    public void CurriculumDataScheduler_GetAvailableIndices()
    {
        var scheduler = new CurriculumDataScheduler(new CurriculumDataSchedulerOptions
        {
            InitialFraction = 0.4,
            FullDataEpoch = 5,
            Order = CurriculumOrder.EasyToHard
        });

        var difficulties = new double[] { 5.0, 1.0, 3.0, 2.0, 4.0 };

        // Epoch 0: 40% of 5 = 2 easiest
        var epoch0 = scheduler.GetAvailableIndices(difficulties, 0);
        Assert.Equal(2, epoch0.Count);

        // Epoch 5: all available
        var epoch5 = scheduler.GetAvailableIndices(difficulties, 5);
        Assert.Equal(5, epoch5.Count);
    }

    [Fact]
    public void ActiveLearningQueryStrategy_DefaultOptions()
    {
        var options = new ActiveLearningQueryStrategyOptions();
        Assert.Equal(100, options.QueryBatchSize);
        Assert.Equal(QueryStrategy.Uncertainty, options.Strategy);
        Assert.Equal(10, options.NumMcDropoutPasses);
    }

    [Fact]
    public void ActiveLearningQueryStrategy_UncertaintySampling()
    {
        var strategy = new ActiveLearningQueryStrategy(new ActiveLearningQueryStrategyOptions
        {
            QueryBatchSize = 2,
            Strategy = QueryStrategy.Uncertainty
        });

        var predictions = new double[][]
        {
            new double[] { 0.9, 0.1 },   // Confident
            new double[] { 0.5, 0.5 },   // Most uncertain
            new double[] { 0.8, 0.2 },   // Somewhat confident
            new double[] { 0.45, 0.55 }  // Very uncertain
        };

        var selected = strategy.Query(predictions);

        Assert.Equal(2, selected.Count);
        // Should select most uncertain samples (indices 1 and 3)
        Assert.Contains(1, selected);
        Assert.Contains(3, selected);
    }

    [Fact]
    public void ActiveLearningQueryStrategy_MarginSampling()
    {
        var strategy = new ActiveLearningQueryStrategy(new ActiveLearningQueryStrategyOptions
        {
            QueryBatchSize = 1,
            Strategy = QueryStrategy.Margin
        });

        var predictions = new double[][]
        {
            new double[] { 0.9, 0.1 },   // Margin = 0.8
            new double[] { 0.51, 0.49 },  // Margin = 0.02 (smallest)
            new double[] { 0.7, 0.3 }     // Margin = 0.4
        };

        var selected = strategy.Query(predictions);

        Assert.Single(selected);
        Assert.Equal(1, selected[0]); // Smallest margin
    }

    [Fact]
    public void ActiveLearningQueryStrategy_BALDQuery()
    {
        var strategy = new ActiveLearningQueryStrategy(new ActiveLearningQueryStrategyOptions
        {
            QueryBatchSize = 1
        });

        // 3 MC passes, 2 samples, 2 classes
        var mcPredictions = new double[][][]
        {
            new double[][] { new double[] { 0.9, 0.1 }, new double[] { 0.6, 0.4 } },
            new double[][] { new double[] { 0.85, 0.15 }, new double[] { 0.3, 0.7 } },
            new double[][] { new double[] { 0.92, 0.08 }, new double[] { 0.5, 0.5 } }
        };

        var selected = strategy.QueryBALD(mcPredictions);

        Assert.Single(selected);
        // Sample 1 has higher disagreement across passes
        Assert.Equal(1, selected[0]);
    }
}
