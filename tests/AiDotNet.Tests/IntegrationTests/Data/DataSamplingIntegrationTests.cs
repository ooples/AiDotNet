using AiDotNet.Data.Sampling;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

/// <summary>
/// Integration tests for advanced data sampling classes:
/// StratifiedSampler, StratifiedBatchSampler, ImportanceSampler,
/// ActiveLearningSampler, Samplers factory.
/// Basic samplers (RandomSampler, SequentialSampler, SubsetSampler,
/// CurriculumSampler, SelfPacedSampler, WeightedSampler) are covered
/// by existing unit tests in DataSamplerTests.cs.
/// </summary>
public class DataSamplingIntegrationTests
{
    #region StratifiedSampler

    [Fact]
    public void StratifiedSampler_ReturnsAllIndices()
    {
        var labels = new[] { 0, 0, 0, 1, 1, 2 };
        var sampler = new StratifiedSampler(labels, numClasses: 3, seed: 42);

        var indices = sampler.GetIndices().ToList();

        Assert.Equal(6, indices.Count);
        Assert.Equal(6, indices.Distinct().Count());
    }

    [Fact]
    public void StratifiedSampler_Length_MatchesDataSize()
    {
        var labels = new[] { 0, 1, 0, 1, 0, 1, 2, 2 };
        var sampler = new StratifiedSampler(labels, numClasses: 3);

        Assert.Equal(8, sampler.Length);
    }

    [Fact]
    public void StratifiedSampler_NumClasses_Property()
    {
        var labels = new[] { 0, 1, 2 };
        var sampler = new StratifiedSampler(labels, numClasses: 3);

        Assert.Equal(3, sampler.NumClasses);
    }

    [Fact]
    public void StratifiedSampler_NullLabels_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new StratifiedSampler(null, numClasses: 2));
    }

    [Fact]
    public void StratifiedSampler_LessThan2Classes_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new StratifiedSampler(new[] { 0, 0 }, numClasses: 1));
    }

    [Fact]
    public void StratifiedSampler_LabelsCanBeUpdated()
    {
        var sampler = new StratifiedSampler(new[] { 0, 0, 1, 1 }, numClasses: 2, seed: 42);
        Assert.Equal(4, sampler.Length);

        sampler.Labels = new[] { 0, 0, 0, 1, 1, 1 };
        Assert.Equal(6, sampler.Length);

        var indices = sampler.GetIndices().ToList();
        Assert.Equal(6, indices.Count);
    }

    [Fact]
    public void StratifiedSampler_ContainsAllClasses()
    {
        var labels = new[] { 0, 0, 0, 0, 0, 1, 1, 2 };
        var sampler = new StratifiedSampler(labels, numClasses: 3, seed: 42);

        var indices = sampler.GetIndices().ToList();

        // Verify indices from all classes are present
        var resultLabels = indices.Select(i => labels[i]).ToHashSet();
        Assert.Contains(0, resultLabels);
        Assert.Contains(1, resultLabels);
        Assert.Contains(2, resultLabels);
    }

    #endregion

    #region StratifiedBatchSampler

    [Fact]
    public void StratifiedBatchSampler_CreatesBatches()
    {
        var labels = Enumerable.Repeat(0, 50).Concat(Enumerable.Repeat(1, 50)).ToArray();
        var sampler = new StratifiedBatchSampler(labels, numClasses: 2, batchSize: 10, seed: 42);

        var batches = sampler.GetBatchIndices().ToList();

        Assert.True(batches.Count >= 1);
        // Full batches should have batchSize elements
        foreach (var batch in batches.Take(batches.Count - 1))
        {
            Assert.Equal(10, batch.Length);
        }
    }

    [Fact]
    public void StratifiedBatchSampler_DropLast_SkipsIncompleteBatch()
    {
        // 15 samples, batchSize 4 => 3 full batches (12), 1 incomplete (3)
        var labels = new[] { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
        var sampler = new StratifiedBatchSampler(labels, numClasses: 2, batchSize: 4, dropLast: true, seed: 42);

        var batches = sampler.GetBatchIndices().ToList();
        foreach (var batch in batches)
        {
            Assert.Equal(4, batch.Length);
        }
    }

    [Fact]
    public void StratifiedBatchSampler_NullLabels_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new StratifiedBatchSampler(null, numClasses: 2, batchSize: 4));
    }

    [Fact]
    public void StratifiedBatchSampler_LessThan2Classes_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new StratifiedBatchSampler(new[] { 0, 0 }, numClasses: 1, batchSize: 2));
    }

    [Fact]
    public void StratifiedBatchSampler_ZeroBatchSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new StratifiedBatchSampler(new[] { 0, 1 }, numClasses: 2, batchSize: 0));
    }

    [Fact]
    public void StratifiedBatchSampler_Properties()
    {
        var labels = new[] { 0, 1, 0, 1 };
        var sampler = new StratifiedBatchSampler(labels, numClasses: 2, batchSize: 2, dropLast: true);

        Assert.Equal(4, sampler.Length);
        Assert.Equal(2, sampler.NumClasses);
        Assert.Equal(2, sampler.BatchSize);
        Assert.True(sampler.DropLast);
    }

    [Fact]
    public void StratifiedBatchSampler_GetIndices_ReturnsAllIndices()
    {
        var labels = new[] { 0, 0, 1, 1, 2, 2 };
        var sampler = new StratifiedBatchSampler(labels, numClasses: 3, batchSize: 3, seed: 42);

        var indices = sampler.GetIndices().ToList();
        Assert.Equal(6, indices.Count);
    }

    #endregion

    #region ImportanceSampler

    [Fact]
    public void ImportanceSampler_DefaultUniform_ReturnsAllIndices()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 20, seed: 42);

        var indices = sampler.GetIndices().ToList();

        Assert.Equal(20, indices.Count);
        Assert.Equal(20, sampler.Length);
    }

    [Fact]
    public void ImportanceSampler_UpdateImportance_AffectsSampling()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 10, smoothingFactor: 0.0, seed: 42);

        // Make index 0 much more important
        sampler.UpdateImportance(0, 100.0);
        for (int i = 1; i < 10; i++)
        {
            sampler.UpdateImportance(i, 0.01);
        }

        var indices = sampler.GetIndices().ToList();
        int count0 = indices.Count(i => i == 0);

        // Index 0 should appear much more often than average
        Assert.True(count0 > 1, $"Index 0 should appear more than once with high importance, got {count0}");
    }

    [Fact]
    public void ImportanceSampler_BatchUpdateImportances()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 5, seed: 42);

        sampler.UpdateImportances(
            new[] { 0, 1, 2 },
            new[] { 10.0, 5.0, 1.0 });

        Assert.Equal(10.0, sampler.ImportanceScores[0]);
        Assert.Equal(5.0, sampler.ImportanceScores[1]);
        Assert.Equal(1.0, sampler.ImportanceScores[2]);
    }

    [Fact]
    public void ImportanceSampler_SetImportances_ReplacesAll()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 3, seed: 42);

        sampler.SetImportances(new[] { 1.0, 2.0, 3.0 });

        Assert.Equal(1.0, sampler.ImportanceScores[0]);
        Assert.Equal(2.0, sampler.ImportanceScores[1]);
        Assert.Equal(3.0, sampler.ImportanceScores[2]);
    }

    [Fact]
    public void ImportanceSampler_SetImportances_WrongSize_Throws()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 3);

        Assert.Throws<ArgumentException>(() =>
            sampler.SetImportances(new[] { 1.0, 2.0 }));
    }

    [Fact]
    public void ImportanceSampler_ZeroDatasetSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ImportanceSampler<double>(datasetSize: 0));
    }

    [Fact]
    public void ImportanceSampler_GetIndicesWithoutReplacement()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 10, seed: 42);
        sampler.SetImportances(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        var indices = sampler.GetIndicesWithoutReplacement(5).ToList();

        Assert.Equal(5, indices.Count);
        Assert.Equal(5, indices.Distinct().Count()); // No duplicates
    }

    [Fact]
    public void ImportanceSampler_GetCorrectionFactor_NonZero()
    {
        var sampler = new ImportanceSampler<double>(datasetSize: 5, seed: 42);
        sampler.SetImportances(new[] { 1.0, 1.0, 1.0, 1.0, 1.0 });

        var correction = sampler.GetCorrectionFactor(0);
        Assert.True(correction > 0.0, "Correction factor should be positive");
    }

    #endregion

    #region ActiveLearningSampler

    [Fact]
    public void ActiveLearningSampler_InitialState()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 20);

        Assert.Equal(20, sampler.Length);
        Assert.Equal(0, sampler.LabeledCount);
        Assert.Equal(20, sampler.UnlabeledCount);
    }

    [Fact]
    public void ActiveLearningSampler_MarkAsLabeled_UpdatesCounts()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 10);

        sampler.MarkAsLabeled(0);
        sampler.MarkAsLabeled(5);

        Assert.Equal(2, sampler.LabeledCount);
        Assert.Equal(8, sampler.UnlabeledCount);
    }

    [Fact]
    public void ActiveLearningSampler_MarkMultipleLabeled()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 10);

        sampler.MarkAsLabeled(new[] { 0, 1, 2, 3 });

        Assert.Equal(4, sampler.LabeledCount);
        Assert.Equal(6, sampler.UnlabeledCount);
    }

    [Fact]
    public void ActiveLearningSampler_GetIndices_ReturnsOnlyLabeled()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 10, seed: 42);

        sampler.MarkAsLabeled(new[] { 2, 5, 8 });

        var indices = sampler.GetIndices().ToList();

        Assert.Equal(3, indices.Count);
        Assert.All(indices, i => Assert.True(i == 2 || i == 5 || i == 8));
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_Uncertainty()
    {
        var sampler = new ActiveLearningSampler<double>(
            datasetSize: 10,
            strategy: ActiveLearningStrategy.Uncertainty,
            seed: 42);

        // Set varying uncertainty
        for (int i = 0; i < 10; i++)
        {
            sampler.UpdateUncertainty(i, i * 0.1);
        }

        var selected = sampler.SelectForLabeling(3).ToList();

        Assert.Equal(3, selected.Count);
        // Most uncertain samples should be selected (highest indices)
        Assert.Contains(9, selected); // uncertainty 0.9
        Assert.Contains(8, selected); // uncertainty 0.8
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_Random()
    {
        var sampler = new ActiveLearningSampler<double>(
            datasetSize: 10,
            strategy: ActiveLearningStrategy.Random,
            seed: 42);

        var selected = sampler.SelectForLabeling(5).ToList();

        Assert.Equal(5, selected.Count);
        Assert.Equal(5, selected.Distinct().Count()); // All unique
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_Diversity()
    {
        var sampler = new ActiveLearningSampler<double>(
            datasetSize: 20,
            strategy: ActiveLearningStrategy.Diversity,
            seed: 42);

        var selected = sampler.SelectForLabeling(5).ToList();

        Assert.Equal(5, selected.Count);
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_Hybrid()
    {
        var sampler = new ActiveLearningSampler<double>(
            datasetSize: 20,
            strategy: ActiveLearningStrategy.Hybrid,
            diversityWeight: 0.5,
            seed: 42);

        for (int i = 0; i < 20; i++)
        {
            sampler.UpdateUncertainty(i, i * 0.05);
        }

        var selected = sampler.SelectForLabeling(6).ToList();

        Assert.True(selected.Count >= 1);
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_SkipsLabeled()
    {
        var sampler = new ActiveLearningSampler<double>(
            datasetSize: 10,
            strategy: ActiveLearningStrategy.Uncertainty,
            seed: 42);

        sampler.MarkAsLabeled(new[] { 7, 8, 9 }); // Mark most uncertain as labeled
        for (int i = 0; i < 10; i++)
        {
            sampler.UpdateUncertainty(i, i * 0.1);
        }

        var selected = sampler.SelectForLabeling(3).ToList();

        // Should not include already-labeled samples
        Assert.DoesNotContain(7, selected);
        Assert.DoesNotContain(8, selected);
        Assert.DoesNotContain(9, selected);
    }

    [Fact]
    public void ActiveLearningSampler_SelectForLabeling_AllLabeled_ReturnsEmpty()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 3);

        sampler.MarkAsLabeled(new[] { 0, 1, 2 });

        var selected = sampler.SelectForLabeling(2).ToList();

        Assert.Empty(selected);
    }

    [Fact]
    public void ActiveLearningSampler_ZeroDatasetSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ActiveLearningSampler<double>(datasetSize: 0));
    }

    [Fact]
    public void ActiveLearningSampler_UpdateUncertainties_Batch()
    {
        var sampler = new ActiveLearningSampler<double>(datasetSize: 5);

        sampler.UpdateUncertainties(
            new[] { 0, 1, 2 },
            new[] { 0.9, 0.8, 0.7 });

        // Verify by selecting - high uncertainty samples should be selected first
        var selected = sampler.SelectForLabeling(2).ToList();
        Assert.Contains(0, selected); // highest uncertainty
    }

    #endregion

    #region Samplers Factory

    [Fact]
    public void Samplers_Random_CreatesRandomSampler()
    {
        var sampler = Samplers.Random(100, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(100, sampler.Length);
        var indices = sampler.GetIndices().ToList();
        Assert.Equal(100, indices.Count);
    }

    [Fact]
    public void Samplers_Sequential_CreatesSequentialSampler()
    {
        var sampler = Samplers.Sequential(50);

        Assert.NotNull(sampler);
        Assert.Equal(50, sampler.Length);
        var indices = sampler.GetIndices().ToList();
        Assert.Equal(Enumerable.Range(0, 50).ToList(), indices);
    }

    [Fact]
    public void Samplers_Subset_CreatesSubsetSampler()
    {
        var sampler = Samplers.Subset(new[] { 5, 10, 15 }, shuffle: false);

        Assert.NotNull(sampler);
        Assert.Equal(3, sampler.Length);
        Assert.Equal(new[] { 5, 10, 15 }, sampler.GetIndices().ToList());
    }

    [Fact]
    public void Samplers_Stratified_CreatesStratifiedSampler()
    {
        var labels = new[] { 0, 0, 1, 1, 2, 2 };
        var sampler = Samplers.Stratified(labels, numClasses: 3, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(6, sampler.Length);
    }

    [Fact]
    public void Samplers_Weighted_CreatesWeightedSampler()
    {
        var sampler = Samplers.Weighted(new[] { 1.0, 2.0, 3.0 }, numSamples: 10, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(10, sampler.Length);
    }

    [Fact]
    public void Samplers_Balanced_CreatesBalancedSampler()
    {
        var labels = Enumerable.Repeat(0, 90).Concat(Enumerable.Repeat(1, 10)).ToList();
        var sampler = Samplers.Balanced(labels, numClasses: 2, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(100, sampler.Length);
    }

    [Fact]
    public void Samplers_Curriculum_CreatesCurriculumSampler()
    {
        var difficulties = new[] { 0.1, 0.3, 0.5, 0.7, 0.9 };
        var sampler = Samplers.Curriculum(difficulties, totalEpochs: 50, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(5, sampler.Length);
    }

    [Fact]
    public void Samplers_SelfPaced_CreatesSelfPacedSampler()
    {
        var sampler = Samplers.SelfPaced(datasetSize: 100, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(100, sampler.Length);
    }

    [Fact]
    public void Samplers_Importance_CreatesImportanceSampler()
    {
        var sampler = Samplers.Importance(datasetSize: 50, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(50, sampler.Length);
    }

    [Fact]
    public void Samplers_ActiveLearning_CreatesActiveLearningtSampler()
    {
        var sampler = Samplers.ActiveLearning(datasetSize: 50, seed: 42);

        Assert.NotNull(sampler);
        Assert.Equal(50, sampler.Length);
    }

    #endregion
}
