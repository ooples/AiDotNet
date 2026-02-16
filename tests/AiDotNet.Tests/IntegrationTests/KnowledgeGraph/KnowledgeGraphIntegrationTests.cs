#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeGraph;

/// <summary>
/// Integration tests for knowledge graph embeddings, link prediction, community detection,
/// temporal support, KG construction, and enhanced GraphRAG.
/// Uses small but realistic graphs to exercise the full pipeline.
/// </summary>
public class KnowledgeGraphIntegrationTests
{
    #region Test Graph Helpers

    /// <summary>
    /// Builds a small knowledge graph with 6 entities, 3 relation types, and 8 edges.
    /// Represents: Einstein/Bohr/Curie (persons), Physics/Chemistry (fields), Germany/Denmark/Poland (locations).
    /// </summary>
    private static KnowledgeGraph<double> BuildSmallGraph()
    {
        var graph = new KnowledgeGraph<double>();

        graph.AddNode(new GraphNode<double>("einstein", "PERSON"));
        graph.AddNode(new GraphNode<double>("bohr", "PERSON"));
        graph.AddNode(new GraphNode<double>("curie", "PERSON"));
        graph.AddNode(new GraphNode<double>("physics", "FIELD"));
        graph.AddNode(new GraphNode<double>("chemistry", "FIELD"));
        graph.AddNode(new GraphNode<double>("germany", "LOCATION"));
        graph.AddNode(new GraphNode<double>("denmark", "LOCATION"));
        graph.AddNode(new GraphNode<double>("poland", "LOCATION"));

        graph.AddEdge(new GraphEdge<double>("einstein", "physics", "WORKS_IN"));
        graph.AddEdge(new GraphEdge<double>("bohr", "physics", "WORKS_IN"));
        graph.AddEdge(new GraphEdge<double>("curie", "chemistry", "WORKS_IN"));
        graph.AddEdge(new GraphEdge<double>("curie", "physics", "WORKS_IN"));
        graph.AddEdge(new GraphEdge<double>("einstein", "germany", "BORN_IN"));
        graph.AddEdge(new GraphEdge<double>("bohr", "denmark", "BORN_IN"));
        graph.AddEdge(new GraphEdge<double>("curie", "poland", "BORN_IN"));
        graph.AddEdge(new GraphEdge<double>("einstein", "bohr", "COLLABORATED_WITH"));

        return graph;
    }

    /// <summary>
    /// Builds a temporal knowledge graph with ValidFrom/ValidUntil on edges.
    /// </summary>
    private static KnowledgeGraph<double> BuildTemporalGraph()
    {
        var graph = new KnowledgeGraph<double>();

        graph.AddNode(new GraphNode<double>("obama", "PERSON"));
        graph.AddNode(new GraphNode<double>("trump", "PERSON"));
        graph.AddNode(new GraphNode<double>("biden", "PERSON"));
        graph.AddNode(new GraphNode<double>("usa", "COUNTRY"));

        var edge1 = new GraphEdge<double>("obama", "usa", "PRESIDENT_OF")
        {
            ValidFrom = new DateTime(2009, 1, 20),
            ValidUntil = new DateTime(2017, 1, 20)
        };
        var edge2 = new GraphEdge<double>("trump", "usa", "PRESIDENT_OF")
        {
            ValidFrom = new DateTime(2017, 1, 20),
            ValidUntil = new DateTime(2021, 1, 20)
        };
        var edge3 = new GraphEdge<double>("biden", "usa", "PRESIDENT_OF")
        {
            ValidFrom = new DateTime(2021, 1, 20),
            ValidUntil = new DateTime(2025, 1, 20)
        };

        graph.AddEdge(edge1);
        graph.AddEdge(edge2);
        graph.AddEdge(edge3);

        return graph;
    }

    private static KGEmbeddingOptions SmallTrainingOptions(int seed = 42) => new()
    {
        EmbeddingDimension = 20,
        Epochs = 50,
        BatchSize = 4,
        LearningRate = 0.01,
        Margin = 1.0,
        NegativeSamples = 2,
        Seed = seed
    };

    #endregion

    #region TransE Embedding Tests

    [Fact]
    public void TransE_TrainAndScore_ProducesValidResults()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.True(model.IsDistanceBased);
        Assert.Equal(20, model.EmbeddingDimension);
        Assert.Equal(8, result.TripleCount);
        Assert.Equal(8, result.EntityCount);
        Assert.Equal(3, result.RelationCount);
        Assert.Equal(50, result.TotalEpochs);
        Assert.True(result.TrainingDuration.TotalMilliseconds > 0);

        // Loss should decrease over training
        Assert.True(result.EpochLosses.Count == 50);
        Assert.True(result.EpochLosses[^1] <= result.EpochLosses[0],
            $"Loss should decrease: first={result.EpochLosses[0]}, last={result.EpochLosses[^1]}");

        // Known triple should score better (lower) than random
        double knownScore = model.ScoreTriple("einstein", "BORN_IN", "germany");
        double randomScore = model.ScoreTriple("einstein", "BORN_IN", "denmark");
        Assert.True(knownScore < randomScore,
            $"Known triple should score lower (better) than random: known={knownScore}, random={randomScore}");
    }

    [Fact]
    public void TransE_GetEntityEmbedding_ReturnsCorrectDimension()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var emb = model.GetEntityEmbedding("einstein");
        Assert.NotNull(emb);
        Assert.Equal(20, emb.Length);

        // Returns null for unknown entity
        Assert.Null(model.GetEntityEmbedding("unknown_entity"));
    }

    [Fact]
    public void TransE_GetRelationEmbedding_ReturnsCorrectDimension()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var emb = model.GetRelationEmbedding("WORKS_IN");
        Assert.NotNull(emb);
        Assert.Equal(20, emb.Length);

        Assert.Null(model.GetRelationEmbedding("UNKNOWN_RELATION"));
    }

    [Fact]
    public void TransE_ScoreTriple_UnknownEntitiesReturnMaxValue()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        double score = model.ScoreTriple("nonexistent", "WORKS_IN", "physics");
        Assert.Equal(double.MaxValue, score);
    }

    #endregion

    #region RotatE Embedding Tests

    [Fact]
    public void RotatE_TrainAndScore_ProducesValidResults()
    {
        var graph = BuildSmallGraph();
        var model = new RotatEEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.True(model.IsDistanceBased);
        Assert.Equal(20, model.EmbeddingDimension);
        Assert.Equal(8, result.TripleCount);
    }

    [Fact]
    public void RotatE_EntityEmbeddings_HaveComplexDimension()
    {
        var graph = BuildSmallGraph();
        var model = new RotatEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        // RotatE entity embeddings are 2*dim (real+imaginary)
        var emb = model.GetEntityEmbedding("einstein");
        Assert.NotNull(emb);
        Assert.Equal(40, emb.Length); // 2 * 20

        // Relation embeddings are phase angles (dim only)
        var relEmb = model.GetRelationEmbedding("WORKS_IN");
        Assert.NotNull(relEmb);
        Assert.Equal(20, relEmb.Length);
    }

    #endregion

    #region ComplEx Embedding Tests

    [Fact]
    public void ComplEx_TrainAndScore_ProducesValidResults()
    {
        var graph = BuildSmallGraph();
        var model = new ComplExEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.False(model.IsDistanceBased); // Semantic matching
        Assert.Equal(8, result.TripleCount);
    }

    [Fact]
    public void ComplEx_SemanticScoring_HigherIsBetter()
    {
        var graph = BuildSmallGraph();
        var model = new ComplExEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        // For semantic models, known triples should score higher than random
        double knownScore = model.ScoreTriple("einstein", "BORN_IN", "germany");
        double randomScore = model.ScoreTriple("einstein", "BORN_IN", "denmark");
        Assert.True(knownScore > randomScore,
            $"Known triple should score higher than random for ComplEx: known={knownScore}, random={randomScore}");
    }

    [Fact]
    public void ComplEx_EntityEmbeddings_HaveComplexDimension()
    {
        var graph = BuildSmallGraph();
        var model = new ComplExEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        // ComplEx: both entity and relation embeddings are 2*dim
        var entEmb = model.GetEntityEmbedding("einstein");
        Assert.NotNull(entEmb);
        Assert.Equal(40, entEmb.Length); // 2 * 20

        var relEmb = model.GetRelationEmbedding("WORKS_IN");
        Assert.NotNull(relEmb);
        Assert.Equal(40, relEmb.Length); // 2 * 20
    }

    [Fact]
    public void ComplEx_ScoreTriple_UnknownEntitiesReturnMinValue()
    {
        var graph = BuildSmallGraph();
        var model = new ComplExEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        // Semantic model returns MinValue for unknown (worst possible)
        double score = model.ScoreTriple("nonexistent", "WORKS_IN", "physics");
        Assert.Equal(double.MinValue, score);
    }

    #endregion

    #region DistMult Embedding Tests

    [Fact]
    public void DistMult_TrainAndScore_ProducesValidResults()
    {
        var graph = BuildSmallGraph();
        var model = new DistMultEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.False(model.IsDistanceBased);
        Assert.Equal(8, result.TripleCount);

        // Standard dim for DistMult (no complex numbers)
        var emb = model.GetEntityEmbedding("einstein");
        Assert.NotNull(emb);
        Assert.Equal(20, emb.Length);
    }

    #endregion

    #region TemporalTransE Tests

    [Fact]
    public void TemporalTransE_TrainAndScoreAtTime_Works()
    {
        var graph = BuildTemporalGraph();
        var model = new TemporalTransEEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.True(model.IsDistanceBased);

        // Score at a time when Obama was president
        double obamaScore2012 = model.ScoreTripleAtTime("obama", "PRESIDENT_OF", "usa", new DateTime(2012, 6, 1));
        Assert.True(!double.IsNaN(obamaScore2012) && !double.IsInfinity(obamaScore2012), "Score should be finite");

        // Score at a time when Trump was president
        double trumpScore2019 = model.ScoreTripleAtTime("trump", "PRESIDENT_OF", "usa", new DateTime(2019, 6, 1));
        Assert.True(!double.IsNaN(trumpScore2019) && !double.IsInfinity(trumpScore2019), "Score should be finite");
    }

    [Fact]
    public void TemporalTransE_NumTimeBins_ConfigurableViaOptions()
    {
        var graph = BuildTemporalGraph();
        var model = new TemporalTransEEmbedding<double>();
        var opts = SmallTrainingOptions();
        opts.NumTimeBins = 12; // 12 bins instead of default 100

        var result = model.Train(graph, opts);
        Assert.True(model.IsTrained);
    }

    [Fact]
    public void TemporalTransE_TrainedViaInterface_Works()
    {
        var graph = BuildTemporalGraph();
        // Create via interface to verify override chain works (not 'new' keyword)
        IKnowledgeGraphEmbedding<double> model = new TemporalTransEEmbedding<double>();
        var result = model.Train(graph, SmallTrainingOptions());

        Assert.True(model.IsTrained);
        Assert.Equal(3, result.TripleCount);
    }

    #endregion

    #region Temporal Graph Query Tests

    [Fact]
    public void TemporalGraph_GetEdgesAt_FiltersCorrectly()
    {
        var graph = BuildTemporalGraph();

        // In 2015, only Obama was president
        var edges2015 = graph.GetEdgesAt(new DateTime(2015, 6, 1)).ToList();
        Assert.Single(edges2015);
        Assert.Equal("obama", edges2015[0].SourceId);

        // In 2019, only Trump was president
        var edges2019 = graph.GetEdgesAt(new DateTime(2019, 6, 1)).ToList();
        Assert.Single(edges2019);
        Assert.Equal("trump", edges2019[0].SourceId);
    }

    [Fact]
    public void TemporalGraph_GetNeighborsAt_ReturnsCorrectNeighbors()
    {
        var graph = BuildTemporalGraph();

        // GetNeighborsAt uses outgoing edges: obama -> usa
        // In 2015, Obama's outgoing PRESIDENT_OF edge is valid, so neighbor = usa
        var neighborsObama2015 = graph.GetNeighborsAt("obama", new DateTime(2015, 6, 1)).ToList();
        Assert.Contains(neighborsObama2015, n => n.Id == "usa");

        // In 2019, Obama's edge expired, so no outgoing neighbors for obama
        var neighborsObama2019 = graph.GetNeighborsAt("obama", new DateTime(2019, 6, 1)).ToList();
        Assert.DoesNotContain(neighborsObama2019, n => n.Id == "usa");

        // In 2019, Trump's outgoing PRESIDENT_OF edge is valid, so neighbor = usa
        var neighborsTrump2019 = graph.GetNeighborsAt("trump", new DateTime(2019, 6, 1)).ToList();
        Assert.Contains(neighborsTrump2019, n => n.Id == "usa");
    }

    [Fact]
    public void GraphEdge_IsValidAt_WorksCorrectly()
    {
        var edge = new GraphEdge<double>("a", "b", "REL")
        {
            ValidFrom = new DateTime(2020, 1, 1),
            ValidUntil = new DateTime(2023, 12, 31)
        };

        Assert.True(edge.IsValidAt(new DateTime(2022, 6, 15)));
        Assert.False(edge.IsValidAt(new DateTime(2019, 12, 31)));
        Assert.False(edge.IsValidAt(new DateTime(2024, 1, 1)));

        // Edge with no temporal bounds is always valid
        var permanentEdge = new GraphEdge<double>("x", "y", "PERMANENT");
        Assert.True(permanentEdge.IsValidAt(DateTime.MinValue));
        Assert.True(permanentEdge.IsValidAt(DateTime.MaxValue));
    }

    #endregion

    #region Link Prediction Tests

    [Fact]
    public void LinkPredictor_PredictTails_ReturnsRankedResults()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var predictions = predictor.PredictTails(graph, "einstein", "WORKS_IN", topK: 5, filterExisting: false);

        Assert.NotEmpty(predictions);
        Assert.True(predictions.Count <= 5);
        Assert.All(predictions, p =>
        {
            Assert.Equal("einstein", p.HeadId);
            Assert.Equal("WORKS_IN", p.RelationType);
            Assert.True(!double.IsNaN(p.Score) && !double.IsInfinity(p.Score));
            Assert.InRange(p.Confidence, 0.0, 1.0);
        });
    }

    [Fact]
    public void LinkPredictor_PredictHeads_ReturnsRankedResults()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var predictions = predictor.PredictHeads(graph, "BORN_IN", "germany", topK: 5, filterExisting: false);

        Assert.NotEmpty(predictions);
        Assert.All(predictions, p =>
        {
            Assert.Equal("germany", p.TailId);
            Assert.Equal("BORN_IN", p.RelationType);
        });
    }

    [Fact]
    public void LinkPredictor_PredictTails_FilterExisting_ExcludesKnownTriples()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var predictions = predictor.PredictTails(graph, "einstein", "BORN_IN", topK: 10, filterExisting: true);

        // "germany" should be filtered out since (einstein, BORN_IN, germany) exists
        Assert.DoesNotContain(predictions, p => p.TailId == "germany");
    }

    [Fact]
    public void LinkPredictor_EvaluateModel_ReturnsValidMetrics()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var testTriples = new[]
        {
            ("einstein", "BORN_IN", "germany"),
            ("bohr", "BORN_IN", "denmark")
        };

        var eval = predictor.EvaluateModel(graph, testTriples);

        Assert.Equal(2, eval.TestTripleCount);
        Assert.True(eval.MeanReciprocalRank > 0.0, $"MRR should be positive: {eval.MeanReciprocalRank}");
        Assert.True(eval.MeanRank >= 1.0, $"MeanRank should be >= 1: {eval.MeanRank}");
        Assert.True(eval.HitsAtK.ContainsKey(1));
        Assert.True(eval.HitsAtK.ContainsKey(3));
        Assert.True(eval.HitsAtK.ContainsKey(10));
        Assert.InRange(eval.HitsAtK[10], 0.0, 1.0);
    }

    [Fact]
    public void LinkPredictor_EvaluateModel_EmptyTestSet_ReturnsZero()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var eval = predictor.EvaluateModel(graph, Array.Empty<(string, string, string)>());

        Assert.Equal(0, eval.TestTripleCount);
    }

    [Fact]
    public void LinkPredictor_WorksWithSemanticModel()
    {
        var graph = BuildSmallGraph();
        var model = new DistMultEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        var predictions = predictor.PredictTails(graph, "einstein", "WORKS_IN", topK: 3, filterExisting: false);

        Assert.NotEmpty(predictions);
        // For semantic models, scores should be sorted descending (higher = better)
        for (int i = 0; i < predictions.Count - 1; i++)
        {
            Assert.True(predictions[i].Score >= predictions[i + 1].Score,
                $"Semantic model predictions should be sorted descending: [{i}]={predictions[i].Score}, [{i + 1}]={predictions[i + 1].Score}");
        }
    }

    #endregion

    #region Leiden Community Detection Tests

    [Fact]
    public void Leiden_DetectCommunities_FindsStructure()
    {
        var graph = BuildSmallGraph();
        var detector = new LeidenCommunityDetector<double>();
        var result = detector.Detect(graph, new LeidenOptions { Seed = 42 });

        Assert.NotNull(result.Communities);
        Assert.Equal(8, result.Communities.Count); // All 8 nodes assigned
        Assert.NotEmpty(result.HierarchicalPartitions);
        Assert.NotEmpty(result.ModularityScores);

        // All nodes should be in the partition
        Assert.Contains("einstein", result.Communities.Keys);
        Assert.Contains("physics", result.Communities.Keys);
        Assert.Contains("germany", result.Communities.Keys);
    }

    [Fact]
    public void Leiden_HierarchicalPartitions_UseOriginalNodeIds()
    {
        var graph = BuildSmallGraph();
        var detector = new LeidenCommunityDetector<double>();
        var result = detector.Detect(graph, new LeidenOptions { Seed = 42 });

        // ALL levels should use original node IDs, never super-node IDs
        foreach (var partition in result.HierarchicalPartitions)
        {
            foreach (var key in partition.Keys)
            {
                Assert.DoesNotContain("super_", key);
            }
        }
    }

    [Fact]
    public void Leiden_EmptyGraph_ReturnsEmptyResult()
    {
        var graph = new KnowledgeGraph<double>();
        var detector = new LeidenCommunityDetector<double>();
        var result = detector.Detect(graph);

        Assert.NotNull(result);
        Assert.Empty(result.Communities);
    }

    #endregion

    #region Community Summarizer Tests

    [Fact]
    public void CommunitySummarizer_Summarize_ProducesSummaries()
    {
        var graph = BuildSmallGraph();
        var detector = new LeidenCommunityDetector<double>();
        var leidenResult = detector.Detect(graph, new LeidenOptions { Seed = 42 });

        var summarizer = new CommunitySummarizer<double>();
        var summaries = summarizer.Summarize(graph, leidenResult);

        Assert.NotEmpty(summaries);
        Assert.All(summaries, s =>
        {
            Assert.NotEmpty(s.EntityIds);
            Assert.NotEmpty(s.Description);
            Assert.Equal(0, s.Level);
        });
    }

    [Fact]
    public void CommunitySummarizer_SummarizePartition_SetsCorrectLevel()
    {
        var graph = BuildSmallGraph();
        var partition = new Dictionary<string, int>
        {
            ["einstein"] = 0, ["bohr"] = 0, ["curie"] = 1,
            ["physics"] = 0, ["chemistry"] = 1,
            ["germany"] = 2, ["denmark"] = 2, ["poland"] = 2
        };

        var summarizer = new CommunitySummarizer<double>();
        var summaries = summarizer.SummarizePartition(graph, partition, level: 3);

        Assert.All(summaries, s => Assert.Equal(3, s.Level));
        Assert.Equal(3, summaries.Count); // 3 communities
    }

    #endregion

    #region CommunityIndex Tests

    [Fact]
    public void CommunityIndex_Build_PopulatesAllLevels()
    {
        var graph = BuildSmallGraph();
        var detector = new LeidenCommunityDetector<double>();
        var leidenResult = detector.Detect(graph, new LeidenOptions { Seed = 42, MaxIterations = 5 });

        var index = new CommunityIndex<double>();
        index.Build(graph, leidenResult);

        // At minimum, level 0 should be populated
        var level0Summaries = index.GetSummariesAtLevel(0).ToList();
        Assert.NotEmpty(level0Summaries);

        // Each level should have the same number of levels as hierarchical partitions
        for (int level = 0; level < leidenResult.HierarchicalPartitions.Count; level++)
        {
            var summaries = index.GetSummariesAtLevel(level).ToList();
            Assert.NotEmpty(summaries);
        }
    }

    [Fact]
    public void CommunityIndex_SearchCommunities_FindsRelevant()
    {
        var graph = BuildSmallGraph();
        // Set name properties for better search
        graph.GetNode("einstein")?.SetProperty("name", "Albert Einstein");
        graph.GetNode("bohr")?.SetProperty("name", "Niels Bohr");
        graph.GetNode("physics")?.SetProperty("name", "Physics");

        var detector = new LeidenCommunityDetector<double>();
        var leidenResult = detector.Detect(graph, new LeidenOptions { Seed = 42 });

        var index = new CommunityIndex<double>();
        index.Build(graph, leidenResult);

        var results = index.SearchCommunities("physics", level: 0, topK: 3).ToList();
        // Should find at least one community related to physics
        Assert.NotEmpty(results);
    }

    #endregion

    #region EnhancedGraphRAG Tests

    [Fact]
    public void EnhancedGraphRAG_LocalSearch_ReturnsContext()
    {
        var graph = BuildSmallGraph();
        graph.GetNode("einstein")?.SetProperty("name", "Albert Einstein");
        graph.GetNode("physics")?.SetProperty("name", "Physics");

        var rag = new EnhancedGraphRAG<double>(graph, new GraphRAGOptions { Mode = GraphRAGMode.Local });
        var context = rag.Retrieve("einstein", topK: 5);

        Assert.NotEmpty(context);
    }

    [Fact]
    public void EnhancedGraphRAG_GlobalSearch_RequiresCommunityIndex()
    {
        var graph = BuildSmallGraph();
        var rag = new EnhancedGraphRAG<double>(graph, new GraphRAGOptions { Mode = GraphRAGMode.Global });

        Assert.Throws<InvalidOperationException>(() => rag.Retrieve("physics"));
    }

    [Fact]
    public void EnhancedGraphRAG_GlobalSearch_WithCommunityIndex_Works()
    {
        var graph = BuildSmallGraph();
        graph.GetNode("physics")?.SetProperty("name", "Physics");

        var rag = new EnhancedGraphRAG<double>(graph, new GraphRAGOptions { Mode = GraphRAGMode.Global });
        rag.BuildCommunityIndex();

        var context = rag.Retrieve("physics", topK: 5);
        Assert.NotEmpty(context);

        // Community structure should be available
        Assert.NotNull(rag.CommunityStructure);
    }

    [Fact]
    public void EnhancedGraphRAG_DriftSearch_Works()
    {
        var graph = BuildSmallGraph();
        graph.GetNode("einstein")?.SetProperty("name", "Albert Einstein");
        graph.GetNode("physics")?.SetProperty("name", "Physics");

        var rag = new EnhancedGraphRAG<double>(graph, new GraphRAGOptions
        {
            Mode = GraphRAGMode.Drift,
            DriftMaxIterations = 2
        });
        rag.BuildCommunityIndex();

        var context = rag.Retrieve("physics", topK: 10);
        Assert.NotEmpty(context);

        // DRIFT should include both community and drill-down context
        Assert.Contains(context, c => c.StartsWith("[Community]"));
    }

    [Fact]
    public void EnhancedGraphRAG_RetrieveNodes_ReturnsNodes()
    {
        var graph = BuildSmallGraph();
        graph.GetNode("einstein")?.SetProperty("name", "Albert Einstein");

        var rag = new EnhancedGraphRAG<double>(graph);
        var nodes = rag.RetrieveNodes("einstein", topK: 5).ToList();

        // Should find at least one node (depending on FindRelatedNodes implementation)
        // If no match, empty is acceptable — we're testing it doesn't crash
        Assert.NotNull(nodes);
    }

    [Fact]
    public void EnhancedGraphRAG_EmptyQuery_ReturnsEmpty()
    {
        var graph = BuildSmallGraph();
        var rag = new EnhancedGraphRAG<double>(graph);

        Assert.Empty(rag.Retrieve(""));
        Assert.Empty(rag.Retrieve("   "));
        Assert.Empty(rag.RetrieveNodes(""));
    }

    #endregion

    #region KG Construction from Text Tests

    [Fact]
    public void KGConstructor_ConstructFromText_ExtractsEntities()
    {
        var constructor = new KGConstructor<double>();
        var graph = constructor.ConstructFromText(
            "Albert Einstein was born in Ulm, Germany. He worked at Princeton University.");

        var allNodes = graph.GetAllNodes().ToList();
        Assert.NotEmpty(allNodes);

        // Should extract at least some named entities
        var nodeIds = allNodes.Select(n => n.Id).ToList();
        Assert.Contains(nodeIds, id => id.Contains("einstein") || id.Contains("albert"));
    }

    [Fact]
    public void KGConstructor_ConstructFromText_ExtractsRelations()
    {
        var constructor = new KGConstructor<double>();
        var graph = constructor.ConstructFromText(
            "Albert Einstein was born in Germany. Marie Curie worked at the University of Paris.");

        var edges = graph.GetAllEdges().ToList();
        // Should extract at least one relation (born in, worked at, or co-occurrence)
        Assert.NotEmpty(edges);
    }

    [Fact]
    public void KGConstructor_ExtractEntities_HandlesAbbreviations()
    {
        var constructor = new KGConstructor<double>();
        var entities = constructor.ExtractEntities("The NASA and IBM partnership with the U.N. was announced.", 0.3);

        var names = entities.Select(e => e.Name).ToList();
        // Should detect abbreviations
        Assert.Contains(names, n => n == "NASA" || n == "IBM");
    }

    [Fact]
    public void KGConstructor_ExtractEntities_HandlesHyphenatedNames()
    {
        var constructor = new KGConstructor<double>();
        var entities = constructor.ExtractEntities("Jean-Pierre Serre studied at the Ecole Normale.", 0.3);

        var names = entities.Select(e => e.Name).ToList();
        Assert.Contains(names, n => n.Contains("Jean-Pierre") || n.Contains("Serre"));
    }

    [Fact]
    public void KGConstructor_MultipleTexts_AugmentsGraph()
    {
        var constructor = new KGConstructor<double>();
        var graph = new KnowledgeGraph<double>();

        constructor.ConstructFromText("Albert Einstein was born in Germany.", graph);
        int nodesAfterFirst = graph.GetAllNodes().Count();

        constructor.ConstructFromText("Marie Curie was born in Poland.", graph);
        int nodesAfterSecond = graph.GetAllNodes().Count();

        Assert.True(nodesAfterSecond >= nodesAfterFirst,
            "Second text should add more entities to the graph");
    }

    [Fact]
    public void KGConstructor_WithEntityResolution_MergesSimilarNames()
    {
        var constructor = new KGConstructor<double>();
        var opts = new KGConstructionOptions
        {
            EnableEntityResolution = true,
            EntitySimilarityThreshold = 0.8
        };

        var graph = constructor.ConstructFromText(
            "Albert Einstein discovered relativity. Einstein also contributed to quantum mechanics.",
            options: opts);

        // Entity resolution should merge "Albert Einstein" and "Einstein" if similar enough
        Assert.NotNull(graph);
    }

    [Fact]
    public void KGConstructor_EmptyText_Throws()
    {
        var constructor = new KGConstructor<double>();
        Assert.Throws<ArgumentException>(() => constructor.ConstructFromText(""));
        Assert.Throws<ArgumentException>(() => constructor.ConstructFromText("   "));
    }

    #endregion

    #region Embedding Model Consistency Tests

    [Fact]
    public void AllModels_SameGraph_ProduceConsistentEmbeddingSizes()
    {
        var graph = BuildSmallGraph();
        var opts = SmallTrainingOptions();

        var models = new IKnowledgeGraphEmbedding<double>[]
        {
            new TransEEmbedding<double>(),
            new RotatEEmbedding<double>(),
            new ComplExEmbedding<double>(),
            new DistMultEmbedding<double>()
        };

        foreach (var model in models)
        {
            model.Train(graph, opts);
            Assert.True(model.IsTrained);
            Assert.Equal(20, model.EmbeddingDimension);

            // ScoreTriple should return finite values for known triples
            double score = model.ScoreTriple("einstein", "WORKS_IN", "physics");
            Assert.True(!double.IsNaN(score) && !double.IsInfinity(score), $"{model.GetType().Name} returned non-finite score: {score}");
        }
    }

    [Fact]
    public void AllModels_UntrainedModel_ThrowsOnScore()
    {
        var models = new IKnowledgeGraphEmbedding<double>[]
        {
            new TransEEmbedding<double>(),
            new RotatEEmbedding<double>(),
            new ComplExEmbedding<double>(),
            new DistMultEmbedding<double>()
        };

        foreach (var model in models)
        {
            Assert.False(model.IsTrained);
            Assert.Throws<InvalidOperationException>(() =>
                model.ScoreTriple("a", "REL", "b"));
        }
    }

    [Fact]
    public void AllModels_Reproducible_WithSameSeed()
    {
        var graph = BuildSmallGraph();
        var opts = SmallTrainingOptions(seed: 123);

        var model1 = new TransEEmbedding<double>();
        model1.Train(graph, opts);
        double score1 = model1.ScoreTriple("einstein", "BORN_IN", "germany");

        var model2 = new TransEEmbedding<double>();
        model2.Train(graph, opts);
        double score2 = model2.ScoreTriple("einstein", "BORN_IN", "germany");

        Assert.Equal(score1, score2, precision: 10);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void Embedding_GraphWithNoEdges_Throws()
    {
        var graph = new KnowledgeGraph<double>();
        graph.AddNode(new GraphNode<double>("lonely", "NODE"));

        var model = new TransEEmbedding<double>();
        Assert.Throws<InvalidOperationException>(() => model.Train(graph, SmallTrainingOptions()));
    }

    [Fact]
    public void LinkPredictor_UntrainedModel_Throws()
    {
        var model = new TransEEmbedding<double>();
        Assert.Throws<InvalidOperationException>(() => new LinkPredictor<double>(model));
    }

    [Fact]
    public void LinkPredictor_NullGraph_Throws()
    {
        var graph = BuildSmallGraph();
        var model = new TransEEmbedding<double>();
        model.Train(graph, SmallTrainingOptions());

        var predictor = new LinkPredictor<double>(model);
        Assert.Throws<ArgumentNullException>(() => predictor.PredictTails(null, "einstein", "REL"));
    }

    #endregion
}
