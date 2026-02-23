using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Contains the results of training a knowledge graph embedding model.
/// </summary>
/// <remarks>
/// <para>
/// This class captures training metrics including per-epoch loss values,
/// entity/relation counts, and training duration for diagnostics and evaluation.
/// </para>
/// <para><b>For Beginners:</b> After training, this tells you:
/// - EpochLosses: How the error decreased over time (should go down)
/// - EntityCount/RelationCount: How many entities and relation types were learned
/// - TripleCount: Total number of facts used for training
/// - TrainingDuration: How long training took
/// </para>
/// </remarks>
public class KGEmbeddingTrainingResult
{
    /// <summary>
    /// Average loss value for each training epoch.
    /// </summary>
    public List<double> EpochLosses { get; set; } = [];

    /// <summary>
    /// Total number of epochs completed.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Number of unique entities in the training data.
    /// </summary>
    public int EntityCount { get; set; }

    /// <summary>
    /// Number of unique relation types in the training data.
    /// </summary>
    public int RelationCount { get; set; }

    /// <summary>
    /// Total number of triples used for training.
    /// </summary>
    public int TripleCount { get; set; }

    /// <summary>
    /// Wall-clock training duration.
    /// </summary>
    public TimeSpan TrainingDuration { get; set; }
}
