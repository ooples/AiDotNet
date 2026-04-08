namespace AiDotNet.Enums;

/// <summary>
/// Defines which stage of an AI pipeline a component operates in.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> AI systems often process data through multiple stages, like an assembly line.
/// This tells you where in the pipeline a component fits. Components in the same stage can be
/// swapped; components in different stages are composed sequentially.
/// </para>
/// <para>
/// A typical RAG pipeline: DataIngestion → Indexing → Retrieval → PostRetrieval → Generation.
/// A typical ML pipeline: Preprocessing → Training → Evaluation.
/// </para>
/// </remarks>
public enum PipelineStage
{
    /// <summary>
    /// Data ingestion stage: parsing, chunking, and preparing raw data.
    /// Components: document parsers, chunking strategies, data loaders.
    /// </summary>
    DataIngestion,

    /// <summary>
    /// Indexing stage: embedding and storing processed data for retrieval.
    /// Components: embedding models, vector stores, document stores.
    /// </summary>
    Indexing,

    /// <summary>
    /// Retrieval stage: searching and filtering stored data given a query.
    /// Components: retrievers, vector search, sparse search.
    /// </summary>
    Retrieval,

    /// <summary>
    /// Post-retrieval stage: refining retrieved results before generation.
    /// Components: rerankers, context compressors, result filters.
    /// </summary>
    PostRetrieval,

    /// <summary>
    /// Generation stage: producing final output from retrieved context.
    /// Components: RAG generators, response synthesizers.
    /// </summary>
    Generation,

    /// <summary>
    /// Preprocessing stage: transforming raw features before model training/inference.
    /// Components: scalers, encoders, feature selectors, dimension reducers.
    /// </summary>
    Preprocessing,

    /// <summary>
    /// Training stage: optimizing model parameters from data.
    /// Components: optimizers, schedulers, distillation strategies, meta-learners.
    /// </summary>
    Training,

    /// <summary>
    /// Evaluation stage: measuring model/pipeline quality.
    /// Components: metrics calculators, benchmark runners, evaluators.
    /// </summary>
    Evaluation,

    /// <summary>
    /// Query processing stage: transforming user queries before retrieval.
    /// Components: query expanders, query decomposers, HyDE.
    /// </summary>
    QueryProcessing
}
