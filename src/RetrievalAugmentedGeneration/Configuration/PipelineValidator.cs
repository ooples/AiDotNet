using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;

namespace AiDotNet.RetrievalAugmentedGeneration.Configuration
{
    /// <summary>
    /// Validates that a set of components forms a valid AI pipeline at runtime.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PipelineValidator provides static methods to check whether a collection of components
    /// can form a valid pipeline before execution. It catches common misconfigurations early
    /// with clear error messages, preventing cryptic runtime failures.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this like a pre-flight checklist for your AI pipeline.
    ///
    /// Before an airplane takes off, pilots check that all systems are working:
    /// - Engines? Check.
    /// - Navigation? Check.
    /// - Fuel? Check.
    ///
    /// Similarly, before running a RAG pipeline, this validator checks:
    /// - Do you have a retriever? Check.
    /// - Do you have a generator? Check.
    /// - Are components in the right stages? Check.
    ///
    /// If something is missing or misconfigured, you get a clear error message
    /// instead of a confusing crash at runtime.
    /// </para>
    /// </remarks>
    public static class PipelineValidator
    {
        /// <summary>
        /// The result of a pipeline validation, containing any errors and warnings.
        /// </summary>
        /// <param name="IsValid">Whether the configuration has no errors (warnings are still allowed).</param>
        /// <param name="Errors">Fatal misconfigurations that will prevent the pipeline from working.</param>
        /// <param name="Warnings">Non-fatal issues that may indicate accidental misconfiguration.</param>
        public sealed class ValidationResult
        {
            public bool IsValid { get; }
            public IReadOnlyList<string> Errors { get; }
            public IReadOnlyList<string> Warnings { get; }

            public ValidationResult(bool isValid, IReadOnlyList<string> errors, IReadOnlyList<string> warnings)
            {
                IsValid = isValid;
                Errors = errors;
                Warnings = warnings;
            }
        }

        // Component types that are valid for each pipeline stage.
        private static readonly Dictionary<PipelineStage, HashSet<ComponentType>> ValidStageComponents = new()
        {
            [PipelineStage.DataIngestion] = new HashSet<ComponentType>
            {
                ComponentType.Chunker,
                ComponentType.EntityRecognizer
            },
            [PipelineStage.Indexing] = new HashSet<ComponentType>
            {
                ComponentType.DocumentStore,
                ComponentType.VectorIndex
            },
            [PipelineStage.QueryProcessing] = new HashSet<ComponentType>
            {
                ComponentType.QueryProcessor,
                ComponentType.QueryExpander
            },
            [PipelineStage.Retrieval] = new HashSet<ComponentType>
            {
                ComponentType.Retriever
            },
            [PipelineStage.PostRetrieval] = new HashSet<ComponentType>
            {
                ComponentType.Reranker,
                ComponentType.ContextCompressor
            },
            [PipelineStage.Generation] = new HashSet<ComponentType>
            {
                ComponentType.Generator
            },
            [PipelineStage.Evaluation] = new HashSet<ComponentType>
            {
                ComponentType.Evaluator
            },
            [PipelineStage.Preprocessing] = new HashSet<ComponentType>
            {
                ComponentType.Scaler,
                ComponentType.Encoder,
                ComponentType.DimensionReducer,
                ComponentType.FeatureSelector,
                ComponentType.FeatureGenerator
            },
            [PipelineStage.Training] = new HashSet<ComponentType>
            {
                ComponentType.Optimizer,
                ComponentType.Scheduler,
                ComponentType.Regularizer,
                ComponentType.DistillationStrategy,
                ComponentType.FederatedAggregator,
                ComponentType.TransferAlgorithm,
                ComponentType.DomainAdapter,
                ComponentType.MetaLearner,
                ComponentType.ActiveLearner,
                ComponentType.ContinualLearner
            }
        };

        // Stages that should only have one component (singleton stages).
        private static readonly HashSet<PipelineStage> SingletonStages = new()
        {
            PipelineStage.Generation
        };

        // The logical ordering of RAG pipeline stages.
        private static readonly PipelineStage[] RAGStageOrder =
        {
            PipelineStage.DataIngestion,
            PipelineStage.Indexing,
            PipelineStage.QueryProcessing,
            PipelineStage.Retrieval,
            PipelineStage.PostRetrieval,
            PipelineStage.Generation,
            PipelineStage.Evaluation
        };

        /// <summary>
        /// Validates the RAG components that have been configured on the builder.
        /// </summary>
        /// <param name="hasRetriever">Whether an <c>IRetriever&lt;T&gt;</c> was provided.</param>
        /// <param name="hasReranker">Whether an <c>IReranker&lt;T&gt;</c> was provided.</param>
        /// <param name="hasGenerator">Whether an <c>IGenerator&lt;T&gt;</c> was provided.</param>
        /// <param name="hasQueryProcessors">Whether any <c>IQueryProcessor</c> instances were provided.</param>
        /// <param name="hasDocumentStore">Whether an <c>IDocumentStore&lt;T&gt;</c> was provided.</param>
        /// <param name="hasKnowledgeGraph">Whether a <c>KnowledgeGraph&lt;T&gt;</c> was provided or constructed.</param>
        /// <param name="hasGraphStore">Whether an <c>IGraphStore&lt;T&gt;</c> was provided.</param>
        /// <returns>A <see cref="ValidationResult"/> with any errors and warnings found.</returns>
        public static ValidationResult ValidateRAGConfiguration(
            bool hasRetriever,
            bool hasReranker,
            bool hasGenerator,
            bool hasQueryProcessors,
            bool hasDocumentStore,
            bool hasKnowledgeGraph,
            bool hasGraphStore)
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            bool hasAnyComponent = hasRetriever || hasReranker || hasGenerator
                || hasQueryProcessors || hasDocumentStore || hasKnowledgeGraph || hasGraphStore;

            // If nothing is configured there is nothing to validate.
            if (!hasAnyComponent)
            {
                return new ValidationResult(true, errors, warnings);
            }

            // --- Errors (fatal) ---

            // A RAG pipeline needs at least one retrieval source.
            if (!hasRetriever && !hasKnowledgeGraph)
            {
                errors.Add(
                    "RAG pipeline has no retrieval source. " +
                    "Provide an IRetriever<T> or a KnowledgeGraph<T> so the pipeline has something to retrieve from.");
            }

            // --- Warnings (non-fatal) ---

            if (hasRetriever && !hasDocumentStore)
            {
                warnings.Add(
                    "A retriever is configured without a document store. " +
                    "The retriever may have no documents to search unless they are loaded separately.");
            }

            if (hasReranker && !hasRetriever)
            {
                warnings.Add(
                    "A reranker is configured without a retriever. " +
                    "The reranker needs initial retrieval results to reorder.");
            }

            if (hasGenerator && !hasRetriever && !hasKnowledgeGraph)
            {
                warnings.Add(
                    "A generator is configured but there is no retriever or knowledge graph to supply context. " +
                    "The generator will have no retrieved context to augment its output.");
            }

            if (hasKnowledgeGraph && !hasGraphStore)
            {
                warnings.Add(
                    "A knowledge graph is configured without a graph store. " +
                    "The knowledge graph will not be persisted between sessions.");
            }

            return new ValidationResult(errors.Count == 0, errors, warnings);
        }

        /// <summary>
        /// Validates a RAG pipeline has all required stages and compatible components.
        /// </summary>
        /// <param name="stages">The pipeline stages present in the pipeline.</param>
        /// <param name="componentTypes">The component types corresponding to each stage (same order as stages).</param>
        /// <returns>A <see cref="ValidationResult"/> indicating whether the pipeline is valid, with any errors and warnings.</returns>
        /// <remarks>
        /// <para>
        /// A valid RAG pipeline requires at minimum a Retrieval stage and a Generation stage.
        /// This method also checks that each component type is appropriate for its declared stage
        /// and warns about missing recommended stages (DataIngestion, Indexing, PostRetrieval).
        /// </para>
        /// <para><b>For Beginners:</b> This checks if your RAG pipeline has everything it needs.
        ///
        /// Required (errors if missing):
        /// - A retriever (to find relevant documents)
        /// - A generator (to produce answers from those documents)
        ///
        /// Recommended (warnings if missing):
        /// - A chunker / data ingestion (to prepare documents)
        /// - A document store / indexing (to store and search documents)
        /// - A reranker / post-retrieval (to improve result quality)
        /// </para>
        /// </remarks>
        public static ValidationResult ValidateRAGPipeline(
            IReadOnlyList<PipelineStage> stages,
            IReadOnlyList<ComponentType> componentTypes)
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            if (stages == null)
            {
                errors.Add("Pipeline stages list cannot be null.");
                return new ValidationResult(false, errors, warnings);
            }

            if (componentTypes == null)
            {
                errors.Add("Component types list cannot be null.");
                return new ValidationResult(false, errors, warnings);
            }

            if (stages.Count != componentTypes.Count)
            {
                errors.Add(
                    $"Stage count ({stages.Count}) does not match component type count ({componentTypes.Count}). " +
                    "Each component must be associated with exactly one stage.");
                return new ValidationResult(false, errors, warnings);
            }

            // Check required stages: Retrieval and Generation
            var stageSet = new HashSet<PipelineStage>(stages);

            if (!stageSet.Contains(PipelineStage.Retrieval))
            {
                errors.Add(
                    "RAG pipeline is missing the Retrieval stage. " +
                    "A retriever component (e.g., DenseRetriever, BM25Retriever) is required to find relevant documents.");
            }

            if (!stageSet.Contains(PipelineStage.Generation))
            {
                errors.Add(
                    "RAG pipeline is missing the Generation stage. " +
                    "A generator component is required to produce answers from retrieved context.");
            }

            // Warn about missing recommended stages
            if (!stageSet.Contains(PipelineStage.DataIngestion))
            {
                warnings.Add(
                    "RAG pipeline has no DataIngestion stage. " +
                    "Consider adding a chunking strategy to split documents into retrievable segments.");
            }

            if (!stageSet.Contains(PipelineStage.Indexing))
            {
                warnings.Add(
                    "RAG pipeline has no Indexing stage. " +
                    "Consider adding a document store and embedding model to enable efficient vector search.");
            }

            if (!stageSet.Contains(PipelineStage.PostRetrieval))
            {
                warnings.Add(
                    "RAG pipeline has no PostRetrieval stage. " +
                    "Consider adding a reranker or context compressor to improve result quality.");
            }

            // Check that each component type is appropriate for its declared stage
            for (int i = 0; i < stages.Count; i++)
            {
                var stage = stages[i];
                var componentType = componentTypes[i];

                if (ValidStageComponents.TryGetValue(stage, out var validTypes))
                {
                    if (!validTypes.Contains(componentType))
                    {
                        errors.Add(
                            $"Component type '{componentType}' is not valid for pipeline stage '{stage}'. " +
                            $"Valid component types for '{stage}' are: {string.Join(", ", validTypes)}.");
                    }
                }
            }

            return new ValidationResult(errors.Count == 0, errors, warnings);
        }

        /// <summary>
        /// Validates a generic pipeline has no conflicting or missing stages.
        /// </summary>
        /// <param name="components">A list of component type and pipeline stage pairs representing the pipeline.</param>
        /// <returns>A <see cref="ValidationResult"/> indicating whether the pipeline is valid, with any errors and warnings.</returns>
        /// <remarks>
        /// <para>
        /// This method performs general pipeline validation including:
        /// <list type="bullet">
        /// <item>Checking that singleton stages (like Generation) do not have multiple components.</item>
        /// <item>Checking that stage ordering is logical (no stage appears before a stage it depends on).</item>
        /// <item>Checking that component types are appropriate for their declared stages.</item>
        /// <item>Warning about common misconfigurations.</item>
        /// </list>
        /// </para>
        /// <para><b>For Beginners:</b> This checks any AI pipeline for common configuration mistakes.
        ///
        /// It catches issues like:
        /// - Having two generators (only one is allowed)
        /// - Putting a retriever in the Generation stage (wrong stage)
        /// - Having stages in the wrong order (like Generation before Retrieval)
        /// </para>
        /// </remarks>
        public static ValidationResult ValidatePipeline(
            IReadOnlyList<(ComponentType type, PipelineStage stage)> components)
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            if (components == null)
            {
                errors.Add("Components list cannot be null.");
                return new ValidationResult(false, errors, warnings);
            }

            if (components.Count == 0)
            {
                errors.Add("Pipeline must contain at least one component.");
                return new ValidationResult(false, errors, warnings);
            }

            // Check for duplicate singleton stages
            var stageCounts = new Dictionary<PipelineStage, int>();
            foreach (var (_, stage) in components)
            {
                if (!stageCounts.ContainsKey(stage))
                {
                    stageCounts[stage] = 0;
                }
                stageCounts[stage]++;
            }

            foreach (var singletonStage in SingletonStages)
            {
                if (stageCounts.TryGetValue(singletonStage, out var count) && count > 1)
                {
                    errors.Add(
                        $"Pipeline stage '{singletonStage}' allows only one component but {count} were registered. " +
                        "Remove the duplicate components to fix this.");
                }
            }

            // Check that component types are valid for their declared stages
            foreach (var (componentType, stage) in components)
            {
                if (ValidStageComponents.TryGetValue(stage, out var validTypes))
                {
                    if (!validTypes.Contains(componentType))
                    {
                        errors.Add(
                            $"Component type '{componentType}' is not valid for pipeline stage '{stage}'. " +
                            $"Valid component types for '{stage}' are: {string.Join(", ", validTypes)}.");
                    }
                }
            }

            // Check stage ordering is logical
            var stageOrderIndex = new Dictionary<PipelineStage, int>();
            for (int i = 0; i < RAGStageOrder.Length; i++)
            {
                stageOrderIndex[RAGStageOrder[i]] = i;
            }

            int lastStageIndex = -1;
            PipelineStage lastStage = default;
            bool orderChecked = false;

            foreach (var (_, stage) in components)
            {
                if (stageOrderIndex.TryGetValue(stage, out var currentIndex))
                {
                    if (orderChecked && currentIndex < lastStageIndex)
                    {
                        warnings.Add(
                            $"Pipeline stage '{stage}' appears after '{lastStage}', but '{stage}' " +
                            $"typically runs before '{lastStage}'. Consider reordering your components " +
                            "to follow the standard pipeline flow: " +
                            "DataIngestion -> Indexing -> QueryProcessing -> Retrieval -> PostRetrieval -> Generation -> Evaluation.");
                    }

                    if (!orderChecked || currentIndex >= lastStageIndex)
                    {
                        lastStageIndex = currentIndex;
                        lastStage = stage;
                    }
                    orderChecked = true;
                }
            }

            // Warn if pipeline has retrieval but no generation, or vice versa
            var presentStages = new HashSet<PipelineStage>(components.Select(c => c.stage));

            if (presentStages.Contains(PipelineStage.Retrieval) && !presentStages.Contains(PipelineStage.Generation))
            {
                warnings.Add(
                    "Pipeline has a Retrieval stage but no Generation stage. " +
                    "Retrieved documents will not be used to generate responses unless a generator is added.");
            }

            if (presentStages.Contains(PipelineStage.Generation) && !presentStages.Contains(PipelineStage.Retrieval))
            {
                warnings.Add(
                    "Pipeline has a Generation stage but no Retrieval stage. " +
                    "The generator will not have retrieved context to ground its responses.");
            }

            if (presentStages.Contains(PipelineStage.PostRetrieval) && !presentStages.Contains(PipelineStage.Retrieval))
            {
                warnings.Add(
                    "Pipeline has a PostRetrieval stage but no Retrieval stage. " +
                    "Post-retrieval components (rerankers, compressors) require retrieved documents to process.");
            }

            return new ValidationResult(errors.Count == 0, errors, warnings);
        }
    }
}
