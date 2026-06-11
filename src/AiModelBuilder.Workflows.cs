using AiDotNet.Augmentation;
using AiDotNet.AutoML.NAS;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet;

/// <summary>
/// Workflow/build orchestration partial (RL, AutoML, federated, fine-tuning, agents) of <see cref="AiModelBuilder{T, TInput, TOutput}"/>. Split out of the
/// 9.5k-LoC main file (audit-2026-05 finding #12) for reviewability; no behaviour change.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{

    /// <summary>
    /// Converts a trained model into a byte array for storage or transmission.
    /// </summary>
    /// <param name="modelResult">The trained predictive model to serialize.</param>
    /// <returns>A byte array representing the serialized model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Serialization converts your model into a format (a series of bytes) that can be
    /// easily stored or sent over a network. This is the underlying mechanism that makes saving models possible.
    /// 
    /// You might use this directly if you want to store the model in a database or send it over a network,
    /// rather than saving it to a file.
    /// </remarks>
    public byte[] SerializeModel(AiModelResult<T, TInput, TOutput> modelResult)
    {
        if (modelResult is null)
            throw new ArgumentNullException(nameof(modelResult));

        ModelPersistenceGuard.EnforceBeforeSave();
        using (ModelPersistenceGuard.InternalOperation())
        {
            return modelResult.Serialize();
        }
    }

    /// <summary>
    /// Converts a byte array back into a usable predictive model.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <returns>The deserialized predictive model that can be used to make predictions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the byte array
    /// representation of your model and converts it back into a usable model object. This is what happens
    /// behind the scenes when you load a model from a file.
    /// 
    /// You might use this directly if you retrieved a serialized model from a database or received it over a network.
    /// </remarks>
    public AiModelResult<T, TInput, TOutput> DeserializeModel(byte[] modelData)
    {
        if (modelData is null)
            throw new ArgumentNullException(nameof(modelData));
        if (modelData.Length == 0)
            throw new ArgumentException("Model data cannot be empty.", nameof(modelData));

        ModelPersistenceGuard.EnforceBeforeLoad();
        var result = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>());
        using (ModelPersistenceGuard.InternalOperation())
        {
            result.Deserialize(modelData);
        }

        // Automatically reattach Graph RAG components if they were configured on this builder
        // Graph RAG components cannot be serialized (file handles, WAL, etc.), so we reattach
        // them from the builder's configuration to provide a seamless experience for users
        if (_knowledgeGraph != null || _graphStore != null || _hybridGraphRetriever != null)
        {
            result.AttachGraphComponents(_knowledgeGraph, _graphStore, _hybridGraphRetriever);
        }

        // Reattach tokenizer if configured
        if (_tokenizer != null)
        {
            result.AttachTokenizer(_tokenizer, _tokenizationConfig);
        }

        return result;
    }

    /// <summary>
    /// Configures the bias detector component for ethical AI evaluation.
    /// </summary>
    /// <param name="detector">The bias detector implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Bias detection helps ensure your model treats different groups fairly.
    /// You can choose from different bias detection strategies like Disparate Impact (80% rule),
    /// Demographic Parity, or Equal Opportunity. This component will be used to evaluate your
    /// trained model's fairness across demographic groups.
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureBiasDetector(IBiasDetector<T> detector)
    {
        _compliance.ConfigureBiasDetector(detector);
        _biasDetector = _compliance.BiasDetector;
        return this;
    }

    /// <summary>
    /// Configures the fairness evaluator component for ethical AI evaluation.
    /// </summary>
    /// <param name="evaluator">The fairness evaluator implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Fairness evaluation measures how equitably your model performs.
    /// You can choose evaluators that compute different sets of fairness metrics, from basic
    /// (just key metrics) to comprehensive (all fairness measures). This helps ensure your
    /// AI system is not only accurate but also ethical.
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFairnessEvaluator(IFairnessEvaluator<T> evaluator)
    {
        _compliance.ConfigureFairnessEvaluator(evaluator);
        _fairnessEvaluator = _compliance.FairnessEvaluator;
        return this;
    }

    /// <summary>
    /// Configures model interpretability and explainability features.
    /// </summary>
    /// <param name="options">The interpretability configuration options. When null, uses default settings.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This configures model-agnostic explanation methods that help understand how the model makes predictions:
    /// </para>
    /// <list type="bullet">
    /// <item><term>SHAP</term><description>Kernel SHAP for local and global feature attribution using Shapley values</description></item>
    /// <item><term>LIME</term><description>Local Interpretable Model-agnostic Explanations for individual predictions</description></item>
    /// <item><term>Permutation Importance</term><description>Global feature importance by measuring score drop when features are shuffled</description></item>
    /// <item><term>Global Surrogate</term><description>Train a simple model to approximate the complex model's behavior</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> These methods help you understand why your model makes certain predictions.
    ///
    /// After training, you can ask questions like:
    /// - "Why did the model predict this person would default on their loan?"
    /// - "Which features are most important for the model overall?"
    /// - "Can I explain the model's behavior with simple rules?"
    ///
    /// Example:
    /// <code>
    /// builder.ConfigureInterpretability(new InterpretabilityOptions
    /// {
    ///     EnableSHAP = true,
    ///     EnablePermutationImportance = true,
    ///     FeatureNames = new[] { "Age", "Income", "CreditScore" }
    /// });
    ///
    /// // After training
    /// var result = builder.Build();
    /// var shapExplanation = result.ExplainWithSHAP(inputInstance, backgroundData);
    /// // Result is available in the returned value
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureInterpretability(InterpretabilityOptions? options = null)
    {
        _compliance.ConfigureInterpretability(options);
        _interpretabilityOptions = _compliance.InterpretabilityOptions;
        return this;
    }

    /// <summary>
    /// Configures adversarial robustness and AI safety features for the model.
    /// </summary>
    /// <param name="configuration">The adversarial robustness configuration. When null, uses industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This unified configuration provides comprehensive control over all aspects of adversarial robustness and AI safety:
    /// </para>
    /// <list type="bullet">
    /// <item><term>Safety Filtering</term><description>Input validation and output filtering for harmful content</description></item>
    /// <item><term>Adversarial Attacks</term><description>FGSM, PGD, CW, AutoAttack for robustness testing</description></item>
    /// <item><term>Adversarial Defenses</term><description>Adversarial training, input preprocessing, ensemble methods</description></item>
    /// <item><term>Certified Robustness</term><description>Randomized smoothing, IBP, CROWN for provable guarantees</description></item>
    /// <item><term>Content Moderation</term><description>Prompt injection detection, PII filtering for LLMs</description></item>
    /// <item><term>Red Teaming</term><description>Automated adversarial prompt generation for evaluation</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> This is your one-stop configuration for making your model safe and robust.
    /// When called with no parameters (null), industry-standard defaults are applied automatically.
    /// You can use factory methods like <c>AdversarialRobustnessConfiguration.BasicSafety()</c> for common setups,
    /// or customize individual options for your specific needs.</para>
    /// <example>
    /// <code>
    /// // Use industry-standard defaults
    /// builder.ConfigureAdversarialRobustness();
    ///
    /// // Basic safety filtering
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;.BasicSafety());
    ///
    /// // Comprehensive robustness with certified guarantees
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;.Comprehensive());
    ///
    /// // LLM safety with content moderation
    /// builder.ConfigureAdversarialRobustness(AdversarialRobustnessConfiguration&lt;double, string, string&gt;.ForLLM());
    ///
    /// // Custom configuration
    /// builder.ConfigureAdversarialRobustness(new AdversarialRobustnessConfiguration&lt;double, Vector&lt;double&gt;, int&gt;
    /// {
    ///     Enabled = true,
    ///     Options = new AdversarialRobustnessOptions&lt;double&gt;
    ///     {
    ///         EnableSafetyFiltering = true,
    ///         EnableAdversarialTraining = true,
    ///         EnableCertifiedRobustness = true
    ///     },
    ///     UseCertifiedInference = true
    /// });
    /// </code>
    /// </example>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAdversarialRobustness(
        AdversarialRobustnessConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _compliance.ConfigureAdversarialRobustness(configuration);
        _adversarialRobustnessConfiguration = _compliance.AdversarialRobustnessConfiguration;
        return this;
    }

    /// <summary>
    /// Internal accessor exposing the most recently configured
    /// <see cref="AdversarialRobustnessConfiguration{T, TInput, TOutput}"/> so
    /// unit tests can verify <see cref="ConfigureAdversarialRobustness"/>
    /// retained the user-supplied (or default) instance. The configuration
    /// is consumed at build time by <c>AttachAdversarialRobustness</c>;
    /// this accessor lets tests assert the configure-time storage step
    /// without instantiating the full result pipeline.
    /// </summary>
    internal AdversarialRobustnessConfiguration<T, TInput, TOutput>? ConfiguredAdversarialRobustness
        => _adversarialRobustnessConfiguration;

    /// <summary>
    /// Configures fine-tuning for the model using preference learning, RLHF, or other alignment methods.
    /// </summary>
    /// <param name="configuration">The fine-tuning configuration including training data. When null, uses industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This configuration enables post-training fine-tuning using various alignment techniques:
    /// </para>
    /// <list type="bullet">
    /// <item><term>Supervised Fine-Tuning (SFT)</term><description>Traditional fine-tuning on labeled examples</description></item>
    /// <item><term>Direct Preference Optimization (DPO)</term><description>Learn from human preferences without reward models</description></item>
    /// <item><term>Simple Preference Optimization (SimPO)</term><description>Reference-free, length-normalized preference learning</description></item>
    /// <item><term>Group Relative Policy Optimization (GRPO)</term><description>Memory-efficient RL without critic models</description></item>
    /// <item><term>Odds Ratio Preference Optimization (ORPO)</term><description>Combined SFT + preference in one step</description></item>
    /// <item><term>Identity Preference Optimization (IPO)</term><description>Regularized preference optimization</description></item>
    /// <item><term>Kahneman-Tversky Optimization (KTO)</term><description>Utility-maximizing preference learning</description></item>
    /// <item><term>Contrastive Preference Optimization (CPO)</term><description>Contrastive learning for preferences</description></item>
    /// <item><term>Constitutional AI (CAI)</term><description>Self-improvement with constitutional principles</description></item>
    /// <item><term>Reinforcement Learning from Human Feedback (RLHF)</term><description>Classic PPO-based alignment</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Fine-tuning helps align your model with human preferences.
    /// When called with no parameters (null), industry-standard defaults are applied automatically.
    /// Training data should be set in the configuration's TrainingData property.
    /// Use factory methods like <c>FineTuningConfiguration.ForDPO(data)</c> for quick setup.
    /// DPO and SimPO are simpler (no reward model needed), while RLHF and GRPO provide more control.</para>
    /// <example>
    /// <code>
    /// // Use industry-standard defaults (training data set separately)
    /// builder.ConfigureFineTuning();
    ///
    /// // DPO fine-tuning with preference pairs
    /// var preferenceData = new FineTuningData&lt;double, string, string&gt;
    /// {
    ///     Inputs = prompts,
    ///     ChosenOutputs = preferredResponses,
    ///     RejectedOutputs = rejectedResponses
    /// };
    /// builder.ConfigureFineTuning(FineTuningConfiguration&lt;double, string, string&gt;.ForDPO(preferenceData));
    ///
    /// // GRPO for RL-based alignment
    /// var rlData = new FineTuningData&lt;double, string, string&gt;
    /// {
    ///     Inputs = prompts,
    ///     Rewards = rewardScores
    /// };
    /// builder.ConfigureFineTuning(FineTuningConfiguration&lt;double, string, string&gt;.ForGRPO(rlData));
    ///
    /// // Custom fine-tuning configuration
    /// builder.ConfigureFineTuning(new FineTuningConfiguration&lt;double, Vector&lt;double&gt;, int&gt;
    /// {
    ///     Enabled = true,
    ///     Options = new FineTuningOptions&lt;double&gt;
    ///     {
    ///         MethodType = FineTuningMethodType.SimPO,
    ///         LearningRate = 1e-5,
    ///         Epochs = 3,
    ///         SimPOGamma = 1.0
    ///     },
    ///     TrainingData = myPreferenceData
    /// });
    /// </code>
    /// </example>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFineTuning(
        FineTuningConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _fineTuningConfiguration = configuration ?? new FineTuningConfiguration<T, TInput, TOutput>();
        return this;
    }

    /// <summary>
    /// Configures a multi-stage training pipeline for advanced training workflows.
    /// </summary>
    /// <param name="configuration">
    /// The training pipeline configuration defining the stages to execute.
    /// When null, uses the default single-stage training based on other configured settings.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// ConfigureTrainingPipeline enables advanced multi-stage training workflows where each stage
    /// can have its own training method, optimizer, learning rate, and dataset. Stages execute
    /// sequentially, with each stage's output model becoming the next stage's input.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as a recipe with multiple cooking steps.
    /// Just like you might marinate, then sear, then bake - training can have multiple
    /// phases where each phase teaches the model something different.</para>
    /// <para>
    /// <b>Common Training Pipelines:</b>
    /// <list type="bullet">
    /// <item><term>Standard Alignment</term><description>SFT → DPO (most common for chat models)</description></item>
    /// <item><term>Full RLHF</term><description>SFT → Reward Model → PPO</description></item>
    /// <item><term>Constitutional AI</term><description>SFT → CAI critique/revision → preference</description></item>
    /// <item><term>Curriculum Learning</term><description>Easy data → Medium → Hard (progressive difficulty)</description></item>
    /// <item><term>Iterative Refinement</term><description>Multiple DPO rounds with decreasing beta</description></item>
    /// </list>
    /// </para>
    /// <example>
    /// <code>
    /// // Standard alignment pipeline (SFT → DPO)
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.StandardAlignment(sftData, preferenceData));
    ///
    /// // Automatic pipeline based on available data
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.Auto(myData));
    ///
    /// // Custom multi-stage pipeline with builder pattern
    /// var pipeline = new TrainingPipelineConfiguration&lt;double, string, string&gt;()
    ///     .AddSFTStage(stage => {
    ///         stage.TrainingData = sftData;
    ///         stage.Options = new FineTuningOptions&lt;double&gt; { Epochs = 3 };
    ///     })
    ///     .AddPreferenceStage(FineTuningMethodType.DPO, stage => {
    ///         stage.TrainingData = preferenceData;
    ///         stage.Options = new FineTuningOptions&lt;double&gt; { Beta = 0.1 };
    ///     })
    ///     .AddEvaluationStage();
    /// builder.ConfigureTrainingPipeline(pipeline);
    ///
    /// // Iterative refinement with multiple DPO rounds
    /// builder.ConfigureTrainingPipeline(
    ///     TrainingPipelineConfiguration&lt;double, string, string&gt;.IterativeRefinement(3, sftData, preferenceData));
    ///
    /// // Custom stage with user-defined training logic
    /// var customPipeline = new TrainingPipelineConfiguration&lt;double, string, string&gt;()
    ///     .AddSFTStage()
    ///     .AddCustomStage("My Custom Training", async (model, data, ct) => {
    ///         // Custom training logic
    ///         return model;
    ///     });
    /// builder.ConfigureTrainingPipeline(customPipeline);
    /// </code>
    /// </example>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTrainingPipeline(
        TrainingPipelineConfiguration<T, TInput, TOutput>? configuration = null)
    {
        _trainingCore.ConfigureTrainingPipeline(configuration);
        _trainingPipelineConfiguration = _trainingCore.TrainingPipelineConfiguration;
        return this;
    }

    /// <summary>
    /// Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    /// </summary>
    /// <param name="loraConfiguration">The LoRA configuration implementation to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> LoRA enables parameter-efficient fine-tuning by adding small "correction layers"
    /// to your neural network. This lets you adapt large pre-trained models with 100x fewer parameters,
    /// making fine-tuning much faster and more memory-efficient. The configuration determines which layers
    /// get LoRA adaptations and how they behave (rank, scaling, freezing).
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLoRA(ILoRAConfiguration<T> loraConfiguration)
    {
        _loraConfiguration = loraConfiguration;
        return this;
    }

    /// <summary>
    /// Configures the retrieval-augmented generation (RAG) components for use during model inference.
    /// </summary>
    /// <param name="retriever">Optional retriever for finding relevant documents. If not provided, standard RAG won't be available.</param>
    /// <param name="reranker">Optional reranker for improving document ranking quality. If not provided, a default reranker will be used if RAG is configured.</param>
    /// <param name="generator">Optional generator for producing grounded answers. If not provided, a default generator will be used if RAG is configured.</param>
    /// <param name="queryProcessors">Optional query processors for improving search quality.</param>
    /// <param name="graphStore">Optional graph storage backend for Graph RAG (e.g., MemoryGraphStore, FileGraphStore).</param>
    /// <param name="knowledgeGraph">Optional pre-configured knowledge graph. If null but graphStore is provided, a new one is created.</param>
    /// <param name="documentStore">Optional document store for hybrid vector + graph retrieval.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RAG combines retrieval and generation to create answers backed by real documents.
    /// Configure it with:
    /// <list type="bullet">
    /// <item><description>A retriever (finds relevant documents from your collection) - required for standard RAG</description></item>
    /// <item><description>A reranker (improves the ordering of retrieved documents) - optional, defaults provided</description></item>
    /// <item><description>A generator (creates answers based on the documents) - optional, defaults provided</description></item>
    /// <item><description>Optional query processors (improve search queries before retrieval)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Graph RAG:</b> When graphStore or knowledgeGraph is provided, enables knowledge graph-based
    /// retrieval that finds related entities and their relationships, providing richer context than
    /// vector similarity alone. Traditional RAG finds similar documents using vectors. Graph RAG goes further by
    /// also exploring relationships between entities. For example, if you ask about "Paris", it can find
    /// not just documents mentioning Paris, but also related concepts like France, Eiffel Tower, and Seine River.
    /// </para>
    /// <para>
    /// <b>Hybrid Retrieval:</b> When both knowledgeGraph and documentStore are provided, creates a
    /// HybridGraphRetriever that combines vector search and graph traversal for optimal results.
    /// </para>
    /// <para>
    /// <b>Disabling RAG:</b> Call with all parameters as null to disable RAG functionality completely.
    /// </para>
    /// <para>
    /// RAG operations are performed during inference (after model training) via the AiModelResult.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
        IRetriever<T>? retriever = null,
        IReranker<T>? reranker = null,
        IGenerator<T>? generator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        IGraphStore<T>? graphStore = null,
        KnowledgeGraph<T>? knowledgeGraph = null,
        IDocumentStore<T>? documentStore = null)
    {
        // Configure standard RAG components
        _ragRetriever = retriever;
        _ragReranker = reranker;
        _ragGenerator = generator;
        _queryProcessors = queryProcessors;

        // Configure Graph RAG components
        // If all Graph RAG parameters are null, clear Graph RAG fields
        if (graphStore == null && knowledgeGraph == null && documentStore == null)
        {
            _graphStore = null;
            _knowledgeGraph = null;
            _hybridGraphRetriever = null;
            return this;
        }

        _graphStore = graphStore;

        // Use provided knowledge graph or create one from the store
        if (knowledgeGraph != null)
        {
            _knowledgeGraph = knowledgeGraph;
        }
        else if (graphStore != null)
        {
            _knowledgeGraph = new KnowledgeGraph<T>(graphStore);
        }
        else
        {
            // No knowledge graph source provided, clear the field
            _knowledgeGraph = null;
        }

        // Create or clear hybrid retriever based on available components
        if (_knowledgeGraph != null && documentStore != null)
        {
            _hybridGraphRetriever = new HybridGraphRetriever<T>(_knowledgeGraph, documentStore);
        }
        else
        {
            // Clear hybrid retriever if dependencies are missing
            _hybridGraphRetriever = null;
        }

        return this;
    }

    /// <summary>
    /// Configures advanced knowledge graph capabilities including embeddings, community detection,
    /// link prediction, temporal queries, and KG construction.
    /// </summary>
    /// <param name="configure">An action that configures the knowledge graph options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This is separate from <see cref="ConfigureRetrievalAugmentedGeneration"/>, which handles
    /// low-level plumbing (IGraphStore, KnowledgeGraph, HybridGraphRetriever). This method
    /// configures higher-level features built on top of the existing infrastructure.
    /// </para>
    /// <para><b>For Beginners:</b> After setting up your knowledge graph via <c>ConfigureRetrievalAugmentedGeneration()</c>,
    /// use this method to enable advanced features:
    ///
    /// <code>
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureRetrievalAugmentedGeneration(graphStore: new MemoryGraphStore&lt;double&gt;())
    ///     .ConfigureKnowledgeGraph(options =&gt; {
    ///         options.TrainEmbeddings = true;
    ///         options.EmbeddingType = KGEmbeddingType.TransE;
    ///         options.GraphRAGMode = GraphRAGMode.Global;
    ///     })
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureKnowledgeGraph(
        Action<KnowledgeGraphOptions>? configure = null)
    {
        _knowledgeGraphOptions = new KnowledgeGraphOptions();
        configure?.Invoke(_knowledgeGraphOptions);
        return this;
    }

    /// <summary>
    /// Configures the cross-validation strategy for model evaluation.
    /// </summary>
    /// <param name="crossValidator">The cross-validation strategy to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Cross-validation tests how well your model will perform on new data
    /// by training and testing it multiple times on different subsets of your training data.
    /// Use the evaluation methods on AiModelResult to perform cross-validation after building.
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCrossValidation(ICrossValidator<T, TInput, TOutput> crossValidator)
    {
        _crossValidation.ConfigureCrossValidation(crossValidator);
        _crossValidator = _crossValidation.CrossValidator;
        return this;
    }

    /// <summary>
    /// Configures an AutoML model for automatic machine learning optimization.
    /// </summary>
    /// <param name="autoMLModel">The AutoML model instance to use for hyperparameter search and model selection.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML (Automated Machine Learning) automatically searches for the best
    /// model and hyperparameters for your problem. Instead of manually trying different models and settings,
    /// AutoML does this for you.
    /// </para>
    /// <para>
    /// When you configure an AutoML model:
    /// - The Build() method will run the AutoML search process
    /// - AutoML will try different models and hyperparameters
    /// - The best model found will be returned as your trained model
    /// - You can configure search time limits, candidate models, and optimization metrics
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Advanced usage: plug in your own AutoML implementation.
    /// // Most users should prefer the ConfigureAutoML(AutoMLOptions&lt;...&gt;) overload instead.
    /// var autoML = new RandomSearchAutoML&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;();
    /// autoML.SetTimeLimit(TimeSpan.FromMinutes(30));
    /// autoML.SetCandidateModels(new List&lt;ModelType&gt; { ModelType.RandomForest, ModelType.GradientBoosting });
    ///
    /// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAutoML(autoML)
    ///     .Build(trainingData, trainingLabels);
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAutoML(IAutoMLModel<T, TInput, TOutput> autoMLModel)
    {
        _autoMLModel = autoMLModel;
        _autoMLOptions = null;
        return this;
    }

    /// <summary>
    /// Configures AutoML using facade-style options (recommended for most users).
    /// </summary>
    /// <param name="options">AutoML options (budget, strategy, and optional overrides). If null, defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML automatically tries different models/settings to find a strong result.
    /// With this overload you only choose a budget (how much time to spend), and AiDotNet handles the rest.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAutoML(AutoMLOptions<T, TInput, TOutput>? options = null)
    {
        _autoMLOptions = options ?? new AutoMLOptions<T, TInput, TOutput>();

        var (defaultTimeLimit, defaultTrialLimit) = AiDotNet.AutoML.AutoMLBudgetDefaults.Resolve(_autoMLOptions.Budget.Preset);
        var timeLimit = _autoMLOptions.Budget.TimeLimitOverride ?? defaultTimeLimit;
        var trialLimit = _autoMLOptions.Budget.TrialLimitOverride ?? defaultTrialLimit;

        if (_autoMLOptions.TaskFamilyOverride == AutoMLTaskFamily.ReinforcementLearning)
        {
            if (_autoMLOptions.SearchStrategy != AutoMLSearchStrategy.RandomSearch)
            {
                throw new NotSupportedException(
                    $"RL AutoML currently supports only '{AutoMLSearchStrategy.RandomSearch}'. Received '{_autoMLOptions.SearchStrategy}'.");
            }

            // RL AutoML runs through the RL training path and selects an IRLAgent; it does not use the supervised IAutoMLModel pipeline.
            _autoMLModel = null;
            return this;
        }

        if (_autoMLOptions.TaskFamilyOverride is AutoMLTaskFamily taskFamilyOverride
            && !IsBuiltInSupervisedTaskFamilySupported(taskFamilyOverride))
        {
            throw new NotSupportedException(
                $"Facade AutoML options currently support only Regression/Binary/MultiClass/TimeSeriesForecasting/TimeSeriesAnomalyDetection/Ranking/Recommendation task families. " +
                $"Received '{taskFamilyOverride}'. Use {nameof(ConfigureAutoML)}(IAutoMLModel<...>) to plug in a custom implementation.");
        }

        _autoMLModel = CreateBuiltInAutoMLModel(_autoMLOptions.SearchStrategy);
        _autoMLModel.TimeLimit = timeLimit;
        _autoMLModel.TrialLimit = trialLimit;

        if (_autoMLModel is AiDotNet.AutoML.SupervisedAutoMLModelBase<T, TInput, TOutput> supervised)
        {
            supervised.EnsembleOptions = _autoMLOptions.Ensembling ?? ResolveDefaultEnsembling(_autoMLOptions.Budget.Preset);
            supervised.BudgetPreset = _autoMLOptions.Budget.Preset;
        }

        if (_autoMLOptions.OptimizationMetricOverride.HasValue)
        {
            var metric = _autoMLOptions.OptimizationMetricOverride.Value;
            _autoMLModel.SetOptimizationMetric(metric, maximize: IsHigherBetter(metric));
        }
        else if (_autoMLOptions.TaskFamilyOverride is AutoMLTaskFamily familyOverride)
        {
            var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(familyOverride);
            _autoMLModel.SetOptimizationMetric(metric, maximize);
        }

        return this;
    }

    /// <summary>
    /// Configures curriculum learning for training with ordered sample difficulty.
    /// </summary>
    /// <param name="options">Curriculum learning options (schedule type, phases, difficulty estimation).
    /// If null, sensible defaults are used (Linear schedule, 5 phases, loss-based difficulty).</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Curriculum Learning trains models by presenting samples in order of difficulty,
    /// starting with easy examples and gradually introducing harder ones. This often leads to faster
    /// convergence and better final performance compared to random training order.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCurriculumLearning(
        CurriculumLearningOptions<T, TInput, TOutput>? options = null)
    {
        _curriculumLearningOptions = options ?? new CurriculumLearningOptions<T, TInput, TOutput>();
        return this;
    }

    private static AutoMLEnsembleOptions ResolveDefaultEnsembling(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.Standard => new AutoMLEnsembleOptions { Enabled = true, MaxModelCount = 3 },
            AutoMLBudgetPreset.Thorough => new AutoMLEnsembleOptions { Enabled = true, MaxModelCount = 5 },
            _ => new AutoMLEnsembleOptions { Enabled = false, MaxModelCount = 3 }
        };
    }

    private static bool IsHigherBetter(MetricType metric)
    {
        return metric switch
        {
            MetricType.MeanSquaredError => false,
            MetricType.RootMeanSquaredError => false,
            MetricType.MeanAbsoluteError => false,
            MetricType.MSE => false,
            MetricType.RMSE => false,
            MetricType.MAE => false,
            MetricType.MAPE => false,
            MetricType.SMAPE => false,
            MetricType.MeanSquaredLogError => false,
            MetricType.CrossEntropyLoss => false,
            MetricType.AIC => false,
            MetricType.BIC => false,
            MetricType.AICAlt => false,
            MetricType.Perplexity => false,
            _ => true
        };
    }

    private static bool IsBuiltInSupervisedTaskFamilySupported(AutoMLTaskFamily taskFamily)
    {
        return taskFamily == AutoMLTaskFamily.Regression
               || taskFamily == AutoMLTaskFamily.BinaryClassification
               || taskFamily == AutoMLTaskFamily.MultiClassClassification
               || taskFamily == AutoMLTaskFamily.TimeSeriesForecasting
               || taskFamily == AutoMLTaskFamily.TimeSeriesAnomalyDetection
               || taskFamily == AutoMLTaskFamily.Ranking
               || taskFamily == AutoMLTaskFamily.Recommendation;
    }

    private IAutoMLModel<T, TInput, TOutput> CreateBuiltInAutoMLModel(AutoMLSearchStrategy strategy)
    {
        return strategy switch
        {
            AutoMLSearchStrategy.RandomSearch => new AiDotNet.AutoML.RandomSearchAutoML<T, TInput, TOutput>(RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.BayesianOptimization => new AiDotNet.AutoML.BayesianOptimizationAutoML<T, TInput, TOutput>(RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.Evolutionary => new AiDotNet.AutoML.EvolutionaryAutoML<T, TInput, TOutput>(RandomHelper.CreateSecureRandom()),
            AutoMLSearchStrategy.MultiFidelity => new AiDotNet.AutoML.MultiFidelityAutoML<T, TInput, TOutput>(RandomHelper.CreateSecureRandom(), _autoMLOptions?.MultiFidelity),
            AutoMLSearchStrategy.NeuralArchitectureSearch or
            AutoMLSearchStrategy.DARTS or
            AutoMLSearchStrategy.GDAS or
            AutoMLSearchStrategy.OnceForAll => CreateNasAutoMLModel(strategy),
            _ => throw new NotSupportedException(
                $"AutoML search strategy '{strategy}' is not available via the facade options overload. " +
                $"Use {nameof(ConfigureAutoML)}(IAutoMLModel<...>) to plug in a custom implementation.")
        };
    }

    /// <summary>
    /// Creates a NAS-based AutoML model with the specified strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// NAS strategies require TInput and TOutput to be <see cref="Tensor{T}"/> types.
    /// If the types don't match, this method throws a helpful exception.
    /// </para>
    /// <para>
    /// <b>Industry Defaults:</b> When NAS options are not specified, sensible defaults are used:
    /// <list type="bullet">
    /// <item><description>SearchSpace: <see cref="MobileNetSearchSpace{T}"/> (efficient for most use cases)</description></item>
    /// <item><description>NumNodes: 4 (balanced complexity)</description></item>
    /// <item><description>GDAS temperature: 5.0 initial, 0.1 final (proven values from research)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private IAutoMLModel<T, TInput, TOutput> CreateNasAutoMLModel(AutoMLSearchStrategy strategy)
    {
        // NAS models specifically work with Tensor<T> inputs/outputs.
        // Validate type compatibility at runtime.
        if (typeof(TInput) != typeof(Tensor<T>) || typeof(TOutput) != typeof(Tensor<T>))
        {
            throw new NotSupportedException(
                $"Neural Architecture Search strategies ({strategy}) require TInput and TOutput to be Tensor<T>. " +
                $"Current types are TInput={typeof(TInput).Name}, TOutput={typeof(TOutput).Name}. " +
                $"Consider using AiModelBuilder<{typeof(T).Name}, Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>> for NAS.");
        }

        // Resolve NAS options with industry-standard defaults.
        var nasOptions = _autoMLOptions?.NAS;
        var searchSpace = nasOptions?.SearchSpace ?? new MobileNetSearchSpace<T>();
        var numNodes = Math.Max(searchSpace.MaxNodes, 4);

        // Create the appropriate NAS model based on strategy.
        NasAutoMLModelBase<T> nasModel = strategy switch
        {
            AutoMLSearchStrategy.DARTS => new GDAS<T>(
                searchSpace,
                numNodes,
                nasOptions?.ArchitectureLearningRate ?? 5.0,    // Initial temperature (GDAS uses temp, not LR)
                0.1),                                            // Final temperature

            AutoMLSearchStrategy.GDAS => new GDAS<T>(
                searchSpace,
                numNodes,
                nasOptions?.ArchitectureLearningRate ?? 5.0,    // Initial temperature
                0.1),                                            // Final temperature

            AutoMLSearchStrategy.OnceForAll => new OnceForAll<T>(
                searchSpace,
                nasOptions?.ElasticDepths,
                nasOptions?.ElasticWidths,
                nasOptions?.ElasticKernelSizes,
                nasOptions?.ElasticExpansionRatios),

            AutoMLSearchStrategy.NeuralArchitectureSearch => SelectBestNasStrategy(searchSpace, numNodes, nasOptions),

            _ => throw new NotSupportedException($"NAS strategy '{strategy}' is not implemented.")
        };

        // Configure NAS model with time/trial limits from parent options.
        // The SearchAsync will receive proper limits when called.

        // Cast via object to satisfy generic constraints (we validated types above).
        return (IAutoMLModel<T, TInput, TOutput>)(object)nasModel;
    }

    /// <summary>
    /// Auto-selects the best NAS strategy based on task characteristics.
    /// </summary>
    /// <remarks>
    /// <para><b>Selection Heuristics:</b></para>
    /// <list type="bullet">
    /// <item><description>Mobile/Edge platforms: OnceForAll (elastic deployment)</description></item>
    /// <item><description>Quick search (&lt;2 hours): GDAS (fast gradient-based)</description></item>
    /// <item><description>Default: GDAS (proven balance of speed and quality)</description></item>
    /// </list>
    /// </remarks>
    private NasAutoMLModelBase<T> SelectBestNasStrategy(SearchSpaceBase<T> searchSpace, int numNodes, NASOptions<T>? nasOptions)
    {
        // If targeting mobile/edge, use OFA for elastic deployment.
        if (nasOptions?.TargetPlatform is HardwarePlatform.Mobile or HardwarePlatform.EdgeTPU)
        {
            return new OnceForAll<T>(
                searchSpace,
                nasOptions?.ElasticDepths,
                nasOptions?.ElasticWidths,
                nasOptions?.ElasticKernelSizes,
                nasOptions?.ElasticExpansionRatios);
        }

        // Default to GDAS - good balance of speed and architecture quality.
        return new GDAS<T>(
            searchSpace,
            numNodes,
            nasOptions?.ArchitectureLearningRate ?? 5.0,
            0.1);
    }

    /// <summary>
    /// Configures a meta-learning algorithm for training models that can quickly adapt to new tasks.
    /// </summary>
    /// <param name="metaLearner">The meta-learning algorithm to use (e.g., ReptileTrainer).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> If you configure this, Build() will do meta-training instead of regular training.
    /// The meta-learner should be created with all its dependencies (model, loss function, episodic data loader).
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureMetaLearning(IMetaLearner<T, TInput, TOutput> metaLearner)
    {
        _metaLearner = metaLearner;
        return this;
    }

    /// <summary>
    /// Configures distributed training across multiple GPUs or machines.
    /// </summary>
    /// <param name="backend">Communication backend to use. If null, uses InMemoryCommunicationBackend.</param>
    /// <param name="strategy">Distributed training strategy. Default is DDP.</param>
    /// <param name="configuration">Optional sharding configuration for advanced settings like gradient compression, parameter grouping, etc.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When distributed training is configured, the Build() method will automatically wrap
    /// the model and optimizer with their distributed counterparts based on the chosen strategy.
    /// This enables training across multiple GPUs or machines with automatic parameter
    /// sharding and gradient synchronization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This enables distributed training across multiple GPUs or machines.
    /// You can call it with no parameters for sensible defaults, or customize as needed.
    ///
    /// When you configure this, the builder automatically handles all the complexity:
    /// - Your model gets split across GPUs (parameter sharding)
    /// - Gradients are synchronized automatically
    /// - Training is coordinated across all processes
    ///
    /// You just train as normal - the distributed magic happens behind the scenes!
    /// </para>
    /// <para>
    /// For pipeline parallelism, call <see cref="ConfigurePipelineParallelism"/> after this method
    /// to customize scheduling, partitioning, and activation checkpointing.
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDistributedTraining(
        ICommunicationBackend<T>? backend = null,
        DistributedStrategy strategy = DistributedStrategy.DDP,
        IShardingConfiguration<T>? configuration = null)
    {
        _distributedBackend = backend;
        _distributedStrategy = strategy;
        _distributedConfiguration = configuration;
        return this;
    }

    /// <summary>
    /// Configures pipeline-specific options for pipeline parallel training.
    /// </summary>
    /// <param name="schedule">
    /// Pipeline execution schedule. If null, uses GPipeSchedule.
    /// Use <see cref="DistributedTraining.OneForwardOneBackwardSchedule{T}"/> for reduced pipeline bubble (~12-15% vs ~50%).
    /// </param>
    /// <param name="partitionStrategy">
    /// Strategy for partitioning layers across pipeline stages.
    /// If null, uses uniform partitioning. Use <see cref="DistributedTraining.LoadBalancedPartitionStrategy{T}"/>
    /// to balance computational cost across stages.
    /// </param>
    /// <param name="checkpointConfig">
    /// Activation checkpointing configuration.
    /// If null, checkpointing is disabled. Enable to reduce memory from O(L) to O(sqrt(L)).
    /// </param>
    /// <param name="microBatchCount">
    /// Number of micro-batches to split the full batch into for pipeline execution.
    /// Higher values reduce pipeline bubble but increase memory. Default: 1.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Call this after <see cref="ConfigureDistributedTraining"/> with
    /// <c>DistributedStrategy.PipelineParallel</c> to customize pipeline scheduling,
    /// partitioning, activation checkpointing, and micro-batch count.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method fine-tunes how pipeline parallelism works.
    /// You only need to call it if you want to change the defaults (GPipe schedule,
    /// uniform partitioning, no checkpointing, 1 micro-batch).
    /// </para>
    /// <para>
    /// <b>Example:</b>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureDistributedTraining(strategy: DistributedStrategy.PipelineParallel)
    ///     .ConfigurePipelineParallelism(
    ///         schedule: new OneForwardOneBackwardSchedule(),
    ///         partitionStrategy: new LoadBalancedPartitionStrategy&lt;double&gt;(estimatedLayerSize: 1024),
    ///         checkpointConfig: new ActivationCheckpointConfig { Enabled = true },
    ///         microBatchCount: 8)
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePipelineParallelism(
        IPipelineSchedule<T>? schedule = null,
        IPipelinePartitionStrategy<T>? partitionStrategy = null,
        ActivationCheckpointConfig? checkpointConfig = null,
        int microBatchCount = 1)
    {
        if (microBatchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(microBatchCount),
                $"Micro-batch count must be at least 1, but was {microBatchCount}.");
        }

        _pipelineSchedule = schedule;
        _pipelinePartitionStrategy = partitionStrategy;
        _pipelineCheckpointConfig = checkpointConfig;
        _pipelineMicroBatchCount = microBatchCount;
        return this;
    }


    /// <summary>
    /// Configures reinforcement learning options for training an RL agent.
    /// </summary>
    /// <param name="options">The reinforcement learning configuration options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Reinforcement learning trains an agent through trial and error
    /// in an environment. This method configures all aspects of RL training:
    /// - The environment (simulation/game for the agent to learn from)
    /// - Training parameters (episodes, steps, batch size)
    /// - Exploration strategies (how to balance trying new things vs using learned behavior)
    /// - Replay buffers (how to store and sample past experiences)
    /// - Callbacks for monitoring training progress
    ///
    /// After configuring RL options, use BuildAsync(episodes) to train the agent.
    ///
    /// Example:
    /// <code>
    /// var options = new RLTrainingOptions&lt;double&gt;
    /// {
    ///     Environment = new CartPoleEnvironment&lt;double&gt;(),
    ///     Episodes = 1000,
    ///     MaxStepsPerEpisode = 500,
    ///     OnEpisodeComplete = (metrics) =&gt; Console.WriteLine($"Episode {metrics.Episode}: {metrics.TotalReward}")
    /// };
    ///
    /// var result = await new AiModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureReinforcementLearning(options)
    ///     .ConfigureModel(new DQNAgent&lt;double&gt;())
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureReinforcementLearning(RLTrainingOptions<T> options)
    {
        _rlOptions = options;
        return this;
    }


    /// <summary>
    /// Configures knowledge distillation to train a smaller, faster student model from a larger teacher model.
    /// </summary>
    /// <param name="options">The knowledge distillation configuration options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Knowledge distillation is a technique to compress a large, accurate "teacher" model
    /// into a smaller, faster "student" model while preserving most of the teacher's accuracy. Think of it like
    /// an expert (teacher) teaching a student - the student learns not just the answers, but also the reasoning process.</para>
    ///
    /// <para><b>Benefits:</b>
    /// - **Model Compression**: 40-90% size reduction with 90-97% accuracy preserved
    /// - **Faster Inference**: Smaller models run 2-10x faster
    /// - **Edge Deployment**: Deploy on mobile devices, IoT, browsers
    /// - **Cost Reduction**: Lower compute and memory costs</para>
    ///
    /// <para><b>Common Use Cases:</b>
    /// - Deploy BERT/GPT models on mobile devices (DistilBERT is 40% smaller, 60% faster)
    /// - Run vision models on edge devices (MobileNet distilled from ResNet)
    /// - Reduce cloud compute costs for inference
    /// - Multi-teacher ensembles distilled into single student</para>
    ///
    /// <para><b>Quick Start Example:</b>
    /// <code>
    /// // Configure knowledge distillation with default settings (good for most cases)
    /// var distillationOptions = new KnowledgeDistillationOptions&lt;Vector&lt;double&gt;, Vector&lt;double&gt;, double&gt;
    /// {
    ///     TeacherModelType = TeacherModelType.NeuralNetwork,
    ///     StrategyType = DistillationStrategyType.ResponseBased,
    ///     Temperature = 3.0,      // Soften predictions (2-5 typical)
    ///     Alpha = 0.3,            // 30% hard labels, 70% teacher knowledge
    ///     Epochs = 20,
    ///     BatchSize = 32,
    ///     LearningRate = 0.001
    /// };
    ///
    /// var result = await new AiModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(studentModel)
    ///     .ConfigureKnowledgeDistillation(distillationOptions)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    ///
    /// <para><b>Advanced Techniques:</b>
    /// - **Response-Based**: Standard Hinton distillation (recommended start)
    /// - **Feature-Based**: Match intermediate layer representations
    /// - **Attention-Based**: For transformers (BERT, GPT)
    /// - **Relational**: Preserve relationships between samples
    /// - **Self-Distillation**: Model teaches itself for better calibration
    /// - **Ensemble**: Multiple teachers for richer knowledge</para>
    ///
    /// <para><b>Key Parameters:</b>
    /// - **Temperature** (2-5): Higher = softer predictions, more knowledge transfer
    /// - **Alpha** (0.2-0.5): Lower = rely more on teacher, higher = rely more on labels
    /// - **Strategy**: ResponseBased (standard), FeatureBased (deeper), AttentionBased (transformers)
    /// - **Teacher Type**: NeuralNetwork (single), Ensemble (multiple), Self (no separate teacher)</para>
    ///
    /// <para><b>Success Stories:</b>
    /// - DistilBERT: 40% smaller than BERT, 97% performance, 60% faster
    /// - TinyBERT: 7.5x smaller than BERT for mobile deployment
    /// - MobileNet: Distilled from ResNet, 10x fewer parameters
    /// - SqueezeNet: AlexNet-level accuracy at 50x smaller size</para>
    ///
    /// <para><b>References:</b>
    /// - Hinton et al. (2015). Distilling the Knowledge in a Neural Network
    /// - Sanh et al. (2019). DistilBERT
    /// - Park et al. (2019). Relational Knowledge Distillation</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureKnowledgeDistillation(
        KnowledgeDistillationOptions<T, TInput, TOutput>? options = null)
    {
        _knowledgeDistillationOptions = options ?? new KnowledgeDistillationOptions<T, TInput, TOutput>();
        return this;
    }

    // ============================================================================
    // Deployment Configuration Methods
    // ============================================================================

    public IAiModelBuilder<T, TInput, TOutput> ConfigureQuantization(QuantizationConfig? config = null)
    {
        _quantizationConfig = config;
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureCompression(CompressionConfig? config = null)
    {
        _compressionConfig = config ?? new CompressionConfig();
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureCaching(CacheConfig? config = null)
    {
        _storage.ConfigureCaching(config);
        _cacheConfig = _storage.CacheConfig;
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureVersioning(VersioningConfig? config = null)
    {
        _storage.ConfigureVersioning(config);
        _versioningConfig = _storage.VersioningConfig;
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureABTesting(ABTestingConfig? config = null)
    {
        _storage.ConfigureABTesting(config);
        _abTestingConfig = _storage.ABTestingConfig;
        return this;
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureTelemetry(TelemetryConfig? config = null)
    {
        _observability.ConfigureTelemetry(config);
        _telemetryConfig = _observability.TelemetryConfig;
        return this;
    }

    /// <inheritdoc/>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureWeightStreaming(WeightStreamingConfig? config = null)
    {
        // Validate at the boundary: a non-positive ThresholdParameters
        // would silently be ignored by ApplyWeightStreamingConfig (the
        // `custom > 0` guard there). Surface the error here instead so
        // the caller sees the typo / misconfiguration loudly at the
        // ConfigureWeightStreaming call site rather than as a "why isn't
        // streaming engaging?" mystery later. Closes review-comment
        // #1271.s-Ne.
        if (config is not null
            && config.ThresholdParameters is long t
            && t <= 0)
        {
            // ParamName points at the specific property whose value is
            // invalid (ThresholdParameters), not just the outer config
            // wrapper, so IDE error squiggles / debugger paramName lookup
            // highlights the exact field the caller mis-set.
            // Closes review-comment #1271.yAYQ.
            throw new ArgumentOutOfRangeException(
                $"{nameof(config)}.{nameof(WeightStreamingConfig.ThresholdParameters)}",
                t,
                "WeightStreamingConfig.ThresholdParameters must be positive when set " +
                "(it is the parameter-count threshold above which auto-detect engages " +
                "streaming). Pass null on the property to use the env-var / default " +
                "threshold, or pass a positive value (e.g. 1_000_000_000L for 1B " +
                "params) to override per-instance.");
        }
        // Stash for application during BuildAsync. Stored as the typed
        // nullable config so a subsequent call to this method without a
        // config (i.e. user changed their mind) clears prior intent and
        // returns to default auto-detect.
        _weightStreamingConfig = config;
        return this;
    }

    /// <summary>
    /// User-supplied weight-streaming overrides. Null means
    /// "use the auto-detect default": models above the parameter
    /// threshold (10B by default) auto-engage streaming; models below
    /// stay eager. Set via <see cref="ConfigureWeightStreaming"/> and
    /// applied during model construction in <see cref="BuildAsync"/>.
    /// </summary>
    private WeightStreamingConfig? _weightStreamingConfig;

    /// <summary>
    /// Constructs a <see cref="WeightStreamingReport"/> from the model's
    /// streaming state if streaming was engaged (auto-detect or explicit
    /// opt-in), otherwise returns null. Called during result construction
    /// in BuildAsync. The report's counters come from
    /// <c>WeightRegistry.GetStreamingReport</c>; the AiDotNet-side wrapper
    /// adds the auto-detect / threshold context that the Tensors-side
    /// pool report doesn't know about.
    /// </summary>
    /// <remarks>
    /// Returns null when:
    /// <list type="bullet">
    /// <item>The model isn't a <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>
    /// (classical models / non-NN regressors don't stream).</item>
    /// <item>The model never engaged streaming (small enough to fit
    /// resident, or explicitly opted out via Enabled=false).</item>
    /// </list>
    /// The Tensors-side pool report is fetched via reflection-free direct
    /// call; we wrap it to insulate AiDotNet callers from Tensors-side
    /// schema rewrites of the underlying pool report struct.
    /// </remarks>
    private AiDotNet.Deployment.Configuration.WeightStreamingReport? BuildWeightStreamingReport()
    {
        if (_model is not NeuralNetworks.NeuralNetworkBase<T> nnBase) return null;
        // Streaming actually engaged? IsWeightStreamingActive is true iff
        // ConfigureWeightLifetime ran on this instance — covers both
        // "auto-detect engaged" and "user forced via
        // ConfigureWeightStreaming(Enabled:true)" branches with one read.
        if (!nnBase.IsWeightStreamingActive) return null;

        // ParameterCount returns long since #1244 + this PR's #1271 migration —
        // no longer throws on >int.MaxValue (was the source of misleading
        // ModelParameterCount=0 reports for >2.1B-param models flagged by
        // review-comment #1271.tgVo). The defensive catch survives only to
        // cover truly-broken model state (a subclass override raising on a
        // partially-constructed instance during reporting); under normal
        // construction, paramCount carries the real long value.
        long paramCount = 0;
        try { paramCount = nnBase.ParameterCount; }
        catch { /* unreachable on the supported path; defensive only. */ }

        // Pull live counters from the Tensors-side streaming pool. Any
        // exception here (Tensors-side schema mismatch, transient
        // pool-state inconsistency) collapses to a "streaming engaged
        // but counters unavailable" report — the StreamingEnabled +
        // AutoDetected fields still tell the caller streaming is on,
        // and the per-counter fields stay 0 rather than failing the
        // entire build at reporting time.
        long diskReads = 0, evictions = 0, prefetchHit = 0, prefetchMiss = 0, prefetchIssue = 0, residentBytes = 0;
        double compressionRatio = 1.0;
        string? countersUnavailableReason = null;
        try
        {
            var pool = WeightRegistry.GetStreamingReport();
            diskReads = pool.DiskReadCount;
            evictions = pool.EvictionCount;
            prefetchHit = pool.PrefetchHitCount;
            prefetchMiss = pool.PrefetchMissCount;
            prefetchIssue = pool.PrefetchIssueCount;
            residentBytes = pool.ResidentBytes;
            compressionRatio = pool.CompressionRatio;
        }
        catch (Exception ex)
        {
            // Capture the reason so dashboards can distinguish "no streaming
            // activity" (counters legitimately zero) from "failed to read
            // counters" (an actual runtime mismatch / pool error). Counters
            // stay at their defaults so the StreamingEnabled + AutoDetected
            // fields still tell the caller streaming is on.
            countersUnavailableReason = $"{ex.GetType().Name}: {ex.Message}";
        }

        // Effective threshold: per-instance override beats env-var beats
        // compiled default. Match the precedence used by the auto-detect
        // path so the report reflects the value that ACTUALLY drove the
        // decision.
        long threshold = _weightStreamingConfig?.ThresholdParameters
                         ?? NeuralNetworks.NeuralNetworkBase<T>.DefaultStreamingThresholdParamsForReport;

        return new AiDotNet.Deployment.Configuration.WeightStreamingReport
        {
            StreamingEnabled = true,
            AutoDetected = nnBase.WeightStreamingAutoDetected,
            ModelParameterCount = paramCount,
            EffectiveThresholdParameters = threshold,
            DiskReadCount = diskReads,
            EvictionCount = evictions,
            PrefetchIssueCount = prefetchIssue,
            PrefetchHitCount = prefetchHit,
            PrefetchMissCount = prefetchMiss,
            ResidentBytes = residentBytes,
            CompressionRatio = compressionRatio,
            CountersUnavailableReason = countersUnavailableReason,
        };
    }

    /// <summary>
    /// Applies <see cref="_weightStreamingConfig"/> to the constructed
    /// neural-network model. Called from BuildAsync immediately after
    /// the model is set up so the config takes effect before any
    /// Predict / Train call. Cases:
    /// <list type="bullet">
    /// <item><c>_weightStreamingConfig == null</c> — no-op; the model's
    /// own ctor-time auto-detect (with the env-var or 10B default
    /// threshold) decides whether to stream.</item>
    /// <item><c>config.ThresholdParameters</c> set — applied via
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.ApplyAutoDetectThresholdOverride"/>
    /// regardless of <c>Enabled</c>'s value, so a custom threshold drives
    /// the auto-detect comparison on the upcoming first-forward retry.</item>
    /// <item><c>config.Enabled == false</c> — calls
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.DisableAutoStreaming"/>
    /// so the next Predict's lazy-retry path doesn't re-engage. If the
    /// ctor's eager path already engaged (because we're at extreme
    /// scale and the model is trained from scratch in the builder),
    /// this is a documented edge case — we don't tear down a process-
    /// wide WeightRegistry config that other models may share.</item>
    /// <item><c>config.Enabled == true</c> — calls
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.ConfigureWeightLifetime"/>
    /// directly with default <c>GpuOffloadOptions</c>, forcing streaming
    /// on regardless of size. Useful for integration tests that need
    /// predictable streaming behavior on small models.</item>
    /// </list>
    /// </summary>
    private void ApplyWeightStreamingConfig()
    {
        if (_model is not NeuralNetworks.NeuralNetworkBase<T> nnBase) return;
        ApplyWeightStreamingConfigTo(nnBase);
    }

    /// <summary>
    /// Applies the builder's <see cref="_weightStreamingConfig"/> to a
    /// specific neural-network instance. Used both for the build's
    /// final <c>_model</c> and for AutoML candidate models (the latter
    /// via the <see cref="AutoML.AutoMLModelBase{T,TInput,TOutput}.OnCandidateCreated"/>
    /// hook so candidates respect the user's threshold / force-on /
    /// force-off intent during the search itself, not just the winner).
    /// </summary>
    /// <remarks>
    /// Idempotent: no-op when <see cref="_weightStreamingConfig"/> is
    /// null. Safe to call multiple times against the same instance —
    /// the post-search apply against the final model and the
    /// per-candidate apply during the search both flow through here.
    /// </remarks>
    private void ApplyWeightStreamingConfigTo(NeuralNetworks.NeuralNetworkBase<T> nnBase)
    {
        if (_weightStreamingConfig is null) return;
        if (nnBase is null) return;

        // Per-instance threshold override: applied BEFORE we route to
        // any of the explicit-on / explicit-off / null branches because
        // (a) the auto-detect retry on first forward needs to see the
        // user's threshold, not the env-var/default, and (b) it's a
        // pure data-flow operation with no side effect when Enabled is
        // explicit (the threshold is only consulted on the auto-detect
        // path).
        if (_weightStreamingConfig.ThresholdParameters is long custom && custom > 0)
        {
            nnBase.ApplyAutoDetectThresholdOverride(custom);
        }

        if (_weightStreamingConfig.Enabled == false)
        {
            nnBase.DisableAutoStreaming();
            return;
        }

        if (_weightStreamingConfig.Enabled == true)
        {
            // Force streaming on. Uses parameterless GpuOffloadOptions
            // so any future Tensors-side default updates flow through
            // without freezing the AiDotNet-side config at today's
            // values.
            nnBase.ConfigureWeightLifetime(new GpuOffloadOptions());
            return;
        }

        // Enabled == null: leave the auto-detect machinery to do its
        // thing on the upcoming first forward. The threshold override
        // (if set) was already applied above and will drive the
        // post-first-forward retry.
    }

    /// <summary>
    /// Controls whether GPU backend diagnostic output is written to
    /// <see cref="System.Console"/> or routed through a custom sink.
    /// Addresses all three controls from github.com/ooples/AiDotNet#1122.
    /// </summary>
    /// <remarks>
    /// Forwards to <see cref="AiDotNet.Configuration.GpuDiagnosticsConfig"/>
    /// which in turn forwards to the underlying Tensors-package flag. The
    /// settings are process-global — setting here affects every AiDotNet
    /// call in the process. <see cref="AiDotNet.Configuration.GpuDiagnosticsOptions.Level"/>
    /// takes precedence over <see cref="AiDotNet.Configuration.GpuDiagnosticsOptions.Verbose"/>
    /// when both are set.
    /// </remarks>
    /// <param name="options">
    /// The GPU-diagnostics options, or <c>null</c> to leave all current
    /// settings unchanged. Each options property is nullable and only
    /// applied when non-null (preserve semantics).
    /// </param>
    /// <returns>The builder instance for method chaining.</returns>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureGpuDiagnostics(
        AiDotNet.Configuration.GpuDiagnosticsOptions? options = null)
    {
        _observability.ConfigureGpuDiagnostics(options);
        return this;
    }

    /// <summary>
    /// Configures benchmarking to run standardized benchmark suites and attach a structured report to the built model.
    /// </summary>
    /// <param name="options">Benchmarking options (suites, sampling, failure policy). If null, sensible defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This integrates benchmarking into the facade flow: users select suites via enums and receive a structured report,
    /// without wiring benchmark implementations manually.
    /// </para>
    /// <para><b>For Beginners:</b> This is like running a standardized test after building your model to see how it performs.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureBenchmarking(BenchmarkingOptions? options = null)
    {
        _observability.ConfigureBenchmarking(options);
        _benchmarkingOptions = _observability.BenchmarkingOptions;
        return this;
    }

    /// <summary>
    /// Configures performance profiling for training and inference operations.
    /// </summary>
    /// <param name="config">The profiling configuration, or null to use industry-standard defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Profiling measures how long different parts of your ML code take to run.
    /// Think of it like a stopwatch for your code - it helps you find bottlenecks and optimize performance.
    ///
    /// The profiling report will be available in the result after training:
    /// <code>
    /// var result = await builder
    ///     .ConfigureProfiling() // Enable with defaults
    ///     .Build(features, labels);
    ///
    /// // Access the profiling report
    /// var report = result.ProfilingReport;
    /// // Result is available in the returned value
    /// </code>
    ///
    /// Features tracked:
    /// - Operation timing: How long each training step, forward pass, backward pass takes
    /// - Memory allocations: How much memory is used during training
    /// - Call hierarchy: Which operations call which other operations
    /// - Percentiles: P50 (median), P95, P99 timing for statistical analysis
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProfiling(ProfilingConfig? config = null)
    {
        _observability.ConfigureProfiling(config);
        _profilingConfig = _observability.ProfilingConfig;
        return this;
    }

    /// <summary>
    /// Configures the comprehensive safety pipeline for input validation and output filtering.
    /// </summary>
    /// <param name="configure">
    /// Action to configure safety settings. If null, safety is enabled with default settings
    /// (text toxicity, PII detection, and jailbreak detection are all enabled).
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The safety pipeline provides modular, composable content safety checks across text,
    /// image, audio, and video modalities. All settings use nullable types with industry-standard
    /// defaults — if you don't configure something, a sensible default is used automatically.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is your one-stop safety control panel. Enable the checks
    /// you need and the pipeline handles the rest:
    /// <code>
    /// builder.ConfigureSafety(safety =&gt;
    /// {
    ///     safety.Text.ToxicityDetection = true;
    ///     safety.Text.PIIDetection = true;
    ///     safety.Image.NSFWDetection = true;
    ///     safety.Guardrails.InputGuardrails = true;
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSafety(Action<AiDotNet.Safety.SafetyConfig>? configure = null)
    {
        _compliance.ConfigureSafety(configure);
        _safetyPipelineConfig = _compliance.SafetyPipelineConfig;
        return this;
    }

    /// <summary>
    /// Creates a ProfilerSession if profiling is enabled; otherwise returns null.
    /// </summary>
    private ProfilerSession? CreateProfilerSession()
    {
        if (_profilingConfig?.Enabled != true)
        {
            return null;
        }

        return new ProfilerSession(_profilingConfig);
    }

    public IAiModelBuilder<T, TInput, TOutput> ConfigureExport(ExportConfig? config = null)
    {
        _exportConfig = config;
        return this;
    }

    /// <summary>
    /// Configures experiment tracking for logging and organizing training runs.
    /// </summary>
    /// <param name="tracker">The experiment tracker implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An experiment tracker is like a lab notebook for your ML experiments.
    /// It logs parameters, metrics, and artifacts so you can compare runs and reproduce results.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureExperimentTracker(IExperimentTracker<T> tracker)
    {
        _storage.ConfigureExperimentTracker(tracker);
        _experimentTracker = _storage.ExperimentTracker;
        return this;
    }

    /// <summary>
    /// Configures checkpoint management for saving and restoring training state.
    /// </summary>
    /// <param name="manager">The checkpoint manager implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Checkpoints are like save points in a video game.
    /// They let you pause training and resume later, or go back to an earlier state if something goes wrong.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCheckpointManager(ICheckpointManager<T, TInput, TOutput> manager)
    {
        _trainingCore.ConfigureCheckpointManager(manager);
        _checkpointManager = _trainingCore.CheckpointManager;
        return this;
    }

    /// <summary>
    /// Configures memory management for training including gradient checkpointing,
    /// activation pooling, and model sharding.
    /// </summary>
    /// <param name="configuration">The memory configuration to use. If null, uses default settings.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training large neural networks requires a lot of memory.
    /// Memory management helps you train bigger models by:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Gradient Checkpointing:</b> Trades compute for memory by recomputing
    /// activations during backpropagation instead of storing them all.</description></item>
    /// <item><description><b>Activation Pooling:</b> Reuses memory buffers to reduce garbage collection.</description></item>
    /// <item><description><b>Model Sharding:</b> Splits large models across multiple GPUs.</description></item>
    /// </list>
    /// <para>
    /// <b>Available Presets:</b>
    /// <list type="bullet">
    /// <item><description><c>TrainingMemoryConfig.MemoryEfficient()</c> - Maximum memory savings</description></item>
    /// <item><description><c>TrainingMemoryConfig.SpeedOptimized()</c> - Maximum speed</description></item>
    /// <item><description><c>TrainingMemoryConfig.MultiGpu(n)</c> - Multi-GPU training</description></item>
    /// <item><description><c>TrainingMemoryConfig.ForTransformers()</c> - Optimized for transformers</description></item>
    /// <item><description><c>TrainingMemoryConfig.ForConvNets()</c> - Optimized for CNNs</description></item>
    /// </list>
    /// </para>
    /// <example>
    /// <code>
    /// // Using a preset configuration
    /// builder.ConfigureMemoryManagement(TrainingMemoryConfig.MemoryEfficient());
    ///
    /// // Using a custom configuration
    /// builder.ConfigureMemoryManagement(new TrainingMemoryConfig
    /// {
    ///     UseGradientCheckpointing = true,
    ///     CheckpointEveryNLayers = 2,
    ///     UseActivationPooling = true,
    ///     MaxPoolMemoryMB = 2048
    /// });
    ///
    /// // Multi-GPU training
    /// builder.ConfigureMemoryManagement(TrainingMemoryConfig.MultiGpu(4));
    /// </code>
    /// </example>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureMemoryManagement(
        Training.Memory.TrainingMemoryConfig? configuration = null)
    {
        _trainingCore.ConfigureMemoryManagement(configuration);
        _memoryConfig = _trainingCore.MemoryConfig;
        return this;
    }

    /// <summary>
    /// Configures training monitoring for real-time visibility into training progress.
    /// </summary>
    /// <param name="monitor">The training monitor implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A training monitor is like a dashboard for your model training.
    /// It shows you how training is progressing, what resources are being used, and if there are any problems.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTrainingMonitor(ITrainingMonitor<T> monitor)
    {
        _trainingCore.ConfigureTrainingMonitor(monitor);
        _trainingMonitor = _trainingCore.TrainingMonitor;
        return this;
    }

    /// <summary>
    /// Configures model registry for centralized model storage and versioning.
    /// </summary>
    /// <param name="registry">The model registry implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A model registry is like a library for your trained models.
    /// It keeps track of all your models, their versions, and which ones are in production.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModelRegistry(IModelRegistry<T, TInput, TOutput> registry)
    {
        _storage.ConfigureModelRegistry(registry);
        _modelRegistry = _storage.ModelRegistry;
        return this;
    }

    /// <summary>
    /// Configures data version control for tracking dataset changes.
    /// </summary>
    /// <param name="dataVersionControl">The data version control implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Data version control is like Git, but for your datasets.
    /// It tracks what data was used for training each model and lets you reproduce experiments.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDataVersionControl(IDataVersionControl<T> dataVersionControl)
    {
        _storage.ConfigureDataVersionControl(dataVersionControl);
        _dataVersionControl = _storage.DataVersionControl;
        return this;
    }

    /// <summary>
    /// Configures hyperparameter optimization for automatic tuning of model settings.
    /// </summary>
    /// <param name="optimizer">The hyperparameter optimizer implementation to use.</param>
    /// <param name="searchSpace">The hyperparameter search space defining parameter ranges. If null, hyperparameter optimization is disabled.</param>
    /// <param name="nTrials">Number of trials to run. Default is 10.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hyperparameter optimization automatically finds the best settings
    /// for your model (like learning rate, number of layers, etc.) instead of you having to guess.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureHyperparameterOptimizer(
        IHyperparameterOptimizer<T, TInput, TOutput> optimizer,
        HyperparameterSearchSpace? searchSpace = null,
        int nTrials = 10)
    {
        _hyperparameterOptimizer = optimizer;
        _hyperparameterSearchSpace = searchSpace;
        _hyperparameterTrials = nTrials;
        return this;
    }

    /// <summary>
    /// Configures data augmentation for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Data augmentation creates variations of training data on-the-fly to help models
    /// generalize better. This configuration covers both training-time augmentation
    /// and Test-Time Augmentation (TTA) for improved inference accuracy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Augmentation is like showing the model many variations of
    /// the same data. For images, this might include rotations, flips, and color changes.
    /// The model learns to recognize objects regardless of these variations.
    /// </para>
    /// <para><b>Key features:</b>
    /// <list type="bullet">
    /// <item>Automatic data-type detection (image, tabular, audio, text, video)</item>
    /// <item>Industry-standard defaults that work well out-of-the-box</item>
    /// <item>Test-Time Augmentation (TTA) enabled by default for better predictions</item>
    /// </list>
    /// </para>
    /// <para>
    /// Example - Simple usage with defaults:
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureAugmentation()  // Uses auto-detected defaults
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// <para>
    /// Example - Custom configuration:
    /// <code>
    /// // Strongly-typed (recommended — review #1368): use the generic
    /// // AugmentationConfig&lt;T, TInput&gt; subclass for compile-time-checked
    /// // augmenter type matching the builder's generics.
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureAugmentation(new AugmentationConfig&lt;float, Tensor&lt;float&gt;&gt;
    ///     {
    ///         Augmenter = new MyTensorAugmenter(), // IntelliSense + compile check
    ///         EnableTTA = true,
    ///         TTANumAugmentations = 8,
    ///         ImageSettings = new ImageAugmentationSettings
    ///         {
    ///             EnableFlips = true,
    ///             EnableRotation = true,
    ///             RotationRange = 20.0
    ///         }
    ///     })
    ///     .Build(images, labels);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="config">
    /// Augmentation configuration. Prefer the strongly-typed
    /// <see cref="AugmentationConfig{T, TInput}"/> subclass (which inherits
    /// from this base type so it slots into this method) for compile-time
    /// validation of the custom augmenter. If null, uses industry-standard
    /// defaults with automatic data-type detection.
    /// </param>
    /// <returns>The builder instance for method chaining.</returns>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAugmentation(
        Augmentation.AugmentationConfig? config = null)
    {
        _dataPipeline.ConfigureAugmentation(config);
        _augmentationConfig = _dataPipeline.AugmentationConfig;
        return this;
    }

    /// <summary>
    /// Strongly-typed overload of <see cref="ConfigureAugmentation(Augmentation.AugmentationConfig?)"/>
    /// that accepts the generic <see cref="AugmentationConfig{T, TInput}"/>
    /// (introduced in review #1368 to replace the <c>object?</c>-typed
    /// custom-augmenter slot). Provides IDE-discoverable
    /// <see cref="AugmentationConfig{T, TInput}.Augmenter"/> property type
    /// so users get IntelliSense and compile-time checks without having
    /// to drill into the non-generic base. Delegates to the base overload
    /// after the typed slot is captured.
    /// </summary>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAugmentation(
        Augmentation.AugmentationConfig<T, TInput>? config)
        => ConfigureAugmentation((Augmentation.AugmentationConfig?)config);

    /// <summary>
    /// Creates a default augmentation configuration with auto-detected modality settings.
    /// </summary>
    private Augmentation.AugmentationConfig CreateDefaultAugmentationConfig()
    {
        var config = new Augmentation.AugmentationConfig();

        // Auto-detect data type from TInput and apply appropriate defaults
        var dataType = Augmentation.DataModalityDetector.Detect<TInput>();

        switch (dataType)
        {
            case Augmentation.DataModality.Image:
                config.ImageSettings = new Augmentation.ImageAugmentationSettings();
                break;
            case Augmentation.DataModality.Tabular:
                config.TabularSettings = new Augmentation.TabularAugmentationSettings();
                break;
            case Augmentation.DataModality.Audio:
                config.AudioSettings = new Augmentation.AudioAugmentationSettings();
                break;
            case Augmentation.DataModality.Text:
                config.TextSettings = new Augmentation.TextAugmentationSettings();
                break;
            case Augmentation.DataModality.Video:
                config.VideoSettings = new Augmentation.VideoAugmentationSettings();
                break;
            default:
                // Unknown type - use generic settings, user can configure manually
                break;
        }

        return config;
    }

    /// <summary>
    /// Configures self-supervised learning for unsupervised representation learning.
    /// </summary>
    /// <param name="configure">Optional action to configure SSL settings.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Self-supervised learning (SSL) allows training powerful representations from unlabeled data.
    /// The learned representations can then be fine-tuned on smaller labeled datasets, often
    /// achieving better results than training from scratch.
    /// </para>
    /// <para><b>For Beginners:</b> SSL is like teaching a model to understand patterns in data
    /// without needing human labels. Think of it as the model learning to "see" or "understand"
    /// images/text before being taught specific tasks. This makes it much better at learning
    /// new tasks with less labeled data.</para>
    ///
    /// <para><b>Supported Methods:</b></para>
    /// <list type="bullet">
    /// <item><b>SimCLR:</b> Contrastive learning with in-batch negatives (large batch sizes)</item>
    /// <item><b>MoCo/MoCoV2/MoCoV3:</b> Momentum contrastive with memory queue (efficient)</item>
    /// <item><b>BYOL:</b> No negatives required, uses momentum teacher</item>
    /// <item><b>SimSiam:</b> Simple Siamese networks with stop-gradient</item>
    /// <item><b>BarlowTwins:</b> Decorrelation-based, no negatives needed</item>
    /// <item><b>DINO:</b> Self-distillation for Vision Transformers</item>
    /// <item><b>MAE:</b> Masked autoencoding for ViT pretraining</item>
    /// </list>
    ///
    /// <para><b>Example - Basic SSL pretraining:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning()  // Uses SimCLR by default
    ///     .Build(unlabeledImages);
    /// </code>
    ///
    /// <para><b>Example - Custom SSL configuration:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning(ssl =>
    ///     {
    ///         ssl.Method = SSLMethodType.MoCoV3;
    ///         ssl.PretrainingEpochs = 300;
    ///         ssl.Temperature = 0.2;
    ///         ssl.ProjectorOutputDimension = 256;
    ///         ssl.MoCo = new MoCoConfig { Momentum = 0.99 };
    ///     })
    ///     .Build(unlabeledImages);
    /// </code>
    ///
    /// <para><b>Example - BYOL without negative samples:</b></para>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(encoder)
    ///     .ConfigureSelfSupervisedLearning(ssl =>
    ///     {
    ///         ssl.Method = SSLMethodType.BYOL;
    ///         ssl.BYOL = new BYOLConfig { Momentum = 0.996 };
    ///     })
    ///     .Build(unlabeledImages);
    /// </code>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSelfSupervisedLearning(
        Action<SelfSupervisedLearning.SSLConfig>? configure = null)
    {
        _sslConfig = new SelfSupervisedLearning.SSLConfig();
        configure?.Invoke(_sslConfig);
        return this;
    }

    /// <summary>
    /// Configures self-supervised learning with a typed pretraining hook
    /// (<see cref="AiDotNet"/>#1361).
    /// </summary>
    /// <param name="configure">Optional <see cref="SelfSupervisedLearning.SSLConfig"/>
    /// configurator. When null, a default <c>SSLConfig</c> is used.</param>
    /// <param name="pretrainAction">User-supplied pretraining hook invoked BEFORE
    /// main training. Receives the current base model + SSLConfig + cancellation
    /// token; returns the model that should feed into main training (typically the
    /// same model with its encoder updated via <see cref="SelfSupervisedLearning
    /// .ISSLMethod{T}"/>'s TrainStep loop). The configured-but-no-action pattern
    /// preserves backwards compatibility — SSL settings are stored on the result
    /// without forcing any pretraining stage to run.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// The two-argument overload is the wire-up entry point — the single-argument
    /// overload above stores SSLConfig but does NOT run a pretraining stage (the
    /// SSL subsystem requires an encoder-shaped <c>INeuralNetwork&lt;T&gt;</c> which
    /// is not interchangeable with arbitrary <c>IFullModel&lt;T, TInput, TOutput&gt;
    /// </c>; the user-supplied action is where the conversion happens).
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSelfSupervisedLearning(
        Action<SelfSupervisedLearning.SSLConfig>? configure,
        Func<IFullModel<T, TInput, TOutput>, SelfSupervisedLearning.SSLConfig, CancellationToken,
            Task<IFullModel<T, TInput, TOutput>>> pretrainAction)
    {
        if (pretrainAction is null) throw new ArgumentNullException(nameof(pretrainAction));
        _sslConfig = new SelfSupervisedLearning.SSLConfig();
        configure?.Invoke(_sslConfig);
        _sslPretrainAction = pretrainAction;
        return this;
    }

    /// <summary>
    /// Configures program synthesis (code generation / repair) settings with sensible defaults.
    /// </summary>
    /// <param name="options">Optional configuration options. If null, safe industry-standard defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Program synthesis focuses on code-oriented tasks such as generation, completion, and repair.
    /// This method wires up the default program-synthesis components and chooses safe default values when
    /// options are not provided (for example, a safe maximum sequence length and vocabulary size).
    /// </para>
    /// <para>
    /// Tokenizer selection:
    /// - If <paramref name="options"/> provides a tokenizer, it is used.
    /// - Otherwise, if a tokenizer was configured earlier via <see cref="ConfigureTokenizer"/>, that tokenizer is reused.
    /// - If no tokenizer is available, a code-aware tokenizer is created automatically based on the target language.
    /// </para>
    /// <para>
    /// Model selection:
    /// - The builder creates a program-synthesis model based on the configured model kind (for example CodeBERT / GraphCodeBERT / CodeT5).
    /// - If the created model is compatible with this builder’s <typeparamref name="TInput"/> and <typeparamref name="TOutput"/>, the model is applied.
    ///   If not compatible, the tokenizer/options are still configured, but the existing model is left unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you want a ready-to-use setup for code tasks and you do not want to
    /// manually choose every low-level component (tokenizer, defaults, and model configuration).
    ///
    /// Simple usage (defaults):
    /// <code>
    /// var result = await new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis()
    ///     .BuildAsync();
    /// </code>
    ///
    /// Custom usage:
    /// <code>
    /// var result = await new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis(new ProgramSynthesisOptions
    ///     {
    ///         TargetLanguage = ProgramLanguage.CSharp,
    ///         ModelKind = ProgramSynthesisModelKind.CodeT5,
    ///         MaxSequenceLength = 1024
    ///     })
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProgramSynthesis(
        AiDotNet.ProgramSynthesis.Options.ProgramSynthesisOptions? options = null)
    {
        options ??= new AiDotNet.ProgramSynthesis.Options.ProgramSynthesisOptions();

        // Defaults-first: clamp invalid user inputs to safe industry-standard values.
        var maxSequenceLength = options.MaxSequenceLength > 0 ? options.MaxSequenceLength : 512;
        var vocabularySize = options.VocabularySize > 0 ? options.VocabularySize : 50000;
        var numEncoderLayers = Math.Max(0, options.NumEncoderLayers);
        var numDecoderLayersConfigured = Math.Max(0, options.NumDecoderLayers);

        var tokenizer = options.Tokenizer ?? _tokenizer;

        if (tokenizer is null)
        {
            var baseTokenizer = AiDotNet.Tokenization.Algorithms.CharacterTokenizer.CreateAscii(
                AiDotNet.Tokenization.Models.SpecialTokens.Bert(),
                lowercase: false);

            var codeLanguage = options.TargetLanguage switch
            {
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.CSharp => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.CSharp,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Python => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Python,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Java => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Java,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.JavaScript => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.JavaScript,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.TypeScript => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.TypeScript,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.C => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.C,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.CPlusPlus => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Cpp,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Go => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Go,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Rust => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Rust,
                AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.SQL => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.SQL,
                _ => AiDotNet.Tokenization.CodeTokenization.ProgrammingLanguage.Generic
            };

            tokenizer = new AiDotNet.Tokenization.CodeTokenization.CodeTokenizer(
                baseTokenizer,
                codeLanguage,
                splitIdentifiers: true);
        }

        _tokenizer = tokenizer;
        _tokenizationConfig ??= new TokenizationConfig();

        bool useDataFlow = options.ModelKind == AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.GraphCodeBERT;

        int numDecoderLayers = options.ModelKind == AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.CodeT5
            ? Math.Max(1, numDecoderLayersConfigured)
            : 0;

        var architecture = new AiDotNet.ProgramSynthesis.Models.CodeSynthesisArchitecture<T>(
            synthesisType: options.SynthesisType,
            targetLanguage: options.TargetLanguage,
            codeTaskType: options.DefaultTask,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: numDecoderLayers,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize,
            useDataFlow: useDataFlow);

        AiDotNet.ProgramSynthesis.Interfaces.ICodeModel<T> codeModel = options.ModelKind switch
        {
            AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.CodeBERT =>
                new AiDotNet.ProgramSynthesis.Engines.CodeBERT<T>(architecture, tokenizer: tokenizer),
            AiDotNet.ProgramSynthesis.Options.ProgramSynthesisModelKind.GraphCodeBERT =>
                new AiDotNet.ProgramSynthesis.Engines.GraphCodeBERT<T>(architecture, tokenizer: tokenizer),
            _ =>
                new AiDotNet.ProgramSynthesis.Engines.CodeT5<T>(architecture, tokenizer: tokenizer)
        };

        // Store the program-synthesis model separately so it is available regardless of the primary model's TInput/TOutput types.
        _programSynthesisModel = codeModel;

        // If compatible, also apply as the primary model (supports code-only workflows).
        if (codeModel is IFullModel<T, TInput, TOutput> fullModel)
        {
            _model = fullModel;
        }

        return this;
    }
}
