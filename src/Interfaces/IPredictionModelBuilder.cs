using AiDotNet.Configuration;
using AiDotNet.Deployment.Configuration;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.MixedPrecision;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Models.Options;
using AiDotNet.PromptEngineering.FewShot;
using AiDotNet.Reasoning.Models;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a builder pattern interface for creating and configuring predictive models.
/// </summary>
/// <remarks>
/// This interface provides a fluent API for setting up all components of a machine learning model.
/// 
/// <b>For Beginners:</b> Think of this as a step-by-step recipe builder for creating AI models.
/// Just like building a custom sandwich where you choose the bread, fillings, and condiments,
/// this builder lets you choose different components for your AI model.
/// 
/// The builder pattern makes it easy to:
/// - Configure your model piece by piece
/// - Change only the parts you want while keeping default settings for the rest
/// - Create different variations of models without writing repetitive code
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IPredictionModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Configures the feature selector component for the model.
    /// </summary>
    /// <remarks>
    /// A feature selector helps identify which input variables (features) are most important
    /// for making predictions.
    /// 
    /// <b>For Beginners:</b> Imagine you're trying to predict house prices. You have many possible 
    /// factors: size, location, age, number of rooms, etc. A feature selector helps figure out 
    /// which of these factors actually matter for making good predictions. This can improve 
    /// your model's accuracy and make it run faster by focusing only on what's important.
    /// </remarks>
    /// <param name="selector">The feature selector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFeatureSelector(IFeatureSelector<T, TInput> selector);

    /// <summary>
    /// Configures the data normalizer component for the model.
    /// </summary>
    /// <remarks>
    /// A normalizer transforms data to a standard scale, which helps many machine learning
    /// algorithms perform better.
    /// 
    /// <b>For Beginners:</b> Different features in your data might use different scales. For example, 
    /// a person's age (0-100) and income (thousands or millions) are on very different scales. 
    /// Normalization converts all features to a similar scale (like 0-1), which prevents features 
    /// with larger numbers from dominating the learning process just because they have bigger values.
    /// </remarks>
    /// <param name="normalizer">The normalizer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureNormalizer(INormalizer<T, TInput, TOutput> normalizer);

    /// <summary>
    /// Configures the regularization component for the model.
    /// </summary>
    /// <remarks>
    /// Regularization helps prevent overfitting by adding a penalty for complexity in the model.
    /// 
    /// <b>For Beginners:</b> Overfitting happens when a model learns the training data too well, including 
    /// all its noise and peculiarities, making it perform poorly on new data. Regularization is like 
    /// adding training wheels that prevent the model from becoming too complex. It's like telling the 
    /// model "keep it simple" so it learns general patterns rather than memorizing specific examples.
    /// </remarks>
    /// <param name="regularization">The regularization implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization);

    /// <summary>
    /// Configures the fitness calculator component for the model.
    /// </summary>
    /// <remarks>
    /// A fitness calculator measures how well the model is performing during training.
    /// 
    /// <b>For Beginners:</b> The fitness calculator is like a scorekeeper that tells you how well your 
    /// model is doing. It compares the model's predictions to the actual correct answers and 
    /// calculates a score. This score helps determine if changes to the model are making it 
    /// better or worse.
    /// </remarks>
    /// <param name="calculator">The fitness calculator implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator);

    /// <summary>
    /// Configures the fit detector component for the model.
    /// </summary>
    /// <remarks>
    /// A fit detector determines whether the model is underfitting, well-fitted, or overfitting.
    /// 
    /// <b>For Beginners:</b> This component checks if your model is learning properly. It's like a 
    /// teacher who can tell if:
    /// - Your model is "underfitting" (too simple and missing important patterns)
    /// - Your model is "just right" (learning the important patterns without memorizing noise)
    /// - Your model is "overfitting" (memorizing the training data instead of learning general rules)
    /// 
    /// This helps you know when to stop training or when to adjust your model's complexity.
    /// </remarks>
    /// <param name="detector">The fit detector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector);

    /// <summary>
    /// Configures the prediction model algorithm to use.
    /// </summary>
    /// <remarks>
    /// This method lets you specify which machine learning algorithm will be used as the core of your predictive model.
    /// 
    /// <b>For Beginners:</b> This is where you choose the specific type of AI model for your prediction task.
    /// You can select from various algorithms depending on your needs:
    /// 
    /// - <b>Regression models</b> for predicting numeric values:
    ///   - Linear regression (for simple straight-line relationships)
    ///   - Polynomial regression (for curved relationships)
    ///   - Ridge or Lasso regression (to prevent overfitting)
    /// 
    /// - <b>Classification models</b> for categorizing data:
    ///   - Logistic regression (for yes/no predictions)
    ///   - Decision trees (for rule-based decisions)
    ///   - Support vector machines (for complex boundaries)
    /// 
    /// - <b>Neural networks</b> for complex pattern recognition:
    ///   - Simple neural networks (for moderate complexity)
    ///   - Deep learning models (for highly complex patterns)
    /// 
    /// - <b>Time series models</b> for sequential data:
    ///   - ARIMA (for forecasting trends)
    ///   - LSTM networks (for long-term patterns)
    /// 
    /// Different models excel at different types of problems, so choosing the right one
    /// depends on your specific data and prediction goals.
    /// </remarks>
    /// <param name="model">The prediction model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureModel(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Configures the optimization algorithm for the model.
    /// </summary>
    /// <remarks>
    /// An optimizer determines how the model's parameters are updated during training.
    /// 
    /// <b>For Beginners:</b> The optimizer is like the "learning strategy" for your model. It decides:
    /// - How quickly the model should learn (learning rate)
    /// - How to adjust the model's parameters to improve predictions
    /// - When to stop trying to improve further
    /// 
    /// Common optimizers include Gradient Descent, Adam, and L-BFGS, each with different 
    /// strengths and weaknesses.
    /// </remarks>
    /// <param name="optimizationAlgorithm">The optimization algorithm implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm);

    /// <summary>
    /// Enables federated learning training using the provided options.
    /// </summary>
    /// <remarks>
    /// Federated learning is orchestrated internally by the builder to preserve the public facade API.
    /// Users typically only provide an options object; optional strategy injection is available for advanced scenarios.
    /// </remarks>
    /// <param name="options">Federated learning configuration options.</param>
    /// <param name="aggregationStrategy">Optional aggregation strategy override (null uses defaults based on options).</param>
    /// <param name="clientSelectionStrategy">Optional client selection strategy override (null uses defaults based on options).</param>
    /// <param name="serverOptimizer">Optional server-side optimizer override (null uses defaults based on options).</param>
    /// <param name="heterogeneityCorrection">Optional heterogeneity correction strategy override (null uses defaults based on options).</param>
    /// <param name="homomorphicEncryptionProvider">Optional homomorphic encryption provider for encrypted aggregation (null uses plaintext aggregation).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFederatedLearning(
        FederatedLearningOptions options,
        IAggregationStrategy<IFullModel<T, TInput, TOutput>>? aggregationStrategy = null,
        IClientSelectionStrategy? clientSelectionStrategy = null,
        IFederatedServerOptimizer<T>? serverOptimizer = null,
        IFederatedHeterogeneityCorrection<T>? heterogeneityCorrection = null,
        IHomomorphicEncryptionProvider<T>? homomorphicEncryptionProvider = null);

    /// <summary>
    /// Configures the data preprocessing component for the model.
    /// </summary>
    /// <remarks>
    /// A data preprocessor cleans and transforms raw data before it's used for training.
    /// 
    /// <b>For Beginners:</b> Data preprocessing is like preparing ingredients before cooking. 
    /// It involves:
    /// - Cleaning data (removing or fixing errors)
    /// - Transforming data (converting text to numbers, etc.)
    /// - Organizing data (putting it in the right format)
    /// 
    /// Good preprocessing can dramatically improve your model's performance by ensuring 
    /// it learns from high-quality data.
    /// </remarks>
    /// <param name="dataPreprocessor">The data preprocessor implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataPreprocessor(IDataPreprocessor<T, TInput, TOutput> dataPreprocessor);

    /// <summary>
    /// Configures the outlier removal component for the model.
    /// </summary>
    /// <remarks>
    /// An outlier removal component identifies and handles unusual data points that might
    /// negatively impact the model's performance.
    /// 
    /// <b>For Beginners:</b> Outliers are unusual data points that don't follow the general pattern. 
    /// For example, if you're analyzing house prices and most houses cost $100,000-$500,000, 
    /// a $10 million mansion would be an outlier. These unusual points can confuse your model 
    /// and make it perform worse. Outlier removal helps identify and handle these unusual cases.
    /// </remarks>
    /// <param name="outlierRemoval">The outlier removal implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureOutlierRemoval(IOutlierRemoval<T, TInput, TOutput> outlierRemoval);

    /// <summary>
    /// Uses a trained model to make predictions on new data.
    /// </summary>
    /// <remarks>
    /// This method applies a previously trained model to new data to generate predictions.
    /// 
    /// <b>For Beginners:</b> Once your model is trained, you can use it to make predictions on new data 
    /// it hasn't seen before. For example, if you trained a model to predict house prices based on 
    /// features like size and location, you can now give it information about new houses and it will 
    /// estimate their prices.
    /// </remarks>
    /// <param name="newData">The new input data to make predictions for.</param>
    /// <param name="model">The trained model to use for making predictions.</param>
    /// <returns>A vector of predicted values.</returns>
    TOutput Predict(TInput newData, PredictionModelResult<T, TInput, TOutput> model);

    /// <summary>
    /// Saves a trained model to a file.
    /// </summary>
    /// <remarks>
    /// This method persists a model to disk so it can be reused later without retraining.
    /// 
    /// <b>For Beginners:</b> Training a model can take a lot of time and computing power. This method 
    /// lets you save your trained model to a file on your computer, so you can use it again later 
    /// without having to retrain it. It's like saving a document you've been working on.
    /// </remarks>
    /// <param name="model">The trained model to save.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    void SaveModel(PredictionModelResult<T, TInput, TOutput> model, string filePath);

    /// <summary>
    /// Loads a previously saved model from a file.
    /// </summary>
    /// <remarks>
    /// This method retrieves a model that was previously saved to disk.
    /// 
    /// <b>For Beginners:</b> This method lets you load a previously saved model from a file. It's like 
    /// opening a document you worked on earlier. Once loaded, you can immediately use the model 
    /// to make predictions without having to train it again.
    /// </remarks>
    /// <param name="filePath">The file path where the model is stored.</param>
    /// <returns>The loaded predictive model.</returns>
    PredictionModelResult<T, TInput, TOutput> LoadModel(string filePath);

    /// <summary>
    /// Converts a trained model into a byte array for storage or transmission.
    /// </summary>
    /// <remarks>
    /// This method transforms a model into a compact binary format that can be stored in memory,
    /// databases, or transmitted over networks.
    /// 
    /// <b>For Beginners:</b> Serialization is like packing your model into a compact digital suitcase.
    /// Instead of saving to a file (like with SaveModel), this method converts your model into
    /// a series of bytes that can be:
    /// - Stored in a database
    /// - Sent over the internet
    /// - Kept in computer memory
    /// - Embedded in other applications
    /// 
    /// This is useful when you need to store models in places other than files or when you
    /// want to send models between different parts of your application.
    /// </remarks>
    /// <param name="model">The trained model to serialize.</param>
    /// <returns>A byte array containing the serialized model data.</returns>
    byte[] SerializeModel(PredictionModelResult<T, TInput, TOutput> model);

    /// <summary>
    /// Reconstructs a model from a previously serialized byte array.
    /// </summary>
    /// <remarks>
    /// This method converts a byte array back into a usable model object.
    /// 
    /// <b>For Beginners:</b> Deserialization is like unpacking your model from the digital suitcase
    /// created by SerializeModel. It takes the compact byte format and rebuilds your complete
    /// model so you can use it for making predictions again.
    /// 
    /// This is the counterpart to SerializeModel - first you serialize to create the byte array,
    /// then you deserialize to recreate the model when needed.
    /// 
    /// For example, if you stored your model in a database or received it over a network,
    /// you would use this method to convert it back into a working model.
    /// </remarks>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <returns>The reconstructed predictive model.</returns>
    PredictionModelResult<T, TInput, TOutput> DeserializeModel(byte[] modelData);

    /// <summary>
    /// Configures the bias detector component for ethical AI evaluation.
    /// </summary>
    /// <remarks>
    /// A bias detector analyzes model predictions to identify potential bias across different
    /// demographic groups defined by sensitive features.
    ///
    /// <b>For Beginners:</b> Bias detection helps ensure your model treats different groups fairly.
    /// For example, if your model predicts loan approvals, bias detection checks whether it
    /// unfairly favors or discriminates against certain demographic groups (like age, gender, or race).
    /// This is crucial for ethical AI and regulatory compliance.
    /// </remarks>
    /// <param name="detector">The bias detector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureBiasDetector(IBiasDetector<T> detector);

    /// <summary>
    /// Configures the fairness evaluator component for ethical AI evaluation.
    /// </summary>
    /// <remarks>
    /// A fairness evaluator computes multiple fairness metrics to assess how equitably
    /// a model performs across different demographic groups.
    ///
    /// <b>For Beginners:</b> Fairness evaluation goes beyond basic accuracy to measure whether
    /// your model is fair to all groups. It calculates metrics like demographic parity (do all
    /// groups get positive outcomes at similar rates?) and equal opportunity (do qualified individuals
    /// from all groups have equal chances?). This helps you build AI systems that are not only
    /// accurate but also ethical and compliant with regulations.
    /// </remarks>
    /// <param name="evaluator">The fairness evaluator implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFairnessEvaluator(IFairnessEvaluator<T> evaluator);

    /// <summary>
    /// Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    /// </summary>
    /// <remarks>
    /// LoRA enables efficient fine-tuning of neural networks by learning low-rank decompositions
    /// of weight updates instead of modifying all weights directly. This dramatically reduces
    /// the number of trainable parameters while maintaining model performance.
    ///
    /// <b>For Beginners:</b> LoRA is a technique that lets you adapt large pre-trained models
    /// with 100x fewer parameters than traditional fine-tuning. Instead of updating all weights,
    /// LoRA adds small "correction layers" that learn what adjustments are needed.
    ///
    /// Think of it like:
    /// - The original model has the base knowledge (optionally frozen)
    /// - LoRA layers learn small corrections for your specific task
    /// - The final output combines both: original + correction
    ///
    /// This is especially useful when:
    /// - You want to fine-tune a large model with limited memory
    /// - You need to create multiple task-specific versions of the same model
    /// - You want to adapt pre-trained models without retraining everything
    ///
    /// The configuration determines which layers get LoRA adaptations, what rank to use,
    /// and whether to freeze the base layers during training.
    /// </remarks>
    /// <param name="loraConfiguration">The LoRA configuration implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureLoRA(ILoRAConfiguration<T> loraConfiguration);

    /// <summary>
    /// Configures uncertainty quantification (UQ) for inference-time uncertainty estimates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uncertainty quantification augments point predictions with uncertainty signals (for example: variance and predictive entropy).
    /// This can be used to detect low-confidence outputs and make safer decisions.
    /// </para>
    /// <para>
    /// Some uncertainty features optionally use a separate calibration dataset (held out from training) to compute
    /// calibration artifacts (for example: conformal thresholds or temperature scaling).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This enables a "confidence signal" alongside predictions. If you're not sure what to choose,
    /// call this method with no parameters to enable industry-standard defaults.
    /// </para>
    /// </remarks>
    /// <param name="options">Optional options; when null, defaults are used and UQ is enabled.</param>
    /// <param name="calibrationData">Optional calibration data for conformal/prediction calibration features.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureUncertaintyQuantification(
        UncertaintyQuantificationOptions? options = null,
        UncertaintyCalibrationData<TInput, TOutput>? calibrationData = null);

    /// <summary>
    /// Configures the retrieval-augmented generation (RAG) components for use during model inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RAG enhances text generation by retrieving relevant documents from a knowledge base
    /// and using them as context for generating grounded, factual answers.
    /// </para>
    /// <para>
    /// <b>Graph RAG:</b> When graphStore or knowledgeGraph is provided, enables knowledge graph-based
    /// retrieval that finds related entities and their relationships, providing richer context than
    /// vector similarity alone. If documentStore is also provided, hybrid retrieval combines both
    /// vector search and graph traversal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> RAG is like giving your AI access to a library before answering questions.
    /// Instead of relying only on what it learned during training, it can:
    /// <list type="number">
    /// <item><description>Search a document collection for relevant information</description></item>
    /// <item><description>Read the relevant documents</description></item>
    /// <item><description>Generate an answer based on those documents</description></item>
    /// <item><description>Cite its sources</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Graph RAG Example:</b> If you ask about "Paris", Graph RAG can find not just documents
    /// mentioning Paris, but also related concepts like France, Eiffel Tower, and Seine River
    /// by traversing the knowledge graph.
    /// </para>
    /// <para>
    /// RAG operations (GenerateAnswer, RetrieveDocuments, GraphQuery, etc.) are performed during
    /// inference via PredictionModelResult, not during model building.
    /// </para>
    /// </remarks>
    /// <param name="retriever">Optional retriever for finding relevant documents. If not provided, standard RAG won't be available.</param>
    /// <param name="reranker">Optional reranker for improving document ranking quality. Default provided if retriever is set.</param>
    /// <param name="generator">Optional generator for producing grounded answers. Default provided if retriever is set.</param>
    /// <param name="queryProcessors">Optional query processors for improving search quality.</param>
    /// <param name="graphStore">Optional graph storage backend for Graph RAG (e.g., MemoryGraphStore, FileGraphStore).</param>
    /// <param name="knowledgeGraph">Optional pre-configured knowledge graph. If null but graphStore is provided, a new one is created.</param>
    /// <param name="documentStore">Optional document store for hybrid vector + graph retrieval.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
        IRetriever<T>? retriever = null,
        IReranker<T>? reranker = null,
        IGenerator<T>? generator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null,
        IGraphStore<T>? graphStore = null,
        KnowledgeGraph<T>? knowledgeGraph = null,
        IDocumentStore<T>? documentStore = null);

    /// <summary>
    /// Configures AI agent assistance during model building and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Agent assistance adds AI-powered help during model creation.
    /// The agent can analyze your data, suggest which model type to use, recommend hyperparameters,
    /// and provide insights about feature importance.
    ///
    /// The configuration is stored securely and will be reused during inference if you call AskAsync() on the trained model.
    /// </para>
    /// <para>
    /// Example usage:
    /// <code>
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true
    /// };
    ///
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="configuration">The agent configuration containing API keys, provider settings, and options.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureAgentAssistance(AgentConfiguration<T> configuration);

    /// <summary>
    /// Configures a meta-learning algorithm (MAML, Reptile, SEAL) for training models that can quickly adapt to new tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Meta-learning trains models to quickly learn new tasks from just a few examples.
    /// If you configure this, Build() will do meta-training instead of regular training.
    ///
    /// Only configure this if you need few-shot learning capabilities. For standard machine learning,
    /// just use ConfigureModel() and Build() as usual.
    /// </para>
    /// </remarks>
    /// <param name="metaLearner">The meta-learning algorithm to use (e.g., ReptileTrainer with its episodic data loader).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureMetaLearning(IMetaLearner<T, TInput, TOutput> metaLearner);

    /// <summary>
    /// Configures distributed training across multiple GPUs or machines.
    /// </summary>
    /// <param name="backend">Communication backend to use. If null, uses InMemoryCommunicationBackend.</param>
    /// <param name="strategy">Distributed training strategy. Default is FSDP.</param>
    /// <param name="autoSyncGradients">Whether to automatically synchronize gradients. Default is true.</param>
    /// <param name="minimumParameterGroupSize">Minimum parameter group size for communication. Default is 1024.</param>
    /// <param name="enableGradientCompression">Whether to enable gradient compression. Default is false.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When distributed training is configured, the builder automatically wraps the model and optimizer
    /// with their distributed counterparts based on the chosen strategy. This enables:
    /// - Training models too large to fit on a single GPU
    /// - Faster training by distributing work across multiple processes
    /// - Automatic gradient synchronization and parameter sharding
    /// </para>
    /// <para>
    /// <b>Important:</b> The strategy parameter controls BOTH the model and optimizer as a matched pair.
    /// You cannot mix and match strategies between model and optimizer because they must be compatible:
    /// </para>
    /// <para>
    /// - <b>DDP</b> → Uses DDPModel + DDPOptimizer (replicated parameters, AllReduce gradients)
    /// </para>
    /// <para>
    /// - <b>FSDP</b> → Uses FSDPModel + FSDPOptimizer (fully sharded parameters)
    /// </para>
    /// <para>
    /// - <b>ZeRO1/2/3</b> → Uses matching ZeRO models + optimizers (progressive sharding)
    /// </para>
    /// <para>
    /// - <b>PipelineParallel</b> → Uses PipelineParallelModel + PipelineParallelOptimizer
    /// </para>
    /// <para>
    /// - <b>TensorParallel</b> → Uses TensorParallelModel + TensorParallelOptimizer
    /// </para>
    /// <para>
    /// - <b>Hybrid</b> → Uses HybridShardedModel + HybridShardedOptimizer (3D parallelism)
    /// </para>
    /// <para>
    /// This design follows industry standards (PyTorch DDP/FSDP, DeepSpeed ZeRO, Megatron-LM) where
    /// the distributed training strategy is a cohesive unit that applies to both model and optimizer.
    /// Mixing strategies would cause incompatibilities - for example, a DDP model (replicated parameters)
    /// cannot work with an FSDP optimizer (expects sharded parameters).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Call this method to enable distributed training across multiple GPUs.
    /// You can use it with no parameters for sensible defaults, or customize each aspect.
    /// The strategy you choose automatically configures both the model and optimizer to work together.
    /// </para>
    /// <para>
    /// <b>Beginner Usage (no parameters):</b>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureDistributedTraining()  // InMemory backend, DDP strategy
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </para>
    /// <para>
    /// <b>Intermediate Usage (specify backend):</b>
    /// <code>
    /// var backend = new MPICommunicationBackend&lt;double&gt;();
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureDistributedTraining(backend)  // MPI backend, DDP strategy
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </para>
    /// <para>
    /// <b>Advanced Usage (specify strategy):</b>
    /// <code>
    /// var result = builder
    ///     .ConfigureModel(myModel)
    ///     .ConfigureDistributedTraining(
    ///         backend: new NCCLCommunicationBackend&lt;double&gt;(),
    ///         strategy: DistributedStrategy.FSDP)  // Use FSDP instead of DDP
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </para>
    /// <para>
    /// <b>Expert Usage (full control):</b>
    /// <code>
    /// var backend = new NCCLCommunicationBackend&lt;double&gt;();
    /// var config = new ShardingConfiguration&lt;double&gt;(backend)
    /// {
    ///     AutoSyncGradients = true,
    ///     MinimumParameterGroupSize = 2048,
    ///     EnableGradientCompression = true
    /// };
    /// var result = builder
    ///     .ConfigureDistributedTraining(
    ///         backend: backend,
    ///         strategy: DistributedStrategy.ZeRO2,
    ///         configuration: config)  // Full control over all options
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="backend">Communication backend. If null, uses InMemoryCommunicationBackend.</param>
    /// <param name="strategy">Distributed training strategy. Default is DDP (most common).</param>
    /// <param name="configuration">Sharding configuration. If null, created from backend with defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureDistributedTraining(
        ICommunicationBackend<T>? backend = null,
        DistributedStrategy strategy = DistributedStrategy.DDP,
        IShardingConfiguration<T>? configuration = null);

    /// <summary>
    /// Configures the model evaluator component for comprehensive model evaluation and cross-validation.
    /// </summary>
    /// <remarks>
    /// A model evaluator provides methods to evaluate model performance on different datasets and
    /// perform cross-validation to assess generalization.
    ///
    /// <b>For Beginners:</b> The model evaluator helps you understand how well your model performs.
    /// If you configure both a model evaluator and cross-validator (via ConfigureCrossValidation),
    /// cross-validation will automatically run during Build() and the results will be included
    /// in your trained model.
    /// </remarks>
    /// <param name="evaluator">The model evaluator implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureModelEvaluator(IModelEvaluator<T, TInput, TOutput> evaluator);

    /// <summary>
    /// Configures the cross-validation strategy for automatic model evaluation during training.
    /// </summary>
    /// <remarks>
    /// A cross-validator determines how data should be split into folds for cross-validation.
    /// Different strategies (K-Fold, Leave-One-Out, Stratified, Time Series, etc.) are appropriate
    /// for different types of data and problems.
    ///
    /// <b>For Beginners:</b> Cross-validation tests how well your model will perform on new data
    /// by training and testing it multiple times on different subsets of your training data.
    /// If you configure both a cross-validator and model evaluator (via ConfigureModelEvaluator),
    /// cross-validation will automatically run during Build() and the results will be included
    /// in your trained model.
    ///
    /// Common strategies:
    /// - StandardCrossValidator (K-Fold): General purpose, splits data into K equal parts
    /// - LeaveOneOutCrossValidator: For small datasets, uses each sample once as test
    /// - StratifiedKFoldCrossValidator: For classification, maintains class proportions
    /// - TimeSeriesCrossValidator: For sequential data, respects temporal ordering
    /// </remarks>
    /// <param name="crossValidator">The cross-validation strategy to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureCrossValidation(ICrossValidator<T, TInput, TOutput> crossValidator);

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
    /// var builder = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAutoML(autoML)
    ///     .Build(trainingData, trainingLabels);
    /// </code>
    /// </para>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoML(IAutoMLModel<T, TInput, TOutput> autoMLModel);

    /// <summary>
    /// Configures AutoML using facade-style options (recommended for most users).
    /// </summary>
    /// <param name="options">AutoML options (budget, strategy, and optional overrides). If null, defaults are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This overload follows the AiDotNet facade pattern: you provide a small options object, and the library
    /// chooses an appropriate built-in AutoML implementation and industry-standard defaults for you.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this overload if you want AutoML without having to manually instantiate
    /// an AutoML implementation. Pick a budget preset (Fast/Standard/Thorough) and let AiDotNet handle the rest.
    /// </para>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureAutoML(AutoMLOptions<T, TInput, TOutput>? options = null);

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
    /// var result = await new PredictionModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureReinforcementLearning(options)
    ///     .ConfigureModel(new DQNAgent&lt;double&gt;())
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureReinforcementLearning(RLTrainingOptions<T> options);

    /// <summary>
    /// Configures knowledge distillation for training a smaller student model from a larger teacher model.
    /// </summary>
    /// <remarks>
    /// Knowledge distillation enables model compression by transferring knowledge from a large,
    /// accurate teacher model to a smaller, faster student model. The student learns to mimic
    /// the teacher's predictions and internal representations.
    ///
    /// <b>For Beginners:</b> Knowledge distillation is like having an expert teacher help train
    /// a smaller, faster student. The student model learns not just from the training labels,
    /// but also from the teacher's "soft" predictions which contain richer information about
    /// relationships between classes.
    ///
    /// Benefits:
    /// - Model compression: Deploy 10x smaller models with 90%+ of original accuracy
    /// - Faster inference: Smaller models run significantly faster
    /// - Lower memory: Fits on edge devices and mobile platforms
    /// - Better generalization: Learning from soft labels often improves accuracy
    ///
    /// Common use cases:
    /// - DistilBERT: 40% smaller than BERT, 97% performance, 60% faster
    /// - MobileNet: Distilled from ResNet for mobile deployment
    /// - Edge AI: Deploy powerful models on resource-constrained devices
    ///
    /// <b>Quick Start Example:</b>
    /// <code>
    /// var distillationOptions = new KnowledgeDistillationOptions&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;
    /// {
    ///     TeacherModelType = TeacherModelType.NeuralNetwork,
    ///     StrategyType = DistillationStrategyType.ResponseBased,
    ///     Temperature = 3.0,
    ///     Alpha = 0.3,
    ///     Epochs = 20,
    ///     BatchSize = 32
    /// };
    /// 
    /// var builder = new PredictionModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureKnowledgeDistillation(distillationOptions);
    /// </code>
    ///
    /// <b>Note:</b> Current implementation requires student model to use Vector&lt;T&gt; for both input and output types.
    /// </remarks>
    /// <param name="options">The knowledge distillation configuration options (optional, uses sensible defaults if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureKnowledgeDistillation(
        KnowledgeDistillationOptions<T, TInput, TOutput>? options = null);

    /// <summary>
    /// Configures model quantization for reducing model size and improving inference speed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Quantization compresses your model by using smaller numbers (like 8-bit instead of 32-bit).
    /// This makes your model:
    /// - Smaller (50-75% size reduction)
    /// - Faster (2-4x speedup)
    /// - Use less memory
    ///
    /// The trade-off is a small accuracy loss (usually 1-5%). For most applications, this is acceptable.
    ///
    /// Example:
    /// <code>
    /// // Use Float16 quantization (recommended for most cases)
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureQuantization(new QuantizationConfig { Mode = QuantizationMode.Float16 })
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The quantization configuration (optional, uses no quantization if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureQuantization(QuantizationConfig? config = null);

    /// <summary>
    /// Configures model compression for reducing model size during serialization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Compression makes your model smaller for storage and faster to load.
    /// When you save (serialize) your model, compression automatically reduces its size.
    /// When you load (deserialize) it, decompression happens transparently.
    ///
    /// Benefits:
    /// - 50-90% smaller model files
    /// - Faster model loading and deployment
    /// - Lower storage and bandwidth costs
    /// - Enables deployment on resource-constrained devices
    ///
    /// Compression is applied during serialization (saving) and reversed during deserialization (loading).
    /// You never need to handle compression manually - it happens behind the scenes.
    ///
    /// Example:
    /// <code>
    /// // Use automatic compression (recommended for most cases)
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureCompression()  // Uses industry-standard defaults
    ///     .BuildAsync();
    ///
    /// // Model is now configured to compress on save
    /// builder.SaveModel(result, "model.bin");  // Compressed automatically
    /// var loaded = builder.LoadModel("model.bin");  // Decompressed automatically
    ///
    /// // Or customize compression settings
    /// var result = await builder
    ///     .ConfigureCompression(new CompressionConfig
    ///     {
    ///         Mode = ModelCompressionMode.Full,
    ///         Type = CompressionType.HybridHuffmanClustering,
    ///         NumClusters = 256
    ///     })
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The compression configuration (optional, uses automatic mode if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureCompression(CompressionConfig? config = null);

    /// <summary>
    /// Configures model caching to avoid reloading models from disk repeatedly.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Caching keeps frequently-used models in memory so they load instantly.
    /// Like keeping your favorite apps open on your phone instead of closing and reopening them.
    ///
    /// Benefits:
    /// - Much faster inference (no model loading time)
    /// - Better throughput for multiple requests
    /// - Configurable cache size and eviction policies
    ///
    /// Example:
    /// <code>
    /// // Enable caching with default settings (10 models, LRU eviction)
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureCaching()
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The caching configuration (optional, uses default cache settings if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureCaching(CacheConfig? config = null);

    /// <summary>
    /// Configures model versioning for managing multiple versions of the same model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Versioning helps you manage different versions of your model as it improves over time.
    /// You can:
    /// - Keep track of which version is deployed
    /// - Roll back to previous versions if needed
    /// - Use "latest" to always get the newest version
    /// - Compare performance between versions
    ///
    /// Example:
    /// <code>
    /// // Enable versioning (defaults to "latest")
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureVersioning()
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The versioning configuration (optional, uses "latest" version if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureVersioning(VersioningConfig? config = null);

    /// <summary>
    /// Configures A/B testing to compare multiple model versions by splitting traffic.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A/B testing lets you safely test a new model version on a small percentage
    /// of users before fully deploying it. For example, you might send 10% of traffic to a new model
    /// and 90% to the current model, then compare performance metrics to decide which is better.
    ///
    /// This is useful for:
    /// - Testing new models in production safely
    /// - Gradually rolling out changes
    /// - Making data-driven decisions about which model to use
    ///
    /// Example:
    /// <code>
    /// // 90% on v1.0 (stable), 10% on v2.0 (experimental)
    /// var abConfig = new ABTestingConfig
    /// {
    ///     Enabled = true,
    ///     TrafficSplit = new Dictionary&lt;string, double&gt; { { "1.0", 0.9 }, { "2.0", 0.1 } },
    ///     ControlVersion = "1.0"
    /// };
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureABTesting(abConfig)
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The A/B testing configuration (optional, disables A/B testing if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureABTesting(ABTestingConfig? config = null);

    /// <summary>
    /// Configures telemetry for tracking and monitoring model inference metrics.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Telemetry collects performance data about your model in production, like:
    /// - How long each inference takes (latency)
    /// - How many inferences per second (throughput)
    /// - When errors occur
    /// - Cache hit/miss rates
    /// - Which model versions are being used
    ///
    /// This helps you:
    /// - Detect performance problems before users complain
    /// - Understand usage patterns
    /// - Debug production issues
    /// - Make informed decisions about model updates
    ///
    /// Example:
    /// <code>
    /// // Enable telemetry with default settings
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureTelemetry()
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="config">The telemetry configuration (optional, uses default telemetry settings if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureTelemetry(TelemetryConfig? config = null);

    /// <summary>
    /// Configures benchmarking to run standardized benchmark suites and attach a structured report to the built model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This follows the AiDotNet facade pattern: users select benchmark suites using enums and receive a structured report,
    /// without wiring benchmark implementations manually.
    /// </para>
    /// <para><b>For Beginners:</b> This is like running a standardized test after training/building your model.</para>
    /// </remarks>
    /// <param name="options">Benchmarking options. If null, sensible defaults are used.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureBenchmarking(BenchmarkingOptions? options = null);

    /// <summary>
    /// Configures export settings for deploying the model to different platforms.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Export settings determine how your trained model will be saved for deployment.
    /// Different platforms need different formats:
    /// - **ONNX**: Universal format, works everywhere (recommended)
    /// - **TensorRT**: NVIDIA GPUs, maximum performance
    /// - **CoreML**: Apple devices (iPhone, iPad, Mac)
    /// - **TFLite**: Android devices and edge hardware
    /// - **WASM**: Run models in web browsers
    ///
    /// Configure this BEFORE training if you know your target platform, so the model can be
    /// optimized accordingly. After training, use the Export methods on PredictionModelResult.
    ///
    /// Example:
    /// <code>
    /// // Configure for TensorRT deployment with FP16 quantization
    /// var exportConfig = new ExportConfig
    /// {
    ///     TargetPlatform = TargetPlatform.TensorRT,
    ///     Quantization = QuantizationMode.Float16
    /// };
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureExport(exportConfig)
    ///     .BuildAsync();
    ///
    /// // After training, export the model
    /// result.ExportToTensorRT("model.trt");
    /// </code>
    /// </remarks>
    /// <param name="config">The export configuration (optional, uses CPU/ONNX if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureExport(ExportConfig? config = null);

    /// <summary>
    /// Configures tokenization for text-based input processing.
    /// </summary>
    /// <remarks>
    /// Tokenization is the process of breaking text into smaller pieces (tokens) that can be processed
    /// by machine learning models. This is essential for NLP and text-based models.
    ///
    /// <b>For Beginners:</b> Tokenization converts human-readable text into numbers that AI models understand.
    ///
    /// Different tokenization strategies include:
    /// - BPE (Byte Pair Encoding): Used by GPT models, learns subword units from data
    /// - WordPiece: Used by BERT, splits unknown words into known subwords
    /// - SentencePiece: Language-independent tokenization used by many multilingual models
    ///
    /// Example:
    /// <code>
    /// var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 32000);
    /// var builder = new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureTokenizer(tokenizer)
    ///     .ConfigureModel(new TransformerModel())
    ///     .BuildAsync();
    /// </code>
    /// </remarks>
    /// <param name="tokenizer">The tokenizer to use for text processing. If null, no tokenizer is configured.</param>
    /// <param name="config">Optional tokenization configuration. If null, default settings are used.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizer(ITokenizer? tokenizer = null, TokenizationConfig? config = null);

    /// <summary>
    /// Configures tokenization using a pretrained tokenizer from HuggingFace Hub.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the easiest and most type-safe way to use industry-standard tokenizers.
    /// Using the enum ensures you always specify a valid model name.
    ///
    /// Simply call without parameters for sensible defaults:
    /// <code>
    /// var builder = new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureTokenizerFromPretrained()  // Uses BertBaseUncased by default
    ///     .ConfigureModel(new BertModel())
    ///     .BuildAsync();
    /// </code>
    ///
    /// Or specify a model using the enum:
    /// <code>
    /// builder.ConfigureTokenizerFromPretrained(PretrainedTokenizerModel.Gpt2)
    /// </code>
    ///
    /// Available models include:
    /// - BertBaseUncased: BERT tokenizer for English text (default)
    /// - Gpt2, Gpt2Medium, Gpt2Large: GPT-2 tokenizers for text generation
    /// - RobertaBase, RobertaLarge: RoBERTa tokenizers (improved BERT)
    /// - T5Small, T5Base, T5Large: T5 tokenizers for text-to-text tasks
    /// - DistilBertBaseUncased: Faster, smaller BERT
    /// - CodeBertBase: For code understanding tasks
    /// </remarks>
    /// <param name="model">The pretrained tokenizer model to use. Defaults to BertBaseUncased.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizerFromPretrained(PretrainedTokenizerModel model = PretrainedTokenizerModel.BertBaseUncased, TokenizationConfig? config = null);

    /// <summary>
    /// Configures tokenization using a pretrained tokenizer from a custom HuggingFace model name or local path.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this overload when you need to specify a custom model name or path
    /// that isn't in the PretrainedTokenizerModel enum. For common models, prefer the enum-based overload
    /// for type safety.
    ///
    /// Example with custom model:
    /// <code>
    /// // Use a custom or community model from HuggingFace
    /// builder.ConfigureTokenizerFromPretrained("sentence-transformers/all-MiniLM-L6-v2")
    /// </code>
    ///
    /// If null or empty, defaults to "bert-base-uncased".
    /// </remarks>
    /// <param name="modelNameOrPath">The HuggingFace model name or local path. Defaults to "bert-base-uncased" if not specified.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureTokenizerFromPretrained(string? modelNameOrPath = null, TokenizationConfig? config = null);

    /// <summary>
    /// Asynchronously configures the tokenizer by loading a pretrained model from HuggingFace Hub.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the async version of ConfigureTokenizerFromPretrained.
    /// Use this when you want to avoid blocking the thread while downloading tokenizer files
    /// from HuggingFace Hub. This is especially important in UI applications or web servers.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Async configuration
    /// await builder.ConfigureTokenizerFromPretrainedAsync(PretrainedTokenizerModel.BertBaseUncased);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="model">The pretrained tokenizer model to use.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>A task that completes with the builder instance for method chaining.</returns>
    Task<IPredictionModelBuilder<T, TInput, TOutput>> ConfigureTokenizerFromPretrainedAsync(PretrainedTokenizerModel model = PretrainedTokenizerModel.BertBaseUncased, TokenizationConfig? config = null);

    /// <summary>
    /// Asynchronously configures the tokenizer by loading a pretrained model from HuggingFace Hub using a model name or path.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the async version that accepts a custom model name or path.
    /// Use this when loading custom or community models without blocking the thread.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Async configuration with custom model
    /// await builder.ConfigureTokenizerFromPretrainedAsync("sentence-transformers/all-MiniLM-L6-v2");
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="modelNameOrPath">The HuggingFace model name or local path. Defaults to "bert-base-uncased" if not specified.</param>
    /// <param name="config">Optional tokenization configuration.</param>
    /// <returns>A task that completes with the builder instance for method chaining.</returns>
    Task<IPredictionModelBuilder<T, TInput, TOutput>> ConfigureTokenizerFromPretrainedAsync(string? modelNameOrPath = null, TokenizationConfig? config = null);

    /// <summary>
    /// Enables GPU acceleration for training and inference with optional configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPU acceleration makes your model train 10-100x faster on large datasets
    /// by using your graphics card (GPU) for parallel computation. It automatically uses GPU for large
    /// operations and CPU for small ones, with zero code changes required.
    /// </para>
    /// <para>
    /// Benefits:
    /// - 10-100x faster training for large neural networks
    /// - Automatic size-based routing (GPU for large ops, CPU for small)
    /// - Supports NVIDIA (CUDA) and AMD/Intel (OpenCL) GPUs
    /// - Automatic CPU fallback if GPU unavailable
    /// - Works transparently with existing models
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Enable with defaults (recommended)
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureGpuAcceleration()
    ///     .BuildAsync();
    ///
    /// // Or with aggressive settings for high-end GPUs
    /// builder.ConfigureGpuAcceleration(GpuAccelerationConfig.Aggressive());
    ///
    /// // Or CPU-only for debugging
    /// builder.ConfigureGpuAcceleration(GpuAccelerationConfig.CpuOnly());
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="config">GPU acceleration configuration (optional, uses defaults if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureGpuAcceleration(GpuAccelerationConfig? config = null);

    /// <summary>
    /// Configures Just-In-Time (JIT) compilation for neural network forward and backward passes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> JIT compilation is an optimization technique that converts your neural network's
    /// operations into highly optimized native code at runtime, similar to how modern browsers optimize JavaScript.
    /// </para>
    /// <para>
    /// Benefits:
    /// - 2-10x faster inference through operation fusion and vectorization
    /// - Reduced memory allocations during forward/backward passes
    /// - Automatic optimization of computation graphs
    /// - Zero code changes required - just enable the config
    /// </para>
    /// <para>
    /// JIT compilation works by:
    /// 1. Analyzing your neural network's computation graph
    /// 2. Fusing compatible operations together (e.g., MatMul + Bias + ReLU)
    /// 3. Generating optimized native code using System.Reflection.Emit
    /// 4. Caching compiled code for subsequent runs
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Enable JIT with defaults (recommended)
    /// var result = await builder
    ///     .ConfigureModel(model)
    ///     .ConfigureJitCompilation()
    ///     .BuildAsync();
    ///
    /// // Or with custom settings
    /// builder.ConfigureJitCompilation(new JitCompilationConfig
    /// {
    ///     Enabled = true,
    ///     CompilerOptions = new JitCompilerOptions
    ///     {
    ///         EnableOperationFusion = true,
    ///         EnableVectorization = true
    ///     }
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="config">JIT compilation configuration (optional, enables with defaults if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureJitCompilation(AiDotNet.Configuration.JitCompilationConfig? config = null);

    /// <summary>
    /// Configures inference-time optimizations for faster predictions.
    /// </summary>
    /// <param name="config">Inference optimization configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inference optimization makes your model's predictions faster and more efficient.
    ///
    /// Key features enabled:
    /// - <b>KV Cache:</b> Speeds up transformer/attention models by 2-10x
    /// - <b>Batching:</b> Groups predictions for higher throughput
    /// - <b>Speculative Decoding:</b> Speeds up text generation by 1.5-3x
    ///
    /// Example:
    /// <code>
    /// var result = await new PredictionModelBuilder&lt;double, ...&gt;()
    ///     .ConfigureModel(myModel)
    ///     .ConfigureInferenceOptimizations()  // Uses sensible defaults
    ///     .BuildAsync();
    ///
    /// // Or with custom settings:
    /// var config = new InferenceOptimizationConfig
    /// {
    ///     EnableKVCache = true,
    ///     MaxBatchSize = 64,
    ///     EnableSpeculativeDecoding = true
    /// };
    ///
    /// var result = await builder
    ///     .ConfigureInferenceOptimizations(config)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureInferenceOptimizations(AiDotNet.Configuration.InferenceOptimizationConfig? config = null);

    /// <summary>
    /// Configures mixed-precision training for faster neural network training with reduced memory usage.
    /// </summary>
    /// <param name="config">Mixed precision configuration (optional, uses defaults if null).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training is a powerful optimization technique that uses
    /// both 16-bit (half precision) and 32-bit (full precision) floating-point numbers during training.
    /// This provides:
    /// - **Up to 50% memory savings** allowing larger batch sizes or bigger models
    /// - **2-3x faster training** on modern GPUs with Tensor Cores (NVIDIA Volta+)
    /// - **Maintained accuracy** through careful precision management and loss scaling
    ///
    /// <b>Requirements:</b>
    /// - Type parameter T must be float (FP32)
    /// - Requires gradient-based optimizers (SGD, Adam, etc.)
    /// - Best suited for neural networks with large parameter counts
    ///
    /// Example:
    /// <code>
    /// // Enable with default settings (recommended)
    /// var result = await new PredictionModelBuilder&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;()
    ///     .ConfigureModel(network)
    ///     .ConfigureOptimizer(optimizer)
    ///     .ConfigureMixedPrecision()  // Enable mixed-precision
    ///     .BuildAsync();
    ///
    /// // Or with custom configuration
    /// builder.ConfigureMixedPrecision(MixedPrecisionConfig.Conservative());
    /// </code>
    /// </para>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureMixedPrecision(MixedPrecisionConfig? config = null);

    /// <summary>
    /// Configures advanced reasoning capabilities for the model using Chain-of-Thought, Tree-of-Thoughts, and Self-Consistency strategies.
    /// </summary>
    /// <param name="config">The reasoning configuration (optional, uses defaults if null).</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reasoning capabilities make AI models "think step by step" instead of
    /// giving quick answers that might be wrong. Just like a student showing their work on a math test,
    /// reasoning strategies help the AI:
    /// - Break down complex problems into manageable steps
    /// - Explore multiple solution approaches
    /// - Verify and refine its answers
    /// - Provide transparent, explainable reasoning
    ///
    /// After building your model, use the reasoning methods on PredictionModelResult:
    /// - ReasonAsync(): Solve problems with configurable reasoning strategies
    /// - QuickReasonAsync(): Fast answers for simple problems
    /// - DeepReasonAsync(): Thorough analysis for complex problems
    ///
    /// Example:
    /// <code>
    /// // Configure reasoning during model building
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .ConfigureReasoning()
    ///     .BuildAsync();
    ///
    /// // Use reasoning on the trained model
    /// var reasoningResult = await result.ReasonAsync(
    ///     "Explain why this prediction was made and what factors contributed most?",
    ///     ReasoningMode.ChainOfThought
    /// );
    /// Console.WriteLine(reasoningResult.FinalAnswer);
    /// </code>
    /// </para>
    /// </remarks>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureReasoning(ReasoningConfig? config = null);

    /// <summary>
    /// Configures the data loader for providing training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A data loader handles loading data from various sources (files, databases, memory, URLs)
    /// and provides it in a format suitable for model training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Instead of passing raw arrays or matrices directly to BuildAsync,
    /// you can configure a data loader that handles loading your data for you. This is useful when:
    /// - Your data comes from a file (CSV, JSON, etc.)
    /// - Your data needs to be downloaded from the internet
    /// - You want automatic batching and shuffling
    /// - You want train/validation/test splitting handled for you
    ///
    /// Example:
    /// <code>
    /// // Load data from CSV
    /// var loader = DataLoaders.FromCsv("housing.csv", labelColumn: "price");
    ///
    /// var result = await builder
    ///     .ConfigureDataLoader(loader)
    ///     .ConfigureModel(model)
    ///     .BuildAsync();  // Uses data from the loader
    /// </code>
    ///
    /// You can also use simple in-memory loaders for arrays:
    /// <code>
    /// var loader = DataLoaders.FromArrays(features, labels);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="dataLoader">The data loader that provides training data.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataLoader(IDataLoader<T> dataLoader);

    /// <summary>
    /// Configures the prompt template for language model interactions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A prompt template provides a structured way to create prompts for language models by combining
    /// a template string with runtime variables.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A prompt template is like a form with blanks to fill in. You define the
    /// structure once and fill in different values each time you use it.
    /// </para>
    /// </remarks>
    /// <param name="template">The prompt template to use. If null, no template is configured.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptTemplate(IPromptTemplate? template = null);

    /// <summary>
    /// Configures the prompt chain for composing multiple language model operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A chain orchestrates multiple language model calls, tools, and transformations into a cohesive
    /// workflow. Chains can be sequential, conditional, or parallel.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A chain connects multiple steps into a complete workflow, like a recipe
    /// where each step builds on the previous one.
    /// </para>
    /// </remarks>
    /// <param name="chain">The chain to use for processing prompts. If null, no chain is configured.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptChain(IChain<string, string>? chain = null);

    /// <summary>
    /// Configures the prompt optimizer for automatically improving prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A prompt optimizer automatically refines prompts to achieve better performance on a specific task.
    /// Optimization strategies include discrete search, gradient-based methods, and evolutionary algorithms.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A prompt optimizer automatically improves your prompts by testing variations
    /// and keeping the best-performing ones.
    /// </para>
    /// </remarks>
    /// <param name="optimizer">The prompt optimizer to use. If null, no optimizer is configured.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigurePromptOptimizer(IPromptOptimizer<T>? optimizer = null);

    /// <summary>
    /// Configures the few-shot example selector for selecting examples to include in prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A few-shot example selector chooses the most relevant examples to include in prompts based
    /// on the current query. Different strategies include random selection, fixed order, and
    /// similarity-based selection.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Few-shot learning teaches the model by showing it examples. The selector
    /// picks which examples to show for each new query.
    /// </para>
    /// </remarks>
    /// <param name="selector">The few-shot example selector to use. If null, no selector is configured.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFewShotExampleSelector(IFewShotExampleSelector<T>? selector = null);

    /// <summary>
    /// Asynchronously builds a meta-trained model that can quickly adapt to new tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is used when you've configured a meta-learner using ConfigureMetaLearning(),
    /// or when you've configured a data loader using ConfigureDataLoader().
    /// </para>
    /// <para>
    /// When a data loader is configured:
    /// - The loader's LoadAsync() is called to load data
    /// - Features and Labels are extracted from the loader
    /// - Training proceeds using the loaded data
    /// </para>
    /// <para>
    /// When meta-learning is configured:
    /// - It performs meta-training across many tasks to create a model that can rapidly adapt
    ///   to new tasks with just a few examples.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this method when you've configured either:
    /// - A data loader (via ConfigureDataLoader) - the loader provides the training data
    /// - Meta-learning (via ConfigureMetaLearning) - trains your model to learn NEW tasks quickly
    ///
    /// Example with data loader:
    /// <code>
    /// var result = await builder
    ///     .ConfigureDataLoader(DataLoaders.FromCsv("data.csv", labelColumn: "target"))
    ///     .ConfigureModel(model)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    /// <exception cref="InvalidOperationException">Thrown if no valid training path was configured.</exception>
    Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync();

    // ============================================================================
    // Training Infrastructure Configuration Methods
    // ============================================================================

    /// <summary>
    /// Configures experiment tracking for organizing and logging ML experiments.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Experiment tracking is like a lab notebook for your machine learning work.
    /// It helps you keep track of what you've tried, what worked, and what didn't.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Creating experiments to group related training runs
    /// - Logging hyperparameters, metrics, and artifacts
    /// - Comparing different runs to find the best approach
    /// - Reproducing previous experiments
    /// </para>
    /// </remarks>
    /// <param name="tracker">The experiment tracker implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureExperimentTracker(IExperimentTracker<T> tracker);

    /// <summary>
    /// Configures checkpoint management for saving and restoring training state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checkpoints are like save points in a video game.
    /// They let you pause training and resume later, or go back to an earlier state if something goes wrong.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Saving model state periodically during training
    /// - Restoring from the latest or best checkpoint
    /// - Automatic cleanup of old checkpoints
    /// - Tracking metrics at each checkpoint
    /// </para>
    /// </remarks>
    /// <param name="manager">The checkpoint manager implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureCheckpointManager(ICheckpointManager<T, TInput, TOutput> manager);

    /// <summary>
    /// Configures training monitoring for real-time visibility into training progress.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A training monitor is like a dashboard for your model training.
    /// It shows you how training is progressing, what resources are being used, and if there are any problems.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Real-time metric tracking (loss, accuracy, etc.)
    /// - Resource usage monitoring (CPU, GPU, memory)
    /// - Progress updates and ETA estimation
    /// - Alert thresholds for detecting problems
    /// </para>
    /// </remarks>
    /// <param name="monitor">The training monitor implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureTrainingMonitor(ITrainingMonitor<T> monitor);

    /// <summary>
    /// Configures model registry for centralized model storage and versioning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A model registry is like a library for your trained models.
    /// It keeps track of all your models, their versions, and which ones are in production.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Storing and versioning trained models
    /// - Managing model lifecycle (development → staging → production)
    /// - Tracking model metadata and lineage
    /// - Comparing different model versions
    /// </para>
    /// </remarks>
    /// <param name="registry">The model registry implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureModelRegistry(IModelRegistry<T, TInput, TOutput> registry);

    /// <summary>
    /// Configures data version control for tracking dataset changes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Data version control is like Git, but for your datasets.
    /// It tracks what data was used for training each model and lets you reproduce experiments.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Creating and tracking dataset versions
    /// - Computing dataset hashes for integrity verification
    /// - Tracking data lineage and transformations
    /// - Linking datasets to training runs
    /// </para>
    /// </remarks>
    /// <param name="dataVersionControl">The data version control implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataVersionControl(IDataVersionControl<T> dataVersionControl);

    /// <summary>
    /// Configures hyperparameter optimization for automatic tuning of model settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hyperparameter optimization automatically finds the best settings
    /// for your model (like learning rate, number of layers, etc.) instead of you having to guess.
    /// </para>
    /// <para>
    /// Key features include:
    /// - Systematic search through hyperparameter space
    /// - Multiple search strategies (grid, random, Bayesian)
    /// - Tracking and comparing trial results
    /// - Early stopping of unpromising trials
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var searchSpace = new HyperparameterSearchSpace();
    /// searchSpace.AddContinuous("learning_rate", 0.0001, 0.1, logScale: true);
    /// searchSpace.AddInteger("hidden_units", 32, 256);
    ///
    /// var optimizer = new RandomSearchOptimizer&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(maximize: false);
    /// var result = builder
    ///     .ConfigureModel(model)
    ///     .ConfigureHyperparameterOptimizer(optimizer, searchSpace, nTrials: 20)
    ///     .Build(x, y);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="optimizer">The hyperparameter optimizer implementation to use.</param>
    /// <param name="searchSpace">The hyperparameter search space defining parameter ranges. If null, hyperparameter optimization is disabled.</param>
    /// <param name="nTrials">Number of trials to run. Default is 10.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureHyperparameterOptimizer(
        IHyperparameterOptimizer<T, TInput, TOutput> optimizer,
        HyperparameterSearchSpace? searchSpace = null,
        int nTrials = 10);

}
