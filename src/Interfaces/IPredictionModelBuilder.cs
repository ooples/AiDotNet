using AiDotNet.Models.Results;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.MixedPrecision;

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
    /// Configures the retrieval-augmented generation (RAG) components for use during model inference.
    /// </summary>
    /// <remarks>
    /// RAG enhances text generation by retrieving relevant documents from a knowledge base
    /// and using them as context for generating grounded, factual answers.
    ///
    /// <b>For Beginners:</b> RAG is like giving your AI access to a library before answering questions.
    /// Instead of relying only on what it learned during training, it can:
    /// 1. Search a document collection for relevant information
    /// 2. Read the relevant documents
    /// 3. Generate an answer based on those documents
    /// 4. Cite its sources
    ///
    /// This makes answers more accurate, up-to-date, and traceable to source materials.
    /// 
    /// RAG operations (GenerateAnswer, RetrieveDocuments) are performed during inference via PredictionModelResult,
    /// not during model building.
    /// </remarks>
    /// <param name="retriever">Optional retriever for finding relevant documents. If not provided, RAG won't be available.</param>
    /// <param name="reranker">Optional reranker for improving document ranking quality. Default provided if retriever is set.</param>
    /// <param name="generator">Optional generator for producing grounded answers. Default provided if retriever is set.</param>
    /// <param name="queryProcessors">Optional query processors for improving search quality.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureRetrievalAugmentedGeneration(
        IRetriever<T>? retriever = null,
        IReranker<T>? reranker = null,
        IGenerator<T>? generator = null,
        IEnumerable<IQueryProcessor>? queryProcessors = null);

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
    /// </remarks>
    /// <param name="options">The knowledge distillation configuration options.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureKnowledgeDistillation(
        KnowledgeDistillationOptions<T, TInput, TOutput> options);

    /// <summary>
    /// Configures mixed-precision training for faster training on GPUs with Tensor Cores.
    /// </summary>
    /// <remarks>
    /// Mixed-precision training uses FP16 (half precision) for computations and FP32 (single precision)
    /// for master weights, providing 2-3x training speedup and ~50% memory reduction on modern GPUs.
    ///
    /// <b>For Beginners:</b> Mixed-precision training makes your model train faster on modern GPUs
    /// (NVIDIA V100, A100, H100, RTX 3000+) by using lower precision numbers for some calculations.
    /// This gives you:
    /// - 2-3x faster training
    /// - ~50% less memory usage
    /// - Same model accuracy
    ///
    /// <b>Requirements:</b>
    /// Mixed-precision training has specific technical requirements:
    ///
    /// 1. **Type Constraint: float only**
    ///    - Type parameter T must be float (FP32)
    ///    - Cannot use double, decimal, or integer types
    ///    - Reason: Mixed-precision converts between FP32 (float) and FP16 (Half) representations
    ///
    /// 2. **Gradient-Based Optimizers Only**
    ///    - Requires optimizers that compute gradients (SGD, Adam, RMSProp, etc.)
    ///    - Does NOT work with non-gradient methods (genetic algorithms, random search, Bayesian optimization)
    ///    - Reason: Requires gradient computation for loss scaling, master weights, and gradient accumulation
    ///
    /// 3. **Neural Networks (Recommended)**
    ///    - Best suited for neural networks with large parameter counts
    ///    - GPU with Tensor Core support (V100, A100, H100, RTX 3000/4000+)
    ///
    /// Only enable this if you have:
    /// - A neural network trained with gradient-based optimizer
    /// - Float as your numeric type (T = float)
    /// - A GPU with Tensor Core support
    ///
    /// For CPU training, older GPUs, or non-gradient optimizers, this provides no benefit.
    ///
    /// Example usage:
    /// <code>
    /// var model = builder
    ///     .ConfigureModel(myNetwork)
    ///     .ConfigureOptimizer(myOptimizer)
    ///     .ConfigureMixedPrecision()  // Use default settings
    ///     .Build(xTrain, yTrain);
    ///
    /// // Or with custom configuration
    /// var model = builder
    ///     .ConfigureModel(myNetwork)
    ///     .ConfigureOptimizer(myOptimizer)
    ///     .ConfigureMixedPrecision(MixedPrecisionConfig.Conservative())
    ///     .Build(xTrain, yTrain);
    /// </code>
    /// </remarks>
    /// <param name="config">Optional mixed-precision configuration. If null, uses default settings.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureMixedPrecision(MixedPrecisionConfig? config = null);

    /// <summary>
    /// Asynchronously builds a meta-trained model thatcan quickly adapt to new tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is used when you've configured a meta-learner using ConfigureMetaLearning().
    /// It performs meta-training across many tasks to create a model that can rapidly adapt
    /// to new tasks with just a few examples.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this method when you've configured meta-learning and agent assistance.
    /// Unlike BuildAsync(x, y) which trains on one dataset, this trains your model to be good
    /// at learning NEW tasks quickly. The training data comes from the episodic data loader
    /// you configured in your meta-learner.
    /// </para>
    /// </remarks>
    /// <returns>A task that represents the asynchronous operation, containing the meta-trained model.</returns>
    /// <exception cref="InvalidOperationException">Thrown if ConfigureMetaLearning has not been called.</exception>
    Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync();

    /// <summary>
    /// Asynchronously builds a predictive model using the provided input features and output values.
    /// If agent assistance is enabled, the agent will help with model selection and hyperparameter tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method trains your AI model on your specific dataset. It can leverage agent assistance
    /// to help select appropriate models and tune hyperparameters based on your data characteristics.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method trains your AI model using the data you provide.
    /// It's the async version that works with agent assistance features.
    /// </para>
    /// <para>
    /// Example with agent assistance:
    /// <code>
    /// var agentConfig = new AgentConfiguration&lt;double&gt;
    /// {
    ///     ApiKey = "sk-...",
    ///     Provider = LLMProvider.OpenAI,
    ///     IsEnabled = true
    /// };
    ///
    /// var result = await new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureAgentAssistance(agentConfig)
    ///     .BuildAsync(housingData, prices);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="x">Matrix of input features (required).</param>
    /// <param name="y">Vector of output values (required).</param>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of rows in the features matrix doesn't match the length of the output vector.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no model has been specified for regular training.</exception>
    Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync(TInput x, TOutput y);
}
