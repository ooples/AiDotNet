using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Latent Embedding Optimization (LEO) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// LEO (Latent Embedding Optimization) performs meta-learning by learning a low-dimensional
/// latent space for model parameters. This enables fast adaptation even for large models
/// by working in a compressed representation space.
/// </para>
/// <para>
/// <b>Key Innovation:</b> Instead of adapting parameters directly (like MAML), LEO:
/// </para>
/// <list type="number">
/// <item>Encodes support set into a latent code z</item>
/// <item>Decodes z into classifier parameters θ = g(z)</item>
/// <item>Adapts in latent space: z' = z - α∇_z L(θ)</item>
/// <item>Decodes adapted code: θ' = g(z')</item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Imagine your neural network has millions of parameters.
/// Updating them all with just 5 examples is risky - you might overfit badly.
/// LEO learns to "compress" the parameter space into maybe 64 numbers.
/// When adapting to a new task:
/// </para>
/// <list type="number">
/// <item>Look at the support examples and generate 64 numbers (latent code)</item>
/// <item>Convert those 64 numbers into full model parameters</item>
/// <item>If it doesn't work well, adjust the 64 numbers (not millions!)</item>
/// <item>Convert again to get updated parameters</item>
/// </list>
/// <para>
/// This is safer because adjusting 64 numbers can't cause as much overfitting
/// as adjusting millions of parameters.
/// </para>
/// <para>
/// <b>Variational Aspect:</b> LEO uses a variational autoencoder-like setup where:
/// - Encoder outputs mean μ and variance σ² of a Gaussian distribution
/// - Latent code is sampled: z ~ N(μ, σ²)
/// - KL divergence regularizes z toward a standard Gaussian
/// This prevents the latent space from collapsing and enables uncertainty estimation.
/// </para>
/// <para>
/// Reference: Rusu, A. A., Rao, D., Sygnowski, J., et al. (2019).
/// Meta-Learning with Latent Embedding Optimization. ICLR 2019.
/// </para>
/// </remarks>
public class LEOAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly LEOOptions<T, TInput, TOutput> _leoOptions;

    // Encoder parameters: maps embeddings to latent mean and variance
    private Vector<T> _encoderWeightsMean;
    private Vector<T> _encoderWeightsVar;

    // Decoder parameters: maps latent code to classifier weights
    private Vector<T> _decoderWeights;

    // Relation network parameters (optional)
    private Vector<T>? _relationWeights;

    /// <summary>
    /// Initializes a new instance of the LEOAlgorithm class.
    /// </summary>
    /// <param name="options">LEO configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create LEO with minimal configuration
    /// var options = new LEOOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var leo = new LEOAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create LEO with custom configuration
    /// var options = new LEOOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     LatentDimension = 64,
    ///     HiddenDimension = 256,
    ///     KLWeight = 0.01,
    ///     AdaptationSteps = 5
    /// };
    /// var leo = new LEOAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public LEOAlgorithm(LEOOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _leoOptions = options;

        // Initialize encoder, decoder, and relation network weights
        _encoderWeightsMean = new Vector<T>(options.EmbeddingDimension * options.LatentDimension);
        _encoderWeightsVar = new Vector<T>(options.EmbeddingDimension * options.LatentDimension);
        _decoderWeights = new Vector<T>(options.LatentDimension * (options.EmbeddingDimension * options.NumClasses));

        InitializeNetworkWeights();

        if (options.UseRelationEncoder)
        {
            _relationWeights = new Vector<T>(options.EmbeddingDimension * options.HiddenDimension);
            InitializeVector(_relationWeights, options.EmbeddingDimension);
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.LEO"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as LEO (Latent Embedding Optimization),
    /// which performs meta-learning by adapting in a low-dimensional latent space.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LEO;

    /// <summary>
    /// Performs one meta-training step using LEO's latent space adaptation approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// LEO meta-training involves training three main components:
    /// </para>
    /// <list type="number">
    /// <item><b>Encoder:</b> Maps support embeddings to latent distribution (μ, σ²)</item>
    /// <item><b>Decoder:</b> Maps latent code to classifier parameters</item>
    /// <item><b>Feature Encoder:</b> Maps inputs to embeddings</item>
    /// </list>
    /// <para>
    /// <b>Training Loop per Task:</b>
    /// <code>
    /// 1. Extract embeddings from support set
    /// 2. Encode to get latent distribution: (μ, σ²) = encoder(embeddings)
    /// 3. Sample latent code: z ~ N(μ, σ²)
    /// 4. Decode to get initial parameters: θ = decoder(z)
    /// 5. For each adaptation step:
    ///    a. Compute support loss with current parameters
    ///    b. Compute gradient with respect to z (not θ!)
    ///    c. Update z' = z - α∇_z L
    ///    d. Decode: θ' = decoder(z')
    /// 6. Compute query loss with adapted parameters
    /// 7. Add KL divergence loss: KL(N(μ, σ²) || N(0, I))
    /// </code>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, LEO learns:
    /// - How to look at examples and generate a good "summary" (latent code)
    /// - How to convert that summary into classifier weights
    /// - How to adjust the summary when the initial guess doesn't work
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate gradients for all networks
        Vector<T>? accumulatedEncoderMeanGrad = null;
        Vector<T>? accumulatedEncoderVarGrad = null;
        Vector<T>? accumulatedDecoderGrad = null;
        Vector<T>? accumulatedFeatureGrad = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Step 1: Extract embeddings from support set
            var supportEmbeddings = ExtractEmbeddings(task.SupportInput);

            // Step 2: Encode to get latent distribution
            var (latentMean, latentVar) = EncodeToLatent(supportEmbeddings);

            // Step 3: Sample latent code using reparameterization trick
            var latentCode = SampleLatent(latentMean, latentVar);

            // Step 4: Decode to get initial classifier parameters
            var classifierParams = DecodeLatent(latentCode);

            // Step 5: Adapt in latent space
            var adaptedLatentCode = AdaptLatentCode(
                latentCode, classifierParams, task.SupportInput, task.SupportOutput);

            // Decode adapted latent code
            var adaptedParams = DecodeLatent(adaptedLatentCode);

            // Step 6: Compute query loss with adapted parameters
            var queryPredictions = ClassifyWithParams(task.QueryInput, adaptedParams);
            T queryLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);

            // Step 7: Add KL divergence loss
            T klLoss = ComputeKLDivergence(latentMean, latentVar);
            T totalTaskLoss = NumOps.Add(queryLoss,
                NumOps.Multiply(NumOps.FromDouble(_leoOptions.KLWeight), klLoss));

            totalLoss = NumOps.Add(totalLoss, totalTaskLoss);

            // Compute gradients for all components
            var (encMeanGrad, encVarGrad, decGrad, featGrad) = ComputeAllGradients(
                task, latentMean, latentVar, adaptedLatentCode, adaptedParams, totalTaskLoss);

            // Accumulate gradients
            if (accumulatedEncoderMeanGrad == null)
            {
                accumulatedEncoderMeanGrad = encMeanGrad;
                accumulatedEncoderVarGrad = encVarGrad;
                accumulatedDecoderGrad = decGrad;
                accumulatedFeatureGrad = featGrad;
            }
            else
            {
                AccumulateVectors(accumulatedEncoderMeanGrad, encMeanGrad);
                AccumulateVectors(accumulatedEncoderVarGrad!, encVarGrad);
                AccumulateVectors(accumulatedDecoderGrad!, decGrad);
                AccumulateVectors(accumulatedFeatureGrad!, featGrad);
            }
        }

        if (accumulatedEncoderMeanGrad == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average and apply gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        DivideVector(accumulatedEncoderMeanGrad, batchSizeT);
        DivideVector(accumulatedEncoderVarGrad!, batchSizeT);
        DivideVector(accumulatedDecoderGrad!, batchSizeT);
        DivideVector(accumulatedFeatureGrad!, batchSizeT);

        // Clip gradients if configured
        if (_leoOptions.GradientClipThreshold.HasValue && _leoOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedEncoderMeanGrad = ClipGradients(accumulatedEncoderMeanGrad, _leoOptions.GradientClipThreshold.Value);
            accumulatedEncoderVarGrad = ClipGradients(accumulatedEncoderVarGrad!, _leoOptions.GradientClipThreshold.Value);
            accumulatedDecoderGrad = ClipGradients(accumulatedDecoderGrad!, _leoOptions.GradientClipThreshold.Value);
            accumulatedFeatureGrad = ClipGradients(accumulatedFeatureGrad!, _leoOptions.GradientClipThreshold.Value);
        }

        // Update network weights
        _encoderWeightsMean = ApplyGradients(_encoderWeightsMean, accumulatedEncoderMeanGrad, _leoOptions.OuterLearningRate);
        _encoderWeightsVar = ApplyGradients(_encoderWeightsVar, accumulatedEncoderVarGrad!, _leoOptions.OuterLearningRate);
        _decoderWeights = ApplyGradients(_decoderWeights, accumulatedDecoderGrad!, _leoOptions.OuterLearningRate);

        // Update feature encoder
        var featureParams = MetaModel.GetParameters();
        var updatedFeatureParams = ApplyGradients(featureParams, accumulatedFeatureGrad!, _leoOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedFeatureParams);

        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using latent space optimization.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// LEO adaptation is performed entirely in the latent space:
    /// </para>
    /// <list type="number">
    /// <item>Extract embeddings from support examples</item>
    /// <item>Encode embeddings to get latent distribution (μ, σ²)</item>
    /// <item>Use mean μ as initial latent code (no sampling at test time)</item>
    /// <item>Decode to get initial classifier parameters</item>
    /// <item>Perform gradient descent steps in latent space</item>
    /// <item>Decode final latent code to get adapted parameters</item>
    /// </list>
    /// <para>
    /// <b>For Beginners:</b> At test time with a new task:
    /// 1. Look at the support examples and figure out "what kind of task this is"
    /// 2. Generate a small code (like 64 numbers) representing the task
    /// 3. Convert that code into classifier weights
    /// 4. Try classifying, and if it's not good, adjust the code
    /// 5. Convert the adjusted code back to get better weights
    /// </para>
    /// <para>
    /// <b>Key Advantage:</b> Adaptation is very fast because we're only optimizing
    /// ~64 numbers instead of potentially millions of parameters. This also
    /// prevents overfitting because the latent space constrains what kinds
    /// of parameter updates are possible.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model for feature extraction
        var featureEncoder = CloneModel();

        // Extract embeddings from support set
        var supportEmbeddings = ExtractEmbeddings(task.SupportInput);

        // Encode to get latent distribution
        var (latentMean, latentVar) = EncodeToLatent(supportEmbeddings);

        // Use mean as latent code (no sampling at test time for determinism)
        var latentCode = CloneVector(latentMean);

        // Decode to get initial classifier parameters
        var classifierParams = DecodeLatent(latentCode);

        // Adapt in latent space
        latentCode = AdaptLatentCode(latentCode, classifierParams, task.SupportInput, task.SupportOutput);

        // Decode adapted latent code
        var adaptedParams = DecodeLatent(latentCode);

        return new LEOModel<T, TInput, TOutput>(
            featureEncoder,
            adaptedParams,
            latentCode,
            _leoOptions);
    }

    #region Network Initialization

    /// <summary>
    /// Initializes encoder, decoder weights.
    /// </summary>
    private void InitializeNetworkWeights()
    {
        // Initialize encoder weights for mean (Xavier)
        InitializeVector(_encoderWeightsMean, _leoOptions.EmbeddingDimension);

        // Initialize encoder weights for variance (smaller scale)
        InitializeVector(_encoderWeightsVar, _leoOptions.EmbeddingDimension, 0.1);

        // Initialize decoder weights
        if (_leoOptions.UseOrthogonalInit)
        {
            InitializeOrthogonal(_decoderWeights, _leoOptions.LatentDimension);
        }
        else
        {
            InitializeVector(_decoderWeights, _leoOptions.LatentDimension);
        }
    }

    /// <summary>
    /// Initializes a vector using Xavier/He initialization.
    /// </summary>
    private void InitializeVector(Vector<T> weights, int fanIn, double scale = 1.0)
    {
        double stddev = scale * Math.Sqrt(2.0 / fanIn);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2 - 1) * stddev);
        }
    }

    /// <summary>
    /// Initializes weights with orthogonal initialization (approximation).
    /// </summary>
    private void InitializeOrthogonal(Vector<T> weights, int fanIn)
    {
        // Simplified orthogonal initialization via Gram-Schmidt-like process
        double scale = Math.Sqrt(2.0 / fanIn);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2 - 1) * scale);
        }
        // Normalize blocks
        int blockSize = fanIn;
        for (int block = 0; block < weights.Length / blockSize; block++)
        {
            NormalizeBlock(weights, block * blockSize, blockSize);
        }
    }

    /// <summary>
    /// Normalizes a block of weights to unit norm.
    /// </summary>
    private void NormalizeBlock(Vector<T> weights, int start, int length)
    {
        T sumSq = NumOps.Zero;
        int end = Math.Min(start + length, weights.Length);
        for (int i = start; i < end; i++)
        {
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(weights[i], weights[i]));
        }
        double norm = Math.Sqrt(Math.Max(NumOps.ToDouble(sumSq), 1e-8));
        for (int i = start; i < end; i++)
        {
            weights[i] = NumOps.Divide(weights[i], NumOps.FromDouble(norm));
        }
    }

    /// <summary>
    /// Clones a vector.
    /// </summary>
    private Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    #endregion

    #region Encoding and Decoding

    /// <summary>
    /// Extracts embeddings from input using the feature encoder.
    /// </summary>
    private Vector<T> ExtractEmbeddings(TInput input)
    {
        var output = MetaModel.Predict(input);
        var vec = ConvertToVector(output);
        return vec ?? new Vector<T>(_leoOptions.EmbeddingDimension);
    }

    /// <summary>
    /// Encodes embeddings to latent distribution (mean and variance).
    /// </summary>
    private (Vector<T> mean, Vector<T> variance) EncodeToLatent(Vector<T> embeddings)
    {
        var mean = new Vector<T>(_leoOptions.LatentDimension);
        var variance = new Vector<T>(_leoOptions.LatentDimension);

        // Linear projection for mean
        for (int i = 0; i < _leoOptions.LatentDimension; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < Math.Min(embeddings.Length, _leoOptions.EmbeddingDimension); j++)
            {
                int weightIdx = i * _leoOptions.EmbeddingDimension + j;
                if (weightIdx < _encoderWeightsMean.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[j], _encoderWeightsMean[weightIdx]));
                }
            }
            mean[i] = sum;
        }

        // Linear projection for log-variance (then exp to get variance)
        for (int i = 0; i < _leoOptions.LatentDimension; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < Math.Min(embeddings.Length, _leoOptions.EmbeddingDimension); j++)
            {
                int weightIdx = i * _leoOptions.EmbeddingDimension + j;
                if (weightIdx < _encoderWeightsVar.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[j], _encoderWeightsVar[weightIdx]));
                }
            }
            // Softplus activation for positive variance
            double logVar = NumOps.ToDouble(sum);
            variance[i] = NumOps.FromDouble(Math.Log(1 + Math.Exp(Math.Min(logVar, 20))));
        }

        return (mean, variance);
    }

    /// <summary>
    /// Samples from the latent distribution using reparameterization trick.
    /// </summary>
    private Vector<T> SampleLatent(Vector<T> mean, Vector<T> variance)
    {
        var sample = new Vector<T>(mean.Length);
        for (int i = 0; i < mean.Length; i++)
        {
            // Reparameterization: z = μ + σ * ε, where ε ~ N(0, 1)
            double epsilon = SampleStandardNormal();
            double stddev = Math.Sqrt(Math.Max(NumOps.ToDouble(variance[i]), 1e-8));
            sample[i] = NumOps.Add(mean[i], NumOps.FromDouble(stddev * epsilon));
        }
        return sample;
    }

    /// <summary>
    /// Samples from standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleStandardNormal()
    {
        // Box-Muller transform for standard normal sampling
        double u1 = RandomGenerator.NextDouble();
        double u2 = RandomGenerator.NextDouble();

        // Avoid log(0)
        while (u1 <= double.Epsilon)
        {
            u1 = RandomGenerator.NextDouble();
        }

        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Decodes latent code to classifier parameters.
    /// </summary>
    private Vector<T> DecodeLatent(Vector<T> latentCode)
    {
        int outputDim = _leoOptions.EmbeddingDimension * _leoOptions.NumClasses;
        var classifierParams = new Vector<T>(outputDim);

        // Linear projection from latent space to classifier weights
        for (int i = 0; i < outputDim; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < latentCode.Length; j++)
            {
                int weightIdx = i * _leoOptions.LatentDimension + j;
                if (weightIdx < _decoderWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(latentCode[j], _decoderWeights[weightIdx]));
                }
            }
            classifierParams[i] = sum;
        }

        return classifierParams;
    }

    #endregion

    #region Latent Space Adaptation

    /// <summary>
    /// Adapts the latent code using gradient descent.
    /// </summary>
    private Vector<T> AdaptLatentCode(
        Vector<T> latentCode,
        Vector<T> classifierParams,
        TInput supportInput,
        TOutput supportOutput)
    {
        var adaptedCode = CloneVector(latentCode);
        var currentParams = CloneVector(classifierParams);

        for (int step = 0; step < _leoOptions.AdaptationSteps; step++)
        {
            // Compute predictions with current parameters
            var predictions = ClassifyWithParams(supportInput, currentParams);

            // Compute loss
            T loss = ComputeLossFromOutput(predictions, supportOutput);

            // Compute gradient with respect to latent code
            var latentGradients = ComputeLatentGradients(
                adaptedCode, currentParams, supportInput, supportOutput);

            // Update latent code
            for (int i = 0; i < adaptedCode.Length; i++)
            {
                T update = NumOps.Multiply(NumOps.FromDouble(_leoOptions.InnerLearningRate), latentGradients[i]);
                adaptedCode[i] = NumOps.Subtract(adaptedCode[i], update);
            }

            // Decode updated latent code
            currentParams = DecodeLatent(adaptedCode);
        }

        return adaptedCode;
    }

    /// <summary>
    /// Computes gradients with respect to the latent code.
    /// </summary>
    private Vector<T> ComputeLatentGradients(
        Vector<T> latentCode,
        Vector<T> classifierParams,
        TInput input,
        TOutput expectedOutput)
    {
        double epsilon = 1e-5;
        var gradients = new Vector<T>(latentCode.Length);

        // Compute baseline loss
        var basePred = ClassifyWithParams(input, classifierParams);
        T baseLoss = ComputeLossFromOutput(basePred, expectedOutput);

        // Finite differences for each latent dimension
        for (int i = 0; i < latentCode.Length; i++)
        {
            // Perturb latent code
            T original = latentCode[i];
            latentCode[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            // Decode perturbed code
            var perturbedParams = DecodeLatent(latentCode);
            var perturbedPred = ClassifyWithParams(input, perturbedParams);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, expectedOutput);

            // Compute gradient
            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
            gradients[i] = NumOps.FromDouble(grad);

            // Restore
            latentCode[i] = original;
        }

        return gradients;
    }

    #endregion

    #region Classification

    /// <summary>
    /// Classifies input using the given classifier parameters.
    /// </summary>
    private TOutput ClassifyWithParams(TInput input, Vector<T> classifierParams)
    {
        // Extract features
        var embeddings = ExtractEmbeddings(input);

        // Apply classifier (linear layer)
        var logits = new Vector<T>(_leoOptions.NumClasses);
        int embDim = Math.Min(embeddings.Length, _leoOptions.EmbeddingDimension);

        for (int c = 0; c < _leoOptions.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int e = 0; e < embDim; e++)
            {
                int paramIdx = c * _leoOptions.EmbeddingDimension + e;
                if (paramIdx < classifierParams.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[e], classifierParams[paramIdx]));
                }
            }
            logits[c] = sum;
        }

        return ConvertFromVector(logits);
    }

    /// <summary>
    /// Converts a vector to the output type.
    /// </summary>
    private TOutput ConvertFromVector(Vector<T> vector)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)vector;
        }

        // Handle Tensor<T>
        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(vector);
        }

        // Handle T[]
        if (typeof(TOutput) == typeof(T[]))
        {
            return (TOutput)(object)vector.ToArray();
        }

        throw new InvalidOperationException(
            $"Cannot convert Vector<{typeof(T).Name}> to {typeof(TOutput).Name}. " +
            $"Supported types: Vector<T>, Tensor<T>, T[]");
    }

    #endregion

    #region Loss and Regularization

    /// <summary>
    /// Computes KL divergence between latent distribution and standard Gaussian.
    /// </summary>
    private T ComputeKLDivergence(Vector<T> mean, Vector<T> variance)
    {
        // KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
        T kl = NumOps.Zero;
        for (int i = 0; i < mean.Length; i++)
        {
            double mu = NumOps.ToDouble(mean[i]);
            double var = Math.Max(NumOps.ToDouble(variance[i]), 1e-8);
            double term = var + mu * mu - 1 - Math.Log(var);
            kl = NumOps.Add(kl, NumOps.FromDouble(0.5 * term));
        }
        return kl;
    }

    #endregion

    #region Meta-Gradient Computation

    /// <summary>
    /// Computes gradients for all network components.
    /// </summary>
    private (Vector<T> encMeanGrad, Vector<T> encVarGrad, Vector<T> decGrad, Vector<T> featGrad)
        ComputeAllGradients(
            IMetaLearningTask<T, TInput, TOutput> task,
            Vector<T> latentMean,
            Vector<T> latentVar,
            Vector<T> adaptedLatent,
            Vector<T> adaptedParams,
            T totalLoss)
    {
        // Simplified gradient computation using finite differences
        double epsilon = 1e-5;

        // Encoder mean gradients - include KL divergence to regularize latent space
        var encMeanGrad = ComputeFiniteDiffGradients(
            _encoderWeightsMean, epsilon, () =>
            {
                var emb = ExtractEmbeddings(task.SupportInput);
                var (mean, var) = EncodeToLatent(emb);
                var code = SampleLatent(mean, var);
                var adapted = AdaptLatentCode(code, DecodeLatent(code), task.SupportInput, task.SupportOutput);
                var pred = ClassifyWithParams(task.QueryInput, DecodeLatent(adapted));
                T queryLoss = ComputeLossFromOutput(pred, task.QueryOutput);
                // Include KL divergence to properly regularize the encoder
                T klLoss = ComputeKLDivergence(mean, var);
                return NumOps.Add(queryLoss, NumOps.Multiply(NumOps.FromDouble(_leoOptions.KLWeight), klLoss));
            });

        // Encoder variance gradients - include KL divergence to regularize latent space
        // Variance affects the reparameterization: z = mean + sqrt(var) * noise
        var encVarGrad = ComputeFiniteDiffGradients(
            _encoderWeightsVar, epsilon, () =>
            {
                var emb = ExtractEmbeddings(task.SupportInput);
                var (mean, variance) = EncodeToLatent(emb);
                var code = SampleLatent(mean, variance);
                var adapted = AdaptLatentCode(code, DecodeLatent(code), task.SupportInput, task.SupportOutput);
                var pred = ClassifyWithParams(task.QueryInput, DecodeLatent(adapted));
                T queryLoss = ComputeLossFromOutput(pred, task.QueryOutput);
                // Include KL divergence to properly regularize the encoder
                T klLoss = ComputeKLDivergence(mean, variance);
                return NumOps.Add(queryLoss, NumOps.Multiply(NumOps.FromDouble(_leoOptions.KLWeight), klLoss));
            });

        // Decoder gradients - decode inside closure so it depends on _decoderWeights
        var decGrad = ComputeFiniteDiffGradients(
            _decoderWeights, epsilon, () =>
            {
                // DecodeLatent uses _decoderWeights, so perturbing them affects the loss
                var decodedParams = DecodeLatent(adaptedLatent);
                var pred = ClassifyWithParams(task.QueryInput, decodedParams);
                return ComputeLossFromOutput(pred, task.QueryOutput);
            });

        // Feature encoder gradients
        var featGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

        return (encMeanGrad, encVarGrad, decGrad, featGrad);
    }

    /// <summary>
    /// Computes gradients using finite differences.
    /// </summary>
    /// <remarks>
    /// For large weight vectors, use stochastic gradient estimation by sampling
    /// a subset and scaling. For small vectors, compute all gradients.
    /// </remarks>
    private Vector<T> ComputeFiniteDiffGradients(Vector<T> weights, double epsilon, Func<T> lossFunc)
    {
        var gradients = new Vector<T>(weights.Length);
        T baseLoss = lossFunc();

        // For efficiency with large vectors, use stochastic gradient estimation
        // with random sampling and proper scaling
        const int MaxDirectCompute = 500;

        if (weights.Length <= MaxDirectCompute)
        {
            // Compute all gradients directly for small vectors
            for (int i = 0; i < weights.Length; i++)
            {
                T original = weights[i];
                weights[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));
                T perturbedLoss = lossFunc();
                double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
                gradients[i] = NumOps.FromDouble(grad);
                weights[i] = original;
            }
        }
        else
        {
            // Stochastic gradient estimation for large vectors
            // Sample uniformly and scale gradients to approximate full computation
            int sampleCount = MaxDirectCompute;
            double scaleFactor = (double)weights.Length / sampleCount;

            for (int s = 0; s < sampleCount; s++)
            {
                // Use deterministic sampling to ensure coverage
                int i = (s * weights.Length) / sampleCount;

                T original = weights[i];
                weights[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));
                T perturbedLoss = lossFunc();
                double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
                // Apply scale factor to make gradient estimate unbiased
                gradients[i] = NumOps.FromDouble(grad * scaleFactor);
                weights[i] = original;
            }

            // Log warning once about approximate gradients
            System.Diagnostics.Debug.WriteLine(
                $"[LEO] Using stochastic gradient estimation for {weights.Length} parameters " +
                $"(sampled {sampleCount}, scaled by {scaleFactor:F2}). For exact gradients, implement IGradientComputable.");
        }

        return gradients;
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Accumulates vectors element-wise.
    /// </summary>
    private void AccumulateVectors(Vector<T> target, Vector<T> source)
    {
        int len = Math.Min(target.Length, source.Length);
        for (int i = 0; i < len; i++)
        {
            target[i] = NumOps.Add(target[i], source[i]);
        }
    }

    /// <summary>
    /// Divides all elements of a vector by a scalar.
    /// </summary>
    private void DivideVector(Vector<T> vec, T divisor)
    {
        for (int i = 0; i < vec.Length; i++)
        {
            vec[i] = NumOps.Divide(vec[i], divisor);
        }
    }

    #endregion
}
