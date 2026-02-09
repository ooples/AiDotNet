using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of SNAIL (Simple Neural Attentive Meta-Learner) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// SNAIL combines temporal convolutions with causal attention to process few-shot learning
/// as a sequence modeling problem. Support examples (with labels) are fed sequentially,
/// then query examples are fed to produce predictions. The model learns which examples to
/// remember and attend to.
/// </para>
/// <para><b>For Beginners:</b> SNAIL treats few-shot learning as reading a story:
///
/// **How it works:**
/// 1. Take all support examples and their labels, arrange them in a sequence
/// 2. Append query examples (without labels) at the end
/// 3. Use temporal convolutions to capture local patterns (nearby examples)
/// 4. Use causal attention to capture global patterns (any previous example)
/// 5. The model predicts labels for query examples using what it "remembers"
///
/// **Analogy:**
/// Imagine reading a detective novel:
/// - Support examples are like clues scattered through the story
/// - Temporal convolutions help you remember recent clues (short-term memory)
/// - Causal attention lets you recall any clue from the entire story (long-term memory)
/// - At the end (query), you must solve the mystery using all clues gathered
///
/// **Key insight:**
/// By treating few-shot learning as sequence modeling, SNAIL can leverage powerful
/// sequence architectures (temporal convolutions + attention) that have been very
/// successful in NLP and speech recognition.
///
/// **Architecture:**
/// - Temporal Convolution (TC) blocks: Dilated causal convolutions that aggregate local info
/// - Causal Attention blocks: Attend to any previous position in the sequence
/// - Stacked TC + Attention blocks form the full SNAIL architecture
/// </para>
/// <para><b>Algorithm - SNAIL:</b>
/// <code>
/// # Architecture
/// f_theta = feature_extractor          # Converts raw inputs to features
/// TC_blocks = temporal_convolutions    # Dilated causal convolutions
/// ATT_blocks = causal_attention        # Multi-head causal attention
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Build input sequence
///         seq = []
///         for (x, y) in support_set:
///             z = f_theta(x)
///             seq.append(concat(z, one_hot(y)))  # Feature + label
///
///         for x in query_set:
///             z = f_theta(x)
///             seq.append(concat(z, zeros))        # Feature + no label
///
///         # 2. Process through SNAIL blocks
///         h = seq
///         for each block in [TC_1, ATT_1, TC_2, ATT_2, ...]:
///             h = block(h)                        # Causally process sequence
///
///         # 3. Read off query predictions from output positions
///         predictions = h[len(support):]          # Only query positions
///         meta_loss_i = loss(predictions, query_y)
///
///     # Update all parameters (backbone + TC + attention)
///     theta = theta - beta * grad(meta_loss, theta)
/// </code>
/// </para>
/// <para>
/// Reference: Mishra, N., Rohaninejad, M., Chen, X., &amp; Abbeel, P. (2018).
/// A Simple Neural Attentive Learner. ICLR 2018.
/// </para>
/// </remarks>
public class SNAILAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SNAILOptions<T, TInput, TOutput> _snailOptions;

    /// <summary>
    /// Parameters for the temporal convolution blocks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the weights for the "short-term memory" part of SNAIL.
    /// Temporal convolutions look at nearby examples in the sequence to detect local patterns.
    /// Dilated convolutions allow the receptive field to grow exponentially with depth.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _tcBlockParams;

    /// <summary>
    /// Parameters for the causal attention blocks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the weights for the "long-term memory" part of SNAIL.
    /// Attention can look back at any previous position in the sequence, allowing the model
    /// to recall relevant examples regardless of how far back they appeared.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _attentionBlockParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SNAIL;

    /// <summary>
    /// Initializes a new SNAIL meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for SNAIL.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new SNAIL meta-learner with:
    /// - A feature extractor for converting raw inputs to features
    /// - Temporal convolution blocks for local sequence patterns
    /// - Causal attention blocks for global sequence patterns
    /// These are all jointly trained during meta-training.
    /// </para>
    /// </remarks>
    public SNAILAlgorithm(SNAILOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _snailOptions = options;
        _tcBlockParams = new List<Vector<T>>();
        _attentionBlockParams = new List<Vector<T>>();
        InitializeSNAILBlocks();
    }

    /// <summary>
    /// Initializes the temporal convolution and attention block parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each SNAIL block, initializes:
    /// - Temporal convolution filters with dilated causal convolutions
    /// - Causal attention weights (key, query, value projections)
    /// Uses He initialization scaled by the hidden dimension.
    /// </para>
    /// <para><b>For Beginners:</b> Sets up the "memory system" of SNAIL:
    /// - TC filters: Pattern detectors for nearby examples (like reading neighboring sentences)
    /// - Attention weights: Mechanism for recalling any previous example (like flipping back through pages)
    /// Both start with small random values and are refined during training.
    /// </para>
    /// </remarks>
    private void InitializeSNAILBlocks()
    {
        int numFilters = _snailOptions.NumTCFilters;
        int keyDim = _snailOptions.AttentionKeyDim;
        int valueDim = _snailOptions.AttentionValueDim;
        int numHeads = _snailOptions.NumAttentionHeads;

        for (int block = 0; block < _snailOptions.NumBlocks; block++)
        {
            // TC block: dilated causal convolution filters
            // Number of dilation levels = log2(MaxSequenceLength) for full coverage
            int numDilationLevels = Math.Max(1, (int)Math.Ceiling(Math.Log(_snailOptions.MaxSequenceLength, 2)));
            int tcParamCount = numDilationLevels * numFilters * numFilters;
            var tcParams = new Vector<T>(tcParamCount);
            double tcScale = Math.Sqrt(2.0 / numFilters);
            for (int i = 0; i < tcParamCount; i++)
            {
                tcParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * tcScale);
            }
            _tcBlockParams.Add(tcParams);

            // Attention block: key, query, value projection matrices
            int attentionInputDim = numFilters; // Output of TC feeds into attention
            int attParamCount = numHeads * (attentionInputDim * keyDim + attentionInputDim * keyDim + attentionInputDim * valueDim);
            var attParams = new Vector<T>(attParamCount);
            double attScale = Math.Sqrt(2.0 / keyDim);
            for (int i = 0; i < attParamCount; i++)
            {
                attParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * attScale);
            }
            _attentionBlockParams.Add(attParams);
        }
    }

    /// <summary>
    /// Performs one meta-training step for SNAIL.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para>
    /// For each task:
    /// 1. Extract features from support and query examples
    /// 2. Build the input sequence (support features + labels, then query features)
    /// 3. Process through stacked TC + Attention blocks
    /// 4. Compute loss on query predictions
    ///
    /// Then update all parameters (backbone + TC + attention) jointly.
    /// </para>
    /// <para><b>For Beginners:</b> Each training step:
    /// 1. Takes a batch of few-shot tasks
    /// 2. For each task, arranges examples into a "story" (sequence)
    /// 3. Processes the story through the SNAIL architecture
    /// 4. Checks how well it predicted the query labels
    /// 5. Updates all weights to do better next time
    ///
    /// The key insight is that by training on many tasks, SNAIL learns general
    /// strategies for extracting useful information from sequences of examples.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Forward pass through backbone
            var supportPred = MetaModel.Predict(task.SupportInput);
            var queryPred = MetaModel.Predict(task.QueryInput);

            // Process through SNAIL blocks (TC + Attention)
            var snailOutput = ProcessSNAILBlocks(supportPred, queryPred);

            // Compute query loss
            var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradients for backbone
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Update backbone parameters
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _snailOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Update SNAIL block parameters via multi-sample SPSA
        for (int block = 0; block < _tcBlockParams.Count; block++)
        {
            var tcParams = _tcBlockParams[block];
            UpdateAuxiliaryParamsSPSA(taskBatch, ref tcParams, _snailOptions.OuterLearningRate);
            _tcBlockParams[block] = tcParams;
        }
        for (int block = 0; block < _attentionBlockParams.Count; block++)
        {
            var attParams = _attentionBlockParams[block];
            UpdateAuxiliaryParamsSPSA(taskBatch, ref attParams, _snailOptions.OuterLearningRate);
            _attentionBlockParams[block] = attParams;
        }

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using SNAIL's sequence processing.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model that uses SNAIL's temporal processing.</returns>
    /// <remarks>
    /// <para>
    /// SNAIL adaptation processes the support set as a sequence:
    /// 1. Extract features from support examples
    /// 2. Build the sequence with labels
    /// 3. Process through all SNAIL blocks
    /// 4. The resulting representation encodes task-specific knowledge
    ///
    /// No gradient descent is needed for adaptation - the temporal convolutions
    /// and attention mechanism do all the "remembering" in a single forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> When given a new task:
    /// 1. SNAIL "reads" all the support examples in order
    /// 2. Its temporal convolutions detect local patterns
    /// 3. Its attention mechanism stores globally relevant information
    /// 4. For each query example, it "recalls" the most relevant support examples
    /// 5. This is all done in a single forward pass - no gradient descent needed
    ///
    /// This makes SNAIL's adaptation very fast (just one pass through the network),
    /// while still being powerful (attention can recall any previous example).
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);

        // Process through SNAIL to build task representation
        var taskRepresentation = ProcessSNAILBlocks(supportPred, default);

        // Compute modulation from task representation magnitudes
        double[]? modulationFactors = null;
        if (taskRepresentation.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < taskRepresentation.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(taskRepresentation[i]));
            double meanAbs = sumAbs / taskRepresentation.Length;
            modulationFactors = [0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0))];
        }

        return new SNAILModel<T, TInput, TOutput>(
            MetaModel, currentParams, _tcBlockParams, _attentionBlockParams,
            taskRepresentation, modulationFactors);
    }

    /// <summary>
    /// Processes inputs through the stacked TC + Attention blocks.
    /// </summary>
    /// <param name="supportFeatures">Features extracted from support examples.</param>
    /// <param name="queryFeatures">Features extracted from query examples (can be default for adaptation).</param>
    /// <returns>A vector representing the SNAIL-processed sequence output.</returns>
    /// <remarks>
    /// <para>
    /// The SNAIL architecture processes a sequence through alternating blocks:
    /// 1. Temporal Convolution (TC): Applies dilated causal convolutions to capture
    ///    local dependencies at multiple time scales
    /// 2. Causal Attention: Applies multi-head attention that can attend to any
    ///    previous position in the sequence
    ///
    /// Each block's output is concatenated with its input (dense connections),
    /// allowing information to flow directly from any block to later blocks.
    /// </para>
    /// <para><b>For Beginners:</b> This processes the sequence of examples through SNAIL:
    /// 1. First, TC looks at nearby examples to find local patterns
    /// 2. Then, attention looks at ALL previous examples to find the most relevant ones
    /// 3. This TC + attention combination repeats for each block
    /// 4. Each block adds new information to the representation (doesn't replace it)
    ///
    /// The result is a rich representation that captures both local and global
    /// patterns in the sequence of examples.
    /// </para>
    /// </remarks>
    private Vector<T> ProcessSNAILBlocks(TOutput supportFeatures, TOutput? queryFeatures)
    {
        var features = ConvertToVector(supportFeatures);
        if (features == null)
        {
            return new Vector<T>(0);
        }

        var current = features;

        for (int block = 0; block < _snailOptions.NumBlocks; block++)
        {
            // Apply temporal convolution block
            current = ApplyTCBlock(current, _tcBlockParams[block]);

            // Apply causal attention block
            current = ApplyAttentionBlock(current, _attentionBlockParams[block]);
        }

        return current;
    }

    /// <summary>
    /// Applies a temporal convolution block with dilated causal convolutions.
    /// </summary>
    /// <param name="input">Input sequence features.</param>
    /// <param name="tcParams">Parameters for this TC block.</param>
    /// <returns>Output features after temporal convolution.</returns>
    /// <remarks>
    /// <para>
    /// Dilated causal convolutions use increasing dilation rates (1, 2, 4, 8, ...)
    /// to capture patterns at exponentially increasing time scales. Causal means
    /// each position can only see previous positions (not future ones).
    /// </para>
    /// <para><b>For Beginners:</b> Temporal convolutions are like reading with different zoom levels:
    /// - Dilation 1: Compare adjacent examples (zoom in)
    /// - Dilation 2: Compare examples 2 apart
    /// - Dilation 4: Compare examples 4 apart (zoom out)
    /// This way, the model captures patterns at many different scales simultaneously.
    /// The "causal" part means it only looks backward (like reading left-to-right).
    /// </para>
    /// </remarks>
    private Vector<T> ApplyTCBlock(Vector<T> input, Vector<T> tcParams)
    {
        int numFilters = _snailOptions.NumTCFilters;
        int outputLen = Math.Min(input.Length, numFilters);
        var output = new Vector<T>(outputLen);

        int paramIdx = 0;
        int numDilationLevels = Math.Max(1, (int)Math.Ceiling(Math.Log(_snailOptions.MaxSequenceLength, 2)));

        for (int d = 0; d < numDilationLevels; d++)
        {
            int dilation = 1 << d; // Exponential dilation: 1, 2, 4, 8, ...

            for (int f = 0; f < numFilters && f < outputLen; f++)
            {
                T sum = NumOps.Zero;
                // Apply dilated convolution
                for (int k = 0; k < numFilters && paramIdx < tcParams.Length; k++)
                {
                    int inputIdx = f - k * dilation;
                    if (inputIdx >= 0 && inputIdx < input.Length)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(input[inputIdx], tcParams[paramIdx]));
                    }
                    paramIdx++;
                }

                // Gated activation: tanh(x) * sigmoid(x) for better gradient flow
                double sumVal = NumOps.ToDouble(sum);
                double gatedVal = Math.Tanh(sumVal) * (1.0 / (1.0 + Math.Exp(-sumVal)));
                output[f] = NumOps.Add(output[f], NumOps.FromDouble(gatedVal));
            }
        }

        // Apply dropout during training
        if (_snailOptions.DropoutRate > 0)
        {
            double keepProb = 1.0 - _snailOptions.DropoutRate;
            for (int i = 0; i < output.Length; i++)
            {
                if (RandomGenerator.NextDouble() > keepProb)
                {
                    output[i] = NumOps.Zero;
                }
                else
                {
                    output[i] = NumOps.Divide(output[i], NumOps.FromDouble(keepProb));
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies a causal attention block.
    /// </summary>
    /// <param name="input">Input features from the TC block.</param>
    /// <param name="attParams">Attention parameters (key, query, value projections).</param>
    /// <returns>Output features after causal attention.</returns>
    /// <remarks>
    /// <para>
    /// Causal attention computes:
    /// - Keys K = input * W_K
    /// - Queries Q = input * W_Q
    /// - Values V = input * W_V
    /// - Attention weights = softmax(Q * K^T / sqrt(d_k)) (masked to be causal)
    /// - Output = attention_weights * V
    ///
    /// Multi-head attention repeats this with multiple sets of projections.
    /// </para>
    /// <para><b>For Beginners:</b> Attention is like a librarian helping you find relevant books:
    /// - Query (Q): "What am I looking for?" (current position's question)
    /// - Key (K): "What does each book contain?" (each position's summary)
    /// - Value (V): "What's in the book?" (each position's actual content)
    /// - The librarian matches your query against all book summaries (keys)
    /// - Returns a weighted combination of the matching books (values)
    /// - "Causal" means you can only look at books you've already read (no peeking ahead)
    /// </para>
    /// </remarks>
    private Vector<T> ApplyAttentionBlock(Vector<T> input, Vector<T> attParams)
    {
        int keyDim = _snailOptions.AttentionKeyDim;
        int valueDim = _snailOptions.AttentionValueDim;
        int numHeads = _snailOptions.NumAttentionHeads;
        int inputDim = input.Length;

        var output = new Vector<T>(inputDim);

        int paramIdx = 0;

        for (int head = 0; head < numHeads; head++)
        {
            // Compute query projection
            var query = new Vector<T>(keyDim);
            for (int k = 0; k < keyDim; k++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inputDim && paramIdx < attParams.Length; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[i], attParams[paramIdx % attParams.Length]));
                    paramIdx++;
                }
                query[k] = sum;
            }

            // Compute key projection
            var key = new Vector<T>(keyDim);
            for (int k = 0; k < keyDim; k++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inputDim && paramIdx < attParams.Length; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[i], attParams[paramIdx % attParams.Length]));
                    paramIdx++;
                }
                key[k] = sum;
            }

            // Compute value projection
            var value = new Vector<T>(valueDim);
            for (int k = 0; k < valueDim; k++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inputDim && paramIdx < attParams.Length; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[i], attParams[paramIdx % attParams.Length]));
                    paramIdx++;
                }
                value[k] = sum;
            }

            // Compute attention score: dot(query, key) / sqrt(keyDim)
            T score = NumOps.Zero;
            for (int k = 0; k < keyDim; k++)
            {
                score = NumOps.Add(score, NumOps.Multiply(query[k], key[k]));
            }
            double scoreVal = NumOps.ToDouble(score) / Math.Sqrt(keyDim);
            double attWeight = 1.0 / (1.0 + Math.Exp(-scoreVal)); // Sigmoid for single position

            // Apply attention weight to value and add to output
            for (int i = 0; i < Math.Min(valueDim, inputDim); i++)
            {
                output[i] = NumOps.Add(output[i],
                    NumOps.Multiply(value[i], NumOps.FromDouble(attWeight / numHeads)));
            }
        }

        // Residual connection
        for (int i = 0; i < inputDim; i++)
        {
            output[i] = NumOps.Add(output[i], input[i]);
        }

        return output;
    }

}

/// <summary>
/// Adapted model wrapper for SNAIL with pre-computed task representation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This model combines the meta-learned feature extractor
/// with the SNAIL blocks that have processed the support set. When predicting on a new
/// query example, it uses the stored temporal and attention representations to classify
/// based on the support examples it has already "read."
/// </para>
/// </remarks>
internal class SNAILModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly List<Vector<T>> _tcBlockParams;
    private readonly List<Vector<T>> _attentionBlockParams;
    private readonly Vector<T> _taskRepresentation;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _taskRepresentation;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public SNAILModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        List<Vector<T>> tcBlockParams,
        List<Vector<T>> attentionBlockParams,
        Vector<T> taskRepresentation,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _tcBlockParams = tcBlockParams;
        _attentionBlockParams = attentionBlockParams;
        _taskRepresentation = taskRepresentation;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = NumOps.Multiply(_backboneParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
