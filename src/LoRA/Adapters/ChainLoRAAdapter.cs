using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Chain-of-LoRA adapter that implements sequential composition of multiple LoRA adapters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Chain-of-LoRA (COLA) is an advanced LoRA technique that enables sequential composition
/// of multiple LoRA adaptations through an iterative optimization framework. Unlike standard
/// LoRA which applies a single low-rank adaptation, COLA builds a chain of adaptations where
/// each adapter is trained, merged into the model, and then a new adapter is initialized for
/// further refinement.
/// </para>
/// <para>
/// This approach bridges the performance gap between standard LoRA and full fine-tuning by
/// employing residual learning principles. Each iteration in the chain adds incremental
/// improvements to the model's task-specific performance without incurring additional
/// computational costs or memory overhead during inference.
/// </para>
/// <para><b>Key Concepts:</b>
///
/// <b>Sequential Adaptation:</b>
/// Chain-of-LoRA applies adaptations in sequence (Task A → Task B → Task C), where each
/// stage builds upon the previous one. This is inspired by the Frank-Wolfe optimization
/// algorithm, which makes greedy updates along the direction of maximum improvement.
///
/// <b>Merge and Re-initialize:</b>
/// After training each LoRA adapter, the learned weights are merged back into the base layer,
/// and a new LoRA adapter is initialized. This "tying a knot" process allows the model to
/// consolidate learned knowledge before adding new adaptations.
///
/// <b>Knowledge Preservation:</b>
/// By freezing the base layer and only training the LoRA components, the chain preserves
/// previously learned knowledge while allowing new task-specific adaptations. Each adapter
/// in the chain captures a specific aspect of the task or a refinement step.
///
/// <b>Incremental Fine-tuning Pipeline:</b>
/// COLA enables continual learning scenarios where tasks are presented sequentially, and
/// the model must adapt to new tasks while maintaining performance on previous ones.
/// </para>
/// <para><b>Benefits of Chain-of-LoRA:</b>
///
/// - <b>Better Performance:</b> Achieves up to 6.47% relative accuracy gain over standard LoRA
/// - <b>No Extra Overhead:</b> After merging, inference cost is identical to the base model
/// - <b>Modular Adaptation:</b> Each adapter can be trained, tested, and validated independently
/// - <b>Catastrophic Forgetting Mitigation:</b> Sequential merging helps preserve prior knowledge
/// - <b>Task Chaining:</b> Naturally supports multi-task learning and transfer learning scenarios
/// - <b>Flexible Deployment:</b> Can deploy the full chain or selected adapters as needed
/// </para>
/// <para><b>For Beginners:</b>
///
/// Imagine you're learning a complex skill in stages:
/// 1. First, you learn the basics (Adapter 1)
/// 2. Then you practice and the basics become automatic (Merge)
/// 3. Next, you learn intermediate techniques on top of the basics (Adapter 2)
/// 4. Again, you practice until they're automatic (Merge)
/// 5. Finally, you learn advanced skills building on everything before (Adapter 3)
///
/// Chain-of-LoRA works the same way: each adapter learns something new, then it's consolidated
/// into the model, and the next adapter can focus on the next refinement. This stepwise approach
/// often achieves better results than trying to learn everything at once.
/// </para>
/// <para><b>Research Reference:</b>
///
/// Based on "Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning"
/// (arXiv:2401.04151, January 2024). The paper demonstrates that sequential low-rank adaptations
/// can significantly improve task performance compared to single-stage LoRA, especially on
/// complex reasoning and multi-step tasks.
/// </para>
/// <para><b>Usage Example:</b>
/// <code>
/// // Create a chain with 3 sequential adaptations
/// var chain = new ChainLoRAAdapter&lt;double&gt;(baseLayer, rank: 8, chainLength: 3);
///
/// // Train first adapter on Task A
/// chain.SetActiveAdapterIndex(0);
/// TrainModel(chain, taskAData);
/// chain.FreezeActiveAdapter(); // Freeze Task A adapter
///
/// // Train second adapter on Task B
/// chain.SetActiveAdapterIndex(1);
/// TrainModel(chain, taskBData);
/// chain.FreezeActiveAdapter(); // Freeze Task B adapter
///
/// // Train third adapter on Task C
/// chain.SetActiveAdapterIndex(2);
/// TrainModel(chain, taskCData);
///
/// // Deploy: merge all adapters into base layer for optimized inference
/// ILayer&lt;double&gt; finalLayer = chain.MergeToOriginalLayer();
/// </code>
/// </para>
/// </remarks>
public class ChainLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// The chain of LoRA adapters applied sequentially.
    /// </summary>
    private readonly List<LoRALayer<T>> _adapterChain;

    /// <summary>
    /// The index of the currently active adapter being trained.
    /// </summary>
    private int _activeAdapterIndex;

    /// <summary>
    /// Whether each adapter in the chain has been merged.
    /// </summary>
    private readonly List<bool> _mergedStatus;

    /// <summary>
    /// The total length of the adapter chain.
    /// </summary>
    private readonly int _chainLength;

    /// <summary>
    /// Cached parameter count reflecting current chain state.
    /// </summary>
    /// <remarks>
    /// This field is updated whenever adapters are merged/unmerged to avoid
    /// recomputing the count on every access and to provide a stable value
    /// during base class construction before the chain is fully initialized.
    /// </remarks>
    private int _currentParameterCount;

    /// <summary>
    /// Gets the total number of adapters in the chain.
    /// </summary>
    /// <remarks>
    /// This represents the maximum number of sequential adaptation stages that can be applied.
    /// Each adapter can be trained independently and then merged before proceeding to the next.
    /// </remarks>
    public int ChainLength => _chainLength;

    /// <summary>
    /// Gets the index of the currently active adapter (0-based).
    /// </summary>
    /// <remarks>
    /// The active adapter is the one currently being trained. Other adapters in the chain
    /// are either waiting to be trained (higher indices) or have been merged (lower indices).
    /// </remarks>
    public int ActiveAdapterIndex => _activeAdapterIndex;

    /// <summary>
    /// Gets the list of LoRA adapters in the chain.
    /// </summary>
    /// <remarks>
    /// Each adapter in the chain represents one stage of sequential adaptation.
    /// Adapters are applied in order during forward passes.
    /// </remarks>
    public IReadOnlyList<LoRALayer<T>> AdapterChain => _adapterChain.AsReadOnly();

    /// <summary>
    /// Gets the frozen status of each adapter in the chain.
    /// </summary>
    /// <remarks>
    /// True indicates that an adapter has been frozen and should no longer contribute
    /// trainable parameters. Frozen adapters still contribute to forward/backward passes
    /// until the entire chain is merged via MergeToOriginalLayer().
    /// </remarks>
    public IReadOnlyList<bool> FrozenStatus => _mergedStatus.AsReadOnly();

    /// <summary>
    /// Initializes a new Chain-of-LoRA adapter with the specified configuration.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with the LoRA chain.</param>
    /// <param name="rank">The rank of each LoRA decomposition in the chain.</param>
    /// <param name="chainLength">The number of sequential adapters in the chain (default: 3).</param>
    /// <param name="alpha">The LoRA scaling factor for each adapter (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when chainLength is less than 1.</exception>
    /// <remarks>
    /// <para>
    /// Creates a chain of LoRA adapters for sequential fine-tuning. Each adapter in the chain
    /// can be trained independently, merged into the model, and then the next adapter can be
    /// activated for further refinement.
    /// </para>
    /// <para><b>For Beginners:</b>
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt (e.g., a dense or convolutional layer)
    /// - rank: How compressed each adapter is (lower = fewer parameters per stage)
    /// - chainLength: How many sequential adaptation stages you want (typical: 2-5)
    /// - alpha: Controls adaptation strength (usually equals rank)
    /// - freezeBaseLayer: Lock base weights to preserve pre-trained knowledge (recommended: true)
    ///
    /// Example: chainLength=3 means you can do three rounds of training and merging,
    /// allowing the model to incrementally improve on complex tasks.
    /// </para>
    /// </remarks>
    public ChainLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        int chainLength = 3,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (chainLength < 1)
        {
            throw new ArgumentException("Chain length must be at least 1", nameof(chainLength));
        }

        _chainLength = chainLength;
        _activeAdapterIndex = 0;
        _adapterChain = new List<LoRALayer<T>>(chainLength);
        _mergedStatus = new List<bool>(chainLength);

        // Create the chain of LoRA adapters
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        for (int i = 0; i < chainLength; i++)
        {
            var adapter = new LoRALayer<T>(inputSize, outputSize, rank, alpha);
            _adapterChain.Add(adapter);
            _mergedStatus.Add(false);
        }

        // Update parameter count to reflect all unmerged adapters
        UpdateParameterCount();
    }

    /// <summary>
    /// Sets which adapter in the chain is currently active for training.
    /// </summary>
    /// <param name="index">The 0-based index of the adapter to activate.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// Only the active adapter receives gradient updates during training. Other adapters
    /// are either frozen (already merged) or inactive (waiting to be trained).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like choosing which stage of learning you're currently working on.
    /// Set to 0 for the first stage, 1 for the second, etc. Only that stage's adapter
    /// will be trained while the others remain frozen.
    /// </para>
    /// </remarks>
    public void SetActiveAdapterIndex(int index)
    {
        if (index < 0 || index >= _chainLength)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {_chainLength - 1}");
        }

        _activeAdapterIndex = index;
    }

    /// <summary>
    /// Freezes the currently active adapter to prevent further training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This "ties a knot" in the chain by marking the active adapter as frozen.
    /// The adapter continues to contribute to forward passes but will no longer receive
    /// gradient updates, allowing the next adapter in the chain to build upon this
    /// consolidated knowledge.
    /// </para>
    /// <para>
    /// IMPORTANT: This method does NOT merge weights into the base layer. All adapters
    /// (frozen or not) remain active during forward/backward passes. True weight merging
    /// only occurs when MergeToOriginalLayer() is called at the end of training.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After training an adapter stage, call this to "lock it in" before moving to the
    /// next stage. The adapter's learned knowledge is preserved and it stops training,
    /// but it still contributes to the model's output. Think of it like finishing one
    /// chapter before starting the next - the previous chapter's knowledge remains active.
    /// </para>
    /// </remarks>
    public void FreezeActiveAdapter()
    {
        if (_activeAdapterIndex < 0 || _activeAdapterIndex >= _chainLength)
        {
            throw new InvalidOperationException($"Invalid active adapter index: {_activeAdapterIndex}");
        }

        _mergedStatus[_activeAdapterIndex] = true;
        UpdateParameterCount();
    }

    /// <summary>
    /// Unfreezes a previously frozen adapter, making it trainable again.
    /// </summary>
    /// <param name="index">The index of the adapter to unfreeze.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    /// <remarks>
    /// <para>
    /// This allows re-training a previously frozen adapter if needed for iterative refinement.
    /// Useful for scenarios where you want to go back and adjust an earlier stage.
    /// </para>
    /// </remarks>
    public void UnfreezeAdapter(int index)
    {
        if (index < 0 || index >= _chainLength)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {_chainLength - 1}");
        }

        _mergedStatus[index] = false;
        UpdateParameterCount();
    }

    /// <summary>
    /// Gets the number of adapters that have been frozen.
    /// </summary>
    /// <returns>Count of frozen adapters.</returns>
    public int GetFrozenCount()
    {
        return _mergedStatus.Count(frozen => frozen);
    }

    /// <summary>
    /// Gets the total number of parameters in the chain (base layer + all unfrozen adapters).
    /// </summary>
    /// <remarks>
    /// This count includes parameters from the base layer (if not frozen) plus all unfrozen adapters in the chain.
    /// Frozen adapters don't contribute to the parameter count since they no longer receive gradient updates.
    /// Returns the cached _currentParameterCount once the chain is initialized, or computes it on-the-fly
    /// during construction to handle base class initialization.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // If chain is not yet initialized (during base construction), compute on-the-fly
            if (_adapterChain == null || _currentParameterCount == 0)
            {
                int count = 0;

                // Add base layer parameters if not frozen and baseLayer exists
                if (_baseLayer != null && !_freezeBaseLayer)
                {
                    count += _baseLayer.ParameterCount;
                }

                // Add unmerged adapter parameters from chain
                if (_adapterChain != null && _mergedStatus != null)
                {
                    for (int i = 0; i < _chainLength; i++)
                    {
                        if (!_mergedStatus[i])
                        {
                            count += _adapterChain[i].ParameterCount;
                        }
                    }
                }

                return count;
            }

            // Otherwise return cached value
            return _currentParameterCount;
        }
    }

    /// <summary>
    /// Gets the number of adapters that are still trainable (not frozen).
    /// </summary>
    /// <returns>Count of unfrozen adapters.</returns>
    public int GetTrainableAdapterCount()
    {
        return _mergedStatus.Count(frozen => !frozen);
    }

    /// <summary>
    /// Performs the forward pass through the base layer and all adapters in the chain.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output with all adapter contributions summed.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    /// output = base_layer(input) + adapter_0(input) + adapter_1(input) + ... + adapter_n(input)
    /// </para>
    /// <para>
    /// IMPORTANT: All adapters contribute to the output, regardless of frozen status.
    /// Frozen adapters continue to be computed in every forward pass. They are only
    /// "frozen" in the sense that they don't receive gradient updates during training.
    /// True inference optimization (eliminating frozen adapter computation) only occurs
    /// after calling MergeToOriginalLayer().
    /// </para>
    /// <para><b>For Beginners:</b>
    /// During inference or training, the input goes through the base layer and ALL adapters
    /// in the chain (both frozen and unfrozen). Their outputs are added together to get the
    /// final result. Freezing an adapter stops it from training, but it still contributes
    /// to every prediction.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> result = _baseLayer.Forward(input);

        // Forward through each adapter in the chain and sum contributions
        foreach (var adapter in _adapterChain)
        {
            Tensor<T> adapterOutput = adapter.Forward(input);

            // Add adapter contribution to result
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Add(result[i], adapterOutput[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through all layers in the chain.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// Gradients flow through all adapters and the base layer. Only unfrozen adapters
    /// and the base layer (if not frozen) receive parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// During learning, this figures out how to improve each adapter. Only the active,
    /// unfrozen adapter gets updated - the frozen ones preserve their learned knowledge.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Initialize input gradient accumulator
        Tensor<T> inputGrad = new Tensor<T>(GetInputShape());

        // Backward through each adapter in the chain
        for (int i = 0; i < _adapterChain.Count; i++)
        {
            Tensor<T> adapterInputGrad = _adapterChain[i].Backward(outputGradient);

            // Accumulate input gradients
            for (int j = 0; j < inputGrad.Length; j++)
            {
                inputGrad[j] = NumOps.Add(inputGrad[j], adapterInputGrad[j]);
            }
        }

        // ALWAYS backward through base layer to get input gradients
        // Even when frozen, we need the base layer's Jacobian to propagate gradients to input
        // Freezing only prevents parameter updates, not gradient computation
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Accumulate base layer gradients
        for (int j = 0; j < inputGrad.Length; j++)
        {
            inputGrad[j] = NumOps.Add(inputGrad[j], baseInputGrad[j]);
        }

        // Update parameter gradients
        UpdateParameterGradientsFromChain();

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// Only the active unfrozen adapter receives updates. Frozen adapters and the base layer
    /// (if frozen) do not receive parameter updates.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update base layer only if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update only the active unfrozen adapter
        if (_activeAdapterIndex >= 0 && _activeAdapterIndex < _chainLength && !_mergedStatus[_activeAdapterIndex])
        {
            _adapterChain[_activeAdapterIndex].UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromChain();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters from base layer (if not frozen) and all unfrozen adapters.</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters.</param>
    /// <exception cref="ArgumentException">Thrown when parameter count doesn't match.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateChainFromParameters();
    }

    /// <summary>
    /// Merges all adapters in the chain into the original base layer.
    /// </summary>
    /// <returns>A new layer with all LoRA adaptations merged into the base weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This creates a single layer that includes all the sequential adaptations from the chain.
    /// The resulting layer has the same computational cost as the original base layer but
    /// includes all the learned improvements from each stage of the chain.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After training all stages of the chain, call this to create a final optimized layer.
    /// The result is a regular layer (no LoRA overhead) that performs as well as the full chain.
    /// Perfect for deployment when you want maximum speed with all the learned adaptations.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("ChainLoRAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create merged parameters starting with base layer weights and biases
        Vector<T> mergedParams = baseParams.Clone();

        // Merge each adapter in the chain sequentially
        foreach (var adapter in _adapterChain)
        {
            // Get the merged weights from this adapter (B × A, already scaled by alpha/rank)
            Matrix<T> adapterWeights = adapter.MergeWeights();

            // Add this adapter's contribution to the merged weights
            // DenseLayer uses [inputSize, outputSize] convention
            for (int i = 0; i < weightCount; i++)
            {
                int row = i / outputSize;
                int col = i % outputSize;
                mergedParams[i] = NumOps.Add(mergedParams[i], adapterWeights[row, col]);
            }
            // Note: Biases remain unchanged (indices weightCount to end)
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of the base layer and all adapters in the chain.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        foreach (var adapter in _adapterChain)
        {
            adapter.ResetState();
        }
    }

    /// <summary>
    /// Updates the parameter count based on current frozen status.
    /// </summary>
    private void UpdateParameterCount()
    {
        int count = 0;

        // Add base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            count += _baseLayer.ParameterCount;
        }

        // Add unfrozen adapter parameters
        for (int i = 0; i < _chainLength; i++)
        {
            if (!_mergedStatus[i])
            {
                count += _adapterChain[i].ParameterCount;
            }
        }

        // Update cached parameter count
        _currentParameterCount = count;

        // Reallocate parameter vectors with new size
        Parameters = new Vector<T>(count);
        ParameterGradients = new Vector<T>(count);
    }

    /// <summary>
    /// Updates the parameter vector from the current state of the chain.
    /// </summary>
    private void UpdateParametersFromChain()
    {
        int idx = 0;

        // Pack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack unfrozen adapter parameters
        for (int i = 0; i < _chainLength; i++)
        {
            if (!_mergedStatus[i])
            {
                Vector<T> adapterParams = _adapterChain[i].GetParameters();
                for (int j = 0; j < adapterParams.Length; j++)
                {
                    Parameters[idx++] = adapterParams[j];
                }
            }
        }
    }

    /// <summary>
    /// Updates the chain from the parameter vector.
    /// </summary>
    private void UpdateChainFromParameters()
    {
        int idx = 0;

        // Unpack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = Parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack unfrozen adapter parameters
        for (int i = 0; i < _chainLength; i++)
        {
            if (!_mergedStatus[i])
            {
                int adapterParamCount = _adapterChain[i].ParameterCount;
                Vector<T> adapterParams = new Vector<T>(adapterParamCount);
                for (int j = 0; j < adapterParamCount; j++)
                {
                    adapterParams[j] = Parameters[idx++];
                }
                _adapterChain[i].SetParameters(adapterParams);
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the chain gradients.
    /// </summary>
    private void UpdateParameterGradientsFromChain()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer gradients if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack unfrozen adapter gradients
        for (int i = 0; i < _chainLength; i++)
        {
            if (!_mergedStatus[i])
            {
                Vector<T> adapterGrads = _adapterChain[i].GetParameterGradients();
                for (int j = 0; j < adapterGrads.Length; j++)
                {
                    ParameterGradients[idx++] = adapterGrads[j];
                }
            }
        }
    }
}
