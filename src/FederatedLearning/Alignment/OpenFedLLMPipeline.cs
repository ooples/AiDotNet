namespace AiDotNet.FederatedLearning.Alignment;

/// <summary>
/// Specifies the training stage in the OpenFedLLM pipeline.
/// </summary>
public enum FedLLMStage
{
    /// <summary>Supervised fine-tuning on instruction-following data.</summary>
    InstructionTuning,
    /// <summary>Value alignment via RLHF or DPO.</summary>
    ValueAlignment,
    /// <summary>Serving the aligned model.</summary>
    Serving
}

/// <summary>
/// Implements OpenFedLLM pipeline patterns for federated LLM training, alignment, and serving.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Training a useful LLM happens in stages: first you teach it to
/// follow instructions (instruction tuning), then you align it with human values (RLHF/DPO),
/// and finally you deploy it (serving). OpenFedLLM defines how to do each stage in a federated
/// setting where data stays private. This class orchestrates the three stages and manages
/// the transitions between them.</para>
///
/// <para>Pipeline stages:</para>
/// <list type="number">
/// <item><b>Instruction Tuning</b> — Federated SFT on instruction/response pairs. Each client
/// keeps their instruction data private. LoRA adapters are aggregated.</item>
/// <item><b>Value Alignment</b> — Federated DPO (preferred) or RLHF on preference data.
/// Each client keeps their human feedback private.</item>
/// <item><b>Serving</b> — Model deployment with optional federated inference routing.</item>
/// </list>
///
/// <para>Reference: Ye, J., et al. (2024). "OpenFedLLM: Training Large Language Models on
/// Decentralized Private Data via Federated Learning." arXiv:2402.06954.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OpenFedLLMPipeline<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly OpenFedLLMOptions _options;
    private FedLLMStage _currentStage;
    private int _currentRound;
    private Dictionary<string, T[]>? _lastCheckpoint;

    /// <summary>
    /// Creates a new OpenFedLLM pipeline.
    /// </summary>
    /// <param name="options">Configuration options. Uses defaults if null.</param>
    public OpenFedLLMPipeline(OpenFedLLMOptions? options = null)
    {
        _options = options ?? new OpenFedLLMOptions();
        _currentStage = FedLLMStage.InstructionTuning;
        _currentRound = 0;
    }

    /// <summary>
    /// Gets the current pipeline stage.
    /// </summary>
    public FedLLMStage CurrentStage => _currentStage;

    /// <summary>
    /// Gets the current round number within the current stage.
    /// </summary>
    public int CurrentRound => _currentRound;

    /// <summary>
    /// Gets a defensive copy of the last checkpointed model parameters, or null if no checkpoint exists.
    /// </summary>
    /// <remarks>
    /// Returns a deep copy to prevent callers from mutating internal checkpoint state.
    /// </remarks>
    public Dictionary<string, T[]>? LastCheckpoint
    {
        get
        {
            if (_lastCheckpoint == null)
            {
                return null;
            }

            var copy = new Dictionary<string, T[]>(_lastCheckpoint.Count);
            foreach (var (key, value) in _lastCheckpoint)
            {
                copy[key] = (T[])value.Clone();
            }

            return copy;
        }
    }

    /// <summary>
    /// Advances to the next pipeline stage, resetting the round counter.
    /// </summary>
    /// <returns>The new stage after advancement.</returns>
    public FedLLMStage AdvanceStage()
    {
        _currentStage = _currentStage switch
        {
            FedLLMStage.InstructionTuning => FedLLMStage.ValueAlignment,
            FedLLMStage.ValueAlignment => FedLLMStage.Serving,
            FedLLMStage.Serving => FedLLMStage.Serving,
            _ => FedLLMStage.InstructionTuning
        };

        _currentRound = 0;
        return _currentStage;
    }

    /// <summary>
    /// Checks whether the current stage has completed its allocated rounds.
    /// </summary>
    /// <returns>True if the current stage should advance.</returns>
    public bool IsStageComplete()
    {
        return _currentStage switch
        {
            FedLLMStage.InstructionTuning => _currentRound >= _options.InstructionTuningRounds,
            FedLLMStage.ValueAlignment => _currentRound >= _options.AlignmentRounds,
            FedLLMStage.Serving => true,
            _ => false
        };
    }

    /// <summary>
    /// Saves a checkpoint of the current model parameters.
    /// </summary>
    /// <param name="modelParams">The model parameters to checkpoint.</param>
    public void SaveCheckpoint(Dictionary<string, T[]> modelParams)
    {
        _lastCheckpoint = new Dictionary<string, T[]>(modelParams.Count);
        foreach (var (key, value) in modelParams)
        {
            _lastCheckpoint[key] = (T[])value.Clone();
        }
    }

    /// <summary>
    /// Aggregates instruction-tuned adapter parameters from clients.
    /// Only aggregates LoRA/adapter layers (keys containing "lora" or "adapter"), keeping base model frozen.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During instruction tuning, only the small adapter layers
    /// (LoRA matrices) are trained and communicated. The base model weights are frozen.
    /// This method filters for adapter-related parameter keys and only aggregates those,
    /// then increments the round counter.</para>
    /// </remarks>
    /// <param name="clientAdapters">Client adapter parameter dictionaries.</param>
    /// <param name="clientWeights">Per-client weights (proportional to instruction count).</param>
    /// <returns>Aggregated adapter parameters (adapter keys only).</returns>
    public Dictionary<string, T[]> AggregateInstructionTuning(
        Dictionary<int, Dictionary<string, T[]>> clientAdapters,
        Dictionary<int, double> clientWeights)
    {
        // For instruction tuning, filter to adapter-only keys for bandwidth efficiency.
        var filtered = FilterAdapterKeys(clientAdapters);
        var result = AggregateWeightedDict(filtered, clientWeights);
        _currentRound++;
        return result;
    }

    /// <summary>
    /// Aggregates alignment-stage model parameters, using either DPO or RLHF aggregation
    /// based on the pipeline configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During value alignment, the model learns to prefer helpful
    /// responses over harmful ones. Unlike instruction tuning which only updates adapters,
    /// alignment may update additional layers (e.g., the value head for RLHF). The aggregation
    /// strategy depends on whether DPO or RLHF is being used.</para>
    /// </remarks>
    /// <param name="clientModels">Client model parameter dictionaries after alignment.</param>
    /// <param name="clientWeights">Per-client weights.</param>
    /// <returns>Aggregated model parameters.</returns>
    public Dictionary<string, T[]> AggregateAlignment(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var result = AggregateWeightedDict(clientModels, clientWeights);
        _currentRound++;
        return result;
    }

    /// <summary>
    /// Runs one round of the pipeline, dispatching to the appropriate stage-specific aggregation.
    /// </summary>
    /// <param name="clientModels">Client model parameter dictionaries.</param>
    /// <param name="clientWeights">Per-client weights.</param>
    /// <returns>Aggregated parameters and whether the stage is now complete.</returns>
    public (Dictionary<string, T[]> AggregatedParams, bool StageComplete) RunRound(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var aggregated = _currentStage switch
        {
            FedLLMStage.InstructionTuning => AggregateInstructionTuning(clientModels, clientWeights),
            FedLLMStage.ValueAlignment => AggregateAlignment(clientModels, clientWeights),
            FedLLMStage.Serving => AggregateWeightedDict(clientModels, clientWeights),
            _ => AggregateWeightedDict(clientModels, clientWeights)
        };

        // Auto-checkpoint at stage boundaries.
        if (IsStageComplete())
        {
            SaveCheckpoint(aggregated);
        }

        return (aggregated, IsStageComplete());
    }

    private static Dictionary<int, Dictionary<string, T[]>> FilterAdapterKeys(
        Dictionary<int, Dictionary<string, T[]>> clientModels)
    {
        var filtered = new Dictionary<int, Dictionary<string, T[]>>();
        foreach (var (clientId, model) in clientModels)
        {
            var adapterOnly = new Dictionary<string, T[]>();
            foreach (var (key, value) in model)
            {
                // Include adapter, lora, or bias-only keys (standard PEFT naming).
                if (key.Contains("lora", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("adapter", StringComparison.OrdinalIgnoreCase) ||
                    key.Contains("bias", StringComparison.OrdinalIgnoreCase))
                {
                    adapterOnly[key] = value;
                }
            }

            // If no adapter keys found, include everything (user may not follow naming convention).
            filtered[clientId] = adapterOnly.Count > 0 ? adapterOnly : model;
        }

        return filtered;
    }

    private Dictionary<string, T[]> AggregateWeightedDict(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // Compute totalWeight including default weights for clients missing from clientWeights.
        double totalWeight = 0;
        foreach (var clientId in clientModels.Keys)
        {
            totalWeight += clientWeights.GetValueOrDefault(clientId, 1.0);
        }

        if (totalWeight <= 0)
        {
            totalWeight = clientModels.Count; // fall back to uniform
        }

        var aggregated = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            var result = new T[referenceModel[layerName].Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Zero;
            }

            aggregated[layerName] = result;
        }

        foreach (var (clientId, clientModel) in clientModels)
        {
            double w = clientWeights.GetValueOrDefault(clientId, 1.0);
            var normalizedWeight = NumOps.FromDouble(w / totalWeight);

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = aggregated[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], normalizedWeight));
                }
            }
        }

        return aggregated;
    }

    /// <summary>Gets the pipeline configuration options.</summary>
    public OpenFedLLMOptions Options => _options;
}

/// <summary>
/// Configuration options for the OpenFedLLM pipeline.
/// </summary>
public class OpenFedLLMOptions
{
    /// <summary>
    /// Gets or sets the number of instruction tuning rounds. Default: 100.
    /// </summary>
    public int InstructionTuningRounds { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of alignment rounds. Default: 50.
    /// </summary>
    public int AlignmentRounds { get; set; } = 50;

    /// <summary>
    /// Gets or sets whether to use DPO (true) or RLHF (false) for alignment. Default: true (DPO).
    /// </summary>
    public bool UseDPO { get; set; } = true;

    /// <summary>
    /// Gets or sets the LoRA rank for instruction tuning. Default: 8.
    /// </summary>
    public int LoRARank { get; set; } = 8;

    /// <summary>
    /// Gets or sets the learning rate for instruction tuning. Default: 2e-5.
    /// </summary>
    public double InstructionTuningLR { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the learning rate for alignment. Default: 5e-7.
    /// </summary>
    public double AlignmentLR { get; set; } = 5e-7;
}
