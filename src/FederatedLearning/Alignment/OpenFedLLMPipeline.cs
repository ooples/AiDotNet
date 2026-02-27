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

    /// <summary>
    /// Creates a new OpenFedLLM pipeline.
    /// </summary>
    /// <param name="options">Configuration options. Uses defaults if null.</param>
    public OpenFedLLMPipeline(OpenFedLLMOptions? options = null)
    {
        _options = options ?? new OpenFedLLMOptions();
        _currentStage = FedLLMStage.InstructionTuning;
    }

    /// <summary>
    /// Gets the current pipeline stage.
    /// </summary>
    public FedLLMStage CurrentStage => _currentStage;

    /// <summary>
    /// Advances to the next pipeline stage.
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

        return _currentStage;
    }

    /// <summary>
    /// Aggregates instruction-tuned adapter parameters from clients.
    /// </summary>
    /// <param name="clientAdapters">Client adapter parameter dictionaries.</param>
    /// <param name="clientWeights">Per-client weights (proportional to instruction count).</param>
    /// <returns>Aggregated adapter parameters.</returns>
    public Dictionary<string, T[]> AggregateInstructionTuning(
        Dictionary<int, Dictionary<string, T[]>> clientAdapters,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedDict(clientAdapters, clientWeights);
    }

    /// <summary>
    /// Aggregates alignment-stage model parameters.
    /// </summary>
    /// <param name="clientModels">Client model parameter dictionaries after alignment.</param>
    /// <param name="clientWeights">Per-client weights.</param>
    /// <returns>Aggregated model parameters.</returns>
    public Dictionary<string, T[]> AggregateAlignment(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedDict(clientModels, clientWeights);
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
        double totalWeight = clientWeights.Values.Sum();

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
