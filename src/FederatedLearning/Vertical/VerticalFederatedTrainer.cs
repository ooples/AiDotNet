using System.Diagnostics;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.FederatedLearning.PSI;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Main orchestrator for vertical federated learning training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the central coordinator that manages the entire VFL
/// training pipeline. It brings together all the components:</para>
/// <list type="number">
/// <item><description><b>Entity alignment:</b> Uses PSI to find shared entities across parties.</description></item>
/// <item><description><b>Split training:</b> Coordinates forward/backward passes across the split network.</description></item>
/// <item><description><b>Gradient protection:</b> Optionally encrypts gradients and protects labels.</description></item>
/// <item><description><b>Missing features:</b> Handles entities not available in all parties.</description></item>
/// </list>
///
/// <para><b>Training loop per batch:</b></para>
/// <list type="number">
/// <item><description>Each party computes its bottom model forward pass locally.</description></item>
/// <item><description>Embeddings are sent to the coordinator (optionally encrypted).</description></item>
/// <item><description>Coordinator aggregates embeddings and runs the top model.</description></item>
/// <item><description>Label holder computes loss and gradient.</description></item>
/// <item><description>Coordinator splits gradient back to parties (optionally with DP noise).</description></item>
/// <item><description>Each party updates its bottom model using received gradients.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Based on the FATE framework architecture and VFLAIR (ICLR 2025).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalFederatedTrainer<T> : FederatedLearningComponentBase<T>, IVerticalFederatedTrainer<T>
{
    private readonly VerticalFederatedLearningOptions _options;
    private readonly List<IVerticalParty<T>> _parties;
    private readonly MissingFeatureHandler<T> _missingFeatureHandler;
    private readonly SecureGradientExchange<T>? _gradientExchange;
    private readonly ILabelProtector<T>? _labelProtector;

    private SplitNeuralNetwork<T>? _splitModel;
    private EntityAligner? _entityAligner;
    private VflAlignmentSummary? _alignmentSummary;

    // Alignment indices: for each party, the local row indices of aligned entities
    private Dictionary<string, IReadOnlyList<int>>? _alignedIndicesPerParty;
    private int _alignedEntityCount;
    private bool _isAligned;
    private bool _isTrained;

    /// <summary>
    /// Initializes a new instance of <see cref="VerticalFederatedTrainer{T}"/>.
    /// </summary>
    /// <param name="options">VFL training configuration.</param>
    /// <param name="labelProtector">Optional label privacy protector. If null and label DP is enabled,
    /// a default <see cref="LabelDifferentialPrivacy{T}"/> is created.</param>
    public VerticalFederatedTrainer(
        VerticalFederatedLearningOptions? options = null,
        ILabelProtector<T>? labelProtector = null)
    {
        _options = options ?? new VerticalFederatedLearningOptions();
        _parties = new List<IVerticalParty<T>>();
        _missingFeatureHandler = new MissingFeatureHandler<T>(_options.MissingFeatures);

        if (_options.EncryptGradients)
        {
            _gradientExchange = new SecureGradientExchange<T>(true, _options.RandomSeed);
        }

        if (labelProtector is not null)
        {
            _labelProtector = labelProtector;
        }
        else if (_options.EnableLabelDifferentialPrivacy)
        {
            _labelProtector = new LabelDifferentialPrivacy<T>(
                _options.LabelDpEpsilon, _options.LabelDpDelta, 1.0, _options.RandomSeed);
        }
    }

    /// <inheritdoc/>
    public void RegisterParty(IVerticalParty<T> party)
    {
        if (party is null)
        {
            throw new ArgumentNullException(nameof(party));
        }

        if (_isAligned)
        {
            throw new InvalidOperationException(
                "Cannot register parties after alignment. Create a new trainer.");
        }

        _parties.Add(party);
    }

    /// <inheritdoc/>
    public VflAlignmentSummary AlignEntities(PsiOptions? psiOptions = null)
    {
        if (_parties.Count < 2)
        {
            throw new InvalidOperationException("At least 2 parties must be registered before alignment.");
        }

        var stopwatch = Stopwatch.StartNew();
        var effectivePsiOptions = psiOptions ?? _options.EntityAlignment;

        _entityAligner = new EntityAligner();
        _alignedIndicesPerParty = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal);

        // For two-party case, use direct alignment
        // For multi-party, use iterative pairwise alignment
        if (_parties.Count == 2)
        {
            var result = _entityAligner.AlignEntities(
                _parties[0].GetEntityIds(), _parties[1].GetEntityIds(), effectivePsiOptions);

            var psiResult = result.PsiResult;
            _alignedEntityCount = psiResult.IntersectionSize;

            // Build per-party aligned indices from PSI result
            var party0Indices = new List<int>(_alignedEntityCount);
            var party1Indices = new List<int>(_alignedEntityCount);

            for (int sharedIdx = 0; sharedIdx < _alignedEntityCount; sharedIdx++)
            {
                foreach (var kvp in psiResult.LocalToSharedIndexMap)
                {
                    if (kvp.Value == sharedIdx)
                    {
                        party0Indices.Add(kvp.Key);
                        break;
                    }
                }

                foreach (var kvp in psiResult.RemoteToSharedIndexMap)
                {
                    if (kvp.Value == sharedIdx)
                    {
                        party1Indices.Add(kvp.Key);
                        break;
                    }
                }
            }

            _alignedIndicesPerParty[_parties[0].PartyId] = party0Indices;
            _alignedIndicesPerParty[_parties[1].PartyId] = party1Indices;
        }
        else
        {
            // Multi-party: use leader (party 0) as reference, pairwise align
            var partySets = new List<IReadOnlyList<string>>();
            foreach (var party in _parties)
            {
                partySets.Add(party.GetEntityIds());
            }

            var multiResult = _entityAligner.AlignMultipleParties(partySets, effectivePsiOptions);
            _alignedEntityCount = multiResult.PsiResult.IntersectionSize;

            // Build indices from multi-party alignment
            var intersectionIdSet = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < multiResult.PsiResult.IntersectionIds.Count; i++)
            {
                intersectionIdSet[multiResult.PsiResult.IntersectionIds[i]] = i;
            }

            foreach (var party in _parties)
            {
                var entityIds = party.GetEntityIds();
                var indices = new List<int>();
                for (int i = 0; i < entityIds.Count; i++)
                {
                    if (intersectionIdSet.ContainsKey(entityIds[i]))
                    {
                        indices.Add(i);
                    }
                }

                _alignedIndicesPerParty[party.PartyId] = indices;
            }
        }

        stopwatch.Stop();

        // Build overlap statistics
        var partyCounts = new Dictionary<string, int>();
        var partyOverlaps = new Dictionary<string, double>();
        foreach (var party in _parties)
        {
            int totalIds = party.GetEntityIds().Count;
            partyCounts[party.PartyId] = totalIds;
            partyOverlaps[party.PartyId] = totalIds > 0
                ? (double)_alignedEntityCount / totalIds
                : 0.0;
        }

        _alignmentSummary = new VflAlignmentSummary
        {
            AlignedEntityCount = _alignedEntityCount,
            PartyEntityCounts = partyCounts,
            PartyOverlapRatios = partyOverlaps,
            MeetsMinimumOverlap = _alignedEntityCount > 0 &&
                partyOverlaps.Values.All(r => r >= _options.MissingFeatures.MinimumOverlapRatio),
            AlignmentTime = stopwatch.Elapsed
        };

        // Initialize split model
        int embDim = _options.SplitModel.EmbeddingDimension;
        _splitModel = new SplitNeuralNetwork<T>(
            _parties.Count, embDim, 1, _options.SplitModel, _options.RandomSeed);

        _isAligned = true;
        return _alignmentSummary;
    }

    /// <inheritdoc/>
    public VflEpochResult<T> TrainEpoch()
    {
        if (!_isAligned || _splitModel is null || _alignedIndicesPerParty is null)
        {
            throw new InvalidOperationException("Entities must be aligned before training.");
        }

        if (_alignedEntityCount == 0)
        {
            return new VflEpochResult<T> { SamplesProcessed = 0 };
        }

        var stopwatch = Stopwatch.StartNew();

        // Find the label holder
        var labelHolder = FindLabelHolder();

        int batchSize = Math.Min(_options.BatchSize, _alignedEntityCount);
        int numBatches = (_alignedEntityCount + batchSize - 1) / batchSize;
        double epochLoss = 0.0;
        int samplesProcessed = 0;

        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            int startIdx = batchIdx * batchSize;
            int endIdx = Math.Min(startIdx + batchSize, _alignedEntityCount);
            int currentBatchSize = endIdx - startIdx;

            // Get batch indices for each party
            var batchEmbeddings = new List<Tensor<T>>();
            var batchIndicesPerParty = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal);

            foreach (var party in _parties)
            {
                var partyIndices = _alignedIndicesPerParty[party.PartyId];
                var batchIndices = new List<int>(currentBatchSize);
                for (int i = startIdx; i < endIdx; i++)
                {
                    batchIndices.Add(partyIndices[i]);
                }

                batchIndicesPerParty[party.PartyId] = batchIndices;

                // Forward pass through bottom model
                var embedding = party.ComputeForward(batchIndices);
                batchEmbeddings.Add(embedding);

                // Update missing feature statistics
                _missingFeatureHandler.UpdateStatistics(party.PartyId, embedding);
            }

            // Aggregate embeddings
            var combined = _splitModel.AggregateEmbeddings(batchEmbeddings);

            // Top model forward pass
            var predictions = _splitModel.ForwardTopModel(combined);

            // Compute loss (label holder)
            var batchLabelIndices = batchIndicesPerParty[labelHolder.PartyId];
            var batchLabels = labelHolder.GetLabels(batchLabelIndices);
            var (loss, lossGradient) = labelHolder.ComputeLoss(predictions, batchLabels);

            epochLoss += loss * currentBatchSize;
            samplesProcessed += currentBatchSize;

            // Apply label DP if configured
            if (_labelProtector is not null)
            {
                lossGradient = _labelProtector.ProtectGradients(lossGradient);
            }

            // Update top model parameters
            _splitModel.UpdateFromGradient(lossGradient, _options.LearningRate);

            // Backward through top model to get per-party gradients
            var partyGradients = _splitModel.BackwardTopModel(lossGradient, batchEmbeddings);

            // Send gradients to each party (optionally encrypted)
            for (int p = 0; p < _parties.Count; p++)
            {
                var gradient = partyGradients[p];

                // Optional gradient encryption
                if (_gradientExchange is not null)
                {
                    var (protected_, mask) = _gradientExchange.ProtectGradients(gradient);
                    gradient = _gradientExchange.RecoverGradients(protected_, mask);
                }

                _parties[p].ApplyBackward(gradient, _options.LearningRate);
            }
        }

        stopwatch.Stop();

        var result = new VflEpochResult<T>
        {
            AverageLoss = samplesProcessed > 0 ? epochLoss / samplesProcessed : 0.0,
            SamplesProcessed = samplesProcessed,
            BatchesProcessed = numBatches,
            EpochTime = stopwatch.Elapsed
        };

        if (_labelProtector is not null)
        {
            result.PrivacyBudgetSpent = _labelProtector.GetPrivacyBudgetSpent();
        }

        return result;
    }

    /// <inheritdoc/>
    public VflTrainingResult<T> Train()
    {
        if (!_isAligned)
        {
            AlignEntities();
        }

        var totalStopwatch = Stopwatch.StartNew();
        var epochHistory = new List<VflEpochResult<T>>();

        for (int epoch = 0; epoch < _options.NumberOfEpochs; epoch++)
        {
            var epochResult = TrainEpoch();
            epochResult.Epoch = epoch;
            epochHistory.Add(epochResult);
        }

        totalStopwatch.Stop();
        _isTrained = true;

        return new VflTrainingResult<T>
        {
            EpochHistory = epochHistory,
            FinalLoss = epochHistory.Count > 0 ? epochHistory[epochHistory.Count - 1].AverageLoss : 0.0,
            TotalTrainingTime = totalStopwatch.Elapsed,
            EpochsCompleted = epochHistory.Count,
            AlignmentSummary = _alignmentSummary,
            TrainingCompleted = true,
            NumberOfParties = _parties.Count
        };
    }

    /// <inheritdoc/>
    public Tensor<T> Predict(IReadOnlyList<int> entityIndices)
    {
        if (!_isTrained || _splitModel is null || _alignedIndicesPerParty is null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var embeddings = new List<Tensor<T>>();
        foreach (var party in _parties)
        {
            var partyIndices = new List<int>(entityIndices.Count);
            var alignedIndices = _alignedIndicesPerParty[party.PartyId];
            for (int i = 0; i < entityIndices.Count; i++)
            {
                int idx = entityIndices[i];
                if (idx < alignedIndices.Count)
                {
                    partyIndices.Add(alignedIndices[idx]);
                }
            }

            if (partyIndices.Count > 0)
            {
                embeddings.Add(party.ComputeForward(partyIndices));
            }
        }

        if (embeddings.Count == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var combined = _splitModel.AggregateEmbeddings(embeddings);
        return _splitModel.ForwardTopModel(combined);
    }

    /// <inheritdoc/>
    public int UnlearnEntities(IReadOnlyList<string> entityIds)
    {
        if (!_options.Unlearning.Enabled)
        {
            throw new InvalidOperationException("Unlearning is not enabled in options.");
        }

        if (!_isTrained || _splitModel is null)
        {
            throw new InvalidOperationException("Model must be trained before unlearning.");
        }

        if (entityIds is null || entityIds.Count == 0)
        {
            return 0;
        }

        // Find aligned indices for entities to unlearn
        var labelHolder = FindLabelHolder();
        var partyEntityIds = labelHolder.GetEntityIds();
        var entityIdSet = new HashSet<string>(entityIds, StringComparer.Ordinal);

        var unlearnIndices = new List<int>();
        for (int i = 0; i < partyEntityIds.Count; i++)
        {
            if (entityIdSet.Contains(partyEntityIds[i]))
            {
                unlearnIndices.Add(i);
            }
        }

        if (unlearnIndices.Count == 0)
        {
            return 0;
        }

        // Apply gradient ascent unlearning
        if (_options.Unlearning.Method == VflUnlearningMethod.GradientAscent)
        {
            for (int step = 0; step < _options.Unlearning.GradientAscentSteps; step++)
            {
                // Forward pass on entities to unlearn
                var embeddings = new List<Tensor<T>>();
                foreach (var party in _parties)
                {
                    embeddings.Add(party.ComputeForward(unlearnIndices));
                }

                var combined = _splitModel.AggregateEmbeddings(embeddings);
                var predictions = _splitModel.ForwardTopModel(combined);
                var labels = labelHolder.GetLabels(unlearnIndices);
                var (_, lossGradient) = labelHolder.ComputeLoss(predictions, labels);

                // Negate gradients (gradient ascent = reverse direction)
                int totalElements = 1;
                for (int d = 0; d < lossGradient.Rank; d++)
                {
                    totalElements *= lossGradient.Shape[d];
                }

                var negatedGradient = new Tensor<T>(lossGradient.Shape);
                for (int i = 0; i < totalElements; i++)
                {
                    negatedGradient[i] = NumOps.FromDouble(-NumOps.ToDouble(lossGradient[i]));
                }

                // Update top model in reverse direction
                _splitModel.UpdateFromGradient(negatedGradient, _options.Unlearning.UnlearningLearningRate);

                // Update bottom models in reverse direction
                var partyGradients = _splitModel.BackwardTopModel(negatedGradient, embeddings);
                for (int p = 0; p < _parties.Count; p++)
                {
                    _parties[p].ApplyBackward(partyGradients[p], _options.Unlearning.UnlearningLearningRate);
                }
            }
        }

        return unlearnIndices.Count;
    }

    private VerticalPartyLabelHolder<T> FindLabelHolder()
    {
        foreach (var party in _parties)
        {
            if (party.IsLabelHolder && party is VerticalPartyLabelHolder<T> labelHolder)
            {
                return labelHolder;
            }
        }

        throw new InvalidOperationException(
            "No label holder found. Register a VerticalPartyLabelHolder before training.");
    }
}
