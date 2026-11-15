using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedMD — Model-agnostic federated learning via mutual distillation on a public dataset.
/// </summary>
/// <remarks>
/// <para>
/// FedMD (Li &amp; Wang, 2019) enables heterogeneous model architectures across clients.
/// Each client computes logits on a shared public dataset, the server averages the logits,
/// and clients distill the consensus logits into their local models.
/// </para>
/// <para>
/// <b>For Beginners:</b> All clients are given the same "test questions" (public dataset).
/// Each client submits their answers (predicted probabilities). The server averages all
/// answers to get a "consensus answer," then each client learns from this consensus.
/// This way, a simple model on a phone can learn from a powerful model on a server.
/// </para>
/// <para>
/// Reference: Li &amp; Wang (2019), "FedMD: Heterogeneous Federated Learning via Model Distillation".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedMDDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly double _temperature;
    private readonly double _distillationAlpha;
    private readonly int _distillationEpochs;
    private readonly double _learningRate;

    /// <summary>
    /// Creates a new FedMD distillation strategy.
    /// </summary>
    /// <param name="temperature">Softmax temperature for soft labels. Default: 3.0.</param>
    /// <param name="distillationAlpha">Weight for distillation loss vs task loss. Default: 0.5.</param>
    /// <param name="distillationEpochs">Number of local distillation epochs. Default: 5.</param>
    /// <param name="learningRate">Learning rate for distillation updates. Default: 0.01.</param>
    public FedMDDistillation(double temperature = 3.0, double distillationAlpha = 0.5,
        int distillationEpochs = 5, double learningRate = 0.01)
    {
        _temperature = temperature;
        _distillationAlpha = distillationAlpha;
        _distillationEpochs = distillationEpochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The local model is the flattened linear classifier W [outputDim x C] over the C public-data columns,
    /// W[j, k] = θ[j·C + k], so outputDim = d / C and d must be a multiple of C. The class scores are z / τ
    /// with z = X Wᵀ.
    /// </remarks>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        Guard.NotNull(localModelParameters);
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset for logit extraction.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        return ComputeScaledLogits(localModelParameters, publicData).ToMatrix();
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientKnowledge);
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        // Get dimensions from first client
        var first = clientKnowledge.Values.First();
        int n = first.Rows;
        int outputDim = first.Columns;

        double totalWeight = clientKnowledge.Keys.Sum(id => clientWeights?.GetValueOrDefault(id, 1.0) ?? 1.0);
        if (!(totalWeight > 0) || double.IsInfinity(totalWeight))
            throw new ArgumentException("Client weights must sum to a finite positive value.", nameof(clientWeights));

        // Weighted average of all client logits (weights normalized by their total).
        var aggregated = new Tensor<T>(new[] { n, outputDim });
        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            if (knowledge.Rows != n || knowledge.Columns != outputDim)
            {
                throw new ArgumentException(
                    $"Client {clientId} knowledge is {knowledge.Rows}x{knowledge.Columns}; expected {n}x{outputDim}.",
                    nameof(clientKnowledge));
            }

            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            aggregated = Engine.TensorAdd(aggregated,
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(knowledge), NumOps.FromDouble(w / totalWeight)));
        }

        return aggregated.ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The digest step: gradient descent on α/(2n) Σ ‖z/τ − consensus‖² over the public samples, so
    /// ∇W = α (z/τ − consensus)ᵀ X / (n τ) over every weight of W. The consensus was built from scores at the
    /// constructor temperature by <see cref="ExtractKnowledge"/>, which takes no temperature, so the digest uses
    /// the same one and <paramref name="temperature"/> is not used: a per-call value here would compare scores
    /// at one temperature against a consensus at another.
    /// </remarks>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(globalKnowledge);
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset.", nameof(publicData));
        ValidateInputs(localModelParameters, publicData);

        int d = localModelParameters.Length;
        int n = publicData.Rows;
        int columns = publicData.Columns;
        int outputDim = d / columns;
        if (globalKnowledge.Rows != n || globalKnowledge.Columns != outputDim)
        {
            throw new ArgumentException(
                $"Global knowledge is {globalKnowledge.Rows}x{globalKnowledge.Columns}; expected {n}x{outputDim}.",
                nameof(globalKnowledge));
        }

        var data = Tensor<T>.FromMatrix(publicData);
        var consensus = Tensor<T>.FromMatrix(globalKnowledge);
        var step = NumOps.FromDouble(_learningRate * _distillationAlpha / (n * _temperature));

        // Distillation: move the local scores toward the global consensus
        var parameters = localModelParameters.Clone();
        for (int epoch = 0; epoch < _distillationEpochs; epoch++)
        {
            var scores = ComputeScaledLogits(parameters, publicData);
            var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(scores, consensus), step);
            var gradWeights = Engine.TensorMatMul(Engine.TensorTranspose(residual), data); // [outputDim, C]
            parameters = Engine.Subtract(parameters, gradWeights.Reshape(new[] { d }).ToVector());
        }

        return parameters;
    }

    private static void ValidateInputs(Vector<T> parameters, Matrix<T> publicData)
    {
        if (parameters.Length == 0)
            throw new ArgumentException("Model parameters cannot be empty.", nameof(parameters));
        if (publicData.Rows == 0 || publicData.Columns == 0)
            throw new ArgumentException("Public data cannot be empty.", nameof(publicData));
        if (parameters.Length % publicData.Columns != 0)
        {
            throw new ArgumentException(
                $"FedMD's local model is a linear classifier [outputDim x {publicData.Columns}], so its " +
                $"{parameters.Length} parameters must be a multiple of the {publicData.Columns} public-data columns.",
                nameof(parameters));
        }
    }

    private Tensor<T> ComputeScaledLogits(Vector<T> parameters, Matrix<T> publicData)
    {
        int columns = publicData.Columns;
        var weights = parameters.AsTensor().Reshape(new[] { parameters.Length / columns, columns });

        // Apply temperature scaling
        var logits = Engine.TensorMatMul(Tensor<T>.FromMatrix(publicData), Engine.TensorTranspose(weights));
        return Engine.TensorMultiplyScalar(logits, NumOps.FromDouble(1.0 / _temperature));
    }
}
