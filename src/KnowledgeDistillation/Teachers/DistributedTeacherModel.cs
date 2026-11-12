using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Distributed teacher model that aggregates predictions from multiple distributed workers.
/// </summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Large Scale Distributed Neural Network Training through Online Distillation",
    "https://arxiv.org/abs/1804.03235",
    Year = 2018,
    Authors = "Rohan Anil, Gabriel Pereyra, Alexandre Passos, et al.")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class DistributedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _workers;
    private readonly AggregationMode _aggregation;

    /// <inheritdoc />
    public override int OutputDimension => _workers[0].OutputDimension;

    public DistributedTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] workers,
        AggregationMode aggregation = AggregationMode.Average)
    {
        Guard.NotNull(workers);
        _workers = workers;

        // Validate workers array is non-empty
        if (_workers.Length == 0)
            throw new ArgumentException("Workers array cannot be empty", nameof(workers));

        // Validate no worker is null
        for (int i = 0; i < _workers.Length; i++)
        {
            if (_workers[i] == null)
                throw new ArgumentException($"Worker at index {i} is null", nameof(workers));
        }

        // Validate all workers have the same output dimension
        int expectedOutputDim = _workers[0].OutputDimension;
        for (int i = 1; i < _workers.Length; i++)
        {
            if (_workers[i].OutputDimension != expectedOutputDim)
                throw new ArgumentException(
                    $"Worker at index {i} has OutputDimension {_workers[i].OutputDimension}, " +
                    $"but expected {expectedOutputDim} (from worker 0). " +
                    $"All workers must have the same OutputDimension.",
                    nameof(workers));
        }

        _aggregation = aggregation;
    }

    /// <summary>
    /// Gets aggregated logits from all distributed workers.
    /// </summary>
    /// <param name="input">Input data.</param>
    /// <returns>Aggregated logits from all workers.</returns>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw aggregated logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        int n = _workers[0].OutputDimension;
        var aggregated = new Vector<T>(n);

        switch (_aggregation)
        {
            case AggregationMode.Average:
                // Mean of the workers' first n logits (each worker is queried once, in order)
                var sum = new Vector<T>(n);
                foreach (var worker in _workers)
                {
                    var logits = worker.GetLogits(input);
                    sum = Engine.Add(sum, logits.Length == n ? logits : logits.GetSubVector(0, n));
                }
                aggregated = Engine.Divide(sum, NumOps.FromDouble(_workers.Length));
                break;

            case AggregationMode.Voting:
                // For simplicity, use average as voting
                goto case AggregationMode.Average;
        }

        return aggregated;
    }

    private static IEngine Engine => AiDotNetEngine.Current;
}
