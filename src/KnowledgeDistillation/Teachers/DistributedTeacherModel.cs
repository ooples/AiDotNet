using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Distributed teacher model that aggregates predictions from multiple distributed workers.
/// </summary>
public class DistributedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _workers;
    private readonly AggregationMode _aggregation;

    public override int OutputDimension => _workers[0].OutputDimension;

    public DistributedTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] workers,
        AggregationMode aggregation = AggregationMode.Average)
    {
        _workers = workers ?? throw new ArgumentNullException(nameof(workers));

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
                for (int j = 0; j < n; j++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < _workers.Length; i++)
                    {
                        var logits = _workers[i].GetLogits(input);
                        sum = NumOps.Add(sum, logits[j]);
                    }
                    aggregated[j] = NumOps.Divide(sum, NumOps.FromDouble(_workers.Length));
                }
                break;

            case AggregationMode.Voting:
                // For simplicity, use average as voting
                goto case AggregationMode.Average;
        }

        return aggregated;
    }

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> if Average aggregation mode is used and all workers support JIT compilation;
    /// otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para><b>Note:</b> While "distributed" implies workers on different machines, JIT compilation
    /// is supported when all workers are local models that implement IJitCompilable. This enables
    /// combining their computation graphs for optimized inference.</para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _aggregation == AggregationMode.Average &&
        _workers.All(w => w is IJitCompilable<T> jit && jit.SupportsJitCompilation);

    /// <summary>
    /// Exports the distributed worker computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the averaged worker output.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the aggregation mode is not Average or when any worker does not support JIT.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The distributed graph combines each worker's computation graph using averaging:
    /// output = (worker1_output + worker2_output + ... + workerN_output) / N
    /// </para>
    /// <para><b>Note:</b> JIT compilation creates a single optimized computation graph
    /// combining all worker models. This is beneficial when workers are local models;
    /// for truly distributed inference across machines, use runtime aggregation instead.</para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));

        if (_aggregation != AggregationMode.Average)
        {
            return ThrowJitNotSupported(
                nameof(DistributedTeacherModel<T>),
                $"aggregation mode {_aggregation} involves dynamic operations that cannot be represented in a static computation graph. Only Average mode supports JIT");
        }

        // Check all workers support JIT
        for (int i = 0; i < _workers.Length; i++)
        {
            if (_workers[i] is not IJitCompilable<T> jit || !jit.SupportsJitCompilation)
            {
                return ThrowJitNotSupported(
                    nameof(DistributedTeacherModel<T>),
                    $"worker at index {i} ({_workers[i].GetType().Name}) does not support JIT compilation");
            }
        }

        // Combine worker graphs with sum then divide
        ComputationNode<T>? sumNode = null;

        for (int i = 0; i < _workers.Length; i++)
        {
            var jitWorker = (IJitCompilable<T>)_workers[i];

            // Get worker's computation graph
            var workerInputNodes = new List<ComputationNode<T>>();
            var workerOutput = jitWorker.ExportComputationGraph(workerInputNodes);
            inputNodes.AddRange(workerInputNodes);

            // Add to sum
            if (sumNode == null)
            {
                sumNode = workerOutput;
            }
            else
            {
                sumNode = TensorOperations<T>.Add(sumNode, workerOutput);
            }
        }

        // Divide by number of workers to get average
        var divisorTensor = new Tensor<T>((int[])sumNode!.Value.Shape.Clone());
        divisorTensor.Fill(NumOps.FromDouble(_workers.Length));
        var divisorNode = TensorOperations<T>.Constant(divisorTensor, "worker_count");
        var resultNode = TensorOperations<T>.Divide(sumNode!, divisorNode);

        return resultNode;
    }
}
