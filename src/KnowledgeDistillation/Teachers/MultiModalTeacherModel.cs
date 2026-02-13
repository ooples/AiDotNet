using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Multi-modal teacher that combines multiple input modalities (vision, text, audio).
/// </summary>
public class MultiModalTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _modalityTeachers;
    private readonly double[] _modalityWeights;

    public override int OutputDimension => _modalityTeachers[0].OutputDimension;

    public MultiModalTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] modalityTeachers,
        double[]? modalityWeights = null)
    {
        Guard.NotNull(modalityTeachers);
        _modalityTeachers = modalityTeachers;

        // Validate modality teachers array is non-empty
        if (_modalityTeachers.Length == 0)
            throw new ArgumentException("Modality teachers array cannot be empty", nameof(modalityTeachers));

        // Validate no teacher is null
        for (int i = 0; i < _modalityTeachers.Length; i++)
        {
            if (_modalityTeachers[i] == null)
                throw new ArgumentException($"Modality teacher at index {i} is null", nameof(modalityTeachers));
        }

        // Validate all teachers have the same output dimension
        int expectedOutputDim = _modalityTeachers[0].OutputDimension;
        for (int i = 1; i < _modalityTeachers.Length; i++)
        {
            if (_modalityTeachers[i].OutputDimension != expectedOutputDim)
                throw new ArgumentException(
                    $"Modality teacher at index {i} has OutputDimension {_modalityTeachers[i].OutputDimension}, " +
                    $"but expected {expectedOutputDim} (from teacher 0). " +
                    $"All modality teachers must have the same OutputDimension.",
                    nameof(modalityTeachers));
        }

        // Set or validate modality weights
        if (modalityWeights == null)
        {
            _modalityWeights = Enumerable.Repeat(1.0 / modalityTeachers.Length, modalityTeachers.Length).ToArray();
        }
        else
        {
            if (modalityWeights.Length != modalityTeachers.Length)
                throw new ArgumentException(
                    $"Modality weights length ({modalityWeights.Length}) must match number of teachers ({modalityTeachers.Length})",
                    nameof(modalityWeights));
            _modalityWeights = modalityWeights;
        }
    }

    /// <summary>
    /// Gets combined logits from all modality teachers.
    /// </summary>
    /// <param name="input">Input data.</param>
    /// <returns>Combined logits from all modalities.</returns>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw combined logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        int n = _modalityTeachers[0].OutputDimension;
        var combined = new Vector<T>(n);

        // Cache logits from each teacher to avoid repeated calls
        var teacherLogits = new Vector<T>[_modalityTeachers.Length];
        for (int i = 0; i < _modalityTeachers.Length; i++)
        {
            teacherLogits[i] = _modalityTeachers[i].GetLogits(input);
        }

        // Combine weighted logits
        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < _modalityTeachers.Length; i++)
            {
                var weighted = NumOps.Multiply(teacherLogits[i][j], NumOps.FromDouble(_modalityWeights[i]));
                sum = NumOps.Add(sum, weighted);
            }
            combined[j] = sum;
        }

        return combined;
    }

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> if all modality teachers support JIT compilation; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-modal JIT compilation is supported when all modality
    /// teachers implement IJitCompilable and support JIT. The combined computation graph
    /// weights and sums each modality's contribution.</para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _modalityTeachers.All(t => t is IJitCompilable<T> jit && jit.SupportsJitCompilation);

    /// <summary>
    /// Exports the multi-modal computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the weighted multi-modal output.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when any modality teacher does not support JIT.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The multi-modal graph combines each modality teacher's computation graph using weighted sum:
    /// output = w1 * modality1_output + w2 * modality2_output + ... + wN * modalityN_output
    /// </para>
    /// <para><b>Note:</b> All modality teachers must support JIT compilation. The combined graph
    /// enables optimized inference across all modalities in a single execution.</para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));

        // Check all modality teachers support JIT
        for (int i = 0; i < _modalityTeachers.Length; i++)
        {
            if (_modalityTeachers[i] is not IJitCompilable<T> jit || !jit.SupportsJitCompilation)
            {
                return ThrowJitNotSupported(
                    nameof(MultiModalTeacherModel<T>),
                    $"modality teacher at index {i} ({_modalityTeachers[i].GetType().Name}) does not support JIT compilation");
            }
        }

        // Combine modality teacher graphs with weighted sum
        ComputationNode<T>? resultNode = null;

        for (int i = 0; i < _modalityTeachers.Length; i++)
        {
            var jitTeacher = (IJitCompilable<T>)_modalityTeachers[i];

            // Get modality teacher's computation graph
            var teacherInputNodes = new List<ComputationNode<T>>();
            var teacherOutput = jitTeacher.ExportComputationGraph(teacherInputNodes);
            inputNodes.AddRange(teacherInputNodes);

            // Scale by modality weight
            var weightTensor = new Tensor<T>((int[])teacherOutput.Value.Shape.Clone());
            weightTensor.Fill(NumOps.FromDouble(_modalityWeights[i]));
            var weightNode = TensorOperations<T>.Constant(weightTensor, $"modality_{i}_weight");
            var scaledOutput = TensorOperations<T>.ElementwiseMultiply(teacherOutput, weightNode);

            // Add to result
            if (resultNode == null)
            {
                resultNode = scaledOutput;
            }
            else
            {
                resultNode = TensorOperations<T>.Add(resultNode, scaledOutput);
            }
        }

        return resultNode!;
    }
}
