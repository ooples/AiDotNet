
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Similarity-preserving distillation that preserves pairwise similarity structure.
/// </summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Similarity-Preserving Knowledge Distillation",
    "https://arxiv.org/abs/1907.09682",
    Year = 2019,
    Authors = "Frederick Tung, Greg Mori")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class SimilarityPreservingStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _similarityWeight;

    public SimilarityPreservingStrategy(
        double similarityWeight = 0.5,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        _similarityWeight = similarityWeight;
    }

    /// <inheritdoc />
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Per sample: T²·KL(teacher_soft || student_soft), blended with hard CE when labels are given
        T totalLoss = Engine.Sum(DistillationHelper<T>.StandardSampleLosses(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha));

        // The temperature-softened distributions are the embeddings whose similarities are preserved
        var studentEmbeddings = DistillationHelper<T>.RowsOf(DistillationHelper<T>.SoftmaxRows(studentBatchOutput, Temperature));
        var teacherEmbeddings = DistillationHelper<T>.RowsOf(DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, Temperature));

        var standardLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var similarityLoss = ComputeSimilarityLoss(studentEmbeddings, teacherEmbeddings);

        return NumOps.Add(
            NumOps.Multiply(standardLoss, NumOps.FromDouble(1.0 - _similarityWeight)),
            similarityLoss);
    }

    /// <inheritdoc />
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard gradient, averaged over the batch like the standard loss. The similarity loss is already a
        // batch-level quantity, so its gradient is not divided by the batch size again.
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        var gradientBatch = Engine.MatrixMultiplyScalar(
            DistillationHelper<T>.StandardSampleGradients(studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha),
            oneOverBatch);

        if (_similarityWeight > 0)
        {
            var studentSoft = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, Temperature);

            // Similarity gradient is w.r.t. softmax outputs (embeddings).
            // Must apply softmax Jacobian to convert to gradient w.r.t. logits.
            var simGradEmbedding = Tensor<T>.FromMatrix(ComputeSimilarityGradient(
                DistillationHelper<T>.RowsOf(studentSoft), DistillationHelper<T>.RowsOf(teacherSoft)));

            // Apply softmax Jacobian: ∂L/∂z_k = Σ_m (∂L/∂p_m) * p_m * (δ_{mk} - p_k) / T
            //                                = p_k * (∂L/∂p_k - Σ_m (∂L/∂p_m) p_m) / T
            var dLdz = DistillationHelper<T>.SoftmaxJacobianProduct(Tensor<T>.FromMatrix(studentSoft), simGradEmbedding, Temperature);

            var standardGrad = Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(gradientBatch), NumOps.FromDouble(1.0 - _similarityWeight));
            var simGrad = Engine.TensorMultiplyScalar(dLdz, NumOps.FromDouble(_similarityWeight));
            gradientBatch = Engine.TensorAdd(standardGrad, simGrad).ToMatrix();
        }

        return gradientBatch;
    }

    public T ComputeSimilarityLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        if (studentEmbeddings == null || teacherEmbeddings == null)
            throw new ArgumentNullException("Embeddings cannot be null");
        if (studentEmbeddings.Length != teacherEmbeddings.Length)
            throw new ArgumentException("Student and teacher must have same batch size");

        // Validate all vectors have same dimensions
        if (studentEmbeddings.Length > 0)
        {
            int studentDim = studentEmbeddings[0].Length;
            int teacherDim = teacherEmbeddings[0].Length;

            for (int i = 0; i < studentEmbeddings.Length; i++)
            {
                if (studentEmbeddings[i].Length != studentDim)
                    throw new ArgumentException($"All student embeddings must have same dimension. Expected {studentDim}, got {studentEmbeddings[i].Length} at index {i}");
                if (teacherEmbeddings[i].Length != teacherDim)
                    throw new ArgumentException($"All teacher embeddings must have same dimension. Expected {teacherDim}, got {teacherEmbeddings[i].Length} at index {i}");
                if (studentEmbeddings[i].Length != teacherEmbeddings[i].Length)
                    throw new ArgumentException($"Student and teacher embeddings must have matching dimensions. Got student={studentEmbeddings[i].Length}, teacher={teacherEmbeddings[i].Length} at index {i}");
                if (studentEmbeddings[i].Length == 0)
                    throw new ArgumentException($"Embedding vectors cannot be empty at index {i}");
            }
        }

        int n = studentEmbeddings.Length;
        int pairCount = n * (n - 1) / 2;
        T totalLoss = NumOps.Zero;

        if (pairCount > 0)
        {
            // Sum over pairs i < j of (cos_teacher(i, j) - cos_student(i, j))²
            var teachers = DistillationHelper<T>.StackRows(teacherEmbeddings);
            var students = DistillationHelper<T>.StackRows(studentEmbeddings);
            var diff = Engine.TensorSubtract(
                DistillationHelper<T>.CosineSimilarityMatrix(teachers, teachers),
                DistillationHelper<T>.CosineSimilarityMatrix(students, students));
            totalLoss = Engine.TensorSum(Engine.TensorMultiply(
                Engine.TensorMultiply(diff, diff), DistillationHelper<T>.PairMask(n, upperOnly: true)));
        }

        var loss = pairCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(pairCount)) : NumOps.Zero;
        return NumOps.Multiply(loss, NumOps.FromDouble(_similarityWeight));
    }


    private Matrix<T> ComputeSimilarityGradient(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int n = studentEmbeddings.Length;
        int dim = studentEmbeddings[0].Length;

        int pairCount = n * (n - 1) / 2;
        if (pairCount == 0)
            return new Matrix<T>(n, dim);

        // For every ordered pair i != j, with e = student embeddings, d = cos_t(i, j) - cos_s(i, j):
        //   ∂sim/∂e_i = e_j / (|e_i||e_j| + ε) - (e_i·e_j) / (|e_i|³|e_j| + ε) * e_i
        //   grad_i   += -2 * d * ∂sim/∂e_i / pairCount
        var e = DistillationHelper<T>.StackRows(studentEmbeddings);
        var t = DistillationHelper<T>.StackRows(teacherEmbeddings);
        var simDiff = Engine.TensorMultiply(
            Engine.TensorSubtract(DistillationHelper<T>.CosineSimilarityMatrix(t, t), DistillationHelper<T>.CosineSimilarityMatrix(e, e)),
            DistillationHelper<T>.PairMask(n, upperOnly: false));

        var dot = Engine.TensorMatMul(e, Engine.TensorTranspose(e));
        var norms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(e, e), [1], keepDims: true)); // [n, 1]
        var normsT = Engine.TensorTranspose(norms);
        T epsilon = NumOps.FromDouble(Epsilon);
        var normProduct = Engine.TensorAddScalar(Engine.TensorMatMul(norms, normsT), epsilon);
        var cubedProduct = Engine.TensorAddScalar(
            Engine.TensorMatMul(Engine.TensorMultiply(Engine.TensorMultiply(norms, norms), norms), normsT), epsilon);

        var neighbourWeights = Engine.TensorDivide(simDiff, normProduct);                               // [n, n]
        var selfWeights = Engine.ReduceSum(
            Engine.TensorDivide(Engine.TensorMultiply(simDiff, dot), cubedProduct), [1], keepDims: true); // [n, 1]
        var dSim = Engine.TensorSubtract(
            Engine.TensorMatMul(neighbourWeights, e),
            Engine.TensorMultiply(e, Engine.TensorBroadcastTo(selfWeights, [n, dim])));

        return Engine.TensorMultiplyScalar(dSim, NumOps.FromDouble(-2.0 / pairCount)).ToMatrix();
    }

}


