
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Similarity-preserving distillation that preserves pairwise similarity structure.
/// </summary>
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

    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);

            studentEmbeddings[r] = studentSoft;
            teacherEmbeddings[r] = teacherSoft;

            var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            T sampleLoss;
            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
                var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, trueLabels);
                sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
            }
            else
            {
                sampleLoss = softLoss;
            }

            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        var standardLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var similarityLoss = ComputeSimilarityLoss(studentEmbeddings, teacherEmbeddings);

        return NumOps.Add(
            NumOps.Multiply(standardLoss, NumOps.FromDouble(1.0 - _similarityWeight)),
            similarityLoss);
    }

    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var gradientBatch = new Matrix<T>(batchSize, outputDim);

        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            var gradient = new Vector<T>(outputDim);
            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);

            studentEmbeddings[r] = studentSoft;
            teacherEmbeddings[r] = teacherSoft;

            for (int i = 0; i < outputDim; i++)
            {
                var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
            }

            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);

                for (int i = 0; i < outputDim; i++)
                {
                    var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                    gradient[i] = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
                }
            }

            gradientBatch.SetRow(r, gradient);
        }

        if (_similarityWeight > 0)
        {
            var similarityGradient = ComputeSimilarityGradient(studentEmbeddings, teacherEmbeddings);

            for (int r = 0; r < batchSize; r++)
            {
                for (int c = 0; c < outputDim; c++)
                {
                    var standardGrad = NumOps.Multiply(gradientBatch[r, c], NumOps.FromDouble(1.0 - _similarityWeight));
                    var simGrad = NumOps.Multiply(similarityGradient[r, c], NumOps.FromDouble(_similarityWeight));
                    gradientBatch[r, c] = NumOps.Add(standardGrad, simGrad);
                }
            }
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
        T totalLoss = NumOps.Zero;
        int pairCount = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double teacherSim = CosineSimilarity(teacherEmbeddings[i], teacherEmbeddings[j]);
                double studentSim = CosineSimilarity(studentEmbeddings[i], studentEmbeddings[j]);
                double diff = teacherSim - studentSim;
                totalLoss = NumOps.Add(totalLoss, NumOps.FromDouble(diff * diff));
                pairCount++;
            }
        }

        var loss = pairCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(pairCount)) : NumOps.Zero;
        return NumOps.Multiply(loss, NumOps.FromDouble(_similarityWeight));
    }

    private double CosineSimilarity(Vector<T> v1, Vector<T> v2)
    {
        T dot = NumOps.Zero, norm1 = NumOps.Zero, norm2 = NumOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(v1[i], v2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(v1[i], v1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(v2[i], v2[i]));
        }

        return Convert.ToDouble(dot) / (Math.Sqrt(Convert.ToDouble(norm1)) * Math.Sqrt(Convert.ToDouble(norm2)) + Epsilon);
    }

    private Matrix<T> ComputeSimilarityGradient(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int n = studentEmbeddings.Length;
        int dim = studentEmbeddings[0].Length;

        var gradients = new Matrix<T>(n, dim);

        int pairCount = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                pairCount++;
            }
        }

        if (pairCount == 0)
            return gradients;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                double teacherSim = CosineSimilarity(teacherEmbeddings[i], teacherEmbeddings[j]);
                double studentSim = CosineSimilarity(studentEmbeddings[i], studentEmbeddings[j]);

                double simDiff = teacherSim - studentSim;

                double dot = 0, norm_i = 0, norm_j = 0;
                for (int k = 0; k < dim; k++)
                {
                    double ei_k = Convert.ToDouble(studentEmbeddings[i][k]);
                    double ej_k = Convert.ToDouble(studentEmbeddings[j][k]);
                    dot += ei_k * ej_k;
                    norm_i += ei_k * ei_k;
                    norm_j += ej_k * ej_k;
                }

                norm_i = Math.Sqrt(norm_i);
                norm_j = Math.Sqrt(norm_j);

                for (int k = 0; k < dim; k++)
                {
                    double ei_k = Convert.ToDouble(studentEmbeddings[i][k]);
                    double ej_k = Convert.ToDouble(studentEmbeddings[j][k]);

                    double dSim_dEi = (ej_k / (norm_i * norm_j + Epsilon)) - (dot / (norm_i * norm_i * norm_i * norm_j + Epsilon)) * ei_k;

                    double grad = -2.0 * simDiff * dSim_dEi / pairCount;

                    gradients[i, k] = NumOps.Add(gradients[i, k], NumOps.FromDouble(grad));
                }
            }
        }

        return gradients;
    }

}


