using AiDotNet.Helpers;
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

        int batchSize = studentBatchOutput.RowCount;
        T totalLoss = NumOps.Zero;

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            var studentSoft = Softmax(studentRow, Temperature);
            var teacherSoft = Softmax(teacherRow, Temperature);
            var softLoss = KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, 1.0);
                var hardLoss = CrossEntropy(studentProbs, labelRow);
                var sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
                totalLoss = NumOps.Add(totalLoss, sampleLoss);
            }
            else
            {
                totalLoss = NumOps.Add(totalLoss, softLoss);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.RowCount;
        int numClasses = studentBatchOutput.ColumnCount;
        var gradient = new Matrix<T>(batchSize, numClasses);

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            var studentSoft = Softmax(studentRow, Temperature);
            var teacherSoft = Softmax(teacherRow, Temperature);

            for (int c = 0; c < numClasses; c++)
            {
                var diff = NumOps.Subtract(studentSoft[c], teacherSoft[c]);
                gradient[r, c] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
            }

            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, 1.0);

                for (int c = 0; c < numClasses; c++)
                {
                    var hardGrad = NumOps.Subtract(studentProbs[c], labelRow[c]);
                    gradient[r, c] = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[r, c]));
                }
            }
        }

        // Average gradients over batch
        T oneOverBatchSize = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                gradient[r, c] = NumOps.Multiply(gradient[r, c], oneOverBatchSize);
            }
        }

        return gradient;
    }

    public T ComputeSimilarityLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        if (studentEmbeddings == null) throw new ArgumentNullException(nameof(studentEmbeddings));
        if (teacherEmbeddings == null) throw new ArgumentNullException(nameof(teacherEmbeddings));

        if (studentEmbeddings.Length != teacherEmbeddings.Length)
        {
            throw new ArgumentException(
                $"Student and teacher embedding batches must match. Student: {studentEmbeddings.Length}, Teacher: {teacherEmbeddings.Length}");
        }

        int n = studentEmbeddings.Length;
        if (n == 0)
        {
            return NumOps.Zero;
        }

        int expectedDim = studentEmbeddings[0].Length;
        for (int i = 0; i < n; i++)
        {
            if (studentEmbeddings[i].Length != expectedDim || teacherEmbeddings[i].Length != expectedDim)
            {
                throw new ArgumentException("All embeddings must share the same dimensionality.");
            }
        }

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

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }

    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;

        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Convert.ToDouble(p[i]);
            double qVal = Convert.ToDouble(q[i]);

            if (pVal > Epsilon)
            {
                double contrib = pVal * Math.Log(pVal / (qVal + Epsilon));
                divergence = NumOps.Add(divergence, NumOps.FromDouble(contrib));
            }
        }

        return divergence;
    }

    private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
    {
        T entropy = NumOps.Zero;

        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Convert.ToDouble(predictions[i]);
            double label = Convert.ToDouble(trueLabels[i]);

            if (label > Epsilon)
            {
                double contrib = -label * Math.Log(pred + Epsilon);
                entropy = NumOps.Add(entropy, NumOps.FromDouble(contrib));
            }
        }

        return entropy;
    }
}
