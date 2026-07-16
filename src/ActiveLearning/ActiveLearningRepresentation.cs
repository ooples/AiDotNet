using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Builds the per-sample representation active-learning diversity measures redundancy in, using a
/// three-tier cascade: authentic BADGE gradient embeddings, then a model's exposed representation, then
/// raw input features.
/// </summary>
/// <remarks>
/// <para>
/// Tier 1 (best) uses <see cref="IGradientComputable{T, TInput, TOutput}"/>: each sample's embedding is
/// the gradient of the loss at the model's own prediction (a pseudo-label). This is exactly BADGE — the
/// gradient magnitude encodes uncertainty and its direction encodes what the model finds similar, so
/// diversity in this space picks a genuinely informative, non-redundant batch. Tier 2 uses
/// <see cref="ISupportsRepresentation{T, TInput}"/> (e.g. penultimate activations). Tier 3 falls back to
/// the raw input features, which is always available.
/// </para>
/// </remarks>
internal static class ActiveLearningRepresentation
{
    // Bound embedding dimensionality so a large-parameter model's gradient embedding cannot blow up
    // memory; feature-hashing to this size preserves cosine geometry approximately (Johnson-Lindenstrauss).
    private const int MaxDimension = 4096;

    public static (Matrix<T> representation, string space) Build<T>(
        IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> pool)
    {
        int n = pool.Shape[0];

        // Tier 1: authentic BADGE gradient embeddings.
        if (model is IGradientComputable<T, Tensor<T>, Tensor<T>> gradientModel)
        {
            try
            {
                var rows = new List<Vector<T>>(n);
                for (int i = 0; i < n; i++)
                {
                    var sample = Row(pool, i);
                    var pseudoLabel = model.Predict(sample);
                    rows.Add(gradientModel.ComputeGradients(sample, pseudoLabel));
                }

                if (rows.Count > 0 && rows[0].Length > 0)
                {
                    return (StackRows(rows), "GradientEmbedding");
                }
            }
            catch
            {
                // Fall through to the next tier if gradient extraction is unavailable for this model.
            }
        }

        // Tier 2: a model-exposed representation.
        if (model is ISupportsRepresentation<T, Tensor<T>> representationModel)
        {
            try
            {
                var rows = new List<Vector<T>>(n);
                for (int i = 0; i < n; i++)
                {
                    rows.Add(representationModel.GetRepresentation(Row(pool, i)));
                }

                if (rows.Count > 0 && rows[0].Length > 0)
                {
                    return (StackRows(rows), "ModelRepresentation");
                }
            }
            catch
            {
                // Fall through to input features.
            }
        }

        // Tier 3: raw input features (always available).
        return (InputFeatures(pool), "InputFeatures");
    }

    private static Tensor<T> Row<T>(Tensor<T> pool, int i)
    {
        int d = pool.Length / pool.Shape[0];
        var row = new Tensor<T>(new[] { 1, d });
        int baseIdx = i * d;
        for (int j = 0; j < d; j++)
        {
            row[j] = pool[baseIdx + j];
        }

        return row;
    }

    private static Matrix<T> InputFeatures<T>(Tensor<T> pool)
    {
        int n = pool.Shape[0];
        int d = pool.Length / n;
        var m = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                m[i, j] = pool[(i * d) + j];
            }
        }

        return m;
    }

    private static Matrix<T> StackRows<T>(List<Vector<T>> rows)
    {
        int n = rows.Count;
        int dim = rows[0].Length;
        int outDim = Math.Min(dim, MaxDimension);
        var numOps = MathHelper.GetNumericOperations<T>();
        var m = new Matrix<T>(n, outDim);

        if (dim <= MaxDimension)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    m[i, j] = rows[i][j];
                }
            }

            return m;
        }

        // Deterministic signed feature-hashing to MaxDimension (bounded, geometry-preserving).
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                int bucket = (int)((uint)(j * 2654435761u) % (uint)outDim);
                double sign = ((j * 40503) & 1) == 0 ? 1.0 : -1.0;
                double add = sign * numOps.ToDouble(rows[i][j]);
                m[i, bucket] = numOps.FromDouble(numOps.ToDouble(m[i, bucket]) + add);
            }
        }

        return m;
    }
}
