using AiDotNet.Interfaces;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Standard (z-score) scaling for regression TARGETS — the <see cref="IDataTransformer{T,TInput,TOutput}"/>
/// that <c>AiModelBuilder.ConfigureTargetScaling()</c> installs into the (previously always-null)
/// <see cref="PreprocessingInfo{T,TInput,TOutput}.TargetPipeline"/>.
/// </summary>
/// <remarks>
/// <para><b>Why:</b> the facade has always scaled FEATURES (<c>ConfigurePreprocessing</c>) but never the
/// target — raw targets on wide scales (prices, dollar P&amp;L) make gradient training diverge, so every
/// consumer hand-rolled a target scaler plus the inverse transform on the way out. The carrier
/// (<c>PreprocessingInfo.TargetPipeline</c> + <c>InverseTransformPredictions</c>) existed but nothing ever
/// populated or invoked it; this transformer + the builder/predict wiring completes the feature.</para>
/// <para><b>Shapes:</b> targets are adapted to a column matrix for the underlying
/// <see cref="StandardScaler{T}"/> — <c>Vector&lt;T&gt;</c> ↔ <c>Matrix[n,1]</c>; <c>Tensor&lt;T&gt;</c>
/// rank-1 ↔ <c>Matrix[n,1]</c>; <c>Tensor&lt;T&gt;</c> rank-2 <c>[n,k]</c> ↔ <c>Matrix[n,k]</c> (each
/// output column scaled independently). Other output types are rejected at construction.</para>
/// </remarks>
public class TargetStandardScaler<T, TOutput> : IDataTransformer<T, TOutput, TOutput>
{
    private readonly StandardScaler<T> _scaler;

    public TargetStandardScaler(StandardScaler<T>? scaler = null)
    {
        _scaler = scaler ?? new StandardScaler<T>();
        if (typeof(TOutput) != typeof(Vector<T>) && typeof(TOutput) != typeof(Tensor<T>))
        {
            throw new NotSupportedException(
                $"Target scaling supports Vector<T> and Tensor<T> outputs; got {typeof(TOutput).Name}. " +
                "Classification/label outputs must not be scaled — configure target scaling for regression only.");
        }
    }

    /// <inheritdoc/>
    public bool IsFitted { get; private set; }

    /// <inheritdoc/>
    public bool SupportsInverseTransform => true;

    /// <inheritdoc/>
    public int[]? ColumnIndices => null; // target scaling always applies to every output column

    /// <inheritdoc/>
    public string[] GetFeatureNamesOut(string[]? inputFeatures = null)
        => inputFeatures ?? []; // scaling does not rename target columns

    /// <inheritdoc/>
    public void Fit(TOutput data)
    {
        _scaler.Fit(ToMatrix(data));
        IsFitted = true;
    }

    /// <inheritdoc/>
    public TOutput Transform(TOutput data)
        => FromMatrix(_scaler.Transform(ToMatrix(data)), data);

    /// <inheritdoc/>
    public TOutput FitTransform(TOutput data)
    {
        var scaled = _scaler.FitTransform(ToMatrix(data));
        IsFitted = true;
        return FromMatrix(scaled, data);
    }

    /// <inheritdoc/>
    public TOutput InverseTransform(TOutput data)
        => FromMatrix(_scaler.InverseTransform(ToMatrix(data)), data);

    private static Matrix<T> ToMatrix(TOutput y)
    {
        switch (y)
        {
            case Vector<T> v:
            {
                var m = new Matrix<T>(v.Length, 1);
                for (int i = 0; i < v.Length; i++)
                {
                    m[i, 0] = v[i];
                }

                return m;
            }

            case Tensor<T> t when t.Rank == 1:
            {
                var m = new Matrix<T>(t.Shape[0], 1);
                var span = t.Data.Span;
                for (int i = 0; i < t.Shape[0]; i++)
                {
                    m[i, 0] = span[i];
                }

                return m;
            }

            case Tensor<T> t when t.Rank == 2:
            {
                var m = new Matrix<T>(t.Shape[0], t.Shape[1]);
                var span = t.Data.Span;
                for (int i = 0; i < t.Shape[0]; i++)
                {
                    for (int j = 0; j < t.Shape[1]; j++)
                    {
                        m[i, j] = span[(i * t.Shape[1]) + j];
                    }
                }

                return m;
            }

            default:
                throw new NotSupportedException(
                    $"Target scaling cannot adapt output of type {y?.GetType().Name ?? "null"} (rank > 2 tensors unsupported).");
        }
    }

    private static TOutput FromMatrix(Matrix<T> m, TOutput shapeLike)
    {
        if (shapeLike is Vector<T>)
        {
            var v = new Vector<T>(m.Rows);
            for (int i = 0; i < m.Rows; i++)
            {
                v[i] = m[i, 0];
            }

            return (TOutput)(object)v;
        }

        var rank1 = shapeLike is Tensor<T> t && t.Rank == 1;
        var data = new T[m.Rows * m.Columns];
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                data[(i * m.Columns) + j] = m[i, j];
            }
        }

        var tensorShape = rank1 ? new[] { m.Rows } : new[] { m.Rows, m.Columns };
        return (TOutput)(object)new Tensor<T>(tensorShape, new Vector<T>(data));
    }
}
