using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Regression.MultiOutput;

/// <summary>
/// Wraps a single-output regression model into a MULTI-OUTPUT one: given an n×H target, it trains H independent
/// "head" models (one per target column, the direct-multi-horizon strategy) and predicts an n×H matrix. This is
/// the regression analogue of the multi-label classifier chain, and the substrate the shape-polymorphic facade
/// routes to when the training target has more than one column (e.g. a multi-horizon forecast vector).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> some models (linear/forest/boosting) only predict a single number. To forecast several
/// horizons at once (1, 5, 10, 20 bars ahead), this trains one copy per horizon behind a single object, so the
/// caller trains once with an n×H target and gets an n×H prediction back — no per-horizon plumbing.
/// Deep sequence models that emit a horizon vector natively don't need this; they output n×H directly.
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Regression)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Multi-Target Regression via Input Space Expansion", "https://doi.org/10.1007/s10994-016-5546-z", Year = 2016, Authors = "Eleftherios Spyromitros-Xioufis, Grigorios Tsoumakas, William Groves, Ioannis Vlahavas")]
public sealed class MultiOutputRegressor<T>
    : IFullModel<T, Matrix<T>, Matrix<T>>, IParameterizable<T, Matrix<T>, Matrix<T>>, IFeatureAware
{
    private readonly Func<IFullModel<T, Matrix<T>, Vector<T>>> _baseFactory;
    private readonly List<IFullModel<T, Matrix<T>, Vector<T>>> _heads = new();
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <param name="baseFactory">Creates a fresh single-output model for each output column.</param>
    public MultiOutputRegressor(Func<IFullModel<T, Matrix<T>, Vector<T>>> baseFactory)
        => _baseFactory = baseFactory ?? throw new ArgumentNullException(nameof(baseFactory));

    /// <summary>Number of output columns (horizons) the last <see cref="Train"/> produced.</summary>
    public int OutputCount => _heads.Count;

    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    public void Train(Matrix<T> input, Matrix<T> expectedOutput)
    {
        if (expectedOutput.Columns < 1)
        {
            throw new ArgumentException("Multi-output target must have at least one column.", nameof(expectedOutput));
        }

        _heads.Clear();
        for (var j = 0; j < expectedOutput.Columns; j++)
        {
            var yj = new Vector<T>(expectedOutput.Rows);
            for (var i = 0; i < expectedOutput.Rows; i++)
            {
                yj[i] = expectedOutput[i, j];
            }

            var head = _baseFactory();
            head.Train(input, yj);
            _heads.Add(head);
        }
    }

    public Matrix<T> Predict(Matrix<T> input)
    {
        if (_heads.Count == 0)
        {
            throw new InvalidOperationException(
                "MultiOutputRegressor has not been trained. Call Train(input, expectedOutput) before Predict().");
        }

        var result = new Matrix<T>(input.Rows, _heads.Count);
        for (var j = 0; j < _heads.Count; j++)
        {
            var col = _heads[j].Predict(input);
            if (col.Length != input.Rows)
            {
                throw new InvalidOperationException(
                    $"MultiOutputRegressor head {j} predicted {col.Length} values for {input.Rows} input rows.");
            }
            for (var i = 0; i < input.Rows; i++)
            {
                result[i, j] = col[i];
            }
        }

        return result;
    }

    public ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = $"MultiOutputRegressor[{_heads.Count}]",
        FeatureCount = _heads.Count > 0 ? _heads[0].GetModelMetadata().FeatureCount : 0,
        Complexity = _heads.Sum(h => h.GetModelMetadata().Complexity),
        Description = "H single-output regression heads (one per target column / horizon).",
    };

    // --- IParameterizable: parameters are the concatenation of every head's parameters, in head order. ---

    public long ParameterCount => _heads.Sum(h => AsParameterizable(h)?.ParameterCount ?? 0L);

    public bool SupportsParameterInitialization => ParameterCount > 0;

    public Vector<T> GetParameters()
    {
        var all = new List<T>();
        foreach (var head in _heads)
        {
            var p = AsParameterizable(head)?.GetParameters();
            if (p is null)
            {
                continue;
            }

            for (var i = 0; i < p.Length; i++)
            {
                all.Add(p[i]);
            }
        }

        return new Vector<T>(all.ToArray());
    }

    public void SetParameters(Vector<T> parameters)
    {
        var offset = 0;
        foreach (var head in _heads)
        {
            var param = AsParameterizable(head);
            if (param is null)
            {
                continue;
            }

            var count = (int)param.ParameterCount;
            var seg = new T[count];
            for (var i = 0; i < count; i++)
            {
                seg[i] = parameters[offset + i];
            }

            param.SetParameters(new Vector<T>(seg));
            offset += count;
        }
    }

    private static IParameterizable<T, Matrix<T>, Vector<T>>? AsParameterizable(
        IFullModel<T, Matrix<T>, Vector<T>> head) => head as IParameterizable<T, Matrix<T>, Vector<T>>;

    public IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (MultiOutputRegressor<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    public Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    // --- IFeatureImportance: aggregate across heads. ---

    public Dictionary<string, T> GetFeatureImportance()
    {
        var acc = new Dictionary<string, T>();
        foreach (var head in _heads)
        {
            foreach (var kv in head.GetFeatureImportance())
            {
                acc[kv.Key] = acc.TryGetValue(kv.Key, out var existing) ? NumOps.Add(existing, kv.Value) : kv.Value;
            }
        }

        if (_heads.Count > 1)
        {
            var denom = NumOps.FromDouble(_heads.Count);
            foreach (var key in acc.Keys.ToList())
            {
                acc[key] = NumOps.Divide(acc[key], denom);
            }
        }

        return acc;
    }

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        var idx = featureIndices.ToList();
        foreach (var head in _heads)
        {
            if (head is IFeatureAware fa)
            {
                fa.SetActiveFeatureIndices(idx);
            }
        }
    }

    public bool IsFeatureUsed(int featureIndex)
        => _heads.Any(h => h is IFeatureAware fa && fa.IsFeatureUsed(featureIndex));

    public IEnumerable<int> GetActiveFeatureIndices()
        => _heads.OfType<IFeatureAware>().SelectMany(fa => fa.GetActiveFeatureIndices()).Distinct();

    // --- Serialization: length-prefixed concatenation of each head's bytes. ---

    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(_heads.Count);
        foreach (var head in _heads)
        {
            var bytes = head.Serialize();
            w.Write(bytes.Length);
            w.Write(bytes);
        }

        return ms.ToArray();
    }

    public void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var r = new BinaryReader(ms);
        var count = r.ReadInt32();
        _heads.Clear();
        for (var j = 0; j < count; j++)
        {
            var len = r.ReadInt32();
            var bytes = r.ReadBytes(len);
            var head = _baseFactory();
            head.Deserialize(bytes);
            _heads.Add(head);
        }
    }

    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());

    public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

    public void SaveState(Stream stream)
    {
        // Delegate to the (verified round-tripping) Serialize path, length-prefixed so the stream can carry more.
        var bytes = Serialize();
        var w = new BinaryWriter(stream);
        w.Write(bytes.Length);
        w.Write(bytes);
        w.Flush();
    }

    public void LoadState(Stream stream)
    {
        var r = new BinaryReader(stream);
        var len = r.ReadInt32();
        var bytes = r.ReadBytes(len);
        Deserialize(bytes);
    }

    public IFullModel<T, Matrix<T>, Matrix<T>> Clone()
    {
        var clone = new MultiOutputRegressor<T>(_baseFactory);
        foreach (var head in _heads)
        {
            clone._heads.Add((IFullModel<T, Matrix<T>, Vector<T>>)head.Clone());
        }

        return clone;
    }

    public IFullModel<T, Matrix<T>, Matrix<T>> DeepCopy() => Clone();

    public void Dispose()
    {
        foreach (var head in _heads)
        {
            head.Dispose();
        }

        _heads.Clear();
    }
}
