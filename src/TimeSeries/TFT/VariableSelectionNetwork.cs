using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.TimeSeries.TFT;

/// <summary>
/// Variable Selection Network (VSN) as described in Lim et al. (2021).
/// Produces feature-selection weights via softmax over per-variable GRN outputs,
/// then returns a weighted sum of individually-processed variable embeddings.
/// </summary>
internal class VariableSelectionNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _numVariables;
    private readonly int _inputDimPerVariable;
    private readonly int _hiddenSize;
    private readonly int _contextSize;

    // Per-variable GRNs: process each variable independently
    private GatedResidualNetwork<T>[] _variableGrns;

    // Flattened GRN for computing selection weights
    private GatedResidualNetwork<T> _flattenedGrn;

    // Selection weight projection: [numVariables, flattenedInputSize]
    private Tensor<T> _selectionWeight;
    private Tensor<T> _selectionBias;

    public VariableSelectionNetwork(
        int numVariables,
        int inputDimPerVariable,
        int hiddenSize,
        int contextSize = 0,
        int? seed = null)
    {
        _numVariables = numVariables;
        _inputDimPerVariable = inputDimPerVariable;
        _hiddenSize = hiddenSize;
        _contextSize = contextSize;

        var random = seed.HasValue ? new Random(seed.Value) : RandomHelper.CreateSeededRandom(42);

        // Per-variable GRNs: each transforms one variable from inputDim to hiddenSize
        _variableGrns = new GatedResidualNetwork<T>[numVariables];
        for (int i = 0; i < numVariables; i++)
        {
            _variableGrns[i] = new GatedResidualNetwork<T>(
                inputDimPerVariable, hiddenSize, hiddenSize, contextSize, random.Next());
        }

        // Flattened GRN for selection weights
        int flattenedSize = numVariables * inputDimPerVariable;
        _flattenedGrn = new GatedResidualNetwork<T>(
            flattenedSize, hiddenSize, numVariables, contextSize, random.Next());

        // Selection weight/bias for softmax
        double std = Math.Sqrt(1.0 / (numVariables * inputDimPerVariable));
        _selectionWeight = CreateRandomTensor([numVariables, flattenedSize], std, random);
        _selectionBias = new Tensor<T>([numVariables]);
    }

    /// <summary>
    /// Forward pass: computes variable selection weights and returns weighted combination.
    /// </summary>
    /// <param name="inputs">List of per-variable input tensors, each [inputDimPerVariable]</param>
    /// <param name="context">Optional static context [contextSize]</param>
    /// <returns>Weighted combination of variable embeddings [hiddenSize]</returns>
    public Tensor<T> Forward(Tensor<T>[] inputs, Tensor<T>? context = null)
    {
        if (inputs.Length != _numVariables)
            throw new ArgumentException($"Expected {_numVariables} variables, got {inputs.Length}.");

        // 1. Flatten all inputs for selection weight computation
        int flatLen = _numVariables * _inputDimPerVariable;
        var flattened = new Tensor<T>([flatLen]);
        var flatSpan = flattened.AsWritableSpan();
        for (int v = 0; v < _numVariables; v++)
        {
            var varSpan = inputs[v].Data.Span;
            int offset = v * _inputDimPerVariable;
            for (int i = 0; i < _inputDimPerVariable && i < varSpan.Length; i++)
            {
                flatSpan[offset + i] = varSpan[i];
            }
        }

        // 2. Compute selection weights via GRN + softmax
        var selectionLogits = _flattenedGrn.Forward(flattened, context);
        var selectionWeights = Softmax(selectionLogits);

        // 3. Process each variable through its own GRN
        var processedVars = new Tensor<T>[_numVariables];
        for (int v = 0; v < _numVariables; v++)
        {
            processedVars[v] = _variableGrns[v].Forward(inputs[v], context);
        }

        // 4. Weighted sum of processed variables
        var output = new Tensor<T>([_hiddenSize]);
        var outSpan = output.AsWritableSpan();
        for (int v = 0; v < _numVariables; v++)
        {
            T weight = selectionWeights[v];
            var varSpan = processedVars[v].Data.Span;
            for (int h = 0; h < _hiddenSize && h < varSpan.Length; h++)
            {
                outSpan[h] = NumOps.Add(outSpan[h], NumOps.Multiply(weight, varSpan[h]));
            }
        }

        return output;
    }

    public IEnumerable<Tensor<T>> GetTrainableParameters()
    {
        foreach (var grn in _variableGrns)
            foreach (var p in grn.GetTrainableParameters())
                yield return p;

        foreach (var p in _flattenedGrn.GetTrainableParameters())
            yield return p;

        yield return _selectionWeight;
        yield return _selectionBias;
    }

    private static Tensor<T> Softmax(Tensor<T> x)
    {
        var span = x.Data.Span;
        int len = span.Length;

        // Find max for numerical stability
        double maxVal = double.NegativeInfinity;
        for (int i = 0; i < len; i++)
        {
            double v = NumOps.ToDouble(span[i]);
            if (v > maxVal) maxVal = v;
        }

        // Compute exp and sum
        var expVals = new double[len];
        double sumExp = 0;
        for (int i = 0; i < len; i++)
        {
            expVals[i] = Math.Exp(NumOps.ToDouble(span[i]) - maxVal);
            sumExp += expVals[i];
        }

        // Normalize
        var result = new Tensor<T>([len]);
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < len; i++)
        {
            rSpan[i] = NumOps.FromDouble(expVals[i] / sumExp);
        }
        return result;
    }

    private static Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = NumOps.FromDouble(normal * stddev);
        }
        return new Tensor<T>(shape, new Vector<T>(data));
    }
}
