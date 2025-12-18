using System;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Flora reinterprets LoRA as a gradient compression mechanism and achieves high-rank updates through
/// periodic resampling of projection matrices while maintaining sublinear space complexity for optimizer states.
/// </para>
/// <para><b>Research Paper:</b> "Flora: Low-Rank Adapters Are Secretly Gradient Compressors"
/// by Yongchang Hao et al., ICML 2024. arXiv:2402.03293
/// </para>
/// <para><b>Key Innovation:</b> Unlike standard LoRA which restricts weight updates to a fixed low-rank subspace,
/// Flora periodically resamples the projection matrices (A and B), allowing the effective rank of cumulative
/// updates to grow over time. This achieves performance comparable to full-rank fine-tuning while maintaining
/// the memory efficiency of LoRA.
/// </para>
/// </remarks>
public class FloraAdapter<T> : LoRAAdapterBase<T>
{
    private readonly int _resamplingInterval;
    private readonly int _rank;
    private int _currentStep;
    private Matrix<T>? _compressedMomentum;
    private Matrix<T>? _compressedSecondMoment;
    private readonly Random _random;
    private readonly double _momentumDecay;
    private readonly double _secondMomentDecay;
    private readonly bool _useAdaptiveLearningRate;

    public FloraAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        int resamplingInterval = 1000,
        double momentumDecay = 0.9,
        double secondMomentDecay = 0.999,
        bool useAdaptiveLearningRate = true,
        bool freezeBaseLayer = true,
        int seed = 42)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (resamplingInterval < 1)
        {
            throw new ArgumentException("Resampling interval must be at least 1", nameof(resamplingInterval));
        }

        _resamplingInterval = resamplingInterval;
        _rank = rank;
        _currentStep = 0;
        _momentumDecay = momentumDecay;
        _secondMomentDecay = secondMomentDecay;
        _useAdaptiveLearningRate = useAdaptiveLearningRate;
        _random = RandomHelper.CreateSeededRandom(seed);

        int outputSize = GetOutputShape()[0];
        _compressedMomentum = new Matrix<T>(rank, outputSize);

        if (_useAdaptiveLearningRate)
        {
            _compressedSecondMoment = new Matrix<T>(rank, outputSize);
        }
    }

    public int ResamplingInterval => _resamplingInterval;
    public int CurrentStep => _currentStep;

    public override void UpdateParameters(T learningRate)
    {
        _currentStep++;

        if (_currentStep % _resamplingInterval == 0)
        {
            ResampleProjectionMatrices();
        }

        Vector<T> loraGradients = _loraLayer.GetParameterGradients();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        Matrix<T> gradB = new Matrix<T>(_rank, outputSize);
        int bOffset = inputSize * _rank;

        for (int i = 0; i < _rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradB[i, j] = loraGradients[bOffset + i * outputSize + j];
            }
        }

        T beta1 = NumOps.FromDouble(_momentumDecay);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _momentumDecay);

        for (int i = 0; i < _rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                T oldMomentum = _compressedMomentum![i, j];
                T newMomentum = NumOps.Add(
                    NumOps.Multiply(beta1, oldMomentum),
                    NumOps.Multiply(oneMinusBeta1, gradB[i, j])
                );
                _compressedMomentum[i, j] = newMomentum;
            }
        }

        if (_useAdaptiveLearningRate)
        {
            T beta2 = NumOps.FromDouble(_secondMomentDecay);
            T oneMinusBeta2 = NumOps.FromDouble(1.0 - _secondMomentDecay);

            for (int i = 0; i < _rank; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    T grad = gradB[i, j];
                    T gradSquared = NumOps.Multiply(grad, grad);
                    T oldSecondMoment = _compressedSecondMoment![i, j];
                    T newSecondMoment = NumOps.Add(
                        NumOps.Multiply(beta2, oldSecondMoment),
                        NumOps.Multiply(oneMinusBeta2, gradSquared)
                    );
                    _compressedSecondMoment[i, j] = newSecondMoment;
                }
            }
        }

        _loraLayer.UpdateParameters(learningRate);

        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        SyncParametersFromLayers();
    }

    private void ResampleProjectionMatrices()
    {
        Vector<T> currentParams = _loraLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        Matrix<T> oldA = new Matrix<T>(inputSize, _rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < _rank; j++)
            {
                oldA[i, j] = currentParams[i * _rank + j];
            }
        }

        Matrix<T> newA = new Matrix<T>(inputSize, _rank);
        double stddev = 1.0 / Math.Sqrt(_rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < _rank; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double gaussianValue = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                newA[i, j] = NumOps.FromDouble(gaussianValue * stddev);
            }
        }

        Matrix<T> transferMatrix = ComputeTransferMatrix(oldA, newA);
        // Correct order: transferMatrix * momentum (project momentum into new parameter space)
        Matrix<T> newMomentum = MultiplyMatrices(transferMatrix, _compressedMomentum!);
        _compressedMomentum = newMomentum;

        if (_useAdaptiveLearningRate && _compressedSecondMoment != null)
        {
            // Same for second moment
            Matrix<T> newSecondMoment = MultiplyMatrices(transferMatrix, _compressedSecondMoment);
            _compressedSecondMoment = newSecondMoment;
        }

        Vector<T> newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < _rank; j++)
            {
                newParams[i * _rank + j] = newA[i, j];
            }
        }

        int bOffset = inputSize * _rank;
        for (int i = bOffset; i < currentParams.Length; i++)
        {
            newParams[i] = currentParams[i];
        }

        _loraLayer.SetParameters(newParams);
    }

    private Matrix<T> ComputeTransferMatrix(Matrix<T> oldA, Matrix<T> newA)
    {
        Matrix<T> result = new Matrix<T>(_rank, _rank);
        int inputSize = GetInputShape()[0];

        for (int i = 0; i < _rank; i++)
        {
            for (int j = 0; j < _rank; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < inputSize; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(oldA[k, i], newA[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    private Matrix<T> MultiplyMatrices(Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Columns;

        if (n != b.Rows)
        {
            throw new ArgumentException($"Matrix dimensions incompatible for multiplication: ({m}×{n}) × ({b.Rows}×{p})");
        }

        Matrix<T> result = new Matrix<T>(m, p);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(a[i, k], b[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException(
                "FloraAdapter merging only supports DenseLayer or FullyConnectedLayer base layers. " +
                $"Got: {_baseLayer.GetType().Name}");
        }

        // Get the LoRA weight contribution from the underlying LoRA layer
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters (works for both DenseLayer and FullyConnectedLayer)
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Both DenseLayer and FullyConnectedLayer store parameters as [weights..., biases...]
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights: baseWeight + loraWeight
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged (Flora/LoRA doesn't modify biases)
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    public override void ResetState()
    {
        base.ResetState();
        _currentStep = 0;
        int outputSize = GetOutputShape()[0];
        _compressedMomentum = new Matrix<T>(_rank, outputSize);

        if (_useAdaptiveLearningRate)
        {
            _compressedSecondMoment = new Matrix<T>(_rank, outputSize);
        }
    }

    private void SyncParametersFromLayers()
    {
        int idx = 0;

        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }
}
