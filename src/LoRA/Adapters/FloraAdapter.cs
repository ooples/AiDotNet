using System;
using AiDotNet.Extensions;
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
public partial class FloraAdapter<T> : LoRAAdapterBase<T>
{
    private readonly int _resamplingInterval;
    private readonly int _rank;
    private int _currentStep;
    [AiDotNet.Attributes.Buffer]
    private Matrix<T>? _compressedMomentum;
    [AiDotNet.Attributes.Buffer]
    private Matrix<T>? _compressedSecondMoment;
    private readonly Random _random;
    private readonly double _momentumDecay;
    private readonly double _secondMomentDecay;
    private readonly bool _useAdaptiveLearningRate;

    /// <summary>Construction state: the 'seed' the layer was built with.</summary>
    private readonly int _seed;

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
        _seed = seed;
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

        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        if (_useAdaptiveLearningRate)
        {
            _compressedSecondMoment = new Matrix<T>(rank, outputSize);
        }
    }

    public int ResamplingInterval => _resamplingInterval;
    public int CurrentStep => _currentStep;

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _currentStep++;

        if (_currentStep % _resamplingInterval == 0)
        {
            ResampleProjectionMatrices();
        }

        Vector<T> loraGradients = _loraLayer.GetParameterGradients();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        _compressedMomentum ??= new Matrix<T>(_rank, outputSize);

        int bOffset = inputSize * _rank;
        Tensor<T> gradB = Engine.Reshape(
            Tensor<T>.FromVector(loraGradients.GetSubVector(bOffset, _rank * outputSize)), [_rank, outputSize]);

        T beta1 = NumOps.FromDouble(_momentumDecay);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _momentumDecay);

        // m = beta1 * m + (1 - beta1) * g
        _compressedMomentum = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(_compressedMomentum), beta1),
            Engine.TensorMultiplyScalar(gradB, oneMinusBeta1)).ToMatrix();

        if (_useAdaptiveLearningRate)
        {
            T beta2 = NumOps.FromDouble(_secondMomentDecay);
            T oneMinusBeta2 = NumOps.FromDouble(1.0 - _secondMomentDecay);

            // v = beta2 * v + (1 - beta2) * g^2
            Matrix<T> oldSecondMoment = _compressedSecondMoment
                ?? throw new InvalidOperationException("_compressedSecondMoment has not been initialized.");
            _compressedSecondMoment = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(oldSecondMoment), beta2),
                Engine.TensorMultiplyScalar(Engine.TensorMultiply(gradB, gradB), oneMinusBeta2)).ToMatrix();
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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        int bOffset = inputSize * _rank;
        Matrix<T> oldA = Engine.Reshape(
            Tensor<T>.FromVector(currentParams.GetSubVector(0, bOffset)), [inputSize, _rank]).ToMatrix();


        Matrix<T> newA = new Matrix<T>(inputSize, _rank);
        double stddev = 1.0 / Math.Sqrt(_rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < _rank; j++)
            {
                newA[i, j] = NumOps.FromDouble(_random.NextGaussian() * stddev);
            }
        }

        Matrix<T> transferMatrix = ComputeTransferMatrix(oldA, newA);
        // Correct order: transferMatrix * momentum (project momentum into new parameter space)
        Matrix<T> newMomentum = MultiplyMatrices(transferMatrix, (_compressedMomentum ?? throw new InvalidOperationException("_compressedMomentum is null where it is required.")));
        _compressedMomentum = newMomentum;

        if (_useAdaptiveLearningRate && _compressedSecondMoment != null)
        {
            // Same for second moment
            Matrix<T> newSecondMoment = MultiplyMatrices(transferMatrix, _compressedSecondMoment);
            _compressedSecondMoment = newSecondMoment;
        }

        // New A (row-major) followed by the unchanged B parameters
        Vector<T> newParams = currentParams.Length > bOffset
            ? Vector<T>.Concatenate(newA.ToRowVector(), currentParams.GetSubVector(bOffset, currentParams.Length - bOffset))
            : newA.ToRowVector();

        _loraLayer.SetParameters(newParams);
    }

    private Matrix<T> ComputeTransferMatrix(Matrix<T> oldA, Matrix<T> newA)
    {
        // Transfer matrix = oldA^T @ newA â€” vectorized via Engine.TensorMatMul
        var oldATensor = Tensor<T>.FromMatrix(oldA).Transpose(new[] { 1, 0 });
        var newATensor = Tensor<T>.FromMatrix(newA);
        var resultTensor = Engine.TensorMatMul(oldATensor, newATensor);
        return resultTensor.ToMatrix();
    }

    private Matrix<T> MultiplyMatrices(Matrix<T> a, Matrix<T> b)
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException($"Matrix dimensions incompatible for multiplication: ({a.Rows}Ã—{a.Columns}) Ã— ({b.Rows}Ã—{b.Columns})");
        }

        // a @ b â€” vectorized via Engine.TensorMatMul
        var aTensor = Tensor<T>.FromMatrix(a);
        var bTensor = Tensor<T>.FromMatrix(b);
        var resultTensor = Engine.TensorMatMul(aTensor, bTensor);
        return resultTensor.ToMatrix();
    }

    /// <inheritdoc />
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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int weightCount = inputSize * outputSize;

        // Merge weights. loraWeights is [inputSize, outputSize]; DenseLayer stores its weights
        // input-major and FullyConnectedLayer output-major, so the latter uses the transpose.
        // Biases are kept unchanged.
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        Vector<T> mergedParams = AddFlatWeightsToParameters(
            baseParams, isOutputMajor ? loraWeights.Transpose() : loraWeights, weightCount);

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        base.ResetState();
        _currentStep = 0;
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        _compressedMomentum = new Matrix<T>(_rank, outputSize);

        if (_useAdaptiveLearningRate)
        {
            _compressedSecondMoment = new Matrix<T>(_rank, outputSize);
        }
    }

    private void SyncParametersFromLayers()
    {
        // [base parameters (when trainable)..., LoRA parameters...] written into Parameters in one bulk copy
        Vector<T> loraParams = _loraLayer.GetParameters();
        Vector<T> combined = _freezeBaseLayer
            ? loraParams
            : Vector<T>.Concatenate(_baseLayer.GetParameters(), loraParams);
        combined.AsSpan().CopyTo(Parameters.AsWritableSpan());
    }
}
