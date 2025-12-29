using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

internal class LinearVectorModel : IFullModel<double, Matrix<double>, Vector<double>>, ICloneable
{
    private Vector<double> _parameters;
    protected readonly int _inputFeatures;

    public LinearVectorModel(int inputFeatures)
    {
        _inputFeatures = inputFeatures;
        _parameters = new Vector<double>(inputFeatures + 1);
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] = 0.01 * (i + 1);
        }
    }

    public Vector<double> Predict(Matrix<double> input)
    {
        var output = new Vector<double>(input.Rows);
        for (int r = 0; r < input.Rows; r++)
        {
            double sum = _parameters[_inputFeatures];
            for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
            {
                sum += input[r, c] * _parameters[c];
            }
            output[r] = sum;
        }
        return output;
    }

    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, 0.01);
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            Name = "LinearVectorModel",
            FeatureCount = _inputFeatures,
            Complexity = _parameters.Length
        };
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != _parameters.Length)
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var model = new LinearVectorModel(_inputFeatures);
        model.SetParameters(parameters);
        return model;
    }

    public virtual IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new LinearVectorModel(_inputFeatures);
        copy.SetParameters(_parameters);
        return copy;
    }

    public virtual IFullModel<double, Matrix<double>, Vector<double>> Clone()
    {
        return DeepCopy();
    }

    object ICloneable.Clone()
    {
        return DeepCopy();
    }

    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
    {
        var gradients = new Vector<double>(_parameters.Length);
        int count = Math.Min(input.Rows, target.Length);
        if (count == 0)
        {
            return gradients;
        }

        var predictions = Predict(input);
        double scale = 2.0 / count;
        for (int r = 0; r < count; r++)
        {
            double error = predictions[r] - target[r];
            for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
            {
                gradients[c] += scale * error * input[r, c];
            }
            gradients[_inputFeatures] += scale * error;
        }

        return gradients;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        int length = Math.Min(gradients.Length, _parameters.Length);
        for (int i = 0; i < length; i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    public byte[] Serialize()
    {
        return Encoding.UTF8.GetBytes(SerializeParameters());
    }

    public void Deserialize(byte[] data)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        DeserializeParameters(Encoding.UTF8.GetString(data));
    }

    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path is required.", nameof(filePath));
        File.WriteAllText(filePath, SerializeParameters());
    }

    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path is required.", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Model file not found.", filePath);
        DeserializeParameters(File.ReadAllText(filePath));
    }

    public void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
        writer.Write(SerializeParameters());
        writer.Flush();
    }

    public void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        using var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen: true);
        DeserializeParameters(reader.ReadToEnd());
    }

    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatures);

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
    }

    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputFeatures;

    public Dictionary<string, double> GetFeatureImportance()
    {
        var importance = new Dictionary<string, double>();
        if (_inputFeatures == 0)
        {
            return importance;
        }

        double score = 1.0 / _inputFeatures;
        for (int i = 0; i < _inputFeatures; i++)
        {
            importance[$"feature_{i}"] = score;
        }

        return importance;
    }

    public bool SupportsJitCompilation => false;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported.");
    }

    private string SerializeParameters()
    {
        return string.Join(",", _parameters.Select(p => p.ToString("R", CultureInfo.InvariantCulture)));
    }

    private void DeserializeParameters(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            _parameters = new Vector<double>(_parameters.Length);
            return;
        }

        var parts = content.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        var vector = new Vector<double>(parts.Length);
        for (int i = 0; i < parts.Length; i++)
        {
            vector[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
        }

        _parameters = vector;
    }
}

internal class SecondOrderMatrixModel : LinearVectorModel, ISecondOrderGradientComputable<double, Matrix<double>, Vector<double>>
{
    public SecondOrderMatrixModel(int inputFeatures) : base(inputFeatures)
    {
    }

    public Vector<double> ComputeSecondOrderGradients(
        List<(Matrix<double> input, Vector<double> target)> adaptationSteps,
        Matrix<double> queryInput,
        Vector<double> queryTarget,
        ILossFunction<double> lossFunction,
        double innerLearningRate)
    {
        // Numerical approximation of gradients with respect to the model parameters.
        // We define a scalar meta-loss as the sum of losses on all adaptation steps
        // plus the loss on the query input/target, and compute central-difference
        // derivatives of this loss w.r.t. each parameter.

        // Helper to evaluate the meta-loss for a given parameter vector.
        double EvaluateMetaLoss(Vector<double> parameters)
        {
            // Work on a copy of the model so we do not mutate the original.
            var modelCopy = (SecondOrderMatrixModel)DeepCopy();
            modelCopy.SetParameters(parameters);

            double totalLoss = 0.0;

            // Include losses over all adaptation steps.
            if (adaptationSteps != null)
            {
                foreach (var (input, target) in adaptationSteps)
                {
                    var prediction = modelCopy.Predict(input);
                    totalLoss += lossFunction.CalculateLoss(prediction, target);
                }
            }

            // Include loss on the query input/target.
            var queryPrediction = modelCopy.Predict(queryInput);
            totalLoss += lossFunction.CalculateLoss(queryPrediction, queryTarget);

            return totalLoss;
        }

        var baseParameters = GetParameters();
        var gradients = new Vector<double>(ParameterCount);

        // Finite-difference step size.
        const double epsilon = 1e-4;

        // Compute central-difference gradient for each parameter.
        for (int i = 0; i < gradients.Length; i++)
        {
            var plusParams = new Vector<double>(baseParameters.Length);
            var minusParams = new Vector<double>(baseParameters.Length);

            for (int j = 0; j < baseParameters.Length; j++)
            {
                plusParams[j] = baseParameters[j];
                minusParams[j] = baseParameters[j];
            }

            plusParams[i] += epsilon;
            minusParams[i] -= epsilon;

            double lossPlus = EvaluateMetaLoss(plusParams);
            double lossMinus = EvaluateMetaLoss(minusParams);

            gradients[i] = (lossPlus - lossMinus) / (2.0 * epsilon);
        }

        return gradients;
    }

    public override IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new SecondOrderMatrixModel(_inputFeatures);
        copy.SetParameters(GetParameters());
        return copy;
    }

    public override IFullModel<double, Matrix<double>, Vector<double>> Clone()
    {
        return DeepCopy();
    }
}

internal class TensorEmbeddingModel : IFullModel<double, Matrix<double>, Tensor<double>>, ICloneable
{
    private Vector<double> _parameters;
    private readonly int _inputFeatures;
    private readonly int _embeddingDim;

    public TensorEmbeddingModel(int inputFeatures, int embeddingDim)
    {
        _inputFeatures = inputFeatures;
        _embeddingDim = embeddingDim;
        _parameters = new Vector<double>(inputFeatures * embeddingDim + embeddingDim);
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] = 0.01 * (i + 1);
        }
    }

    public Tensor<double> Predict(Matrix<double> input)
    {
        var output = new Tensor<double>(new[] { input.Rows, _embeddingDim });
        for (int r = 0; r < input.Rows; r++)
        {
            for (int e = 0; e < _embeddingDim; e++)
            {
                double sum = _parameters[_inputFeatures * _embeddingDim + e];
                int weightOffset = e * _inputFeatures;
                for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
                {
                    sum += input[r, c] * _parameters[weightOffset + c];
                }
                output[new[] { r, e }] = sum;
            }
        }
        return output;
    }

    public void Train(Matrix<double> input, Tensor<double> expectedOutput)
    {
        var gradients = ComputeGradients(input, expectedOutput, DefaultLossFunction);
        ApplyGradients(gradients, 0.01);
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            Name = "TensorEmbeddingModel",
            FeatureCount = _inputFeatures,
            Complexity = _parameters.Length
        };
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != _parameters.Length)
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var model = new TensorEmbeddingModel(_inputFeatures, _embeddingDim);
        model.SetParameters(parameters);
        return model;
    }

    public IFullModel<double, Matrix<double>, Tensor<double>> DeepCopy()
    {
        var copy = new TensorEmbeddingModel(_inputFeatures, _embeddingDim);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<double, Matrix<double>, Tensor<double>> Clone()
    {
        return DeepCopy();
    }

    object ICloneable.Clone()
    {
        return DeepCopy();
    }

    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Matrix<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
    {
        var gradients = new Vector<double>(_parameters.Length);
        int count = Math.Min(input.Rows, target.Shape[0]);
        if (count == 0)
        {
            return gradients;
        }

        var predictions = Predict(input);

        // Handle 1D target (class labels) vs 2D target (embeddings)
        bool targetIs2D = target.Shape.Length >= 2 && target.Shape[1] == _embeddingDim;

        if (targetIs2D)
        {
            // Standard MSE gradient when target is embedding
            double scale = 2.0 / (count * _embeddingDim);

            for (int r = 0; r < count; r++)
            {
                for (int e = 0; e < _embeddingDim; e++)
                {
                    double error = predictions[new[] { r, e }] - target[new[] { r, e }];

                    // Gradient for weights
                    int weightOffset = e * _inputFeatures;
                    for (int c = 0; c < _inputFeatures && c < input.Columns; c++)
                    {
                        gradients[weightOffset + c] += scale * error * input[r, c];
                    }

                    // Gradient for bias
                    int biasIndex = _inputFeatures * _embeddingDim + e;
                    gradients[biasIndex] += scale * error;
                }
            }
        }
        else
        {
            // For 1D target (class labels), use numerical gradient approximation
            // This handles the case where target contains class indices rather than embedding vectors
            const double epsilon = 1e-4;
            var baseParams = GetParameters();

            for (int i = 0; i < gradients.Length; i++)
            {
                // Perturb parameter positively
                var plusParams = new Vector<double>(baseParams.Length);
                var minusParams = new Vector<double>(baseParams.Length);
                for (int j = 0; j < baseParams.Length; j++)
                {
                    plusParams[j] = baseParams[j];
                    minusParams[j] = baseParams[j];
                }
                plusParams[i] += epsilon;
                minusParams[i] -= epsilon;

                // Compute loss with perturbed parameters
                SetParameters(plusParams);
                var predPlus = Predict(input);
                double lossPlus = ComputeMSELossForClassLabels(predPlus, target, count);

                SetParameters(minusParams);
                var predMinus = Predict(input);
                double lossMinus = ComputeMSELossForClassLabels(predMinus, target, count);

                // Central difference gradient
                gradients[i] = (lossPlus - lossMinus) / (2.0 * epsilon);
            }

            // Restore original parameters
            SetParameters(baseParams);
        }

        return gradients;
    }

    private double ComputeMSELossForClassLabels(Tensor<double> predictions, Tensor<double> target, int count)
    {
        // Compute loss using first embedding dimension vs class label
        double loss = 0.0;
        for (int r = 0; r < count; r++)
        {
            double pred = predictions[new[] { r, 0 }];
            double label = target[new[] { r }];
            double error = pred - label;
            loss += error * error;
        }
        return loss / count;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        int length = Math.Min(gradients.Length, _parameters.Length);
        for (int i = 0; i < length; i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    public byte[] Serialize()
    {
        return Encoding.UTF8.GetBytes(SerializeParameters());
    }

    public void Deserialize(byte[] data)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        DeserializeParameters(Encoding.UTF8.GetString(data));
    }

    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path is required.", nameof(filePath));
        File.WriteAllText(filePath, SerializeParameters());
    }

    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path is required.", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Model file not found.", filePath);
        DeserializeParameters(File.ReadAllText(filePath));
    }

    public void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        using var writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen: true);
        writer.Write(SerializeParameters());
        writer.Flush();
    }

    public void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        using var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen: true);
        DeserializeParameters(reader.ReadToEnd());
    }

    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatures);

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
    }

    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputFeatures;

    public Dictionary<string, double> GetFeatureImportance()
    {
        var importance = new Dictionary<string, double>();
        if (_inputFeatures == 0)
        {
            return importance;
        }

        double score = 1.0 / _inputFeatures;
        for (int i = 0; i < _inputFeatures; i++)
        {
            importance[$"feature_{i}"] = score;
        }

        return importance;
    }

    public bool SupportsJitCompilation => false;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported.");
    }

    private string SerializeParameters()
    {
        return string.Join(",", _parameters.Select(p => p.ToString("R", CultureInfo.InvariantCulture)));
    }

    private void DeserializeParameters(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            _parameters = new Vector<double>(_parameters.Length);
            return;
        }

        var parts = content.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        var vector = new Vector<double>(parts.Length);
        for (int i = 0; i < parts.Length; i++)
        {
            vector[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
        }

        _parameters = vector;
    }
}
