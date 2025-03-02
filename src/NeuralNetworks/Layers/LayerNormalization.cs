namespace AiDotNet.NeuralNetworks.Layers;

public class LayerNormalization<T> : LayerBase<T>
{
    private Vector<T> _gamma; // Scale parameter
    private Vector<T> _beta;  // Shift parameter
    
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastVariance;
    
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;
    
    private readonly T _epsilon;
    private readonly int _normalizationAxis;
    
    public override bool SupportsTraining => true;

    public LayerNormalization(int[] shape, double epsilon = 1e-5, int normalizationAxis = -1)
        : base(shape, shape)
    {
        // Default epsilon value to prevent division by zero
        _epsilon = NumOps.FromDouble(epsilon);
        
        // Determine the normalization axis
        _normalizationAxis = normalizationAxis < 0 ? shape.Length - 1 : normalizationAxis;
        
        // Initialize gamma (scale) to ones and beta (shift) to zeros
        int featureSize = shape[_normalizationAxis];
        _gamma = new Vector<T>(featureSize);
        _beta = new Vector<T>(featureSize);
        
        for (int i = 0; i < featureSize; i++)
        {
            _gamma[i] = NumOps.One;
            _beta[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[_normalizationAxis];
        
        // Calculate mean and variance along the normalization axis
        _lastMean = CalculateMean(input);
        _lastVariance = CalculateVariance(input, _lastMean);
        
        // Normalize the input
        _lastNormalized = Normalize(input, _lastMean, _lastVariance);
        
        // Scale and shift
        var output = ScaleAndShift(_lastNormalized);
        
        return output;
    }

    private Tensor<T> CalculateMean(Tensor<T> input)
    {
        int[] meanShape = new int[input.Shape.Length];
        for (int i = 0; i < meanShape.Length; i++)
        {
            meanShape[i] = i == _normalizationAxis ? 1 : input.Shape[i];
        }
        
        var mean = new Tensor<T>(meanShape);
        int featureSize = input.Shape[_normalizationAxis];
        
        // Calculate mean for each feature
        for (int i = 0; i < input.Length; i++)
        {
            // Calculate indices for the input tensor
            int[] indices = GetIndices(i, input.Shape);
            
            // Calculate index for the mean tensor
            int[] meanIndices = new int[indices.Length];
            for (int j = 0; j < indices.Length; j++)
            {
                meanIndices[j] = j == _normalizationAxis ? 0 : indices[j];
            }
            
            mean[meanIndices] = NumOps.Add(mean[meanIndices], input[indices]);
        }
        
        // Divide by the number of elements
        for (int i = 0; i < mean.Length; i++)
        {
            mean[i] = NumOps.Divide(mean[i], NumOps.FromDouble(featureSize));
        }
        
        return mean;
    }

    private Tensor<T> CalculateVariance(Tensor<T> input, Tensor<T> mean)
    {
        int[] varianceShape = new int[input.Shape.Length];
        for (int i = 0; i < varianceShape.Length; i++)
        {
            varianceShape[i] = i == _normalizationAxis ? 1 : input.Shape[i];
        }
        
        var variance = new Tensor<T>(varianceShape);
        int featureSize = input.Shape[_normalizationAxis];
        
        // Calculate variance for each feature
        for (int i = 0; i < input.Length; i++)
        {
            // Calculate indices for the input tensor
            int[] indices = GetIndices(i, input.Shape);
            
            // Calculate index for the variance tensor
            int[] varianceIndices = new int[indices.Length];
            for (int j = 0; j < indices.Length; j++)
            {
                varianceIndices[j] = j == _normalizationAxis ? 0 : indices[j];
            }
            
            // Get the corresponding mean
            int[] meanIndices = new int[indices.Length];
            for (int j = 0; j < indices.Length; j++)
            {
                meanIndices[j] = j == _normalizationAxis ? 0 : indices[j];
            }
            
            // Calculate (x - mean)^2
            T diff = NumOps.Subtract(input[indices], mean[meanIndices]);
            variance[varianceIndices] = NumOps.Add(variance[varianceIndices], NumOps.Multiply(diff, diff));
        }
        
        // Divide by the number of elements
        for (int i = 0; i < variance.Length; i++)
        {
            variance[i] = NumOps.Divide(variance[i], NumOps.FromDouble(featureSize));
        }
        
        return variance;
    }

    private Tensor<T> Normalize(Tensor<T> input, Tensor<T> mean, Tensor<T> variance)
    {
        var normalized = new Tensor<T>(input.Shape);
        
        for (int i = 0; i < input.Length; i++)
        {
            // Calculate indices for the input tensor
            int[] indices = GetIndices(i, input.Shape);
            
            // Calculate index for the mean and variance tensors
            int[] meanVarIndices = new int[indices.Length];
            for (int j = 0; j < indices.Length; j++)
            {
                meanVarIndices[j] = j == _normalizationAxis ? 0 : indices[j];
            }
            
            // Normalize: (x - mean) / sqrt(variance + epsilon)
            T diff = NumOps.Subtract(input[indices], mean[meanVarIndices]);
            T stdDev = NumOps.Sqrt(NumOps.Add(variance[meanVarIndices], _epsilon));
            normalized[indices] = NumOps.Divide(diff, stdDev);
        }
        
        return normalized;
    }

    private Tensor<T> ScaleAndShift(Tensor<T> normalized)
    {
        var output = new Tensor<T>(normalized.Shape);
        
        for (int i = 0; i < normalized.Length; i++)
        {
            // Calculate indices for the normalized tensor
            int[] indices = GetIndices(i, normalized.Shape);
            
            // Get the feature index
            int featureIndex = indices[_normalizationAxis];
            
            // Scale and shift: gamma * normalized + beta
            output[indices] = NumOps.Add(
                NumOps.Multiply(normalized[indices], _gamma[featureIndex]),
                _beta[featureIndex]
            );
        }
        
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        
        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[_normalizationAxis];
        
        // Initialize gradients for gamma and beta
        _gammaGradient = new Vector<T>(_gamma.Length);
        _betaGradient = new Vector<T>(_beta.Length);
        
        // Calculate gradients for gamma and beta
        for (int i = 0; i < outputGradient.Length; i++)
        {
            // Calculate indices for the gradient tensor
            int[] indices = GetIndices(i, outputGradient.Shape);
            
            // Get the feature index
            int featureIndex = indices[_normalizationAxis];
            
            // Gradient for gamma: sum(dout * normalized)
            _gammaGradient[featureIndex] = NumOps.Add(
                _gammaGradient[featureIndex],
                NumOps.Multiply(outputGradient[indices], _lastNormalized[indices])
            );
            
            // Gradient for beta: sum(dout)
            _betaGradient[featureIndex] = NumOps.Add(
                _betaGradient[featureIndex],
                outputGradient[indices]
            );
        }
        
        // Calculate gradient for the input
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        
        // For each sample in the batch
        for (int b = 0; b < batchSize; b++)
        {
            // Calculate the gradient for the normalized input
            var dNormalized = new Tensor<T>(_lastInput.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                // Calculate indices for the gradient tensor
                int[] indices = GetIndices(i, outputGradient.Shape);
                
                // Skip if not the current batch
                if (indices[0] != b) continue;
                
                // Get the feature index
                int featureIndex = indices[_normalizationAxis];
                
                // Gradient for normalized: dout * gamma
                dNormalized[indices] = NumOps.Multiply(outputGradient[indices], _gamma[featureIndex]);
            }
            
            // Calculate the gradient for the input
            for (int i = 0; i < _lastInput.Length; i++)
            {
                // Calculate indices for the input tensor
                int[] indices = GetIndices(i, _lastInput.Shape);
                
                // Skip if not the current batch
                if (indices[0] != b) continue;
                
                // Calculate index for the mean and variance tensors
                int[] meanVarIndices = new int[indices.Length];
                for (int j = 0; j < indices.Length; j++)
                {
                    meanVarIndices[j] = j == _normalizationAxis ? 0 : indices[j];
                }
                
                // Get the standard deviation
                T stdDev = NumOps.Sqrt(NumOps.Add(_lastVariance[meanVarIndices], _epsilon));
                
                // Calculate the gradient for the input
                T dxNorm = dNormalized[indices];
                T dxVar = NumOps.Multiply(
                    NumOps.Multiply(
                        NumOps.FromDouble(-0.5),
                        NumOps.Divide(
                            NumOps.Subtract(_lastInput[indices], _lastMean[meanVarIndices]),
                            NumOps.Multiply(
                                NumOps.Power(stdDev, NumOps.FromDouble(3)),
                                dxNorm
                            )
                        )
                    ),
                    NumOps.FromDouble(1.0 / featureSize)
                );
                
                T dxMean = NumOps.Add(
                    NumOps.Multiply(
                        NumOps.Divide(NumOps.FromDouble(-1), stdDev),
                        dxNorm
                    ),
                    NumOps.Multiply(
                        NumOps.FromDouble(-2),
                        NumOps.Multiply(
                            NumOps.Subtract(_lastInput[indices], _lastMean[meanVarIndices]),
                            dxVar
                        )
                    )
                );
                
                inputGradient[indices] = NumOps.Add(
                    NumOps.Divide(dxNorm, stdDev),
                    NumOps.Add(
                        NumOps.Multiply(
                            NumOps.FromDouble(2),
                            NumOps.Multiply(
                                NumOps.Subtract(_lastInput[indices], _lastMean[meanVarIndices]),
                                dxVar
                            )
                        ),
                        NumOps.Divide(dxMean, NumOps.FromDouble(featureSize))
                    )
                );
            }
        }
        
        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        
        // Update gamma and beta
        for (int i = 0; i < _gamma.Length; i++)
        {
            _gamma[i] = NumOps.Subtract(_gamma[i], NumOps.Multiply(learningRate, _gammaGradient[i]));
            _beta[i] = NumOps.Subtract(_beta[i], NumOps.Multiply(learningRate, _betaGradient[i]));
        }
    }

    public override Vector<T> GetParameters()
    {
        // Combine gamma and beta into a single vector
        var parameters = new Vector<T>(_gamma.Length + _beta.Length);
        
        // Copy gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            parameters[i] = _gamma[i];
        }
        
        // Copy beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            parameters[i + _gamma.Length] = _beta[i];
        }
        
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _gamma.Length + _beta.Length)
        {
            throw new ArgumentException($"Expected {_gamma.Length + _beta.Length} parameters, but got {parameters.Length}");
        }
        
        // Set gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            _gamma[i] = parameters[i];
        }
        
        // Set beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = parameters[i + _gamma.Length];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastNormalized = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }

    private static int[] GetIndices(int flatIndex, int[] shape)
    {
        int[] indices = new int[shape.Length];
        int remaining = flatIndex;
        
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        
        return indices;
    }
}