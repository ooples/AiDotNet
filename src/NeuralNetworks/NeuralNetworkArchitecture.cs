namespace AiDotNet.NeuralNetworks;

public class NeuralNetworkArchitecture<T>
{
    public List<ILayer<T>>? Layers { get; }
    public InputType InputType { get; }
    public int InputSize { get; private set; }
    public int OutputSize { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int InputDepth { get; }
    public List<RestrictedBoltzmannMachine<T>> RbmLayers { get; set; }
    public NeuralNetworkTaskType TaskType { get; }
    public NetworkComplexity Complexity { get; }

    public int InputDimension => 
        InputType == InputType.OneDimensional ? 1 :
        InputType == InputType.TwoDimensional ? 2 : 3;

    public int CalculatedInputSize =>
        InputType switch
        {
            InputType.OneDimensional => InputSize > 0 ? InputSize : throw new InvalidOperationException("InputSize must be set for OneDimensional input."),
            InputType.TwoDimensional => InputHeight * InputWidth,
            InputType.ThreeDimensional => InputHeight * InputWidth * InputDepth,
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };

    public NeuralNetworkArchitecture(
        InputType inputType,
        NeuralNetworkTaskType taskType,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int inputHeight = 0,
        int inputWidth = 0,
        int inputDepth = 1,
        int outputSize = 0,
        List<ILayer<T>>? layers = null,
        List<RestrictedBoltzmannMachine<T>>? rbmLayers = null)
    {
        InputType = inputType;
        TaskType = taskType;
        Complexity = complexity;
        InputSize = inputSize;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputDepth = inputDepth;

        Layers = layers;
        RbmLayers = rbmLayers ?? [];
        OutputSize = outputSize;

        ValidateInputDimensions();
    }

    public int[] GetHiddenLayerSizes()
    {
        if (Layers == null || Layers.Count <= 1)
        {
            return [];
        }

        var hiddenLayerSizes = new List<int>();
        for (int i = 1; i < Layers.Count - 1; i++)
        {
            var outputShape = Layers[i].GetOutputShape();
            hiddenLayerSizes.Add(outputShape.Aggregate(1, (a, b) => a * b));
        }

        return [.. hiddenLayerSizes];
    }

    public int[] GetInputShape()
    {
        return InputType switch
        {
            InputType.OneDimensional => [InputSize],
            InputType.TwoDimensional => [InputHeight, InputWidth],
            InputType.ThreeDimensional => [InputDepth, InputHeight, InputWidth],
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };
    }

    public int[] GetOutputShape()
    {
        if (Layers == null || Layers.Count == 0)
        {
            return GetInputShape(); // If no layers, output shape is the same as input shape
        }

        return Layers[Layers.Count - 1].GetOutputShape();
    }

    public int CalculateOutputSize()
    {
        var outputShape = GetOutputShape();
        int result = 1;

        for (int i = 0; i < outputShape.Length; i++)
        {
            result *= outputShape[i];
        }

        return result;
    }

    public int[] GetLayerSizes()
    {
        if (Layers == null || Layers.Count == 0)
        {
            return [CalculatedInputSize];
        }

        var layerSizes = new List<int> { CalculatedInputSize };
        foreach (var layer in Layers)
        {
            layerSizes.Add(layer.GetOutputShape().Aggregate(1, (a, b) => a * b));
        }

        return [.. layerSizes];
    }

    private void ValidateInputDimensions()
    {
        int calculatedSize = InputType switch
        {
            InputType.OneDimensional => InputSize,
            InputType.TwoDimensional => InputHeight * InputWidth,
            InputType.ThreeDimensional => InputHeight * InputWidth * InputDepth,
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };

        switch (InputType)
        {
            case InputType.OneDimensional:
                if (InputSize <= 0)
                {
                    throw new ArgumentException("InputSize must be greater than 0 for OneDimensional input.");
                }
                if (InputHeight != 0 || InputWidth != 0 || InputDepth != 1)
                {
                    throw new ArgumentException("InputHeight, InputWidth, and InputDepth should not be set for OneDimensional input.");
                }
                break;

            case InputType.TwoDimensional:
                if (InputHeight <= 0 || InputWidth <= 0)
                {
                    throw new ArgumentException("Both InputHeight and InputWidth must be greater than 0 for TwoDimensional input.");
                }
                if (InputDepth != 1)
                {
                    throw new ArgumentException("InputDepth should be 1 for TwoDimensional input.");
                }
                break;

            case InputType.ThreeDimensional:
                if (InputHeight <= 0 || InputWidth <= 0 || InputDepth <= 0)
                {
                    throw new ArgumentException("InputHeight, InputWidth, and InputDepth must all be greater than 0 for ThreeDimensional input.");
                }
                break;

            default:
                throw new ArgumentException("Invalid InputDimensionality specified.");
        }

        if (InputSize > 0 && InputSize != calculatedSize)
        {
            throw new ArgumentException($"Provided InputSize ({InputSize}) does not match the calculated size based on dimensions ({calculatedSize}). For {InputType} input, use either InputSize or the appropriate dimension parameters, not both.");
        }

        // If InputSize wasn't provided, set it to the calculated size
        if (InputSize == 0)
        {
            InputSize = calculatedSize;
        }

        // Validate layers if provided
        if (Layers != null && Layers.Count > 0)
        {
            var firstLayer = Layers[0];
            int firstLayerInputSize = firstLayer.GetInputShape().Aggregate(1, (a, b) => a * b);
        
            if (firstLayerInputSize != InputSize)
            {
                throw new ArgumentException($"The first layer's input size ({firstLayerInputSize}) must match the input size ({InputSize}).");
            }
        }

        // Validate RBM layers if provided
        if (RbmLayers.Count > 0)
        {
            int previousLayerSize = InputSize;
            foreach (var rbm in RbmLayers)
            {
                if (rbm.VisibleSize != previousLayerSize)
                {
                    throw new ArgumentException($"RBM visible size ({rbm.VisibleSize}) must match the previous layer size ({previousLayerSize}).");
                }

                previousLayerSize = rbm.HiddenSize;
            }
        }
    }
}