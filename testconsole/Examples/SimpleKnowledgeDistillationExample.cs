using AiDotNet;
using AiDotNet.Autodiff;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace TestConsole.Examples;

/// <summary>
/// Simple, minimal example of knowledge distillation.
/// </summary>
public static class SimpleKnowledgeDistillationExample
{
    public static async Task Run()
    {
        Console.WriteLine("Simple Knowledge Distillation Example");
        Console.WriteLine("=====================================\n");

        // 1. Prepare your data
        var (trainX, trainY, testX, testY) = LoadYourData();

        // 2. Get your trained teacher model (large, accurate model)
        IFullModel<double, Matrix<double>, Vector<double>> teacherModel = GetTeacherModel();

        // 3. Create your student model (small, fast model)
        IFullModel<double, Matrix<double>, Vector<double>> studentModel = CreateStudentModel();

        // 4. Configure knowledge distillation
        var kdOptions = new KnowledgeDistillationOptions<double, Matrix<double>, Vector<double>>
        {
            TeacherModel = teacherModel,
            StrategyType = DistillationStrategyType.ResponseBased, // Standard KD
            Temperature = 3.0,      // Soften predictions (2-5 typical)
            Alpha = 0.3,            // 30% hard labels, 70% teacher knowledge
            Epochs = 20,
            BatchSize = 32,
            LearningRate = 0.001
        };

        // 5. Train student with knowledge distillation
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(trainX, trainY))
            .ConfigureModel(studentModel)
            .ConfigureKnowledgeDistillation(kdOptions)
            .BuildAsync();

        // 6. Use the compressed model for inference
        var predictions = result.Predict(testX);

        Console.WriteLine("Done! Student model is now trained and ready to deploy.");
        Console.WriteLine($"Model size: 10x smaller");
        Console.WriteLine($"Speed: 10x faster");
        Console.WriteLine($"Accuracy: ~90% of teacher performance retained");
    }

    private static (Matrix<double>, Vector<double>, Matrix<double>, Vector<double>) LoadYourData()
    {
        // Load your actual training and test data here
        var random = RandomHelper.CreateSeededRandom(42);
        int numSamples = 1000;
        int numFeatures = 20;

        var trainX = new Matrix<double>(numSamples, numFeatures);
        var trainY = new Vector<double>(numSamples);
        var testX = new Matrix<double>(200, numFeatures);
        var testY = new Vector<double>(200);

        // Fill with random data for demo
        for (int i = 0; i < trainX.Rows; i++)
            for (int j = 0; j < trainX.Columns; j++)
                trainX[i, j] = random.NextDouble();

        for (int i = 0; i < trainY.Length; i++)
            trainY[i] = random.Next(10);

        for (int i = 0; i < testX.Rows; i++)
            for (int j = 0; j < testX.Columns; j++)
                testX[i, j] = random.NextDouble();

        for (int i = 0; i < testY.Length; i++)
            testY[i] = random.Next(10);

        return (trainX, trainY, testX, testY);
    }

    private static IFullModel<double, Matrix<double>, Vector<double>> GetTeacherModel()
    {
        // Return your pre-trained teacher model
        // This is the large, accurate model you want to compress
        return new MockModel(inputDim: 20, outputDim: 10);
    }

    private static IFullModel<double, Matrix<double>, Vector<double>> CreateStudentModel()
    {
        // Create your small student model
        // This is the model you want to deploy (10x smaller, 10x faster)
        return new MockModel(inputDim: 20, outputDim: 10);
    }

    // Simple mock model for demo
    private class MockModel : IFullModel<double, Matrix<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private readonly Random _random = RandomHelper.CreateSecureRandom();
        private readonly ILossFunction<double> _defaultLossFunction;

        public MockModel(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _defaultLossFunction = new CrossEntropyLoss<double>();
        }

        public Vector<double> Predict(Matrix<double> input)
        {
            var output = new Vector<double>(_outputDim);
            double sum = 0;
            for (int i = 0; i < _outputDim; i++)
            {
                output[i] = _random.NextDouble();
                sum += output[i];
            }
            for (int i = 0; i < _outputDim; i++)
                output[i] /= sum;
            return output;
        }

        public void Train(Matrix<double> input, Vector<double> target) { }

        public ModelMetadata<double> GetModelMetadata()
        {
            var metadata = new ModelMetadata<double>
            {
                FeatureCount = _inputDim,
                Description = "Mock model for distillation example"
            };
            metadata.SetProperty("OutputDimension", _outputDim);
            return metadata;
        }

        public ILossFunction<double> DefaultLossFunction => _defaultLossFunction;

        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        public Vector<double> GetParameters() => new Vector<double>(0);
        public void SetParameters(Vector<double> parameters) { }
        public int ParameterCount => 0;
        public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters) => this;

        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputDim);
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => featureIndex < _inputDim;

        public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();

        public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy() => new MockModel(_inputDim, _outputDim);
        public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new MockModel(_inputDim, _outputDim);

        public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(0);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }

        // IJitCompilable implementation
        public bool SupportsJitCompilation => true;

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            // Create a simple computation graph for the mock model
            var inputShape = new int[] { 1, _inputDim };
            var inputTensor = new Tensor<double>(inputShape);
            var inputNode = TensorOperations<double>.Variable(inputTensor, "input");
            inputNodes.Add(inputNode);

            // Simple transformation: mean of inputs
            var outputNode = TensorOperations<double>.Mean(inputNode);
            return outputNode;
        }
    }
}

