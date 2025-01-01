namespace AiDotNet.Models
{
    public class ModelEvaluationData<T>
    {
        public DataSetStats<T> TrainingSet { get; set; } = new();
        public DataSetStats<T> ValidationSet { get; set; } = new();
        public DataSetStats<T> TestSet { get; set; } = new();
        public ModelStats<T> ModelStats { get; set; } = ModelStats<T>.Empty();
        public Matrix<T> Features { get; set; } = Matrix<T>.Empty();
    }
}