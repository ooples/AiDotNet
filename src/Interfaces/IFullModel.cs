namespace AiDotNet.Interfaces;

public interface IFullModel<T> : ITrainableModel<T>, IPredictiveModel<T>
{
}