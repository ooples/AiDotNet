namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Model adapter interface (for LoRA, etc.)
    /// </summary>
    public interface IModelAdapter
    {
        string AdapterType { get; }
        long AdapterParameters { get; }
        void Apply<T>(IFoundationModel<T> model);
    }
}