namespace AiDotNet.Interfaces;

public interface IModelSerializer<T>
{
    byte[] Serialize();
    void Deserialize(byte[] data);
}