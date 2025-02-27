namespace AiDotNet.Interfaces;

public interface IModelSerializer
{
    byte[] Serialize();
    void Deserialize(byte[] data);
}