namespace AiDotNet.Models.Options;

public class VARMAModelOptions<T> : VARModelOptions<T>
{
    public int MaLag { get; set; } = 1;
}