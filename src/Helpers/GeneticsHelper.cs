namespace AiDotNet.Helpers;

internal static class GeneticsHelper
{
    public static void Shuffle<T>(this IList<T> list)
    {
        var provider = RandomNumberGenerator.Create();
        var n = list.Count;

        while (n > 1)
        {
            var box = new byte[1];
            do provider.GetBytes(box);
            while (!(box[0] < n * (byte.MaxValue / n)));
            var k = box[0] % n;
            n--;
            (list[n], list[k]) = (list[k], list[n]);
        }
    }
}