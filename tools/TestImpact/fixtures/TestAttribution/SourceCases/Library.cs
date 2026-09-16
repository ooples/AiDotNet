namespace SourceLibrary;

public static class Subject
{
    public static int Left(int input)
    {
#if SOURCE_ALTERNATIVE
        return (input - 1) + 2;
#else
        return input + 1;
#endif
    }

    public static int Right(int input) => input + 2;
    public static int Third(int input) => input + 3;
}
