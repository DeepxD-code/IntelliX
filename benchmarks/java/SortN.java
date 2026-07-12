import java.util.Arrays;
public class SortN {
    static long f(long n) {
        int size = (int) n;
        long[] arr = new long[size];
        for (int i = 0; i < size; i++) arr[i] = size - i;
        Arrays.sort(arr);
        return arr[0];
    }
    public static void main(String[] args) {
        long n = Long.parseLong(args[0]);
        long t0 = System.nanoTime();
        long r = f(n);
        long t1 = System.nanoTime();
        System.out.println("ELAPSED_MS=" + (t1 - t0) / 1_000_000.0 + " RESULT=" + r);
    }
}
