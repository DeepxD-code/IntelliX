public class QuadraticPairs {
    static long f(long n) {
        long c = 0;
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                c += (i ^ j) & 1;
        return c;
    }
    public static void main(String[] args) {
        long n = Long.parseLong(args[0]);
        long t0 = System.nanoTime();
        long r = f(n);
        long t1 = System.nanoTime();
        System.out.println("ELAPSED_MS=" + (t1 - t0) / 1_000_000.0 + " RESULT=" + r);
    }
}
