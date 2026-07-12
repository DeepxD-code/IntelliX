public class LinearSum {
    static long f(long n) {
        long total = 0;
        for (long i = 0; i < n; i++) total += i;
        return total;
    }
    public static void main(String[] args) {
        long n = Long.parseLong(args[0]);
        long t0 = System.nanoTime();
        long r = f(n);
        long t1 = System.nanoTime();
        System.out.println("ELAPSED_MS=" + (t1 - t0) / 1_000_000.0 + " RESULT=" + r);
    }
}
