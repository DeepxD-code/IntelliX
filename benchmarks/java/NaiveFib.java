public class NaiveFib {
    static long f(long n) {
        if (n <= 1) return n;
        return f(n - 1) + f(n - 2);
    }
    public static void main(String[] args) {
        long n = Long.parseLong(args[0]);
        long t0 = System.nanoTime();
        long r = f(n);
        long t1 = System.nanoTime();
        System.out.println("ELAPSED_MS=" + (t1 - t0) / 1_000_000.0 + " RESULT=" + r);
    }
}
