/*
 * IntelliProfile - Java API Server
 * RESTful API for profiling service and bytecode analysis
 */

import java.io.*;
import java.util.*;
import java.lang.management.*;

class ProfileRequest {
    String language;
    String code;
    Map<String, Integer> metadata;
    
    public ProfileRequest(String language, String code, Map<String, Integer> metadata) {
        this.language = language;
        this.code = code;
        this.metadata = metadata;
    }
}

class ProfileResponse {
    String status;
    Map<String, Object> metrics;
    String prediction;
    List<String> recommendations;
    
    public ProfileResponse() {
        this.metrics = new HashMap<>();
        this.recommendations = new ArrayList<>();
    }
}

class BytecodeAnalyzer {
    // Analyze Java bytecode patterns (simplified)
    public Map<String, Integer> analyzeComplexity(String code) {
        Map<String, Integer> analysis = new HashMap<>();
        
        // Count method invocations
        int methodCalls = countOccurrences(code, "(");
        analysis.put("methodCalls", methodCalls);
        
        // Detect loops
        int loops = countOccurrences(code, "for") + 
                   countOccurrences(code, "while") + 
                   countOccurrences(code, "do");
        analysis.put("loopCount", loops);
        
        // Detect conditionals
        int conditionals = countOccurrences(code, "if") + 
                          countOccurrences(code, "switch");
        analysis.put("conditionals", conditionals);
        
        // Estimate cyclomatic complexity
        int complexity = loops + conditionals + 1;
        analysis.put("cyclomaticComplexity", complexity);
        
        return analysis;
    }
    
    private int countOccurrences(String str, String pattern) {
        int count = 0;
        int index = 0;
        while ((index = str.indexOf(pattern, index)) != -1) {
            count++;
            index += pattern.length();
        }
        return count;
    }
}

class JavaProfiler {
    private BytecodeAnalyzer analyzer;
    private MemoryMXBean memoryBean;
    private ThreadMXBean threadBean;
    
    public JavaProfiler() {
        this.analyzer = new BytecodeAnalyzer();
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.threadBean = ManagementFactory.getThreadMXBean();
    }
    
    public ProfileResponse profileCode(ProfileRequest request) {
        ProfileResponse response = new ProfileResponse();
        
        try {
            // Get initial memory state
            long startMemory = memoryBean.getHeapMemoryUsage().getUsed();
            long startTime = System.nanoTime();
            
            // Analyze code
            Map<String, Integer> codeAnalysis = analyzer.analyzeComplexity(request.code);
            
            // Simulate execution time
            Thread.sleep(10); // 10ms delay
            
            long endTime = System.nanoTime();
            long endMemory = memoryBean.getHeapMemoryUsage().getUsed();
            
            // Calculate metrics
            double executionTime = (endTime - startTime) / 1_000_000.0; // Convert to ms
            long memoryUsed = (endMemory - startMemory) / 1024; // Convert to KB
            
            // Populate metrics
            response.metrics.put("executionTime", executionTime + " ms");
            response.metrics.put("memoryUsage", Math.abs(memoryUsed) + " KB");
            response.metrics.put("cyclomaticComplexity", codeAnalysis.get("cyclomaticComplexity"));
            response.metrics.put("loopCount", codeAnalysis.get("loopCount"));
            response.metrics.put("methodCalls", codeAnalysis.get("methodCalls"));
            
            // Generate prediction based on complexity
            int complexity = codeAnalysis.get("cyclomaticComplexity");
            if (complexity > 10) {
                response.prediction = "NEEDS_OPTIMIZATION";
                response.recommendations.add("High cyclomatic complexity detected. Consider refactoring.");
                response.recommendations.add("Break down complex methods into smaller functions.");
            } else if (complexity > 5) {
                response.prediction = "MODERATE";
                response.recommendations.add("Code complexity is moderate. Review for optimization opportunities.");
            } else {
                response.prediction = "EFFICIENT";
                response.recommendations.add("Code appears well-structured and efficient.");
            }
            
            // Add loop-specific recommendations
            if (codeAnalysis.get("loopCount") > 2) {
                response.recommendations.add("Multiple loops detected. Consider using streams or parallel processing.");
            }
            
            response.status = "SUCCESS";
            
        } catch (Exception e) {
            response.status = "ERROR";
            response.recommendations.add("Error during profiling: " + e.getMessage());
        }
        
        return response;
    }
    
    public void printResponse(ProfileResponse response) {
        System.out.println("\n========================================");
        System.out.println("Java Profiler - Analysis Results");
        System.out.println("========================================");
        System.out.println("Status: " + response.status);
        System.out.println("\nMetrics:");
        for (Map.Entry<String, Object> entry : response.metrics.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }
        System.out.println("\nPrediction: " + response.prediction);
        System.out.println("\nRecommendations:");
        for (String rec : response.recommendations) {
            System.out.println("  - " + rec);
        }
        System.out.println("========================================\n");
    }
}

public class IntelliProfileAPI {
    public static void main(String[] args) {
        System.out.println("IntelliProfile - Java API Server");
        System.out.println("=================================\n");
        
        JavaProfiler profiler = new JavaProfiler();
        
        // Test Case 1: Simple code
        String simpleCode = """
            public int calculateSum(int n) {
                int sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += i;
                }
                return sum;
            }
        """;
        
        Map<String, Integer> metadata1 = new HashMap<>();
        metadata1.put("loopDepth", 1);
        metadata1.put("functionCalls", 2);
        
        ProfileRequest req1 = new ProfileRequest("Java", simpleCode, metadata1);
        System.out.println("Test 1: Profiling Simple Java Code...");
        ProfileResponse resp1 = profiler.profileCode(req1);
        profiler.printResponse(resp1);
        
        // Test Case 2: Complex nested code
        String complexCode = """
            public void complexOperation(int[][] matrix) {
                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < matrix[i].length; j++) {
                        if (matrix[i][j] > 0) {
                            for (int k = 0; k < matrix.length; k++) {
                                if (k != i && k != j) {
                                    matrix[i][j] += matrix[k][j];
                                }
                            }
                        }
                    }
                }
            }
        """;
        
        Map<String, Integer> metadata2 = new HashMap<>();
        metadata2.put("loopDepth", 3);
        metadata2.put("functionCalls", 5);
        
        ProfileRequest req2 = new ProfileRequest("Java", complexCode, metadata2);
        System.out.println("Test 2: Profiling Complex Nested Code...");
        ProfileResponse resp2 = profiler.profileCode(req2);
        profiler.printResponse(resp2);
        
        System.out.println("Profiling complete!");
    }
}
