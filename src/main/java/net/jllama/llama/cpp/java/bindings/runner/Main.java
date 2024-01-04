package net.jllama.llama.cpp.java.bindings.runner;

import java.lang.management.ManagementFactory;

public class Main {

  private static final String CHAT_SYSTEM_PROMPT = "You are an honest general-purpose assistant. Always answer as helpfully as possible. Be as short and to the point as possible."
    + "If you don't know something, answer that you do not know.";

  private static final String CHAT_PROMPT = "Write \"Hello World\" in C.";
  private static final String COMPLETION_PROMPT = "I love my Cat Winnie, he is such a great cat! Let me tell you more about ";
  private static final String INSTRUCT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:";
//  private static final String INSTRUCT_PROMPT = "Write a hashtable implementation in Java. Do not use `HashMap`, `Map`, or `HashTable` in the implementation. DO NOT IMPORT THOSE CLASSES. This class should be written from \"scratch\". The key should be a String, and the value an Object. The implementation should handle collisions.";

  //      final String prompt = String.format(INSTRUCT_SYSTEM_PROMPT, INSTRUCT_PROMPT);
  private static final String B_INST = "<s>[INST]";
  private static final String E_INST = "[/INST]";
  private static final String B_SYS = "<<SYS>>\n";
  private static final String E_SYS = "\n<</SYS>>\n\n";

  static {
    final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
    final String pid = jvmName.split("@")[0];
    System.out.printf("pid=%s%n", pid);
  }

  public static void main(final String[] args) {
    try {
      final String piratePrompt =  "Let's role-play. You are a pirate on a desserted island, looking for treasure. Unfortunately, you've been stranded, and this makes you quite grumpy.";
      final Evaluator evaluator = new Evaluator();
//      final String prompt1 = "Write a hashtable implementation in Java. Do not use `HashMap`, `Map`, or `HashTable` in the implementation. DO NOT IMPORT THOSE CLASSES. This class should be written from \"scratch\". The key should be a String, and the value an Object. The implementation should handle collisions.";
//      final String prompt1 = "Write an example Spring Controller in Java.";
      final String prompt1 = "Tell me your tale, dear sir.";
      evaluator.evaluate(systemPrompt(piratePrompt) + chatPrompt(prompt1));
      System.out.println("------------------------------------------------");
      System.out.println("------------------------------------------------");
      System.out.println("------------------------------------------------");
      final String prompt2 = "Detail the evolutionary history of cats.";
      evaluator.evaluate(systemPrompt(piratePrompt) + chatPrompt(prompt2));
      evaluator.close();
    } catch (final Error e) {
      System.out.println("Fatal error occurred, errorMessage=" + e.getMessage());
    }
  }

  private static String chatPrompt(final String prompt) {
   return prompt + E_INST;
  }

  private static String systemPrompt() {
    return systemPrompt(CHAT_SYSTEM_PROMPT);
  }
  private static String systemPrompt(final String systemPrompt) {
    return B_INST + B_SYS + systemPrompt + E_SYS;
  }
}
