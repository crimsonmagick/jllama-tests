package net.jllama.llama.cpp.java.bindings.runner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import net.jllama.api.Context;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.core.LlamaContext.LlamaBatch;
import net.jllama.core.LlamaCpp;
import net.jllama.core.LlamaContext;
import net.jllama.core.LlamaModel;
import net.jllama.core.LlamaTokenDataArray;

import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import net.jllama.core.exceptions.LlamaCppException;

public class Main {

  private static final String CHAT_SYSTEM_PROMPT = "You are a helpful, respectful and honest general-purpose assistant, "
    + "that also specializes in code generation. Always answer as helpfully as possible. "
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
      final Evaluator evaluator = new Evaluator();
//      final String prompt1 = "Write a hashtable implementation in Java. Do not use `HashMap`, `Map`, or `HashTable` in the implementation. DO NOT IMPORT THOSE CLASSES. This class should be written from \"scratch\". The key should be a String, and the value an Object. The implementation should handle collisions.";
      final String prompt1 = "Write an example Spring Controller in Java.";
      evaluator.evaluate(chatPrompt(prompt1));
      System.out.println("------------------------------------------------");
      System.out.println("------------------------------------------------");
      System.out.println("------------------------------------------------");
      final String prompt2 = "Detail the evolutionary history of cats.";
      evaluator.evaluate(chatPrompt(prompt2));
      evaluator.close();
    } catch (final Error e) {
      System.out.println("Fatal error occurred, errorMessage=" + e.getMessage());
    }
  }

  private static String chatPrompt(final String prompt) {
    return B_INST + B_SYS + CHAT_SYSTEM_PROMPT + E_SYS + prompt + E_INST;
  }

}
