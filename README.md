# jllama-tests

# Running the proof of concept
Make sure that you have the llama.cpp on your dynamic library path for your operating system. An example of this can be found in the top level gradle "runProgram" task. Make sure to specify the model path and your log level using: `-Dmodelpath=/path/to/model/mymodel.gguf` and `-Dloglevel={LOGLEVEL}`. Log level can be one of `OFF|INFO|WARN|ERROR`. 