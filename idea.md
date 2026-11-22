Using the measure of decode TPS as your measure of performance (`Decode TPS = (Output Tokens - 1) / (End-to-End Latency - TTFT)`).
Use the following optimizations to increase the TPS of [this model (WeMM-Video-35BA3B)](https://huggingface.co/tencent/WeMM-Video-35BA3B) by at least 2.5 times.
You should first run tests to measure TPS model without any optimizations and use that as your baseline.
Then you should run the same tests to measure the TPS with optimizations and calculate the TPS.


## The optimizations
The machine you are targeting has an RTX Pro 6000 Blackwell and we are only checking single user performance.
Any optimizations you make should NOT impact the quality of the output of the model.

### Speculative Decoding (Ideally MTP)
There is no official confirmation from Tencent on whether WeMM-Video-35BA3B-AWQ supports MTP, but since it is built on Qwen3.6-35B-A3B which does support it, I am inclined to think that this would of WeMM-Video-35BA3B-AWQ.
Enabling MTP with `K=3` should should reduce per token latency from 22ms to 13ms.
Either way, if there is a perforamce gain you should use a speculative decoding (preferably but not necessarily MTP).

### The rest
These are the other optimizations that should be made:

1. Enable Cuda Graphs and FlashInfer, make sure that `enforce_eager=False` and that FlashInfer is installed and active.
2. Use FP8 KV cache (`--kv-cache-dtype fp8`) to reduce bandwidth used by attention at decode steps.


## Some things to keep in mind
This will run a seperate server that has an RTX Pro 6000 Blackwell server edition, and we will only be testing single user performance.
The GPU on the current machine is too weak to run this model, so if you want to test something, give it to me and I will run it on the other machine.
If you can think of any additional optimizations, say them, but the initial measurements should just use the ones I listed.
While I think this can be done in a single `.sh` file, you can feel free to write additional files to help measure and boost performance.
You can feel free to write the prompts for the tests or ask me to write them.

