# Fine-tuning of Llama 3B Instruct

Instructions are for macOS. Fine-tuning works on Apple M series MacBooks with 16 GB RAM. One full training run (180 iterations) on the LEET-Arg dataset should take less than 30 minutes (as measured on a MacBook Pro M5 with 16 GB RAM). Peak memory usage will be 10 GB, the fans will run fast, and no other work is possible.

It is not necessary to download the model.

1. Create Python virtual environment and activate it.

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. Update pip.

   ```bash
   pip install --upgrade pip
   ```

3. Install dependencies.

   ```bash
   pip install mlx-lm datasets huggingface_hu
   ```

4. Prepare training dataset by creating a folder named `data` with two files, `train.jsonl` and `valid.jsonl`. MLX accepts JSON Lines format. For Llama 3.2 3B Instruct, format your lines using the Llama 3 prompt template.

   ```jsonl
   {
     "text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow do I reset my router?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nUnplug the power cable for 30 seconds and plug it back in.<|eot_id|>"
   }
   ```

   Note that when using MLX’s chat format (`{"messages": [...]}`), MLX converts each line into the Llama 3.2 template using the tokenizer that comes with the model. `mlx_lm.generate` (see below) applies the chat template to `--prompt` by default. So, a hand-written Llama prompt gets wrapped twice unless you add `--ignore-chat-template`.

5. To diagnose and fix problems with your formatted data run the following scripts.

   ```bash
   python3 json_diagnose.py
   python3 json_repair.py
   ```

6. Start training.

   ```bash
   mlx_lm.lora \
                --model mlx-community/Llama-3.2-3B-Instruct-4bit \
                --train \
                --data ./data \
                --iters 600 \
                --batch-size 1 \
                --num-layers 16 \
                --learning-rate 1e-4 \
                --adapter-path ./adapters

   ```

7. Prompt your trained model.

   ```bash
   mlx_lm.generate \
                --model mlx-community/Llama-3.2-3B-Instruct-4bit \
                --adapter-path ./adapters \
                --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert legal reasoning auditor. Analyze the debate provided, extract the core premises and inference rules for each statement, and declare a final verdict (CORRECT or INCORRECT).<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<Theory> For an act to \"cause harm\" to someone means that, if the act had not occurred, that person would have been in a better state. Similarly, for an act to \"benefit\" someone means that, if the act had not occurred, that person would have been in a worse state. Alice and Bob have the following debate regarding <Theory>:\nAlice1: I could have given my friend 50,000 won for no reason but chose not to. If I had done so, my friend would have been in a better state. According to <Theory>, my act harmed my friend. But that seems unreasonable.\nBob1: <Theory> doesn't have such implications. That's because doing nothing—like simply not giving your friend 50,000 won—cannot be considered an act.\nAlice2: If you limit \"act\" in such a way within <Theory>, other unreasonable consequences follow. Consider someone who saw a child struggling in water and had the ability to rescue them but chose not to, resulting in the child's drowning. Not saving the child clearly harmed the child.\nBob2: But this case is different. The choice not to save the child is itself the result of a deliberate decision to avoid the rescue. In this sense, not saving the child should be considered an act.\nAlice3: Then what about this case? Imagine A bought a gift intended for B but became attached to it and decided to keep it instead of giving it to B. In this case, although not giving the gift was a deliberate decision, it doesn't seem like A harmed B.\n\n<question> Which of the following statements in <statements> is/are correct as an analysis of the above passage?\n<statements>\n(a) According to Alice1’s interpretation of <Theory>, if I could have punched my friend and broken their nose but did not do so, then, by not punching my friend, I performed an act that benefited my friend.\n(b) Alice2 and Bob2 differ in their judgment about whether not rescuing the child caused harm to the child.\n(c) If Bob, in response to Alice3, were to assert that “A’s not giving the gift to B did, in fact, harm B,” this would make Bob’s position inconsistent.<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
                --max-tokens 600
   ```
