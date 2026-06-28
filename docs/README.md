# Analysis Framework

The scripts and other artifacts in this repo are for probing a generative AI model for causal reasoning in legal contexts. For the time being, this is just a small proof-of concept for one scenario, which is a patentee claiming lost profit damages. In this scenario there are two causal considerations a model must make:

1. There needs to be an infringing product; without it there can be no damages
2. There must not be a non-infringing substitute product; if such exists, consumers would use that product instead of the patentee's product

Here are the setup instructions for running an analysis of this scenario for the Llama 3.2 1B (1 Billion parameters) model. All instructions are for macOS (and the Fish shell).

## 0. Preliminaries

1. Create a [Hugging Face](https://huggingface.co) account.
2. Log into Hugging Face and request access to the [Llama-3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B). It should take only 30 minutes or so.
3. Create an access token on Hugging Face.
4. Install the [Hugging Face command line tools](https://huggingface.co/docs/huggingface_hub/en/guides/cli) with:

   ```bash
   brew install hf
   ```

   Check with:

   ```bash
   hf version
   ```

   if it worked.

5. Log into Hugging Face with:

   ```bash
   hf auth login
   ```

   If asked, you can add the token as git credential.

6. Install Python and create a Python virtual environment (which is not necessary, but makes dependency management easier) with (assuming you have Homebrew installed):

   ```bash
   /opt/homebrew/bin/python3 -m venv .venv
   ```

   Start the virtual environment with:

   ```bash
   source .venv/bin/activate.fish
   ```

   or just `source .venv/bin/activate` if you are not using Fish. You can stop the virtual environment with `deactivate`.

7. Install all dependencies with:

   ```bash
   pip3 install torch transformers accelerate pyvene transformer-lens
   ```

## 1. Audit

1. Run the baseline audit of the model with:

   ```bash
   python3 run_benchmark.py
   ```

2. From the output we see that the model has problems reasoning causally in the patent damages scenario. We can look inside the hidden layers of the model to extract the mathematical concept vector, `ip_concept_vector.pt`, with:

   ```bash
   python3 probe_activations.py
   ```

## 2. Mitigation

Before re-training the model, fine-tuning it, or attempting other mitigations, we can try to modify the responsible model vector on the fly with:

```bash
python3 steer_inference.py
```

However, from the output we see that does not work. The model is still not reasoning correctly.

I can essentially see the following options for fixing a model:

1. Re-training
2. Guardrails
3. Integrating a solver into the model architecture

## 3. Post-intervention Verification

We can rerun the audit under 1. to verify that any permanent verification was successful.

## 4. Limitations

One limitation I ran into is that the model refused to respond to prompts in employment scenarios, e.g., age discrimination. There are built-in safeguards that, when triggered, make the model refuse to respond to prompts. So, either we would need to find enough scenarios that a model can answer or remove the safeguards (but this latter approach is a whole research project on its own, and we may also inadvertently change the model behavior invalidating our findings as those are no longer for the unchanged model).
