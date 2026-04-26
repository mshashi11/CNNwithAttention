# Research on Convoluational Neural Networks (CNNs) with Attention Mechanism

The goal of this result is to improve the existing implementation of CNN with Attention Mechanism on a given dataset, and achieve the best possible accuracy for an image classification problem

## Setup

The relevant files for your research are:

1. `README.md`: The readme file that provides the context about the problem being solved
2. `common.py`: Script containing a set of common functions for preparing training/test data for the problem, and evaluating the accuracy of the model on the test data.
3. `fashion_mnist_multi_attn.py`: The latest script containing the best possible model I have so far, and the only script you are allowed to modify.

The training/testing data is in the data/ folder, and it should already contain the required data. If not, executing the python script will automatically download data in this folder.

To set-up a new experiment, work with your human to:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**:  Read the relevant files described above.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment will run on a single GPU, and the script has a fixed training budget of 10 minutes. The script can be simply run as: `./fashion_mnist_multi_attn.py`. Here are the specific instructions to you for running a single experiment:

1. You can modify ONLY the `fashion_mnist_multi_attn.py`. All options are available to you for exploring how to improve the model performance, ranging from hyperparameter fine-tuning, to experimenting with radically new approaches for using Attention Mechansim in CNNs.
2. NEVER edit the file `common.py` under any circumstance.
3. DO NOT install any new library or package for the purposes of this research. Use only the existing Python libraries - that should be sufficient for the research work.

**Goal**: Achieve the highest possible classification accuracy on the testing dataset. On your first run, execute the current Python script as-is to establish the baseline accuracy for your research work.

**Simplicity criterion**: Strive for the simplest possible model that gives the best possible accuracy. A complex model that achieves only five basis point improvement over the current best model is not worth it. Aim to build a model that is easy to understand and explain.


## Output format

The output of the script looks something like this:

```
Device: cuda
Epoch: 0 | Loss: 201.20
Epoch: 1 | Loss: 174.87
Epoch: 2 | Loss: 168.35
Epoch: 3 | Loss: 163.75
...
Epoch: 58 | Loss: 128.03
Epoch: 59 | Loss: 128.02
---Summary---
Accuracy: 93.99
Time: 419.75
```

The last two lines of the output are important: they contains two fields, one is accuracy and the other is runtime of the model (in seconds). From the run log, you can use the `grep` command to extract the value of the two fields from the `run.log` log file.


## Logging results

When an experiment is done, log it to results.tsv (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:
```
commit	Accuracy	Time	status	description
```

1. git commit hash (short, 7 chars)
2. Accuracy (e.g. 93.99) — use 00.00 for crashes
3. Training time, in seconds (e.g. 419.75) — use 0.0 for crashes
4. status: keep, discard, or crash
5, short text description of what this experiment tried

Example:

```
commit	Accuracy	Time	status	description
a1b2c3d	93.79	444.0	keep	baseline
b2c3d4e	94.10	444.2	keep	increase epochs to 70
c3d4e5f	93.50	400.0	discard	switch to GeLU activation for final feature creation layer
d4e5f6g	00.00	0.0	crash	double the number of layers in classification model (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `fashion_mnist_multi_attn.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `./fashion_mnist_multi_attn.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^Accuracy:\|^Time:" run.log`
If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
6. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
7. If accuracy improved (higher), you "advance" the branch, keeping the git commit. Update the `README.md` file at the end and give a brief technical description of the changes you made in this commit, so that we can keep track of how you incrementally improved the performance of the model.
8. If accuracy is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for preparing training/testing data). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working **indefinitely** until you are manually stopped. You are autonomous. If you run out of ideas, think harder — search for relevant literature in the field if needed, re-read the in-scope files for new angles, try combining previous near-misses, or try more radical architectural changes. The loop runs until the human interrupts you, period.
