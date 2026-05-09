[Back to Overview](../README.md)

> Copyright (c) 2022 DFKI GmbH - All Rights Reserved
> 
> Written by Michael Fürst <Michael.Fuerst@dfki.de>, October 2022

# Example: Randomness Analysis of Streaming

Instead of a tar we will create a stream of tuples (sequence_id, frame_id). These streams are then shuffled and batched like a datapipe. Once batches are created we analyze how many similar frames are in a single batch.

## Create a dummy pipeline

First we create a configurable pipeline that produces a stream of tuples.
If we give a zero or negative shuffle buffer we do global shuffling to emulate single file access randomness.

Example:
```python
import random

def create_pipeline(num_sequences, sequence_len, shuffle_buffer, batch_size):
    """
    Create a dummy pipeline.

    If shuffle_buffer is 0 or less, global shuffling is done (simulates random file access).
    """
    def _shuffle(stream, buffer_size):
        buffer = []
        for date in stream:
            if len(buffer) < buffer_size:
                buffer.append(date)
            else:
                idx = random.randint(0, buffer_size - 1)
                ret, buffer[idx] = buffer[idx], date
                yield ret
        while len(buffer) > 0:
            idx = random.randint(0, len(buffer) - 1)
            ret = buffer[idx]
            del buffer[idx]
            yield ret

    def _batch(stream, batch_size):
        batch = []
        for date in stream:
            batch.append(date)
            if len(batch) >= batch_size:
                ret = batch
                batch = []
                yield ret

    def _fake_sequence_stream(num_sequences, sequence_len):
        for seq in range(num_sequences):
            for frame in range(sequence_len):
                yield (seq, frame)

    stream = _fake_sequence_stream(num_sequences, sequence_len)
    if shuffle_buffer > 0:
        stream = _shuffle(stream, shuffle_buffer)
    else:
        stream = list(stream)
        random.shuffle(stream)
    stream = _batch(stream, batch_size)
    return stream
```

## Analysis function

We write a function that runs a list of configs using a similarity function and creates plots on how many percent of a batch are similar.
At the end it also prints a table with scores for each config.

Example:
```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

from collections import namedtuple
Config = namedtuple("Config", ["sequences", "seq_len", "shuffle_buffer", "batch_size"])

def analyze(configs, is_same, N=10):
    def _similar_within_batch(batch, is_same):
        same = 0
        for ix, x in enumerate(batch):
            for iy, y in enumerate(batch):
                if ix < iy:
                    same += is_same(x, y)
        return same

    def _similar_across_batches(batch0, batch1, is_same):
        same = 0
        for x in batch0:
            for y in batch1:
                same += is_same(x, y)
        return same

    def _fmt_conf(config):
        return f"{tuple(config)}"
    
    def _compute_stats(data):
        stats = []
        for x in data:
            x = np.array(x, dtype=np.float32)
            stats.append((np.average(x), np.std(x), np.min(x), np.max(x)))
        return stats

    def _update_plot(hfig, fig, ax1, ax2, similarity_within_batch, similarity_across_batches):
        ax1.cla()
        ax1.set_title("Similarity within batch (%)")
        for run, vals in enumerate(similarity_within_batch):
            ax1.plot(vals, label=_fmt_conf(configs[run]))
            ax1.legend()
        ax2.cla()
        ax2.set_title("Similarity of neighbouring batches (%)")
        for run, vals in enumerate(similarity_across_batches):
            ax2.plot(vals, label=_fmt_conf(configs[run]))
            ax2.legend()
        fig.canvas.draw()
        hfig.update(fig)

    similarity_within_batch = []
    similarity_across_batches = []
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    hfig = display(fig, display_id=True)
    for config in configs:
        batch0 = None
        similarity_within_batch.append([0.0])
        similarity_across_batches.append([0.0])
        sequences, seq_len, shuffle_buffer, batch_size = config
        stream = create_pipeline(sequences, seq_len, shuffle_buffer, batch_size)
        for idx, batch in enumerate(stream):
            similarity_within_batch[-1][-1] += _similar_within_batch(batch, is_same)
            if batch0 is not None:
                similarity_across_batches[-1][-1] += _similar_across_batches(batch0, batch, is_same)
            if idx % N == N-1:
                similarity_within_batch[-1][-1] /= float(N) * batch_size / 100.0
                similarity_across_batches[-1][-1] /= float(N) * batch_size / 100.0
                _update_plot(hfig, fig, ax1, ax2, similarity_within_batch, similarity_across_batches)
                similarity_within_batch[-1].append(0.0)
                similarity_across_batches[-1].append(0.0)
            batch0 = batch
        N_remainder = (idx % N) + 1
        similarity_within_batch[-1][-1] /= float(N_remainder) * batch_size / 100.0
        similarity_across_batches[-1][-1] /= float(N_remainder) * batch_size / 100.0
        _update_plot(hfig, fig, ax1, ax2, similarity_within_batch, similarity_across_batches)
        plt.close(fig)

    print("+--------+------+------+------+------+------+------+------+------+")
    print("|        | Within Single Batch       | Neighbouring Batches      |")
    print("| Config | AVG  | STD  | MIN  | MAX  | AVG  | STD  | MIN  | MAX  |")
    print("+--------+------+------+------+------+------+------+------+------+")
    for idx, (a, b) in enumerate(zip(_compute_stats(similarity_within_batch), _compute_stats(similarity_across_batches))):
        out = "| " + f"{idx}".rjust(6, " ") + " |"
        for val in a:
            out += f"{val:.1f}".rjust(5, " ") + " |"
        for val in b:
            out += f"{val:.1f}".rjust(5, " ") + " |"
        print(out)
    print("+--------+------+------+------+------+------+------+------+------+")
```

## Many Small Sequences

First we want to test the use case of nuscenes, where we have many small sequences.
Since the data is captured with 2 Hz in a driving car, we can assume, that frames are sufficiently different, if we have 5 seconds (10 frames) in between or a different sequence.

Thus, similarity is measured by:
1. If two frames are of a different sequence, they are not similar.
2. If two frames have a frame number with a difference larger than a parameter X, they are not similar.
3. Else they are similar.

Example:
```python

MIN_FRAME_DIST = 10

def is_same(x, y):
    s0, f0 = x
    s1, f1 = y
    if s0 != s1:
        return 0
    if abs(f1-f0) > MIN_FRAME_DIST:
        return 0
    return 1

configs = [
    Config(1000, 40, 40 * 32, 32),
    Config(1000, 40, 40 * 32 * 10, 32),
    Config(1000, 40, -1, 32),
]
analyze(configs, is_same, N=20)
```

Output:
```
+--------+------+------+------+------+------+------+------+------+
|        | Within Single Batch       | Neighbouring Batches      |
| Config | AVG  | STD  | MIN  | MAX  | AVG  | STD  | MIN  | MAX  |
+--------+------+------+------+------+------+------+------+------+
|      0 | 10.5 |  1.6 |  7.2 | 16.7 | 21.2 |  2.4 | 17.3 | 32.0 |
|      1 |  1.2 |  0.5 |  0.3 |  2.8 |  2.4 |  0.8 |  0.8 |  5.3 |
|      2 |  0.7 |  0.3 |  0.0 |  1.6 |  1.3 |  0.4 |  0.3 |  2.5 |
+--------+------+------+------+------+------+------+------+------+
```
![data](../../docs/jlabdev_images/394901ba862156f0144b8bb53ff121cf.png)

## Same Sequence

Now we count samples which come from the same sequence as too similar.
This could happen in cases where a camera is mounted on a tripod and thus almost all content of the image is identical.

Example:
```python
MIN_FRAME_DIST = 10

def is_same_seq(x, y):
    s0, f0 = x
    s1, f1 = y
    if s0 != s1:
        return 0
    return 1

configs = [
    Config(1000, 40, 40 * 32, 32),
    Config(1000, 40, 40 * 32 * 10, 32),
    Config(1000, 40, -1, 32),
]
analyze(configs, is_same_seq, N=20)
```

Output:
```
+--------+------+------+------+------+------+------+------+------+
|        | Within Single Batch       | Neighbouring Batches      |
| Config | AVG  | STD  | MIN  | MAX  | AVG  | STD  | MIN  | MAX  |
+--------+------+------+------+------+------+------+------+------+
|      0 | 23.4 |  2.6 | 19.5 | 37.8 | 48.5 |  4.8 | 39.7 | 74.7 |
|      1 |  2.8 |  0.9 |  1.1 |  5.0 |  5.5 |  1.4 |  3.3 | 11.2 |
|      2 |  1.5 |  0.4 |  0.8 |  3.1 |  3.1 |  0.6 |  1.7 |  4.4 |
+--------+------+------+------+------+------+------+------+------+
```
![data](../../docs/jlabdev_images/75f76b32562b959227cea29fc648a990.png)

## Fewer and longer sequences

In thisnext example we have only 200 sequences where each is of length 200.
This simulates datasets with longer sequences, where one sequence would roughly correspond to 1 shard.

Example:
```python

MIN_FRAME_DIST = 10

def is_same(x, y):
    s0, f0 = x
    s1, f1 = y
    if s0 != s1:
        return 0
    if abs(f1-f0) > MIN_FRAME_DIST:
        return 0
    return 1

configs = [
    Config(200, 200, 200 * 32 / 5, 32),
    Config(200, 200, 200 * 32, 32),
    Config(200, 200, -1, 32),
]
analyze(configs, is_same, N=20)
```

Output:
```
+--------+------+------+------+------+------+------+------+------+
|        | Within Single Batch       | Neighbouring Batches      |
| Config | AVG  | STD  | MIN  | MAX  | AVG  | STD  | MIN  | MAX  |
+--------+------+------+------+------+------+------+------+------+
|      0 | 11.5 |  1.8 |  8.6 | 21.7 | 23.7 |  2.6 | 18.3 | 35.9 |
|      1 |  2.5 |  0.7 |  1.4 |  4.5 |  5.2 |  1.3 |  3.0 |  8.9 |
|      2 |  0.8 |  0.3 |  0.0 |  1.7 |  1.6 |  0.4 |  0.8 |  2.8 |
+--------+------+------+------+------+------+------+------+------+
```
![data](../../docs/jlabdev_images/4a4f0ad12d66e086ca2d562c000b5b3b.png)

