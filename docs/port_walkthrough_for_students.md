# How we taught an NVIDIA-only AI training pipeline to run on an Intel GPU

*Article material, beginner-friendly version. Written for a reader who knows how to
program a bit and has heard of AI, but has never trained a model. The technically dense
version with the exact code lives in `port_walkthrough.md`.*

---

## What we were trying to do, in one paragraph

A voice assistant needs a **wake word** — a phrase like "hey Siri" that tells it "this
speech is for you". Detecting it is done by a small neural network that listens to raw
audio all the time. There's a great open-source project for this, openWakeWord, and it
comes in two halves: **inference** (using a trained model — cheap, runs anywhere) and
**training** (creating that model — expensive, needs a powerful graphics card). The
problem: its training half only works on NVIDIA graphics cards. We own an Intel one.
This is the story of making it work anyway — and of everything that broke on the way.

## First, four pieces of background

**Why do you need a graphics card to train AI at all?** Training a neural network is
mostly multiplying huge tables of numbers together, millions of times. A CPU does a few
of those multiplications at once; a GPU (graphics card) was built to color millions of
pixels simultaneously, which turns out to be the same kind of math. So GPUs train
models tens of times faster.

**What is CUDA?** CUDA is NVIDIA's private language for talking to their GPUs. It came
first, it's excellent, and for 15 years almost all AI software was written assuming it.
That's a problem if your GPU is made by anyone else — like game console exclusives:
the game is great, but it only runs on one company's box.

**What is PyTorch, and what's a "device"?** PyTorch is the most popular library for
building neural networks in Python. When your code says *where* a computation should
run, it names a **device**: `"cpu"`, or `"cuda"` for NVIDIA GPUs. The catch: thousands
of programs have the word `"cuda"` literally written into them. Recently PyTorch gained
a third option: `"xpu"`, the name for Intel GPUs. The hardware support exists — but all
that old code still says `"cuda"` in it.

**What does "porting" mean here?** Taking software written for one platform and making
it run on another. Our job: find every place the training pipeline says or assumes
"cuda", and make it work with "xpu" — *without breaking it for people who do have
NVIDIA cards*.

## Step 1 — before changing anything: find out where the enemy actually is

Rule one of porting: don't touch the code until you know every place that needs
touching. We searched the whole chain of software for CUDA references first.

The surprise: the pipeline we forked was almost clean. The CUDA assumptions were hiding
in the *libraries it uses* — in three different places:

1. **The trainer itself** (openwakeword's training script): picks its device with
   literally `'cuda' if available else 'cpu'` — Intel isn't even considered a GPU here.
2. **The speech generator** (Piper): training needs thousands of recordings of people
   saying your wake word. Nobody records those — a text-to-speech engine *generates*
   them in hundreds of synthetic voices. That engine also has `"use the GPU" = "use
   CUDA"` wired in.
3. **The sneaky one.** One component takes a `device="gpu"` option — but "gpu" there
   doesn't mean PyTorch at all. It selects an NVIDIA plugin of a *different* library
   (ONNX Runtime). Lesson: the same word can mean different things in different layers.
   If we'd blindly replaced "cuda" with "xpu" everywhere, this one would have broken in
   a way that's hard to even diagnose.

One more thing we found: the speech generator already contained a special code branch
for Apple's chips (added when Macs got their own GPUs). So "teach old AI code about a
new kind of GPU" had been done before, right there in the file. Intel was just the
branch nobody had written yet.

## Step 2 — prove the thing works BEFORE you change it

Before porting, we ran the entire pipeline unmodified on the plain CPU. Slow, but it
answers a crucial question later: *if something breaks after our changes, was it us —
or was it already broken?*

Good thing we did, because it **was** already broken — six times over, and none of it
had anything to do with CUDA. The pipeline hadn't been updated in five months, and in
the Python world that's an eternity: a math library had deleted a function that an
audio library still used; the speech generator had reorganized its files so imports
pointed at nothing; a download check expected a 600 MB file when the real one is
204 MB… Six fixes just to reach the starting line.

This deserves its own lesson: **open-source AI code rots fast.** The NVIDIA lock-in
was just the loudest symptom of an unmaintained stack. If you ever wonder why "it
worked in the tutorial" fails on your machine — this is why. Pin your versions (write
down exactly which version of everything works) like your project depends on it,
because it does.

## Step 3 — the actual port: one file rewritten, the rest tricked

We used two techniques, and choosing which one where is the design decision of the
whole project.

**Technique A — copy and edit ("vendoring").** The main training script (~900 lines)
got copied into our project and its device logic rewritten. Now, instead of "cuda or
cpu", it asks one small function of ours:

```python
def resolve_device():
    if torch.xpu.is_available():   return "xpu"    # Intel GPU
    if torch.cuda.is_available():  return "cuda"   # NVIDIA GPU
    return "cpu"                                    # neither
```

Everything downstream uses the answer. Note it still says NVIDIA — a port should *add*
a platform, not swap one lock-in for another.

**Technique B — monkeypatching (the trick).** The speech generator has its CUDA calls
buried deep inside functions we'd rather not copy and maintain forever. Python allows
something wild: at runtime, you can *replace functions inside someone else's imported
library*. It's called monkeypatching. Before the pipeline starts, we quietly install
replacements:

- `torch.cuda.is_available()` — the "do you have an NVIDIA card?" question — is
  replaced so it answers **"yes"** on Intel machines,
- and every "move this data to the NVIDIA card" call is replaced so it moves the data
  to the Intel card instead.

So the old code *believes* it's on an NVIDIA machine, calls the usual functions, and
lands on Intel hardware. All of it guarded: on a real NVIDIA box, or a box with no GPU,
the patches don't install and nothing changes.

**The cost of lying.** Monkeypatching means your whole process now believes something
false, and you must catch every consequence. It bit us twice:

1. Some code doesn't call `.cuda()` — it builds a destination named `'cuda:0'` and
   *then* moves data there. Our first patch missed that spelling, so the lie ("yes,
   you have CUDA!") sent that code down a path that crashed. We had to also intercept
   the move operation itself and rewrite the destination.
2. At the very end, the model-export step *also* asks "is CUDA available?" — hears our
   "yes" — and tries to initialize real CUDA. Crash, *after* an hour of successful
   training. Fix: during export, temporarily restore the honest answer. **If you must
   lie to old code, keep the truth saved somewhere, and know exactly when to tell it.**

## Step 4 — the bug that made the whole project worth it

First real run on the Intel GPU: the speech generator crashes deep inside the
text-to-speech model, with an error pointing... nowhere useful.

An annoying property of GPUs: they run asynchronously. The CPU hands work to the GPU
and races ahead; when the GPU hits a problem, the error surfaces *later*, in some
unrelated line. Debugging that is like hearing the smoke alarm twenty minutes after
dinner — which meal burned? Our trick: force the CPU to stop and wait after every
single stage (a "synchronize" call). Now the alarm rings while the pan is still on
fire, and we found the exact operation.

Then we shrank the problem. Out of a full text-to-speech model, down to one PyTorch
operation, down to five numbers:

```python
torch.tensor([1, 0, 1, 0, 1], device="xpu").nonzero()
```

`nonzero` answers one trivial question: *at which positions is this list non-zero?*
The correct answer is positions 0, 2 and 4. Our Intel GPU answered **0, 0 and 4**.

Sit with that for a second. This isn't our code being wrong, or the wake-word project
being wrong. It's one of PyTorch's most basic operations **computing a wrong answer**
on that version (2.13) of its Intel build — like a calculator that says 2+2=5, but
only on Tuesdays, quietly. It didn't even crash most of the time. Everything built on
top of it — and in PyTorch, every "select the elements where condition X holds" runs
through `nonzero` — was silently poisoned. A training run would have produced a
garbage model with no error anywhere.

The fix was undramatic: we tested older versions of PyTorch on the same machine, found
that 2.9.1 computes correctly, and pinned it. But the *practice* we took away is the
valuable part: we wrote a small script that any new machine must pass before we trust
it — it literally checks that counting positions of five numbers gives the right
answer. **Don't just test your program. On new hardware platforms, test the
arithmetic.** ("Should work the same" is precisely the assumption that needs testing —
that sentence is why this project exists.)

## Step 5 — memory, and the assistant that shot itself

Our development machine was a laptop, whose small Intel GPU has no memory of its own —
it borrows the laptop's ordinary RAM. Two consequences we never would have seen on a
big discrete card:

- PyTorch keeps memory it has already used, for speed (why give it back if you'll need
  it again?). On a 24 GB dedicated card, nobody notices. In borrowed laptop RAM, that
  greed eventually left nothing to borrow, and training died mid-run. Fix: tell it to
  release its cache at a few strategic moments.
- Better: at one point the operating system, low on memory, chose a process to kill to
  save the machine — and it picked… the voice assistant's own speech server, running on
  the same laptop. The wake word being trained *for* the assistant took the assistant
  down. On shared hardware, training and using AI compete for the same resources;
  that's the real argument for a workstation card with its own memory.

## Did it work? The numbers

Everything runs, end to end, on the Intel GPU: generating synthetic voices, audio
processing, training, exporting the final model. And the numbers tell an honest,
slightly surprising story (small test run; laptop iGPU vs the same laptop's CPU):

| What | CPU | Intel laptop GPU |
|---|---|---|
| The core training loop | ~1 step/second | **~40 steps/second** |
| The whole small test run | 8.5 min | 11 min (!) |
| Projected full-size run | ~14 hours | ~1–1.5 hours |

How can the GPU be 40× faster at the loop and still lose the small test? Because the
test run is dominated by *other* work (validation checks in tiny pieces, where shuttling
data to the GPU costs more than the math saves). At full scale the loop dominates and
the GPU wins by an order of magnitude. Benchmarks lie if you measure the small version
of your problem — measure what you'll actually run.

And all of this was the *warm-up*: it validated the port on a laptop. The real training
runs on an Intel Arc Pro B70 workstation card (32 GB of its own memory — both memory
problems from Step 5 simply don't exist there). Those numbers are the next chapter.

## The six things to remember

1. **Map before you edit.** Find every platform assumption first; ours hid in three
   libraries and two different meanings of the word "gpu".
2. **Get a known-good baseline before changing anything** — otherwise you can't tell
   your bugs from the ones that were already there.
3. **Version-pin everything.** Most of what broke wasn't CUDA — it was five months of
   ordinary software drift.
4. **Monkeypatching is a loan.** You can lie to old code to make it run, but you must
   track every place the lie leaks, and keep the truth around to restore.
5. **On new hardware, verify the arithmetic itself.** The scariest bug computed wrong
   answers without crashing. A ten-line sanity script now guards every machine we use.
6. **Portability is a public good.** The fix wasn't heroic — resolve the device in one
   place, add one branch. The barrier was never technical difficulty; it was that
   nobody with non-NVIDIA hardware had walked through first and written it down.
