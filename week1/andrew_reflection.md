## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

*Not just "subset" — what makes each layer different from the one above it?*

AI is the umbrella of all forms of AI. Machine learning is the machine finding the rules. 
DL is the specific subset built on neural networks. Kind of inspired by the brain!

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*
A program that detects what kind of car is in a picture. You can make your program look for very specific details 
and say if the wheels look like this and the hood looks like this and so forth. It would take forever to program an algorithm to do that
. For ML, you can just throw in a bunch of examples, and it will train itself on those examples and pop out an answer 


---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**
Unsupervised, parametric: it isn't given labeled correct answers; it finds patterns in the data
. Also, the prompt says fixed params
### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**
Supervised, parametric: 100,000 LABELED correct answers, so it knows what's right, and the model has fixed weights so
it's comparing prediction to truth
### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- **Supervised or unsupervised?**
- **Parametric or nonparametric?**
Supervised nonparametric: explicitly says no params are learned ahead of time, so this is nonparametric activity, and then it's unsupervised
due to the fact that the 10 most similar past customers is an ever-changing variable 
### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario:**
- Say you are a chef trying to find a linguine noodle amongst many different kinds of pasta noodles. You then give your trusty pasta ai a large collection of unlabeled noodles.
-  You then need to group/find patterns among unlabeled noodles based on similarities such as shape, width, and length
- 
- **Supervised or unsupervised?**
- **Parametric or nonparametric?**
unsupervised, nonparametric: we are not giving the AI labeled answers like noodle = linguine; it's finding patterns on its own
the AI is not learning a fixed set of params ahead of time; it's working directly with the examples and their similarities 
---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —**
- this adjusts the weights/ how hot or cold you want it
- **(b) the prediction —**
- the thermostat is predicting what the temp is
- **(c) the error —**
- the difference between what you wanted and what actually happened
- **(d) the learning step —**
- with the knowledge we learn from a-b, we adjust the knob based on that error

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —**
- 8.5 x 0.2 = 1.7
- **(b) weight set to `0` —** *(and what that means conceptually)*
- pred is 0, so there is no influence 
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)*
- -0.85 so as the input increases, it pushes the pred upward
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)*
- 0.85 is smaller than 1.7, so the prediction is too low; if the prediction needs to be larger, increase the weight
- 

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Depending on how many knobs you have, or even if it's just one, the exact position can be very difficult because it may affect preds.
Real-world data is also very complicated, so most things just take time, and trying to calculate things like that is unrealistic

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

One thing deep learning could be weather prediction. Where I work, the weather is a huge factor to how
we plan on going about our day. I work at a construction company running equipment and if the ground is to wet on
site, we cant do any work. A predeciton model to find out when inclement weather is about to hit woul dbe very useful
and provide a scope of when we can work on certain projects and how much time we have until a rain

### 2. Supervised or unsupervised, and why

It would have to be unsuperivesd becasue the weather is super unpredictable and there are no definite labels
or parameters you can give it

### 3. What "success" would look like

*How would you know it was working?*

If my model is correctly predicitng when and how severe a storm is going to be and when it arrives,
we would know when to pull machines of site and prep for other things

### 4. What could go wrong

*At least one way this could fail or do harm if deployed carelessly.*

It could go wrong if just a huge dark cloud maybe gets misclassified and the ai tells us to leave
site for no reason and then we would waste time and money

---

## Before you open the PR

- [ ] All four parts answered — check against the headings above, not your memory
- [ ] Part 2: both axes **and** a justification, for all four scenarios
- [ ] Part 2 #4 uses a combination you didn't already use
- [ ] Part 4 #4 is answered — it's the one people skip
- [ ] Your own words throughout (see below)
- [ ] The file is `week1/yourname_reflection.md` — not this template

Week 1 branches straight off `main`; there's no week branch until Week 2
(`CONTRIBUTING.md` §1).

```bash
git checkout main
git pull origin main
git checkout -b week1/alia-reflection        # your name, not Alia's
git add week1/alia_reflection.md
git commit -m "[Week1] Add reflection for Alia Mehta"
git push -u origin week1/alia-reflection
```

Open the PR into `main`, ask a teammate to review it — they're confirming it's
complete and thoughtful, they are not grading it — and merge before Sunday 11:59 PM.

---

## On AI tools

You're welcome to use them to check your understanding. Every answer has to be in your
own words and reflect your own thinking. If I can paste your answer into a search box
and find it verbatim, that's a problem.

**And say so when you use them.** A line in your PR description naming what you used
and what you used it for is enough — the pull request template asks for it directly.
Acknowledged use is ordinary professional practice. Unacknowledged use is an academic
integrity issue.

Don't let an assistant write your reflection. Write it. *Then* ask it where you've been
unclear — and rewrite.
