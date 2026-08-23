## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

1. Artificial Intelligence is the sector of technology where computers perform in ways that would normally require human intelligence. Machine learning is a part of artifical intelligence where computers learn the rules and find patterns on their own without hard coded rules. Deep learning is using neural networks to find those rules.

### 2. A problem traditional programming can't touch

1. Creating sheet music is something that ML learning would be able to do well, but not traditional programming. With how different music could be written, ML would be necessary to make something unique.

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

- unsupervised - each song is unique, there are no categories
- Parametric - parameters are adjusted during training

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

- **Supervised - X-rays are labeled pneumonia vs no pneumonia
- **Parametric - millions of fixed weights are adjusted during training

### 3. Retail site — recommendations from the 10 most similar past customers

-  Supervised - website knows previous purchases
- nonparametric - no parameters are learned ahead of time. The model compares new customers against previous ones.

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario: Photo libraries grouping photos by similarity
- unsupervised - no labels for the photos, only similarites.
nonparametric - no set parameters. The groups depend on the data.

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob - the weight that gets altered depending on the results and how close they are to the correct answer.
- **(b) the prediction - the temperature that the thermostat is trying to get set to.
- **(c) the error - after the temperature settles, the error is the difference between the actual temp, and the predicted temp.
- **(d) the learning step - how much to change the knob by depending on the error.

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` - The prediction changes to 1.70.
- **(b) weight set to `0` - the input will be zero, thus that neuron doesn't affect the output.
- **(c) weight negative, `-0.1` - the prediction is -0.85, so the neuron would try to push the prediction in the negative direction. If trying to predict if an email is spam, the word "calendar" might have a negative weight because spam emails are less likely to have this word.
- **(d) if the answer should have been `1.7` — the weight of 0.1 is too low, it should be 0.2 because the input (8.5) * (0.2) would be 1.7

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

3. When there are a lot of knobs, it can be hard to see which ones are affecting the output the most in the positive and negative directions. Also, each knob depends on the other ones for predicting the final output.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

1. The problem of procrastination vs productivity. The input data could be a bunch of different things from grades, IQ, attention to detail, and schedules. I would want to be able to predict what percent of assignments get done on time, or maybe even early.

### 2. Supervised or unsupervised, and why

Supervised because the model would be trained on the students whose assignment completion information would already be known.
### 3. What "success" would look like

Success would be able to tell how well students will be able to keep up with their assignemnts if they keep doing the same thing.

### 4. What could go wrong

Students could turn in assignments without working hard on them, and they would be considered productive while failing classes.
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
