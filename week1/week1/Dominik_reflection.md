# Foundations Reflection — template

**Assignment 2 · 20 points · individual.** Merged by **Sunday, Aug 23, 11:59 PM.**

> **Copy this file — don't write in it.** Save your answers as
> `week1/yourname_reflection.md` (your real first name, lowercase — e.g.
> `week1/alia_reflection.md`). Leave this template where it is so the next person
> can find it.
>
> The full prompts live in the Assignment 2 handout on Canvas. What's below is the
> skeleton: every part and question, in order, with the point values. It is not a
> substitute for the handout — keep that open, especially for Part 2, where the
> exact wording of each scenario is what carries the answer.
>
> The four parts are graded separately, and a part you forgot to answer is a zero
> for that part — far and away the most common way people lose points on this one.

*In your copy, delete the block above and the italic hints as you go.*

---

## Part 1 — The Landscape (4 pts)



### 1. AI vs. machine learning vs. deep learning

*Not just "subset" — what makes each layer different from the one above it?*

AI: The enabling of machines to replace human actions
ML: Learning pattern from structured data
Deep Learning: Using neural networks

### 2. A problem traditional programming can't touch

*Original example, and why machine learning suits it.*

Detecting fradulaut card transactions. Traditional programming would not work for this since smaller purchases could look innocent but machine learning could recongize the context of the said transation and tell that for example that it was made right after a password was reset to effectively block the transaction.

---

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

*…and it has a fixed set of internal parameters it adjusts during training.*

- Unsupervised - the songs form groups where the names are auto generated.
- Parametric- all the songs in the playlist are songs that the user has listened to before

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

*…and the model has millions of fixed weights adjusted during training.*

- Supervised - the patient either has pnemounia or doesn't
- Nonparametric - the weights adjust meaning the data changes

### 3. Retail site — recommendations from the 10 most similar past customers

*…those recorded purchases are the answers it learns from, and no parameters are
learned ahead of time.*

- Supervised - the past purchases guide the recommendation
- Nonparametric - there are no parameters learned ahead of time and the similar purchases the reccomendation is based on changes.

### 4. Your own scenario


- Generating the player's next frames in a completely AI based Minecraft game.
- Unsupervised - each pixel in the next frame is a guess as to what should fit there
- Parametric - data is real Minecraft gameplay and video footage from it creating the prediction

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob — also the literal knob on the wall that u turn to adjust the temp (in this case there is only one)**
- **(b) the prediction — the current temp setting (the actual output) **
- **(c) the error — the difference between the temp we wanted and the actual temp**
- **(d) the learning step — turning the knob slightly hotter or colder based on the temp we feel at the moment**

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` — it makes the prediction equal to 1.7**
- **(b) weight set to `0` — the prediction is also 0 making it not important**
- **(c) weight negative, `-0.1` — equal to -0.85 meaning as the input increases the prediction decreases**
- **(d) if the answer should have been `1.7` — to low since 0.85 < 1.7 meaning the weight should be higher** *(too high or too low, and how you know)*

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Data can be extremely large so finding the exact positions is iextremely expensive while we can only use a fraction of the data to set approximate positions that do the job.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

In a competitve matchmaking system for a game, players get teamates and opponents assigned based on their performance in previous games.

### 2. Supervised or unsupervised, and why

Unsupervised, since the players assigned can vary based on all their previous performances.

### 3. What "success" would look like

Teamates and opponents are all balanced skill-wise and all players have a fair competitive experience.

### 4. What could go wrong


A player doesn't receive the desired competitive experience due to not having assigned appropriate teammates, i.e. having too skilled opponents or teamates not matching their own compentence.

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
