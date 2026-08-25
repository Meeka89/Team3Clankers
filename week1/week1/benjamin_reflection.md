# Foundations Reflection — template

**Assignment 2 · 20 points · individual.** Merged by **Sunday, Aug 23, 11:59 PM.**

---

## Part 1 — The Landscape (4 pts)

*Your own words, 2–4 sentences each. If you catch yourself reaching for a phrase you
remember off a slide, treat that as a signal the idea isn't yours yet.*

### 1. AI vs. machine learning vs. deep learning

Artificial Intelligence is a way that a computer produces responses for things or does some simple preordained task. It is generally used for doing minute and tedious work. Machine Learning is a more complex Artificial Intelligence that uses linear equations and specific code to solve higher and more abstract problems. It can help predict things given a dataset. Deep Learning goes even further and dives into the realm of Neural Networks.  

### 2. A problem traditional programming can't touch

A problem with traditional programming (data + rules -> answers) that can easily be solved by Machine Learning (data + answers -> rules) is when there is a large dataset that involves null values and requires a prediction made for future analysis.

## Part 2 — Classifying Algorithms (6 pts)

*Both axes for **every** scenario — supervised/unsupervised **and**
parametric/nonparametric — plus a justification for each. The justification is where
the points are. The classification on its own is a coin flip, and I can't tell a
lucky guess from understanding.*

### 1. Music app — auto-generated playlists

- **Supervised or unsupervised?**
    unsupervised since it is adapting to different music styles per person
- **Parametric or nonparametric?**
    parametric since it has parameters that it adjusts 

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

- **Supervised or unsupervised?**
    supervised since the model is shown what to learn from.
- **Parametric or nonparametric?**
    parametric since the number of the inputs is fixed.

### 3. Retail site — recommendations from the 10 most similar past customers

- **Supervised or unsupervised?**
    unsupervised since there are differing inputs from customers 
- **Parametric or nonparametric?**
    nonparametric since there are no parameters to learn from before

### 4. Your own scenario

- **The scenario:**
    Predicting house prices in a neighborhood by taking the weighted average price of the 5 closest similar homes
- **Supervised or unsupervised?**
    supervised since the model is training on the previous housing prices
- **Parametric or nonparametric?**
    nonparametric since the models make predictions from the known data

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —**
    the parameter
- **(b) the prediction —**
    the output
- **(c) the error —**
    error = target - prediction
- **(d) the learning step —**
    gradient update

### 2. Single neuron arithmetic

`prediction = input * weight` — with `input = 8.5` and `weight = 0.1`, the prediction
is `0.85`.

- **(a) weight doubled to `0.2` —**
    1.7 (prediction doubles)
- **(b) weight set to `0` —** *(and what that means conceptually)*
    0 (prediction and weight = 0 which means the model will not factor it into the output)
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)*
    -0.85 (an inverse or opposing relationship; commparing inversely related things)
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)*
    the answer is too low (the weight needs to be doubled to reach 1.7)

### 3. The big picture

*Why is finding the right knob positions hard? Why can't we just calculate them
directly?*

Finding the perfect fit quickly is hard, especially since we can't always get a clear picture of what is changing with so many variables at play. It is just like trying to find a needle in a haystack: almost impossible.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

*Be specific: what does the input data look like, and what are you predicting?*

Being able to help automated cars avoid hitting pedestrians
Have an automated system to help watch for people even in darkness, rain, etc.

The input data would be from many different types of high definition cameras, radars, and other visual aids to watch for and predict pedestrian traffic

### 2. Supervised or unsupervised, and why

supervised since the model is training on data that is recorded live and from historical records of where pedestrians are likeliest to be walking.

### 3. What "success" would look like

No accidents with hitting pedestrians or causing the car to hit anything in avoidance of hitting a person. Also preserving brakes by smooth slowdown vs emergency braking every time it has to stop.

### 4. What could go wrong

If the model is not prepared for certain edgecases (such as unexpected jay-walking, etc.), the vehicle may have an accident either by hitting a person or another vehicle causing expensive repairs (in the case of vehicle to vehicle) or fatality (in the case of vehicle to person).

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
