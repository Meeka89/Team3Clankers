# Foundations Reflection — template

**Assignment 2 · 20 points · individual.** Merged by **Sunday, Aug 23, 11:59 PM.**


---

## Part 1 — The Landscape (4 pts)


### 1. AI vs. machine learning vs. deep learning

The categories of AI, machine learning, and deep learning parts of each other. Simply, AI encompasses ML and ML encompasses DL. Artificial Intelligence is simply anything that artificially takes the job
of what a human can do, but more nuanced answer would describe it as rule based expert systems that do the jobs of humans. Machine Learning is another step further and more sophisticated solutions to these
problems and solutions can involve decision trees, SVMs, and so on. Deep Learning is level even further where the models are improving upon themselves, "remembering" errors, and most commonly deep learning
takes the forms of neural networks.

### 2. A problem traditional programming can't touch

Machine Learning excels with examples where their large amounts of data and labels. So, for medical data where one is trying to figure out if one testing positive or negative for cancer or some other condition.
In these examples, machine learning excels because machine learning solution identifies the rules from the data and can identify patterns in the data much better than traditional programming.

---

## Part 2 — Classifying Algorithms (6 pts)

### 1. Music app — auto-generated playlists

- **Unsupervised** - app is clustering similar songs into playlists that are not created already (unlabeled)
- **Nonparametric** - the app is creating an unknown amount of playlists (flexible number of parameters).

### 2. Hospital — pneumonia from 100,000 labeled chest X-rays

- **Supervised** - The chest X-rays are already labeled.
- **Parametric** - There is a set amount of samples (100,000 x-rays).

### 3. Retail site — recommendations from the 10 most similar past customers

- **Unsupervised** - The site is creating recommendations based off the other similar customers.
- **Parametric** - The site is remembering any prior similar customers and there's new fixed similar customer list

### 4. Your own scenario

*Pick a combination you have **not** already used in 1–3.*

- **The scenario:** Identifying who is most likely to tear their ACL of everyone in Arkansas 
- **Supervised** - It would be supervised because you are taking everyone and their data trying to label if they tear ACL. They either are labeled as torn their ACL or not.
- **Parametric** - This would be parametric because it would be limited to the fixed number of people in Arkansas only.

---

## Part 3 — The Knobs Mental Model (6 pts)

*No calculator needed. These are about whether you can feel what a weight is doing
before you compute it. A fast answer here is usually a shallow one.*

### 1. The thermostat analogy

- **(a) the knob —** The knob are the weights of the model. When provide your input (turning the knob), how the knob outputs the data (the actual water temperature) is dependent on the makeup of the knob aka the weights.
- **(b) the prediction —** This is the temperature of the water when you turn the knob. When you turn the knob to the prediction of what you think the water will be, that is your prediction of the input of the knob input.
- **(c) the error —** The error is your prediction of how it would be when you turned the knob to the actual temperature of the water. So, now you know that hot is not as hot as you thought and et cetera.
- **(d) the learning step —** The learning step of realizing how hot the actual water is and then adjusting the knob to another spot and predicting how hot it would be more accurately than your first prediction is the learning step. This mirrors how a neural network works the loss of the prediction to create better results next time.

### 2. Single neuron arithmetic

- **(a) weight doubled to `0.2` —** The prediction is then doubled to 1.7 
- **(b) weight set to `0` —** *(and what that means conceptually)* If the weight is 0, then the prediction is zero. This will mean that the input here is not as important, but this is where you have to worry about dead neurons too.
- **(c) weight negative, `-0.1` —** *(and what a negative weight could represent)* It would make the prediction negative, so -0.85. This represents just an inverse relationship given the input, so given the input the prediction would be stronger inclination of the inverse.
- **(d) if the answer should have been `1.7` —** *(too high or too low, and how you know)* In this very simple case, it would simply take the prediction and the answer and find the difference. In a DEEP neural network where there's multiple levels, it would do this across multiple levels to figure the gradients.

### 3. The big picture

Finding the right knob positions is very challenging because it depends on the data set and simply that there are lots of knobs to mess with. Some datasets have different priorities so "adjusting the knob" correctly looks differently and we may have to experiment which datasets need which adjustments.

---

## Part 4 — Your Deep Learning Problem (4 pts)

*Something you're actually fascinated by — from your life, your major, your hometown.
You don't need to know how to build it. You may answer in your padawan's voice; the
thinking still has to be yours.*

### 1. The problem

I love movies and I would want to know what other people like the movies I like. You could use some sort of dataset from either IMDB or Letterboxd (both are movie ranking/grading/reviewing platforms), and type people who are the most similar highest rankings of movies.

### 2. Supervised or unsupervised, and why

This would be unsupervised because you do not have labels for them and the model would categorize people into similar clusters. There is no pre-existing labels.

### 3. What "success" would look like

If it was successful, the model would categorize a person with other people who agree on the rankings/tastes of movies. If you think the Dark Knight is the best movie ever, then it would put you with other people who think similarly. 

### 4. What could go wrong

This could go wrong in a number of ways, but the biggest pitfall that comes to mind is the fact that people rank and grade movies differently. For example, one person may be extremely critical of movies and for the most part rank movies pretty low except for one or two movies, while on the flip side you have someone who generally loves most movies and ranks everything 5 stars. I don't know how that would affect the model and maybe it would be able handle things like that, but even still, the fact that there is no agreeable metric on how to rank movies makes it hard to super accurately group people.

---
