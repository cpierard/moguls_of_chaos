# Moguls of Chaos

What are the moguls and why are they related to chaos? Mogul skiing is a ski discipline that consists of descending a slope full of small regular bumps, where the skills and technique of skiers are tested to its full. Ed Lorenz, an outdoorsman and science legend, discovered that objects descending the moguls exhibit chaotic behavior. He built a small numerical model to explore the dynamics of an idealized board descending such type of slopes and exemplified some of the tools used for characterizing chaos.

This project aims to perform the small numerical experiment that Ed did and explained in his book *The Essence of Chaos*.

I started this small project two years ago after reading the above-mentioned book, but I couldn't finish it, and given that I am required to do a small project for my scientific computing class, I will use it as an opportunity to finish once and for all this mini-project.

**Update:** I created this project seven years ago. I still find it interesting and I wanted to use it as a project to reinforce my Pytorch skills. The idea is to train a NN with the sleds trajectories that exhibit chaos and recreate the Poincaré sections and bifurcation diagrams from a data-driven model.

![](animations/b_05.gif)

## Structure

### Part I : Solving the dynamics

I put the explanation and the results (code-free) in a blog post that you can find [here](https://cpierard.github.io/projects/moguls-of-chaos/). 

**Warning!** The results showed in the blogpost were created from two jupyter notebooks, found in  `notebooks\`, 7 years ago!. I recommend reading the blog post to understand how to build the sleds toy model. I don't recomend running the notebooks, because I didn't save the python or julia environemts used to code them, therefore, they might not run.

### Part II : Trainning the neural network on the solved trajectories

This is a description on how to used this scripts and I recomend executing them on the order I'm mentioning.

The python environment parameters are saved in `environmet.yml` and should be used to create a conda environment for this project.

In `generate_trajectories.py` I solve the trajectories of the sleds and save them in zarr files. The zarr contain the data to train the neural network.

The `prepare_data.py` defines two functions used to load and slit the dataset in training, testing and validation subsets.


## Objectives

- [x] Build the code in Julia.
- [x] Build the code in Python.
- [ ] Train a NN on sled trajectories
- [ ] Train a Neural ODE on trajectories
- [ ] Comparisons physics based & to data driven

## References

- Lorenz, Edward N. *The Essence of Chaos*. Univ. of Washington Press, 1993.
