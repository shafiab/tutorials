
# Bayes Theorem
$\theta -$  Parameters
$X - $ Observations

![enter image description here](https://lh3.googleusercontent.com/xL_wgSlIv21xg6F3xD7wzyld_Pu6sRCuQYScxuBJ7w5Y5UchVeZ0vE3HiqUkrHUc-xCLyLB0VoA=s0 "Screen Shot 2017-11-14 at 9.21.48 PM.png")

### Probability rules:
 - **Sum rules:** Marginalization from joint distribution $$p(X) = \int_{-\infty}^\infty p(X,Y) dY$$
 - **Chain rules:** $$
  P(X, Y) = P(X|Y)P(Y)=P(Y|X)P(X)\\
  P(X,Y,Z)=P(X|Y,Z)P(Y,Z)=P(X|Y,Z)P(Y|Z)P(Z)\\
  P(X_1,\ldots,X_N)=\prod_{i=1}^NP(X_i|X_1,\ldots,X_{i-1})
  $$
### Point Estimation (Frequentist vs. Bayesian)
Rather than estimate the entire distribution $p(θ|x)$, sometimes it is sufficient to find a single ‘good’ value for $θ$. We call this a point estimate.

 - Frequentist thinks parameters $\theta$ are fixed, data $X$ are random. 
 Maximum Likelihood Estimation: $$\hat{\theta}=\arg\max_{\theta} P(X|\theta) $$
 - Bayesian thinks parameters $\theta$ are random, data $X$ are fixed.
 Maximum Aposteriori Estimation (MAP) $$\hat{\theta}=\arg\max_{\theta} P(\theta|X) $$ 
  - MAP estimation is not invariant to non-linear transformations of $θ$. E.g. A non-linear transformation  $\theta^\prime=g(\theta)$, to $\theta$  can shift the posterior mode in such a way that $g^{-1}(\mathrm{mode}[\theta^\prime]) \neq \mathrm{mode}[\theta]$.
  - MAP estimate may not be typical of the posterior. 


## Bayesian Netwok (Graphical Model)
![enter image description here](https://lh3.googleusercontent.com/EbDwfsTX7a-plaotkXFSJ4kyDZ-2BhE6rgpUYRjWaC1ThN_Emp-chhKVhP5-PFUKhEaMqSpHR4M=s400 "Screen Shot 2017-11-14 at 9.48.06 PM.png")

 - Nodes are random variables
 - Edges indicates dependence (e.g. Grass is wet depends on both sprinkler or rain, and whether sprinkler is on or off depends on rain)
 - Observed variables are shaded nodes; unshaded nodes are hidden
 - Plated denote replicated structure

Joint probability over all the variables in the above model is given by:
![enter image description here](https://lh3.googleusercontent.com/TN2FS7xe1saG8IZqYbqUHwC78CZTw-kZGztDlt5oyT1efKU8vYJwMVP4SXTwIbVzosPdqg_RHzY=s400 "Screen Shot 2017-11-14 at 9.52.05 PM.png")

**Example 1:**
![enter image description here](https://lh3.googleusercontent.com/nStGvG60Q40IsXdBcoWiYRkIYG4jX04daZ_io4fhLNiaOZ5ycYw-K9In8eqyzcet45fuPANQvQE=s300 "Screen Shot 2017-11-14 at 9.54.38 PM.png") Here,  $P(S,R,G) = P(G|S, R) P(S|R) P(R)$

**Example 2: Naive Bayes Classifer**
![enter image description here](https://lh3.googleusercontent.com/-8YmaTwGT5Nw/WgvYHhwHclI/AAAAAAAAADs/xit9iNys-K0ekgzOXBlSouASqQdQNoaKwCLcBGAs/s300/Screen+Shot+2017-11-14+at+9.58.55+PM.png "Screen Shot 2017-11-14 at 9.58.55 PM.png") 

Joint Probability $P(c, f_1,\ldots,f_N)=P(c)\prod_{i=1}^N P(f_i|C)$

In plate notation, the figure above can be shortened as follows:

![enter image description here](https://lh3.googleusercontent.com/-34usjTZw2_o/WgvYCVVgoxI/AAAAAAAAADk/nz0kDVbkUUUNzJtWj646AvBG0nZl_DJfwCLcBGAs/s300/Screen+Shot+2017-11-14+at+9.59.02+PM.png "Screen Shot 2017-11-14 at 9.59.02 PM.png")
# Conjugate Prior
 - MIT [lecture note](https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec15.pdf) has a good section on conjugate prior

Point estimation is useful for many applications, however true goal in Bayesian analysis is often to find the full posterior $p(\theta|X)$. In most cases, it is difficult to calculate the denominator $p(X)=\int p(X|\theta)p(\theta)$. One approach to circumventing the integral is to use conjugate priors. Here the idea is, if we choose the ‘right’ prior for a particular likelihood function, then we can compute the posterior without worrying about the integral.

Formally, a prior  $p(\theta)$ is **conjugate to the likelihood $p(X|\theta)$**, if the prior  $p(\theta)$ and the posterior  $p(\theta|x)$ are from the same family of distribution.

Examples:
 

 - Beta distribution is conjugate to Bernoulli likelihood. [Here](http://varianceexplained.org/statistics/beta_distribution_and_baseball/) is a good example of this for baseball batting average calculation.
 - Dirichlet distribution is conjugate to Multinomial likelihood (e.g. application in LDA)
 
# Variational Inference
 - very intuitive explanation in this [blog](http://blog.evjang.com/2016/08/variational-bayes.html)

# Common Probability Distributions
## Gamma Distribution

$$
p(\gamma|a,b) = \dfrac{b^a}{\Gamma(a)}\gamma ^{a-1}e^{-b\gamma}
$$
Here,

-  $ \gamma, a, b > 0$
- support of Gamma distribution is $[0, \infty)$
- $\mathbb{E}[\gamma]=\frac{a}{b}$
- $\text{Var}[\gamma]=\frac{a}{b^2}$

![enter image description here](https://lh3.googleusercontent.com/-qHr1Hzm5u_gI3sS7zT03VfxJW2_50-RqYoLszvmQyLn02oiChCnc8U-JPq3ZADq0TX4iGwXaug=s400 "Screenshot 2017-11-22 01.46.31.png")

**Example:** Suppose I ran 5km $\pm$ 100 m every day, i.e. mean 5km with std 100m. We can model this as Gamma distribution. We can also use Gaussian - however, that means we can run negative distance.

![enter image description here](https://lh3.googleusercontent.com/--Q8IPSsVSpw/WhVIE29Nl-I/AAAAAAAAAE0/sJu5DdSPoVUc0QzU3CPJEEeserY7Kq1MgCLcBGAs/s400/Screenshot+2017-11-22+01.48.32.png "Screenshot 2017-11-22 01.48.32.png")

## Beta Distribution
$$
p(x|a,b)=\dfrac{1}{B(a,b)}x^{a-1}(1-x)^{b-1}
$$
Here,
 - a, b > 0
 - support of beta distribution is [0,1], i.e. $x\in [0,1]$
 - $\mathbb{E}[x]=\frac{a}{a+b}$
 - $\text{Var}[x]=\frac{ab}{(a+b)^2(a+b-1)}$
 
![enter image description here](https://lh3.googleusercontent.com/-bYZ8S1hI0Sg/WhVKBX3iKZI/AAAAAAAAAFg/d1Y9XYKJAFwUCQD7sQeSFrqY_5yGHnDPACLcBGAs/s400/Screenshot+2017-11-22+01.57.03.png "Screenshot 2017-11-22 01.57.03.png")

**Example:** Baseball batting average (its a number between 0 and 1). e.g.$ 0.27\pm 0.1$