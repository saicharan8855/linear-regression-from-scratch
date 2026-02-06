# Maximum Likelihood Estimation for Linear Regression

The Gaussian PDF leads to Mean Squared Error (MSE) as the loss function in regression models via maximum likelihood estimation (MLE), where we assume additive Gaussian noise. This derivation shows why minimizing squared errors is statistically optimal under that noise model step by step.


## Assumptions

Assuming Gaussian noise in the model $y = f(x;\theta) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, we derive the likelihood function of the observed data under this noise model.


## Model Assumption

Consider data points $(x_i, y_i)$ for $i = 1, \ldots, m$, modeled as
$$
y_i = \theta^\top x_i + \epsilon_i,
$$
where the noise terms are independent and identically distributed as
$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2).
$$

The conditional probability density function of $y_i$ given $x_i$ is
$$
p(y_i \mid x_i; \theta)
= \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{(y_i - \theta^\top x_i)^2}{2\sigma^2}
\right).
$$


## Likelihood Function

For independent observations, the joint likelihood is given by the product
$$
\mathcal{L}(\theta)
= \prod_{i=1}^{m} p(y_i \mid x_i; \theta).
$$

Substituting the Gaussian density, we obtain
$$
\mathcal{L}(\theta)
= \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^m
\exp\!\left(
-\frac{1}{2\sigma^2}
\sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\right),
$$
where $\hat{y}_i = \theta^\top x_i$.

Maximum Likelihood Estimation (MLE) finds the parameter vector $\theta$ that maximizes $\mathcal{L}(\theta)$, the probability of observing the data given the parameters.


## Log-Likelihood Simplification

Products are difficult to maximize directly, so we take the natural logarithm, which is a monotonically increasing function and therefore preserves the maximizer. The log-likelihood is
$$
\ell(\theta)
= \log \mathcal{L}(\theta)
= -\frac{m}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}
\sum_{i=1}^{m} (y_i - \hat{y}_i)^2.
$$

The first term is constant with respect to $\theta$, and $\sigma^2$ is typically fixed or estimated separately. Therefore, maximizing the log-likelihood $\ell(\theta)$ is equivalent to minimizing
$$
\sum_{i=1}^{m} (y_i - \hat{y}_i)^2,
$$
which is the sum of squared errors.


## Negative Log-Likelihood as Loss

In machine learning, optimization is typically framed as loss minimization. The negative log-likelihood (NLL) is given by
$$
-\ell(\theta)
= \text{constant}
+ \frac{1}{2\sigma^2}
\sum_{i=1}^{m} (y_i - \hat{y}_i)^2.
$$

Ignoring constants and positive scaling factors (since $\frac{1}{2\sigma^2} > 0$), minimizing the negative log-likelihood is equivalent to minimizing
$$
\frac{1}{m}
\sum_{i=1}^{m} (y_i - \hat{y}_i)^2,
$$
which is the Mean Squared Error (MSE).


## Conclusion

Under the assumption of independent and identically distributed Gaussian noise, maximizing the likelihood of the observed data leads to minimizing the Mean Squared Error. This establishes Mean Squared Error as the statistically optimal loss function for linear regression under the Gaussian noise model.








