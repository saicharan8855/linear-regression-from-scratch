import nbformat as nbf

# Your Geometry content (paste from earlier)
geometry_content = """This section presents a geometric interpretation of linear regression, showing that the optimal prediction vector is obtained by projecting the target vector $y$ onto the column space of the design matrix $X$. This viewpoint explains the normal equation as a consequence of orthogonality between the residual vector and the feature space.

## Vector Space Setup

## Column Space of the Design Matrix

The column space of the design matrix $X$, denoted as $\\text{Col}(X)$, is the subspace of $\\mathbb{R}^m$ spanned by the columns of $X$. Any vector in $\\text{Col}(X)$ can be written as a linear combination of the columns of $X$.

For any parameter vector $\\theta$, the predicted response  
$$\\hat{y} = X\\theta$$  
lies in the column space of $X$. Therefore, linear regression restricts predictions to vectors that belong to $\\text{Col}(X)$.

## Projection of the Target Vector

In general, the observed target vector $y$ does not lie in the column space of $X$. Linear regression therefore seeks a vector $\\hat{y} \\in \\text{Col}(X)$ that is closest to $y$ in the Euclidean sense.

This closest vector is obtained by orthogonally projecting $y$ onto the column space of $X$. The resulting vector $\\hat{y} = X\\theta$ represents the best linear approximation to $y$ within $\\text{Col}(X)$.

## Orthogonality of the Residual

Let the residual vector be defined as  
$$r = y - \\hat{y} = y - X\\theta.$$  
A fundamental property of orthogonal projection is that the residual vector is orthogonal to the subspace onto which the projection is made. Therefore, the residual $r$ is orthogonal to the column space of $X$:  
$$X^\\top r = 0.$$

## Derivation of the Normal Equation

From the orthogonality condition  
$$X^\\top r = 0,$$  
and using the definition of the residual $r = y - X\\theta$, we obtain  
$$X^\\top (y - X\\theta) = 0.$$  
Expanding the expression gives  
$$X^\\top y - X^\\top X\\theta = 0.$$  
Rearranging terms yields  
$$X^\\top X\\theta = X^\\top y.$$  
If $X^\\top X$ is invertible, the solution for $\\theta$ is  
$$\\theta = (X^\\top X)^{-1} X^\\top y.$$

## Geometric Interpretation

Geometrically, linear regression finds the vector $\\hat{y} = X\\theta$ in the column space of $X$ that is closest to the target vector $y$. The normal equation ensures that the residual vector $y - \\hat{y}$ is orthogonal to every column of $X$, meaning no further movement within the column space can reduce the distance to $y$.

Thus, the normal equation represents the condition for an orthogonal projection of $y$ onto $\\text{Col}(X)$.

## Conclusion

From a geometric perspective, linear regression is the problem of projecting the target vector $y$ onto the column space of the design matrix $X$. The normal equation arises naturally from the orthogonality condition of this projection, providing a closed‑form solution for the optimal parameters."""

# Create notebook
nb_geo = nbf.v4.new_notebook()
nb_geo.cells.append(nbf.v4.new_markdown_cell(geometry_content))

# Write with UTF-8
with open('theory/Geometry.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_geo, f)

print('✅ Fixed Geometry.ipynb')

# Do same for MLE
mle_content = """# Maximum Likelihood Estimation for Linear Regression

The Gaussian PDF leads to Mean Squared Error (MSE) as the loss function in regression models via maximum likelihood estimation (MLE), where we assume additive Gaussian noise. This derivation shows why minimizing squared errors is statistically optimal under that noise model step by step.

## Assumptions

Assuming Gaussian noise in the model $y = f(x;\\theta) + \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$, we derive the likelihood function of the observed data under this noise model.

## Model Assumption

Consider data points $(x_i, y_i)$ for $i = 1, \\ldots, m$, modeled as
$$y_i = \\theta^\\top x_i + \\epsilon_i,$$
where the noise terms are independent and identically distributed as
$$\\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2).$$

The conditional probability density function of $y_i$ given $x_i$ is
$$p(y_i \\mid x_i; \\theta)
= \\frac{1}{\\sqrt{2\\pi\\sigma^2}}
\\exp\\!\\left(
-\\frac{(y_i - \\theta^\\top x_i)^2}{2\\sigma^2}
\\right).$$

## Likelihood Function

For independent observations, the joint likelihood is given by the product
$$\\mathcal{L}(\\theta)
= \\prod_{i=1}^{m} p(y_i \\mid x_i; \\theta).$$

Substituting the Gaussian density, we obtain
$$\\mathcal{L}(\\theta)
= \\left(\\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\right)^m
\\exp\\!\\left(
-\\frac{1}{2\\sigma^2}
\\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2
\\right),$$
where $\\hat{y}_i = \\theta^\\top x_i$.

Maximum Likelihood Estimation (MLE) finds the parameter vector $\\theta$ that maximizes $\\mathcal{L}(\\theta)$, the probability of observing the data given the parameters.

## Log-Likelihood Simplification

Products are difficult to maximize directly, so we take the natural logarithm, which is a monotonically increasing function and therefore preserves the maximizer. The log-likelihood is
$$\\ell(\\theta)
= \\log \\mathcal{L}(\\theta)
= -\\frac{m}{2}\\log(2\\pi\\sigma^2)
- \\frac{1}{2\\sigma^2}
\\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2.$$

The first term is constant with respect to $\\theta$, and $\\sigma^2$ is typically fixed or estimated separately. Therefore, maximizing the log-likelihood $\\ell(\\theta)$ is equivalent to minimizing
$$\\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2,$$
which is the sum of squared errors.

## Negative Log-Likelihood as Loss

In machine learning, optimization is typically framed as loss minimization. The negative log-likelihood (NLL) is given by
$$-\\ell(\\theta)
= \\text{constant}
+ \\frac{1}{2\\sigma^2}
\\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2.$$

Ignoring constants and positive scaling factors (since $\\frac{1}{2\\sigma^2} > 0$), minimizing the negative log-likelihood is equivalent to minimizing
$$\\frac{1}{m}
\\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2,$$
which is the Mean Squared Error (MSE).

## Conclusion

Under the assumption of independent and identically distributed Gaussian noise, maximizing the likelihood of the observed data leads to minimizing the Mean Squared Error. This establishes Mean Squared Error as the statistically optimal loss function for linear regression under the Gaussian noise model."""

nb_mle = nbf.v4.new_notebook()
nb_mle.cells.append(nbf.v4.new_markdown_cell(mle_content))

with open('theory/MLE.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_mle, f)

print('✅ Fixed MLE.ipynb')