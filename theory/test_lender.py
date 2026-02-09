import json
import nbformat as nbf

def create_geometry_notebook():
    nb = nbf.v4.new_notebook()
    
    text = """# Geometric Interpretation of Linear Regression

This section presents a geometric interpretation of linear regression, showing that the optimal prediction vector is obtained by projecting the target vector $y$ onto the column space of the design matrix $X$.

## Column Space

The column space of $X$, denoted as $\\text{Col}(X)$, is the subspace of $\\mathbb{R}^m$ spanned by the columns of $X$.

For any parameter vector $\\theta$, the predicted response

$$\\hat{y} = X\\theta$$

lies in the column space of $X$.

## Projection

The observed target vector $y$ does not lie in $\\text{Col}(X)$. Linear regression seeks $\\hat{y} \\in \\text{Col}(X)$ closest to $y$:

$$\\hat{y} = \\text{proj}_{\\text{Col}(X)}(y)$$

## Orthogonality

The residual $r = y - \\hat{y}$ is orthogonal to $\\text{Col}(X)$:

$$X^\\top r = 0$$

## Normal Equation

From $X^\\top (y - X\\theta) = 0$, we get:

$$X^\\top X\\theta = X^\\top y$$"""
    
    nb.cells.append(nbf.v4.new_markdown_cell(text))
    
    with open('theory/Geometry.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created Geometry.ipynb")

def create_mle_notebook():
    nb = nbf.v4.new_notebook()
    
    text = """# Maximum Likelihood Estimation for Linear Regression

## Model

$$y_i = \\theta^\\top x_i + \\epsilon_i, \\quad \\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$$

## Likelihood

$$p(y_i \\mid x_i; \\theta) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(y_i - \\theta^\\top x_i)^2}{2\\sigma^2}\\right)$$

## Log-Likelihood

$$\\ell(\\theta) = -\\frac{m}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2} \\sum_{i=1}^{m} (y_i - \\theta^\\top x_i)^2$$

## Maximizing Likelihood

Maximizing $\\ell(\\theta)$ is equivalent to minimizing:

$$\\text{MSE} = \\frac{1}{m} \\sum_{i=1}^{m} (y_i - \\theta^\\top x_i)^2$$"""
    
    nb.cells.append(nbf.v4.new_markdown_cell(text))
    
    with open('theory/MLE.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created MLE.ipynb")

if __name__ == '__main__':
    create_geometry_notebook()
    create_mle_notebook()