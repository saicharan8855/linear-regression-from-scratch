This section presents a geometric interpretation of linear regression, showing that the optimal prediction vector is obtained by projecting the target vector $y$ onto the column space of the design matrix $X$. This viewpoint explains the normal equation as a consequence of orthogonality between the residual vector and the feature space.

## Vector Space Setup
## Column Space of the Design Matrix

The column space of the design matrix $X$, denoted as $\text{Col}(X)$, is the subspace of $\mathbb{R}^m$ spanned by the columns of $X$. Any vector in $\text{Col}(X)$ can be written as a linear combination of the columns of $X$.

For any parameter vector $\theta$, the predicted response  
$$
\hat{y} = X\theta
$$  
lies in the column space of $X$. Therefore, linear regression restricts predictions to vectors that belong to $\text{Col}(X)$.

## Projection of the Target Vector

In general, the observed target vector $y$ does not lie in the column space of $X$. Linear regression therefore seeks a vector $\hat{y} \in \text{Col}(X)$ that is closest to $y$ in the Euclidean sense.

This closest vector is obtained by orthogonally projecting $y$ onto the column space of $X$. The resulting vector $\hat{y} = X\theta$ represents the best linear approximation to $y$ within $\text{Col}(X)$.

## Orthogonality of the Residual

Let the residual vector be defined as  
$$
r = y - \hat{y} = y - X\theta.
$$  
A fundamental property of orthogonal projection is that the residual vector is orthogonal to the subspace onto which the projection is made. Therefore, the residual $r$ is orthogonal to the column space of $X$:  
$$
X^\top r = 0.
$$

## Derivation of the Normal Equation

From the orthogonality condition  
$$
X^\top r = 0,
$$  
and using the definition of the residual $r = y - X\theta$, we obtain  
$$
X^\top (y - X\theta) = 0.
$$  
Expanding the expression gives  
$$
X^\top y - X^\top X\theta = 0.
$$  
Rearranging terms yields  
$$
X^\top X\theta = X^\top y.
$$  
If $X^\top X$ is invertible, the solution for $\theta$ is  
$$
\theta = (X^\top X)^{-1} X^\top y.
$$

## Geometric Interpretation

Geometrically, linear regression finds the vector $\hat{y} = X\theta$ in the column space of $X$ that is closest to the target vector $y$. The normal equation ensures that the residual vector $y - \hat{y}$ is orthogonal to every column of $X$, meaning no further movement within the column space can reduce the distance to $y$.

Thus, the normal equation represents the condition for an orthogonal projection of $y$ onto $\text{Col}(X)$.

## Conclusion

From a geometric perspective, linear regression is the problem of projecting the target vector $y$ onto the column space of the design matrix $X$. The normal equation arises naturally from the orthogonality condition of this projection, providing a closed‑form solution for the optimal parameters.



