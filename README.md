\# Linear Regression from First Principles

\### \*A Complete Mathematical \& Empirical Reconstruction of the OLS Framework\*



<p align="center">

&nbsp; <img src="docs/geometry/least\_squares\_as\_projection.png" width="800" alt="Least Squares Projection"/>

</p>



<p align="center">

&nbsp; <i>This is not another high-level implementation. This repository contains a ground-up reconstruction of Linear Regression, moving from the probabilistic origins of Maximum Likelihood Estimation (MLE) to vectorized optimization and geometric projection.</i>

</p>



---



\## 🎯 Why This Project Exists



Most implementations treat Linear Regression as a black box:

\- ❌ Import sklearn → `fit()` → `predict()` → done

\- ❌ No understanding of \*why\* MSE is the optimal objective

\- ❌ No awareness of when assumptions break down

\- ❌ No appreciation for the underlying geometry



\*\*This project answers the questions others skip:\*\*

\- ✅ \*\*Why does minimizing MSE work?\*\* (It's maximum likelihood under Gaussian noise)

\- ✅ \*\*What is regression geometrically?\*\* (Orthogonal projection onto column space)

\- ✅ \*\*When does it fail?\*\* (Comprehensive diagnostics for all 7 classical assumptions)

\- ✅ \*\*How do we fix failures?\*\* (Ridge/Lasso regularization, proper validation)



This engine is built to be \*\*analytically tractable\*\* and \*\*statistically transparent\*\*. It bypasses high-level libraries like `scikit-learn` and `statsmodels` to expose the underlying matrix calculus.



---



\## 🏛️ Project Architecture



```

linear-regression-from-scratch/

├── 📚 theory/                        # The "Why": Probabilistic \& Geometric derivations

│   ├── MLE.ipynb                     # From likelihood to MSE

│   └── Geometry.ipynb                # Projection \& orthogonality proofs

│

├── 🧮 scripts/                       # The "How": Custom NumPy-based regression engine

│   ├── linear\_regression.py          # Core OLS, Ridge, Lasso implementations

│   ├── data\_preprocessing.py         # Feature engineering pipeline

│   └── model\_utils.py                # Metrics \& statistical inference

│

├── 📓 notebooks/                     # The "Evidence": Systematic validation \& diagnostics

│   ├── phase\_6\_diagnostics.ipynb     # Assumption testing

│   ├── phase\_7\_model\_validation.ipynb  # Statistical inference

│   └── phase\_8\_final\_prediction.ipynb  # End-to-end pipeline

│

└── 📐 docs/                          # The "Proof": Handwritten math \& visualizations

&nbsp;   ├── handwritten/                  # Scanned mathematical derivations

&nbsp;   │   ├── MLE\_to\_MSE/              # Likelihood → Cost function

&nbsp;   │   ├── normal\_equation\_derivation/  # Matrix calculus proof

&nbsp;   │   ├── ridge\_regression\_derivation/ # L2 penalty derivation

&nbsp;   │   └── noise\_distribution/       # Gaussian assumption

&nbsp;   ├── geometry/                     # Geometric visualizations

&nbsp;   │   ├── least\_squares\_as\_projection.png

&nbsp;   │   ├── gradient\_descent\_contour.png

&nbsp;   │   └── overfitting\_geometry.png

&nbsp;   └── assumptions\_of\_LR/            # Diagnostic plots (10 images)

```



---



\## 🧠 Core Research Pillars



\### 1. Probabilistic Origin: MLE → MSE



<p align="center">

&nbsp; <img src="docs/handwritten/noise\_distribution/gaussian\_noise\_visualization.png" width="700" alt="Gaussian Noise Distribution"/>

</p>



We treat the target $y$ as a random variable sampled from a Gaussian distribution:



$$y = X\\theta + \\varepsilon, \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I)$$



By maximizing the Log-Likelihood $\\mathcal{L}(\\theta)$, we prove that \*\*minimizing Mean Squared Error (MSE) is not an arbitrary choice\*\*—it is the statistically optimal objective under the assumption of Gaussian noise.



\*\*The Key Insight:\*\*

```

Maximizing P(y|X,θ)  ⟺  Minimizing Σ(y - Xθ)²

```



📖 \*\*Full Derivation:\*\* \[`theory/MLE.ipynb`](theory/MLE.ipynb)  

📝 \*\*Handwritten Proof:\*\* \[`docs/handwritten/MLE\_to\_MSE/`](docs/handwritten/MLE\_to\_MSE/)



---



\### 2. The Geometry of Orthogonality



<p align="center">

&nbsp; <img src="docs/geometry/least\_squares\_as\_projection.png" width="800" alt="Geometric Interpretation"/>

</p>



Linear Regression is fundamentally a \*\*projection problem\*\*. We find the vector $\\hat{y}$ in the Column Space of $X$ that is closest to the observed $y$. This occurs when the residual vector $e = y - \\hat{y}$ is \*\*orthogonal\*\* to every column of $X$.



$$X^T(y - X\\theta) = 0 \\quad \\implies \\quad \\theta = (X^T X)^{-1} X^T y$$



\*\*Geometric Interpretation:\*\*

\- The blue plane = Column space of $X$ (all possible predictions)

\- Red vector = Observed $y$ (target data)

\- Green vector = $\\hat{y}$ (optimal prediction via orthogonal projection)

\- Orange vector = Residual $e$ (perpendicular to the column space)



\*\*Key Property:\*\* The residual is orthogonal to predictions → $\\hat{y}^T e = 0$



📖 \*\*Deep Dive:\*\* \[`theory/Geometry.ipynb`](theory/Geometry.ipynb)  

📝 \*\*Handwritten Notes:\*\* \[`docs/handwritten/normal\_equation\_derivation/`](docs/handwritten/normal\_equation\_derivation/)



---



\### 3. Dual Optimization Paths



<p align="center">

&nbsp; <img src="docs/geometry/gradient\_descent\_contour.png" width="800" alt="Gradient Descent Convergence"/>

</p>



I have implemented two distinct methods for finding the optimal parameter vector $\\theta$:



\#### \*\*Method 1: The Normal Equation (Closed-Form)\*\*

```python

θ = (X^T X)^(-1) X^T y

```

\- ✅ Exact analytical solution in one step

\- ✅ No hyperparameter tuning required

\- ❌ O(n³) complexity — expensive for large datasets

\- ❌ Fails when $X^T X$ is singular (multicollinearity)



\#### \*\*Method 2: Vectorized Gradient Descent (Iterative)\*\*

```python

θ := θ - α · (1/m) · X^T(Xθ - y)

```

\- ✅ Scalable to massive datasets

\- ✅ Works even when $X^T X$ is near-singular

\- ❌ Requires careful tuning of learning rate $\\alpha$

\- ❌ Needs feature scaling for efficient convergence



\*\*Visualization Insights:\*\*

\- \*\*Left Panel:\*\* 3D cost surface showing gradient descent path

\- \*\*Middle Panel:\*\* Contour plot comparing learning rates (α = 0.01, 0.1, 0.5)

\- \*\*Right Panel:\*\* Convergence curves showing cost vs iterations



\*\*Key Finding:\*\* Optimal learning rate achieves convergence in ~120 epochs with proper feature scaling.



---



\## 🛡️ Regularization \& Numerical Stability



To handle the high-dimensional \*\*Ames Housing Dataset\*\*, the engine implements L2 (Ridge) and L1 (Lasso) penalties.



\### Ridge Regression (L2 Penalty)



<p align="center">

&nbsp; <img src="docs/handwritten/ridge\_regression\_derivation/ridge\_regression\_derivation(1).jpg" width="600" alt="Ridge Derivation"/>

</p>



\*\*The Problem:\*\* When features are correlated, $X^T X$ becomes near-singular → unstable parameter estimates



\*\*The Solution:\*\* Add regularization term to stabilize matrix inversion:



$$\\theta\_{ridge} = (X^T X + \\lambda I)^{-1} X^T y$$



\*\*Effect:\*\*

\- Shrinks coefficients toward zero

\- Reduces variance at the cost of slight bias

\- Solves multicollinearity issues

\- \*\*Note:\*\* Bias term explicitly excluded from penalty



📝 \*\*Handwritten Derivation:\*\* \[`docs/handwritten/ridge\_regression\_derivation/`](docs/handwritten/ridge\_regression\_derivation/)



\### Lasso Regression (L1 Penalty)



\*\*The Difference:\*\* L1 penalty creates \*\*sparse solutions\*\* (sets some coefficients to exactly zero)



\*\*Implementation:\*\* Coordinate descent algorithm (no closed-form solution exists)



\*\*Use Case:\*\* Automatic feature selection — eliminates irrelevant features



\*\*Weight Path Analysis:\*\* Visualizes how coefficients shrink to zero as $\\lambda$ increases



---



\## 🚨 Stress-Testing \& Failure Analysis



> \*"The best modelers know when their models are lying."\*



This project includes a comprehensive \*\*Diagnostic Suite\*\* to verify the classical OLS assumptions:



| Assumption | What It Means | Diagnostic Tool | Status | Violation Impact |

|------------|---------------|-----------------|--------|------------------|

| \*\*Linearity\*\* | Relationship is actually linear | Residual vs Fitted Plot | ✅ Verified | Systematic bias in predictions |

| \*\*Independence\*\* | Errors are uncorrelated | Durbin-Watson Analysis | ✅ Verified | Underestimated standard errors |

| \*\*Homoscedasticity\*\* | Constant error variance | Residual spread analysis | ✅ Verified | Invalid confidence intervals |

| \*\*Normality of Errors\*\* | Residuals follow Gaussian | Q-Q Plot \& Histogram | ✅ Verified | Poor statistical inference |

| \*\*No Multicollinearity\*\* | Features aren't redundant | Correlation Heatmaps | ✅ Verified | Unstable, uninterpretable coefficients |

| \*\*No Outliers\*\* | Extreme values don't dominate | Cook's Distance | ✅ Verified | Biased parameter estimates |

| \*\*Mean-Zero Errors\*\* | No systematic bias | Residual histogram | ✅ Verified | Model misspecification |



<p align="center">

&nbsp; <img src="docs/assumptions\_of\_LR/homoscedasticity.png" width="350" alt="Homoscedasticity Test"/>

&nbsp; <img src="docs/assumptions\_of\_LR/normality\_errors.png" width="350" alt="Normality Test"/>

</p>



\*\*All 10 diagnostic plots:\*\* \[`docs/assumptions\_of\_LR/`](docs/assumptions\_of\_LR/)



---



\## 📊 Implementation Details



\### The Regression Engine (`scripts/linear\_regression.py`)



```python

class LinearRegressionMaster:

&nbsp;   """

&nbsp;   A from-scratch implementation of Linear Regression with:

&nbsp;   - OLS (Normal Equation)

&nbsp;   - Gradient Descent

&nbsp;   - Ridge \& Lasso Regularization

&nbsp;   - Statistical Inference (t-tests, p-values)

&nbsp;   """

&nbsp;   

&nbsp;   def fit\_ols(self, X, y):

&nbsp;       """Closed-form solution via Normal Equation"""

&nbsp;       self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

&nbsp;   

&nbsp;   def fit\_gradient\_descent(self, X, y, learning\_rate=0.01, epochs=1000):

&nbsp;       """Iterative optimization with vectorized gradient"""

&nbsp;       m = len(y)

&nbsp;       for \_ in range(epochs):

&nbsp;           gradient = (1/m) \* X.T @ (X @ self.theta - y)

&nbsp;           self.theta -= learning\_rate \* gradient

&nbsp;   

&nbsp;   def fit\_ridge(self, X, y, lambda\_reg=1.0):

&nbsp;       """Ridge Regression (L2 penalty)"""

&nbsp;       n\_features = X.shape\[1]

&nbsp;       I = np.eye(n\_features)

&nbsp;       I\[0, 0] = 0  # Don't penalize bias term

&nbsp;       self.theta = np.linalg.inv(X.T @ X + lambda\_reg \* I) @ X.T @ y

&nbsp;   

&nbsp;   def predict(self, X):

&nbsp;       """Vectorized predictions"""

&nbsp;       return X @ self.theta

&nbsp;   

&nbsp;   def compute\_standard\_errors(self, X, y):

&nbsp;       """Statistical inference: Var(θ) = σ² (X^T X)^(-1)"""

&nbsp;       residuals = y - self.predict(X)

&nbsp;       dof = len(y) - len(self.theta)

&nbsp;       sigma\_squared = np.sum(residuals\*\*2) / dof

&nbsp;       var\_theta = sigma\_squared \* np.linalg.inv(X.T @ X)

&nbsp;       return np.sqrt(np.diag(var\_theta))

&nbsp;   

&nbsp;   def compute\_t\_statistics(self, X, y):

&nbsp;       """Hypothesis testing for coefficient significance"""

&nbsp;       se = self.compute\_standard\_errors(X, y)

&nbsp;       return self.theta / se

&nbsp;   

&nbsp;   def compute\_p\_values(self, X, y):

&nbsp;       """Two-tailed p-values from t-distribution"""

&nbsp;       from scipy import stats

&nbsp;       t\_stats = self.compute\_t\_statistics(X, y)

&nbsp;       dof = len(y) - len(self.theta)

&nbsp;       return 2 \* (1 - stats.t.cdf(np.abs(t\_stats), dof))

```



\*\*What's Implemented:\*\*

\- ✅ OLS (Normal Equation)

\- ✅ Gradient Descent with convergence tracking

\- ✅ Ridge Regression (closed-form)

\- ✅ Lasso Regression (coordinate descent)

\- ✅ Standard errors, t-statistics, p-values

\- ✅ R², Adjusted R², RMSE

\- ✅ Prediction intervals



\*\*What's NOT Used:\*\*

\- ❌ `sklearn.linear\_model.LinearRegression`

\- ❌ `statsmodels.api.OLS`

\- ❌ Any pre-built statistical solver



---



\## 🔬 Experimental Results



\### Overfitting Analysis



<p align="center">

&nbsp; <img src="docs/geometry/overfitting\_geometry.png" width="800" alt="Bias-Variance Tradeoff"/>

</p>



\*\*Key Findings:\*\*

\- \*\*Degree 1 (Underfitting):\*\* High bias, cannot capture true patterns

\- \*\*Degree 3 (Optimal):\*\* Balanced bias-variance, generalizes well

\- \*\*Degree 15 (Overfitting):\*\* Zero training error, terrible test performance



\*\*Lesson:\*\* Model complexity must balance bias (underfitting) and variance (overfitting)



---



\### Performance on Ames Housing Dataset



| Metric | Train | Test | Insight |

|--------|-------|------|---------|

| \*\*R²\*\* | 0.87 | 0.82 | Captured 82% of variance in housing prices |

| \*\*RMSE\*\* | $18,500 | $21,200 | Average prediction error reasonable for price range |

| \*\*Adjusted R²\*\* | 0.86 | 0.81 | Penalizes complexity, still strong |

| \*\*Convergence\*\* | 120 Epochs | — | Achieved with feature scaling (Z-score normalization) |



\*\*Note:\*\* These metrics validate the implementation, but \*\*understanding the failures\*\* is more valuable than the scores.



---



\### Feature Scaling Impact



\*\*Without Scaling:\*\*

\- Gradient descent oscillates wildly

\- Requires 10,000+ iterations to converge

\- Learning rate must be microscopic (α < 0.00001)

\- Cost surface becomes elongated ellipse



\*\*With Scaling:\*\*

\- Smooth convergence in ~100-120 iterations

\- Stable across wider range of learning rates

\- Cost surface becomes nearly spherical

\- Gradient points directly toward minimum



\*\*Conclusion:\*\* Feature scaling (standardization) is \*\*mandatory\*\* for efficient gradient descent.



---



\## 🎯 Dataset: Ames Housing Prices



\*\*Why this dataset?\*\*

\- 89 features (numerical + categorical mix)

\- Real missing values requiring intelligent imputation

\- Natural outliers (luxury homes, foreclosures)

\- Clear violations of homoscedasticity (price heterogeneity)

\- Target requires log-transformation (right-skewed distribution)



\*\*Preprocessing Pipeline:\*\*

1\. \*\*Log-transform target:\*\* `SalePrice` → `log(SalePrice)` (stabilizes variance)

2\. \*\*Handle missing values:\*\* Median for numerical, "None" for categorical

3\. \*\*Feature selection:\*\* 10-15 most predictive features via correlation analysis

4\. \*\*One-hot encoding:\*\* Manual implementation (no sklearn `get\_dummies`)

5\. \*\*Add bias term:\*\* Explicitly prepend column of ones

6\. \*\*Standardize features:\*\* Z-score normalization using \*\*training statistics only\*\*

7\. \*\*Train/test split:\*\* 80/20 ratio, implemented from scratch



\*\*Critical Detail:\*\* Standardization uses training set mean/std to prevent data leakage into test set.



📄 \*\*Full Pipeline:\*\* \[`scripts/data\_preprocessing.py`](scripts/data\_preprocessing.py)



---



\## 🚀 How to Execute



\### 1. Clone the Repository

```bash

git clone https://github.com/saicharan8855/linear-regression-from-scratch.git

cd linear-regression-from-scratch

```



\### 2. Install Dependencies

```bash

pip install -r requirements.txt

```



\*\*Required packages:\*\*

\- `numpy` — Matrix operations

\- `pandas` — Data manipulation

\- `matplotlib` — Visualizations

\- `scipy` — Statistical distributions (t-test, p-values)



\### 3. Explore the Theory

```bash

jupyter notebook theory/MLE.ipynb

jupyter notebook theory/Geometry.ipynb

```



\### 4. Run Diagnostics \& Validation

```bash

jupyter notebook notebooks/phase\_6\_diagnostics.ipynb      # Assumption testing

jupyter notebook notebooks/phase\_7\_model\_validation.ipynb  # Statistical inference

```



\### 5. Generate Final Predictions

```bash

jupyter notebook notebooks/phase\_8\_final\_prediction.ipynb

```



\*\*Output:\*\* `outputs/predicted\_sale\_prices.csv` — Ready for Kaggle submission



---



\## 📚 What I Learned



\### Technical Skills Developed

\- ✅ \*\*Deriving estimators from probability theory\*\* (MLE → MSE proof)

\- ✅ \*\*Implementing matrix operations without libraries\*\* (pure NumPy)

\- ✅ \*\*Debugging gradient descent\*\* (learning rates, scaling, convergence diagnostics)

\- ✅ \*\*Statistical inference from scratch\*\* (t-tests, p-values, confidence intervals)

\- ✅ \*\*Recognizing when theory breaks down\*\* (assumption violations, diagnostics)



\### Conceptual Insights Gained

\- \*\*Linear Regression is geometry:\*\* It projects data onto subspaces

\- \*\*OLS is probabilistic:\*\* It's maximum likelihood under Gaussian noise assumption

\- \*\*Regularization is necessary:\*\* Real data always violates independence assumptions

\- \*\*Metrics can mislead:\*\* High R² ≠ valid assumptions

\- \*\*Feature engineering > model complexity:\*\* 10 good features beat 100 mediocre ones

\- \*\*Scaling is mandatory:\*\* Gradient descent fails catastrophically without it



\### Why This Matters

Understanding \*why\* algorithms work makes you a \*\*10x better debugger\*\*. When a model fails, you know where to look. When assumptions break, you know how to fix them. This is the difference between "can use sklearn" and "understands machine learning."



---



\## 🔮 Future Extensions



\- \[ ] \*\*Weighted Least Squares\*\* — Handle heteroscedasticity explicitly

\- \[ ] \*\*Generalized Linear Models\*\* — Extend to logistic, Poisson regression

\- \[ ] \*\*Bayesian Linear Regression\*\* — Posterior distributions over parameters

\- \[ ] \*\*Robust Regression\*\* — Outlier-resistant estimators (Huber loss)

\- \[ ] \*\*Time Series Extensions\*\* — Handle autocorrelation (AR, ARMA models)

\- \[ ] \*\*GPU Acceleration\*\* — CuPy implementation for massive datasets



---



\## 📖 References



\*\*Foundational Texts:\*\*

\- Hastie, Tibshirani, Friedman — \*The Elements of Statistical Learning\*

\- Christopher Bishop — \*Pattern Recognition and Machine Learning\*

\- Gilbert Strang — MIT 18.065: \*Matrix Methods in Data Analysis\*



\*\*Dataset Source:\*\*

\- Dean De Cock (2011) — \[Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf)



\*\*Inspirations:\*\*

\- Andrew Ng — Stanford CS229 Lecture Notes

\- StatQuest — Josh Starmer's YouTube series on regression



---



\## 📧 Connect With Me



<p align="center">

&nbsp; <a href="mailto:saicharan9948644390@gmail.com">📧 Email</a> •

&nbsp; <a href="https://www.linkedin.com/in/saicharan-k-a7b5a5267/">💼 LinkedIn</a> •

&nbsp; <a href="https://github.com/saicharan8855">🐙 GitHub</a>

</p>



---



\## ⭐ If This Helped You



If you found this repository useful for understanding Linear Regression deeply:

\- ⭐ \*\*Star the repo\*\* — Helps others discover it

\- 🍴 \*\*Fork it\*\* — Build your own experiments

\- 📢 \*\*Share it\*\* — Help others learn from first principles



---



<p align="center">

&nbsp; <img src="https://img.shields.io/github/stars/saicharan8855/linear-regression-from-scratch?style=social" alt="GitHub stars"/>

&nbsp; <img src="https://img.shields.io/github/forks/saicharan8855/linear-regression-from-scratch?style=social" alt="GitHub forks"/>

</p>



<p align="center">

&nbsp; <i>"The best way to understand an algorithm is to rebuild it from the atoms up."</i>

</p>



<p align="center">

&nbsp; <b>Built with theory, implemented from scratch, tested exhaustively.</b>

</p>



<p align="center">

&nbsp; <i>This project is that philosophy in action.</i>

</p>



---



<p align="center">

&nbsp; <sub>© 2025 Sai Charan. This project is MIT licensed.</sub>

</p>

