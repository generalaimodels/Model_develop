

| Name | Equation | Explanation |
|------|----------|-------------|
| Maximal Information Coefficient (MIC) | $\text{MIC}(X,Y) = \max_{xy < B(n)} \frac{I^*(X,Y)}{log_2(\min(x,y))}$ | Measures the strength of the linear or non-linear association between two variables. |
| Distance Correlation | $\text{dCor}(X,Y) = \frac{\text{dCov}(X,Y)}{\sqrt{\text{dVar}(X) \cdot \text{dVar}(Y)}}$ | Measures the dependence between two random vectors, including non-linear and non-monotonic relationships. |
| Hilbert-Schmidt Independence Criterion (HSIC) | $\text{HSIC}(X,Y) = \frac{1}{n^2} \text{tr}(KHLH)$ | Measures the dependence between two random variables using kernel methods. |
| Brownian Covariance | $\text{BCov}(X,Y) = \frac{1}{2} \mathbb{E}[\|X-X'\| \|Y-Y'\|] + \frac{1}{2} \mathbb{E}[\|X-X'\|] \mathbb{E}[\|Y-Y'\|] - \mathbb{E}[\|X-X'\| \|Y-Y''\|]$ | Measures the covariance between two random variables using Brownian motion. |
| Copula-based Dependence Measures | $\theta_C = \int_{[0,1]^2} C(u,v) dC(u,v) - 1$ | Measures the dependence between two random variables using copula functions. |
| Randomized Dependence Coefficient (RDC) | $\text{RDC}(X,Y) = \sup_{f,g} \text{Cov}(f(X), g(Y))$ | Measures the dependence between two random variables using random non-linear projections. |
| Canonical Correlation Analysis (CCA) | $\rho_c = \max_{a,b} \text{Corr}(a^TX, b^TY)$ | Finds the linear combinations of two sets of variables that have maximum correlation with each other. |
| Kernel Canonical Correlation Analysis (KCCA) | $\rho_{kc} = \max_{\alpha,\beta} \frac{\alpha^T K_x K_y \beta}{\sqrt{(\alpha^T K_x^2 \alpha)(\beta^T K_y^2 \beta)}}$ | Finds the non-linear combinations of two sets of variables that have maximum correlation with each other. |
| Hirschfeld-Gebelein-Rényi (HGR) Maximal Correlation | $\text{HGR}(X,Y) = \sup_{f,g} \text{Corr}(f(X), g(Y))$ | Measures the maximal correlation between two random variables over all possible non-linear transformations. |
| Alternating Conditional Expectations (ACE) | $\text{ACE}(X,Y) = \max_{f,g} \text{Corr}(f(X), g(Y))$ | Finds the optimal non-linear transformations of two variables that maximize their correlation. |
| Mutual Information Dimension | $\text{MID}(X,Y) = \lim_{\varepsilon \to 0} \frac{I(X;Y)}{\log(1/\varepsilon)}$ | Measures the dimensionality of the relationship between two variables using mutual information. |
| Rényi's Maximal Correlation | $\rho_{\infty}(X,Y) = \sup_{f,g} \|f(X) - g(Y)\|_{\infty}$ | Measures the maximal correlation between two random variables over all possible measurable functions. |
| Kernel Mean Embedding | $\mu_X = \mathbb{E}[\phi(X)]$ | Embeds probability distributions into a reproducing kernel Hilbert space (RKHS). |
| Maximum Mean Discrepancy (MMD) | $\text{MMD}(P,Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}}$ | Measures the distance between two probability distributions using kernel mean embeddings. |
| Hilbert-Schmidt Norm | $\|A\|_{HS} = \sqrt{\sum_{i,j} |a_{ij}|^2}$ | Measures the size of a matrix or operator in a Hilbert space. |
| Wasserstein Distance | $W_p(P,Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y)^p d\gamma(x,y)\right)^{1/p}$ | Measures the distance between two probability distributions using optimal transport. |
| Gromov-Wasserstein Distance | $\text{GW}(X,Y) = \inf_{\mu \in \mathcal{M}(X \times Y)} \int_{X \times Y} |d_X(x,x') - d_Y(y,y')| d\mu(x,y) d\mu(x',y')$ | Measures the distance between two metric spaces using optimal transport. |
| Kernel Alignment | $A(K_1, K_2) = \frac{\langle K_1, K_2 \rangle_F}{\|K_1\|_F \|K_2\|_F}$ | Measures the similarity between two kernel matrices. |
| Centered Kernel Alignment (CKA) | $\text{CKA}(K_1, K_2) = \frac{\langle K_1^c, K_2^c \rangle_F}{\|K_1^c\|_F \|K_2^c\|_F}$ | Measures the similarity between two centered kernel matrices. |
| Kernel Target Alignment | $\text{KTA}(K, y) = \frac{\langle K, yy^T \rangle_F}{\|K\|_F \|yy^T\|_F}$ | Measures the similarity between a kernel matrix and a target matrix. |
| Hilbert-Schmidt Independence Criterion Lasso (HSIC Lasso) | $\min_{\alpha} \frac{1}{2} \|\alpha\|_2^2 + \lambda \|\alpha\|_1 \quad \text{s.t.} \quad \alpha^T K \alpha = 1$ | Performs feature selection using HSIC as a measure of dependence. |
| Kernel Partial Least Squares (KPLS) | $\max_{w,c} \text{Cov}(Kw, Yc) \quad \text{s.t.} \quad \|w\|^2 = \|c\|^2 = 1$ | Finds the directions in the kernel feature space that maximize the covariance with the response variable. |
| Kernel Dimension Reduction | $\min_{W} \text{tr}(W^T K W) \quad \text{s.t.} \quad W^T W = I$ | Finds a low-dimensional embedding of the data that preserves the kernel structure. |
| Kernel Dependence Measures | $\text{KDM}(X,Y) = \frac{\text{HSIC}(X,Y)}{\sqrt{\text{HSIC}(X,X) \cdot \text{HSIC}(Y,Y)}}$ | Measures the dependence between two random variables using kernel-based methods. |

These advanced metrics cover a wide range of techniques for measuring the relationship between two entities, including kernel methods, optimal transport, and information-theoretic approaches. They can capture non





| Name                             | Mathematical Equation                                                 | Explanation                                                                                                                                         |
|----------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Covariance                       | cov(X,Y) = Σ [(xi - μx)(yi - μy)] / (n - 1)                          | Measures the joint variability of two random variables. If the greater values of one variable mainly correspond with the greater values of the other, and likewise for the lesser values, the covariance is positive. |
| Pearson Correlation Coefficient  | ρxy = cov(X,Y) / (σx * σy)                                           | Measures the linear correlation between two variables X and Y, giving a value between +1 and −1 inclusive.                                           |
| Spearman's Rank Correlation      | ρ = 1 - (6 Σ di^2) / (n(n^2 - 1))                                    | Non-parametric measure of rank correlation (statistical dependence between the rankings of two variables).                                           |
| Partial Correlation              | rXY.Z = (rXY - rXZrYZ) / sqrt[(1 - rXZ^2)(1 - rYZ^2)]                | Measures the degree of association between two random variables, after removing the effect of one or more other variables.                            |
| Kendall Tau Coefficient          | τ = (concordant pairs - discordant pairs) / (n(n-1)/2)                | A measure of the correspondence between two rankings and identifying the strength of association between them.                                       |
| Mutual Information               | I(X;Y) = Σ Σ p(x,y) log(p(x,y) / (p(x)p(y)))                         | A measure of the mutual dependence between the two variables.                                                                                        |
| Mahalanobis Distance             | D^2 = (x - μ)'S^(-1)(x - μ)                                          | A multivariate measure of the distance between a point and a distribution.                                                                           |
| Canonical Correlation            | ρ = max corr(a'X, b'Y)                                               | Measures the linear relationship between two multivariate datasets.                                                                                  |
| Singular Value Decomposition (SVD)| X = UΣV^*                                                          | Factorizes a matrix into three matrices, often used to solve least squares problems, compute pseudoinverses, etc.                                    |
| Principle Component Analysis (PCA)| Y = PCAX                                                            | A statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables. |
| Cross-correlation                | (f ⋆ g)(τ) = ∫ f*(t) g(t+τ) dt                                       | A measure of similarity of two series as a function of the displacement of one relative to the other.                                               |
| Point Biserial Correlation       | rpb = (M1 - M2) sqrt(n1n2) / (nσ)                                    | Measures the strength and direction of the association that exists between one continuous-level variable and one binary variable.                    |
| Distance Correlation             | DCor(X,Y)                                                           | A measure of association between two random vectors that is zero if and only if they are independent.                                                |
| Granger Causality                | F-test on VAR model                                                  | A statistical hypothesis test for determining whether one time series can predict another.                                                           |
| Dynamic Time Warping (DTW)       | min Σ d(x(i), y(j))                                                  | An algorithm for measuring similarity between two temporal sequences which may vary in time or speed.                                                |
| Co-integration                   | β'Yt = ut, test for unit root in ut                                  | Used to test for a long-term relationship between two time series.                                                                                   |
| Factor Analysis                  | X = ΛF + ε                                                          | A way to investigate whether a number of variables of interest Y1, Y2, ..., Yk, are linearly related to a smaller number of unobservable factors F1, F2, ..., Fm. |
| Item Response Theory (IRT)       | P(θ) = c + (1 - c) / (1 + exp(-a(θ - b)))                           | A family of models used to explain the relationship between latent traits (abilities, attitudes) and their manifestations (correct/incorrect answers). |
| Structural Equation Modeling (SEM)| Various, depending on model specification                           | A multivariate statistical analysis technique that is used to analyze structural relationships between measured variables and latent constructs.       |
| Time-series Forecasting (ARIMA)  | ARIMA models various depending on order (p,d,q)                      | Used for forecasting future points in time series data using the autoregressive, integrated, moving average method.                                  |
| Multiple Correspondence Analysis (MCA)| Various, depending on data                                     | A data analysis technique for nominal categorical data, used to detect and represent underlying structures in a data set.                            |
| Hierarchical Linear Modeling (HLM)| Y = Xβ + Zγ + ε, where Y is outcome, X and Z are fixed and random effects | A statistical regression model that is used to analyze data with a hierarchical structure.                                                           |
| Quadratic Discriminant Analysis (QDA)| δk(x) = -0.5 log|Σk| - 0.5 (x-μk)'Σk^(-1)(x-μk) + log(πk)             | A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.                  |
| Hotelling's T-squared            | T² = n(X̄ - μ₀)'S⁻¹(X̄ - μ₀)                                         | A multivariate statistical test that is the multivariate analogue of the Student's t-test.                                                          |
| Multidimensional Scaling (MDS)   | Stress minimization or eigenvalue decomposition based on a distance matrix | A means of visualizing the level of similarity of individual cases of a dataset.                                                                     |

Each of these metrics has a specific use case and application depending on the nature of the data and the type of analysis being performed. Some of them are more suited for time-series data, some for multivariate analysis, some for categorical data analysis, etc.




| Name | Equation | Explanation |
|------|----------|-------------|
| Covariance | $\text{Cov}(X,Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n-1}$ | Measures the joint variability of two random variables. |
| Pearson Correlation Coefficient | $\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | Measures the linear correlation between two variables. |
| Spearman's Rank Correlation Coefficient | $r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$ | Measures the monotonic relationship between two variables using rank values. |
| Kendall's Tau | $\tau = \frac{2(C - D)}{n(n-1)}$ | Measures the ordinal association between two variables. |
| Singular Value Decomposition (SVD) | $A = U\Sigma V^T$ | Decomposes a matrix into three matrices to reveal the underlying structure. |
| Euclidean Distance | $d(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$ | Measures the straight-line distance between two points in Euclidean space. |
| Manhattan Distance | $d(x,y) = \sum_{i=1}^{n} |x_i - y_i|$ | Measures the distance between two points by summing the absolute differences of their coordinates. |
| Cosine Similarity | $\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$ | Measures the cosine of the angle between two non-zero vectors. |
| Jaccard Similarity | $J(A,B)$= $\frac{|A \cap B|}{|A \cup B|}$ | Measures the similarity between two sets by dividing the size of their intersection by the size of their union. |
| Dice Coefficient | $\text{Dice}(A,B) = \frac{2|A \cap B|}{|A| + |B|}$ | Measures the similarity between two sets by dividing twice the size of their intersection by the sum of their sizes. |
| Mutual Information | $I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$ | Measures the mutual dependence between two random variables. |
| Pointwise Mutual Information (PMI) | $\text{PMI}(x,y) = \log \frac{p(x,y)}{p(x)p(y)}$ | Measures the association between two events based on their joint probability and individual probabilities. |
| Normalized Pointwise Mutual Information (NPMI) | $\text{NPMI}(x,y) = \frac{\text{PMI}(x,y)}{-\log p(x,y)}$ | Normalizes PMI to have a value between -1 and 1. |
| Kullback-Leibler Divergence | $D_{KL}(P \| Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}$ | Measures the difference between two probability distributions. |
| Jensen-Shannon Divergence | $\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$ | Measures the similarity between two probability distributions, where $M = \frac{1}{2}(P + Q)$. |
| Chi-Square Statistic | $\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}$ | Measures the difference between observed and expected frequencies. |
| Cramér's V | $V = \sqrt{\frac{\chi^2 / n}{\min(k-1, r-1)}}$ | Measures the strength of association between two categorical variables. |
| Contingency Coefficient | $C = \sqrt{\frac{\chi^2}{\chi^2 + n}}$ | Measures the degree of association between two categorical variables. |
| Uncertainty Coefficient | $U(X|Y) = \frac{I(X;Y)}{H(X)}$ | Measures the proportion of uncertainty in one variable that is explained by another variable. |
| Odds Ratio | $\text{OR} = \frac{p_1 / (1 - p_1)}{p_2 / (1 - p_2)}$ | Measures the association between two binary variables. |
| Yule's Q | $Q = \frac{\text{OR} - 1}{\text{OR} + 1}$ | Measures the strength and direction of association between two binary variables. |
| Phi Coefficient | $\phi = \frac{\chi^2}{n}$ | Measures the degree of association between two binary variables. |
| Tschuprow's T | $T = \sqrt{\frac{\phi^2}{\sqrt{(k-1)(r-1)}}}$ | Measures the degree of association between two categorical variables. |
| Goodman and Kruskal's Lambda | $\lambda = \frac{\sum_{i=1}^{r} \max_j n_{ij} - \max_j n_{+j}}{n - \max_j n_{+j}}$ | Measures the proportional reduction in error when predicting one variable based on another. |
| Theil's U | $U = \sqrt{1 - e^{-2I(X;Y)}}$ | Measures the degree of association between two categorical variables. |


-----
## different statistical estimators 



| Estimator Name                   | Explanation and Importance                                                                 |
|----------------------------------|---------------------------------------------------------------------------------------------|
| 1. Maximum Likelihood Estimator (MLE) | Finds parameter values that maximize the likelihood of observing the given data; widely used because of its desirable properties like consistency and efficiency. |
| 2. Method of Moments Estimator   | Uses sample moments to estimate population parameters; important due to its simplicity and ease of computation. |
| 3. Least Squares Estimator (LSE) | Minimizes the sum of the squares of the residuals; fundamental in regression analysis for its simplicity and optimality properties under certain conditions. |
| 4. Weighted Least Squares Estimator | A generalization of LSE that allows for heteroskedasticity by assigning different weights to different observations. |
| 5. Ridge Estimator               | Addresses multicollinearity in regression by adding a degree of bias to the regression estimates, which can reduce variance. |
| 6. Lasso Estimator               | Similar to Ridge, but can set some coefficients to zero, effectively selecting variables; useful in high-dimensional datasets. |
| 7. Elastic Net Estimator         | Combines Lasso and Ridge penalties; useful for model selection and multicollinearity. |
| 8. Generalized Method of Moments (GMM) | Extends the method of moments to more complex models; widely applicable and consistent under weaker conditions. |
| 9. Instrumental Variables Estimator (IV) | Deals with endogeneity in regression models by using instruments; critical for causal inference in econometrics. |
| 10. Maximum a Posteriori Estimator (MAP) | Incorporates prior information into the estimation process; important when prior information is available and relevant. |
| 11. Quantile Estimator           | Estimates the quantiles of a distribution; useful for understanding the distributional properties beyond the mean. |
| 12. Kaplan-Meier Estimator       | Non-parametric statistic used to estimate the survival function; vital in survival analysis. |
| 13. Jackknife Estimator          | Reduces bias and variance by systematically recomputing the estimator leaving out one observation at a time; useful for variance estimation and bias reduction. |
| 14. Bootstrap Estimator          | Uses resampling with replacement to estimate the sampling distribution; important for assessing variability and constructing confidence intervals. |
| 15. Robust Estimators            | Provides estimates that are not unduly affected by outliers or non-normality; crucial for reliability in the presence of model deviations. |
| 16. Kernel Density Estimator     | Estimates the probability density function of a random variable; key for non-parametric density estimation. |
| 17. Expectation-Maximization (EM) Estimator | Iteratively improves estimates in models with latent variables; essential for dealing with incomplete data. |
| 18. Probit and Logit Estimators  | Used for estimating binary outcome models; important in fields like biostatistics and econometrics. |
| 19. Partial Least Squares (PLS) Estimator | Combines features from principal component analysis and regression; useful when predictors exceed observations. |
| 20. Principal Components Regression (PCR) Estimator | Uses principal component analysis for dimensionality reduction before regression; helps to mitigate the effects of multicollinearity. |
| 21. Matched Pair Estimator       | Used in experimental design to pair similar units to isolate treatment effects; fundamental for causal inference in controlled experiments. |
| 22. Propensity Score Estimator   | Used to adjust for confounding in observational studies by matching on the propensity score; key for causal inference when randomization is not possible. |
| 23. Two-Stage Least Squares (2SLS) Estimator | An extension of IV that addresses endogeneity in systems of equations; critical in simultaneous equation models. |
| 24. Dynamic Panel Data Estimator | Deals with unobserved heterogeneity and endogeneity in panel data; important in econometric analysis over time. |
| 25. Random Effects Estimator     | Assumes individual-specific effects are random and uncorrelated with independent variables; used in panel data analysis to control for unobserved heterogeneity. |

These estimators are key tools in statistical analysis and are used across various fields, including economics, finance, engineering, medicine, and social sciences, to make inferences about populations from samples, test hypotheses, and build predictive models.



## estimators 
| Estimator Name | Explanation (Importance) |
|----------------|---------------------------|
| Maximum Likelihood Estimator (MLE) | Finds parameter values that maximize the likelihood function, providing the most probable estimates given the observed data. Important for its desirable properties and wide applicability. |
| Method of Moments Estimator (MME) | Equates sample moments to population moments to estimate parameters. Useful for quick and simple estimation, especially when likelihood-based methods are challenging. |
| Bayesian Estimator | Incorporates prior knowledge and updates it with observed data to obtain posterior estimates. Valuable for integrating prior information and quantifying uncertainty. |
| Least Squares Estimator (LSE) | Minimizes the sum of squared residuals to estimate parameters in regression models. Fundamental for its simplicity and optimality under certain conditions. |
| Minimum Mean Squared Error Estimator (MMSE) | Minimizes the expected squared error between the estimate and the true parameter. Important for achieving the lowest possible mean squared error. |
| Maximum a Posteriori Estimator (MAP) | Bayesian estimator that maximizes the posterior probability density function. Useful for incorporating prior knowledge and obtaining point estimates. |
| Minimum Variance Unbiased Estimator (MVUE) | Unbiased estimator with the lowest variance among all unbiased estimators. Desirable for its efficiency and lack of bias. |
| James-Stein Estimator | Shrinkage estimator that combines individual estimates to improve overall performance. Valuable for high-dimensional settings and reducing estimation risk. |
| Empirical Bayes Estimator | Uses data to estimate prior parameters and combines them with observed data for estimation. Useful when prior information is limited or uncertain. |
| Stein's Unbiased Risk Estimator (SURE) | Provides an unbiased estimate of the mean squared error for a given estimator. Important for estimator selection and risk assessment. |
| Generalized Method of Moments Estimator (GMM) | Generalizes the method of moments to allow for more moment conditions than parameters. Flexible and widely used in econometrics and finance. |
| Adaptive Estimator | Automatically adapts to the underlying data distribution to achieve optimal performance. Valuable for robust estimation in various settings. |
| Penalized Likelihood Estimator (PLE) | Adds a penalty term to the likelihood function to encourage desirable properties like sparsity or smoothness. Useful for variable selection and regularization. |
| Minimax Estimator | Minimizes the maximum possible risk or loss over a class of estimators. Important for robust estimation in worst-case scenarios. |
| Jackknife Estimator | Resampling technique that estimates parameters by leaving out one observation at a time. Useful for bias reduction and variance estimation. |
| Bootstrap Estimator | Resampling technique that estimates parameters by resampling with replacement from the observed data. Valuable for quantifying uncertainty and constructing confidence intervals. |
| Kernel Density Estimator (KDE) | Non-parametric estimator that estimates the probability density function using kernel functions. Useful for exploratory data analysis and visualizing distributions. |
| Spline Estimator | Uses spline functions to estimate smooth curves or surfaces from observed data. Important for non-parametric regression and function approximation. |
| Wavelet Estimator | Estimates parameters using wavelet basis functions, capturing local features and multi-resolution properties. Useful for signal processing and image analysis. |
| Functional Data Estimator | Estimates parameters for data that are functions or curves rather than scalar values. Important for analyzing functional data in various fields. |
| Semiparametric Estimator | Combines parametric and non-parametric components to balance flexibility and efficiency. Useful when the data partially follows a parametric model. |
| Quantile Estimator | Estimates specific quantiles of a distribution, such as the median or percentiles. Valuable for robust estimation and understanding data distributions. |
| Spectral Estimator | Estimates the power spectral density of a time series or spatial process. Important for analyzing periodic patterns and frequency components. |
| Empirical Likelihood Estimator | Non-parametric likelihood-based method that incorporates constraints from the data. Useful for constructing confidence regions and testing hypotheses. |
| Generalized Estimating Equations (GEE) Estimator | Extends GLMs to correlated data by specifying only the mean and variance structures. Widely used in longitudinal and clustered data analysis. |