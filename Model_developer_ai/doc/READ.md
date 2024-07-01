# Deep Learning Techniques and Equations

This document provides a tabular representation of various deep learning techniques and their corresponding mathematical equations.

| **Name** | **Mathematical Equation** |
|---|---|
| **Convolutional Layer** | $$ y_{i,j,k} = \sum_{m,n,l} x_{i+m, j+n, l} \cdot w_{m,n,l,k} + b_k $$ |
| **Pooling Layer** | $$ y_{i,j,k} = \max_{m,n} (x_{i+m, j+n, k}) $$ (Max Pooling) |
| **Fully Connected Layer** | $$ y = W x + b $$ |
| **Batch Normalization** | $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta $$ |
| **Dropout** | $$ y = \frac{1}{p} x \cdot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(p) $$ |
| **ReLU (Rectified Linear Unit)** | $$ y = \max(0, x) $$ |
| **Leaky ReLU** | $$ y = \max(\alpha x, x) $$ |
| **Parametric ReLU (PReLU)** | $$ y = \max(\alpha_i x, x) $$ |
| **ELU (Exponential Linear Unit)** | $$ y = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases} $$ |
| **SELU (Scaled Exponential Linear Unit)** | $$ y = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases} $$ |
| **GELU (Gaussian Error Linear Unit)** | $$ y = x \cdot \Phi(x) $$, where $$ \Phi(x) $$ is the CDF of the standard normal distribution |
| **Swish** | $$ y = x \cdot \sigma(x) $$ |
| **Sigmoid** | $$ y = \frac{1}{1 + e^{-x}} $$ |
| **Tanh** | $$ y = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$ |
| **Residual Connections (Skip Connections)** | $$ y = x + f(x) $$ |
| **Depthwise Separable Convolutions** | $$ y_{i,j,k} = \sum_{m,n} x_{i+m,j+n,k} \cdot w_{m,n,k} + b_k $$ (Depthwise) and $$ y_{i,j,l} = \sum_{k} y_{i,j,k} \cdot w'_{k,l} + b'_l $$ (Pointwise) |
| **Pointwise Convolutions (1x1 Convolutions)** | $$ y_{i,j,k} = \sum_{l} x_{i,j,l} \cdot w_{l,k} + b_k $$ |
| **Dilated (Atrous) Convolutions** | $$ y_{i,j,k} = \sum_{m,n,l} x_{i+dm, j+dn, l} \cdot w_{m,n,l,k} + b_k $$ |
| **Transposed Convolutions (Deconvolutions)** | $$ y = x * W^T $$ |
| **Squeeze-and-Excitation Blocks** | $$ s = \sigma(W_2 \text{ReLU}(W_1 z)) $$, where $$ z = \text{GlobalAvgPool}(x) $$ and $$ y = s \cdot x $$ |
| **Self-Attention** | $$ y = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$ |
| **Channel Attention** | $$ y_c = \sigma(W_2 \text{ReLU}(W_1 \text{GlobalAvgPool}(x))) \cdot x $$ |
| **Spatial Attention** | $$ y = \sigma(\text{Conv2D}([\text{AvgPool}(x), \text{MaxPool}(x)])) \cdot x $$ |
| **Multi-Head Attention** | $$ y = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W^O $$, where $$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$ |
| **Scaled Dot-Product Attention** | $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$ |
| **Feature Pyramid Networks (FPN)** | $$ P_i = \text{Conv}(C_i) + \text{UpSample}(P_{i+1}) $$ |
| **Inception Modules** | Concatenation of multiple convolutions with different kernel sizes |
| **Residual Blocks** | $$ y = x + f(x) $$ where $$ f(x) $$ is a stack of convolutional layers |
| **Bottleneck Layers** | $$ y = \text{Conv1x1}(\text{Conv3x3}(\text{Conv1x1}(x))) + x $$ |
| **Dense Blocks** | $$ y = \text{Concat}(x_0, x_1, \ldots, x_{i-1}) $$ |
| **Upsampling** | $$ y = \text{UpSample}(x) $$ |
| **Subpixel Convolution** | $$ y = \text{PixelShuffle}(\text{Conv2D}(x)) $$ |
| **Layer Normalization** | $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta $$ |
| **Instance Normalization** | $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta $$ |
| **Group Normalization** | $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta $$ |
| **Stochastic Depth** | $$ y = \begin{cases} x + f(x), & \text{with probability } p \\ x, & \text{with probability } 1 - p \end{cases} $$ |
| **Spatial Dropout** | $$ y = x \cdot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(p) $$ applied spatially |
| **Grad-CAM** | $$ \text{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha_k A^k\right) $$, where $$ \alpha_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{i,j}} $$ |
| **Integrated Gradients** | $$ \text{IG}(x) = (x - x') \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x} d\alpha $$ |
| **Attention Maps** | Derived from attention weights in self-attention mechanisms |
| **Multi-Task Learning Layers** | Shared layers among multiple tasks with task-specific output layers |
| **Atrous Spatial Pyramid Pooling (ASPP)** | $$ \text{ASPP}(x) = \text{Concat}(\text{Conv}(x, r=1), \text{Conv}(x, r=6), \text{Conv}(x, r=12), \text{Conv}(x, r=18)) $$ |
| **ROI Pooling (Region of Interest Pooling)** | $$ y = \text{MaxPool}(x) $$ within ROIs |
| **ROI Align** | $$ y = \text{BilinearInterpolate}(x) $$ within ROIs |
| **Non-Maximum Suppression (NMS)** | $$ \text{keep}_i = \text{IoU}(b_i, b_j) < t $$ for all $$ j $$ with higher scores |
| **Anchor Boxes** | Predefined boxes used in object detection |
| **Feature Maps** | Intermediate representations from convolutional layers |
| **ConvLSTM (Convolutional LSTM)** | LSTM with convolution operations in gates and state updates |
| **Gated Convolutions** | $$ y = \text{Conv}(x) \cdot \sigma(\text{Conv}(x)) $$ |
| **Spectral Normalization** | $$ \hat{W} = \frac{W}{\sigma(W)} $$, where $$ \sigma(W) $$ is the spectral norm of $$ W $$ |
| **Weight Normalization** | $$ \hat{W} = \frac{g}{\|v\|} v $$, where $$ W = g \frac{v}{\|v\|} $$ |
| **Weight Sharing** | Sharing weights across different layers or components |
| **Adaptive Average Pooling** | Output size is fixed, $$ y = \text{AdaptiveAvgPool}(x) $$ |
| **Adaptive Max Pooling** | Output size is fixed, $$ y = \text{AdaptiveMaxPool}(x) $$ |
| **Global Average Pooling** | $$ y = \text{AvgPool}(x) $$ over entire spatial dimensions |
| **Global Max Pooling** | $$ y = \text{MaxPool}(x) $$ over entire spatial dimensions |
| **Intermediate Supervision** | Adding loss functions at intermediate layers |
| **Multi-Scale Training** | Training with images of different scales |
| **Skip Connections in U-Net** | Connecting corresponding down-sampling and up-sampling layers |
| **Upsampling via Interpolation** | $$ y = \text{Interpolate}(x) $$ |
| **Dense Skip Connections** | Connecting each layer to every other layer in a feed-forward manner |
| **Spatial Transformer Networks** | $$ y = T(x, \theta) $$, where $$ T $$ is a spatial transformation |
| **Region Proposal Network (RPN)** | Generating object proposals using sliding windows |
| **Anchor-Free Object Detectors** | Detecting objects without predefined anchors |
| **Bounding Box Regression** | $$ \hat{b} = W x + b $$ to predict bounding box coordinates |
| **Segmentation Masks** | Predicting pixel-wise class labels |
| **IoU Loss (Intersection over Union)** | $$ \text{IoU} = \frac{|A \cap B|}{|A \cup B|} $$ |
| **Focal Loss** | $$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$ |
| **Dice Loss** | $$ \text{Dice} = \frac{2|A \cap B|}{|A| + |B|} $$ |
| **Edge Detection Layers** | Applying edge detection filters (e.g., Sobel filters) |
| **Shape Priors** | Using prior knowledge of shapes in segmentation |
| **Graph Convolutions** | Convolution operations on graphs, $$ y = A x W $$ |
| **Mesh Convolutions** | Convolution operations on mesh structures |
| **PointNet Layers** | Layers designed for point cloud processing |
| **Pose Estimation Layers** | Predicting keypoint coordinates for human poses |
| **Keypoint Detection Layers** | Predicting keypoint locations |
| **Triplet Loss** | $$ \mathcal{L} = \max(d(a, p) - d(a, n) + \alpha, 0) $$ |
| **Contrastive Loss** | $$ \mathcal{L} = (1 - y) \frac{1}{2} d^2 + y \frac{1}{2} \max(m - d, 0)^2 $$ |
| **Hinge Loss** | $$ \mathcal{L} = \max(0, 1 - y f(x)) $$ |
| **Pixel Shuffle** | Rearranging elements of the convolution output |
| **Feature Attention** | $$ y = \alpha \cdot x $$, where $$ \alpha $$ are attention weights |
| **Entropy Regularization** | Adding entropy of the predictions as a regularization term |
| **Curriculum Learning** | Training models with increasingly complex examples |
| **Knowledge Distillation** | Transferring knowledge from a large model to a smaller one |
| **Ensemble Methods** | Combining predictions from multiple models |
| **Multi-Branch Networks** | Networks with multiple branches for different tasks |
| **Neural Architecture Search (NAS)** | Automatically searching for optimal architectures |
| **Hyperparameter Optimization** | Optimizing hyperparameters using techniques like grid search, random search, or Bayesian optimization |
| **Gradient Clipping** | Clipping gradients to prevent exploding gradients |
| **Cyclic Learning Rates** | Varying learning rate cyclically |
| **Cosine Annealing** | $$ \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\frac{T_{cur}}{T_{max}} \pi)) $$ |
| **Learning Rate Schedulers** | Adjusting learning rate during training |
| **Transfer Learning Layers** | Using pre-trained layers from other models |
| **Pre-trained Embeddings** | Using pre-trained embeddings as input features |
| **Latent Space Representations** | Representations in the latent space |
| **Reconstruction Loss** | $$ \mathcal{L} = \|x - \hat{x}\|^2 $$ |
| **Pixel-wise Loss** | Loss computed pixel by pixel |
| **Perceptual Loss** | Loss based on feature maps from a pre-trained network |
| **Adversarial Loss** | $$ \mathcal{L} = \log(D(x)) + \log(1 - D(G(z))) $$ |
| **Style Loss** | Loss based on style differences between images |
| **Content Loss** | Loss based on content differences between images |
| **Histogram Loss** | Loss based on histogram differences |
| **Feature Matching Loss** | Loss based on matching intermediate feature representations |
| **Cycle Consistency Loss** | Loss enforcing $$ x \approx G(F(x)) $$ |
| **Temporal Consistency Loss** | Loss ensuring consistency over time |
| **Total Variation Loss** | $$ \mathcal{L} = \sum_{i,j} ((x_{i,j} - x_{i+1,j})^2 + (x_{i,j} - x_{i,j+1})^2) $$ |
| **Network Pruning** | Removing unnecessary weights or neurons |
| **Quantization** | Reducing the precision of weights |
| **Low-Rank Factorization** | Approximating weight matrices by lower-rank matrices |
| **Weight Initialization Techniques** | Various methods for initializing weights |
| **Xavier Initialization** | $$ W \sim \mathcal{U}(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}) $$ |
| **He Initialization** | $$ W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}}) $$ |
| **Orthogonal Initialization** | Initializing weights with orthogonal matrices |
| **Stochastic Gradient Descent (SGD)** | $$ \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta) $$ |
| **Adam** | $$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$, $$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$, $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$, $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$, $$ \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$ |
| **RMSprop** | $$ E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2 $$, $$ \theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{E[g^2]_t} + \epsilon} $$ |
| **Adagrad** | $$ G_{t+1} = G_t + g_t^2 $$, $$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} g_t $$ |
| **L1 Regularization** | $$ \mathcal{L} = \mathcal{L}_0 + \lambda \|w\|_1 $$ |
| **L2 Regularization** | $$ \mathcal{L} = \mathcal{L}_0 + \lambda \|w\|_2^2 $$ |



## Computer Vision Loss Functions

| **Name** | **Mathematical Equation** |
|---|---|
| **Mean Squared Error (MSE)** |  $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ |
| **Mean Absolute Error (MAE)** | $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$ |
| **Root Mean Squared Error (RMSE)** | $L(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$ |
| **Huber Loss** | $L(y, \hat{y}) = \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\ \delta|y_i - \hat{y}_i| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$ |
| **Log-Cosh Loss** | $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N \log(\cosh(y_i - \hat{y}_i))$ |
| **Binary Cross-Entropy** | $L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$ |
| **Categorical Cross-Entropy** | $L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C y_{ij} \log(\hat{y}_{ij})$ |
| **Focal Loss** | $L(y, \hat{y}) = - \alpha (1 - \hat{y}_i)^\gamma \log(\hat{y}_i)$ |
| **Dice Loss** | $L(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^N y_i \hat{y}_i}{\sum_{i=1}^N y_i^2 + \sum_{i=1}^N \hat{y}_i^2}$ |
| **Jaccard Loss (IoU Loss)** | $L(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N y_i \hat{y}_i}{\sum_{i=1}^N y_i + \sum_{i=1}^N \hat{y}_i - \sum_{i=1}^N y_i \hat{y}_i}$ |
| **Tversky Loss** | $L(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N y_i \hat{y}_i}{\alpha \sum_{i=1}^N y_i \hat{y}_i + \beta \sum_{i=1}^N (1 - y_i) \hat{y}_i + \gamma \sum_{i=1}^N y_i (1 - \hat{y}_i)}$ |
| **Generalized Dice Loss** | $L(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^N w_i y_i \hat{y}_i}{\sum_{i=1}^N w_i y_i^2 + \sum_{i=1}^N w_i \hat{y}_i^2}$ |
| **Weighted Cross-Entropy Loss** | $L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N [w_i y_i \log(\hat{y}_i) + (1 - w_i) (1 - y_i) \log(1 - \hat{y}_i)]$ |
| **Lovasz Loss** | $L(y, \hat{y}) = \sum_{c=1}^C \text{Lovasz-hinge}(y_c, \hat{y}_c)$ |
| **Boundary Loss** | $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| + \lambda \sum_{i=1}^N |\nabla y_i - \nabla \hat{y}_i|$ |
| **WGAN-GP Loss** | $L(D) = \mathbb{E}_{x \sim p_r(x)}[D(x)] - \mathbb{E}_{z \sim p_z(z)}[D(G(z))] + \lambda \mathbb{E}_{x \sim p_{\hat{x}}(x)}[(\|\nabla_x D(x)\|_2 - 1)^2]$ |
| **Wasserstein Loss** | $L(D, G) = \mathbb{E}_{x \sim p_r(x)}[D(x)] - \mathbb{E}_{z \sim p_z(z)}[D(G(z))]$ |
| **Perceptual Loss** | $L(y, \hat{y}) = \| \phi(y) - \phi(\hat{y}) \|_2^2$ |
| **Style Loss** | $L(y, \hat{y}) = \| G(y) - G(\hat{y}) \|_F^2$ |
| **Total Variation Loss** | $L(y, \hat{y}) = \sum_{i=1}^{N-1} \sum_{j=1}^{M-1} |y_{i+1,j} - y_{i,j}| + |y_{i,j+1} - y_{i,j}|$ |
| **L1 Loss** | $L(y, \hat{y}) = \sum_{i=1}^N |y_i - \hat{y}_i|$ |
| **L2 Loss** | $L(y, \hat{y}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2$ |
| **Triplet Loss** | $L(a, p, n) = max(d(a, p) - d(a, n) + \alpha, 0)$ |
| **Contrastive Loss** | $L(x_i, x_j) = \frac{1}{2} (d(x_i, x_j)^2 + [max(m - d(x_i, x_j), 0)]^2)$ |
| **Siamese Network Loss** | $L(x_1, x_2) = d(f(x_1), f(x_2))$ |
| **Center Loss** | $L(x, y) = \frac{1}{2} \sum_{i=1}^N \|x_i - c_{y_i}\|^2$ |
| **Angular Loss** | $L(x, y) = \sum_{i=1}^N (1 - cos(\theta_{y_i, x_i}))^2$ |
| **Cross-Correlation Loss** | $L(x, y) = -\sum_{i=1}^N \sum_{j=1}^N x_i y_j$ |
| **Mutual Information Loss** | $L(x, y) = I(x; y)$ |
| **KL Divergence Loss** | $L(p, q) = \sum_{i=1}^N p_i \log \frac{p_i}{q_i}$ |
| **JS Divergence Loss** | $L(p, q) = \frac{1}{2} KL(p || \frac{p + q}{2}) + \frac{1}{2} KL(q || \frac{p + q}{2})$ |
| **Jensen-Shannon Divergence Loss** | $L(p, q) = \frac{1}{2} KL(p || m) + \frac{1}{2} KL(q || m)$ where $m = \frac{1}{2}(p + q)$ |
| **Earth Mover's Distance (EMD) Loss** | $L(p, q) = \min_{\gamma \in \Gamma(p,q)} \sum_{i=1}^N \sum_{j=1}^N \gamma_{ij} d_{ij}$ |
| **Hausdorff Distance Loss** | $L(A, B) = max(h(A, B), h(B, A))$ where $h(A, B) = \max_{a \in A} \min_{b \in B} d(a, b)$ |
| **Chamfer Distance Loss** | $ L(A, B) = \frac{1}{|A|} \sum_{a \in A} \min_{b \in B} d(a, b)^2 + \frac{1}{|B|} \sum_{b \in B} \min_{a \in A} d(a, b)^2 $ |
| **PointNet Loss** | $L(x, y) = \sum_{i=1}^N \|x_i - y_i\|^2$ |
| **PointNet++ Loss** | $L(x, y) = \sum_{i=1}^N \|x_i - y_i\|^2 + \lambda \sum_{i=1}^N \|f(x_i) - f(y_i)\|^2$ |
| **VoxelNet Loss** | $L(x, y) = \sum_{i=1}^N \|x_i - y_i\|^2 + \lambda \sum_{i=1}^N \|f(x_i) - f(y_i)\|^2$ |
| **SSD Loss** | $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N L_{conf}(y_i, \hat{y}_i) + \alpha L_{loc}(y_i, \hat{y}_i)$ |
| **YOLO Loss** | $L(y, \hat{y}) = \lambda_{coord} \sum_{i=1}^N L_{coord}(y_i, \hat{y}_i) + \lambda_{noobj} \sum_{i=1}^N L_{noobj}(y_i, \hat{y}_i) + \lambda_{obj} \sum_{i=1}^N L_{obj}(y_i, \hat{y}_i)$ |
| **RetinaNet Loss** | $L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N L_{cls}(y_i, \hat{y}_i) + \frac{1}{N} \sum_{i=1}^N L_{reg}(y_i, \hat{y}_i)$ |
| **Mask R-CNN Loss** | $L(y, \hat{y}) = L_{cls}(y, \hat{y}) + L_{bbox}(y, \hat{y}) + L_{mask}(y, \hat{y})$ |
| **DeepLab Loss** | $L(y, \hat{y}) = L_{cross\_entropy}(y, \hat{y}) + \lambda L_{aux\_loss}(y, \hat{y})$ |
| **U-Net Loss** | $L(y, \hat{y}) = L_{dice}(y, \hat{y}) + \lambda L_{cross\_entropy}(y, \hat{y})$ |

**Note:**
* $y$ represents the ground truth label.
* $\hat{y}$ represents the predicted output.
* $N$ represents the number of data points.
* $\alpha$, $\beta$, $\gamma$, $\lambda$, and $\delta$ are hyperparameters. 
* $\phi$ represents a feature extractor network.
* $G$ represents a style transfer network.
* $f$ represents a point cloud feature extraction network.
* $D$ represents the discriminator network.
* $G$ represents the generator network.
* $p_r(x)$ represents the real data distribution.
* $p_z(z)$ represents the noise distribution.
* $p_{\hat{x}}(x)$ represents the distribution of generated data.



-----

Applies a 1D convolution over an input signal composed of several input
    planes.

  In the simplest case, the output value ofthe layer with input size
  :math:$(N, C_{\text{in}}, L)$ and output:math:$(N, C_{\text{out}}, L_{\text{out}})$ can be
  precisely described as:


$$\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
\sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
\star \text{input}(N_i, k)$$
# 2D Convolution Layer


## Mathematical Formulation

In the simplest case, the output value of the layer with input size $ (N, C_{\text{in}}, H, W) $ and output $ (N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}}) $ can be precisely described by the following equation:

$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
$$


$$H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor $$


$$W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor$$
                 


## Module Support

- This module supports TensorFloat32 on Ampere architecture.
- On certain ROCm devices, when using float16 inputs, this module will use different precision for backward operations.

## Parameters

- `stride`: Controls the stride for the cross-correlation, a single number or a tuple.
- `padding`: Controls the amount of padding applied to the input. It can be either a string {'valid', 'same'} or an int/a tuple of ints giving the amount of implicit padding applied on both sides.
- `dilation`: Controls the spacing between the kernel points, also known as the à trous algorithm.

## Notes

- `kernel_size`, `stride`, `padding`, `dilation` can either be:
  - a single `int` — in which case the same value is used for the height and width dimension
  - a `tuple` of two ints — in which case, the first `int` is used for the height dimension, and the second `int` for the width dimension

- `padding='valid'` is the same as no padding. `padding='same'` pads the input so the output has the shape as the input. However, this mode doesn't support any stride values other than 1.

- This module supports complex data types i.e., `complex32`, `complex64`, `complex128`.

## Arguments

- `in_channels` (int): Number of channels in the input image
- `out_channels` (int): Number of channels produced by the convolution
- `kernel_size` (int or tuple): Size of the convolving kernel
- `stride` (int or tuple, optional): Stride of the convolution. Default: 1
- `padding` (int, tuple, or str, optional): Padding added to all four sides of the input. Default: 0
- `padding_mode` (str, optional): `'zeros'`, `'reflect'`, `'replicate'`, or `'circular'`. Default: `'zeros'`
- `dilation` (int or tuple, optional): Spacing between kernel elements. Default: 1
- `groups` (int, optional): Number of blocked connections from input channels to output channels. Default: 1
- `bias` (bool, optional): If `True`, adds a learnable bias to the output. Default: `True`


# 3D Convolution Layer

This documentation details the implementation of a 3D convolution layer, which applies a 3D convolution over an input signal composed of several input planes.

## Mathematical Formulation

In the simplest case, the output value of the layer with input size \( (N, C_{\text{in}}, D, H, W) \) and output \( (N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) \) can be described as:

$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
$$

where \( \star \) denotes the valid 3D cross-correlation operator.

## Module Support

- Supports TensorFloat32 on Ampere architecture.
- On certain ROCm devices, when using float16 inputs, this module will use different precision for backward operations.

## Parameters

- `stride`: Controls the stride for the cross-correlation.
- `padding`: Controls the amount of padding applied to the input. It can be either a string {'valid', 'same'} or a tuple of ints giving the amount of implicit padding applied on both sides.
- `dilation`: Controls the spacing between the kernel points, also known as the à trous algorithm.

## Notes

- `kernel_size`, `stride`, `padding`, `dilation` can either be:
  - a single `int` — in which case the same value is used for the depth, height, and width dimensions
  - a `tuple` of three ints — in which case, the first `int` is used for the depth dimension, the second `int` for the height dimension, and the third `int` for the width dimension

- `padding='valid'` is the same as no padding. `padding='same'` pads the input so the output has the shape as the input. However, this mode doesn't support any stride values other than 1.

- This module supports complex data types i.e., `complex32`, `complex64`, `complex128`.

## Arguments

- `in_channels` (int): Number of channels in the input image
- `out_channels` (int): Number of channels produced by the convolution
- `kernel_size` (int or tuple): Size of the convolving kernel
- `stride` (int or tuple, optional): Stride of the convolution. Default: 1
- `padding` (int, tuple, or str, optional): Padding added to all six sides of the input. Default: 0
- `padding_mode` (str, optional): `'zeros'`, `'reflect'`, `'replicate'`, or `'circular'`. Default: `'zeros'`
- `dilation` (int or tuple, optional): Spacing between kernel elements. Default: 1
- `groups` (int, optional): Number of blocked connections from input channels to output channels. Default: 1
- `bias` (bool, optional): If `True`, adds a learnable bias to the output. Default: `True`

## Shape

- Input: $ (N, C_{\text{in}}, D_{\text{in}}, H_{\text{in}}, W_{\text{in}}) $ or $ (C_{\text{in}}, D_{\text{in}}, H_{\text{in}}, W_{\text{in}}) $
- Output: $ (N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) $ or $ (C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) $, where

$$
D_{\text{out}} = \left\lfloor\frac{D_{\text{in}} + 2 \times \text{padding}[0] - \text{dilation}[0]
\times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
$$

$$
H_{\text{out}} = \left\lfloor\frac{H_{\text{in}} + 2 \times \text{padding}[1] - \text{dilation}[1]
\times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
$$

$$
W_{\text{out}} = \left\lfloor\frac{W_{\text{in}} + 2 \times \text{padding}[2] - \text{dilation}[2]
\times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
$$
----
 - Input: :math:$(N, C_{in}, L_{in})$ or :math:$(C_{in}, L_{in})$
        - Output: :math:$(N, C_{out}, L_{out})$ or :math$(C_{out}, L_{out})$, where


$$L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel\_size} - 1) + \text{output\_padding} + 1$$



 Shape:
        - Input: :math:$(N, C_{in}, H_{in}, W_{in})$ or :math:$(C_{in}, H_{in}, W_{in})`
        - Output: :math:$(N, C_{out}, H_{out}, W_{out})$ or :math:`(C_{out}, H_{out}, W_{out})$, where

 
$$H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1$$

$$W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1$$

-------
3D...
 - Input: :math:$(N, C_{in}, D_{in}, H_{in}, W_{in})$ or :math:$(C_{in}, D_{in}, H_{in}, W_{in})$
        - Output: :math:$(N, C_{out}, D_{out}, H_{out}, W_{out})$ or
          :math:$(C_{out}, D_{out}, H_{out}, W_{out})$, where


$$D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1$$
$$H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1$$

$$W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                        \times (\text{kernel\_size}[2] - 1) + \text{output\_padding}[2] + 1$$

---
 - Input: :math:$(N, C, *)$
- Output: :math:$(N, C \times \prod(\text{kernel\_size}), L)$ as described above

$$

L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

$$
Fold

$$
L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor
$$

- Input: :math:$(N, C \times \prod(\text{kernel\_size}), L)$ or :math:$(C \times \prod(\text{kernel\_size}), L)$
- Output: :math:$(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)$
          or :math:$(C, \text{output\_size}[0], \text{output\_size}[1], \dots)$ as described above
### 2DMax_pool
$$   \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned} $$
----
$$
f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}
$$
----
$$
 L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
$$
$$
out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
$$

- Input: :math:$(N, C, H_{in}, W_{in})$ or :math:$(C, H_{in}, W_{in})$.
        - Output: :math:$(N, C, H_{out}, W_{out})$ or :math:$(C, H_{out}, W_{out})$, where


$$H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}$$






$$W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}$$


        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` where

$H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}$

$W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}$


-----
# Activation

$$\text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}$$
---
$$

\text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}
$$
----
$$
\text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}
$$
----
$$
\text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}
$$
----
$$

\text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}
$$
---
$$
\text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

$$
---
$$ 
\text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

$$ 
---
$$
 \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
$$
$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.
--
$\text{PReLU}(x) = \max(0,x) + a * \min(0,x)$

    or

$$\text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}$$

$$ \text{ReLU}(x) = (x)^+ = \max(0, x) $$
$\text{ReLU6}(x) = \min(\max(0,x), 6)$

$$ \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}
$$
$\text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))$

$$ \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))$$

$$ \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))$$
$$  \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}$$

$$\text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}$$
$$\text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))$$
$$\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
$$

$$ \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}$$

$$ y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}
$$
$$
{GLU}(a, b)= a \otimes \sigma(b)
$$

# batch_Normalised

$$ y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
## RNN

$$      h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh}) $$

$$
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}
$$

$$
\begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
        \end{array}
$$
---
$$
h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

$$

$$
\begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f \odot c + i \odot g \\
        h' = o \odot \tanh(c') \\
        \end{array}
$$

$$

\begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
        h' = (1 - z) \odot n + z \odot h
\end{array}

$$




| Component | Mathematical Equation |
|-----------|-----------------------|
| `nn.Transformer` | $$\text{Transformer}(x) = \text{Encoder}(x) \oplus \text{Decoder}(y)$$ |
| `nn.TransformerEncoder` | $$\text{Encoder}(x) = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x)) + \text{FeedForward}(x)$$ |
| `nn.TransformerDecoder` | $$\text{Decoder}(y) = \text{LayerNorm}(y + \text{MultiHeadAttention}(y, y, y)) + \text{MultiHeadAttention}(y, \text{Encoder}(x), \text{Encoder}(x)) + \text{FeedForward}(y)$$ |
| `nn.TransformerEncoderLayer` | $$\text{EncoderLayer}(x) = \text{LayerNorm}(x + \text{SelfAttention}(x, x, x)) + \text{FeedForward}(x)$$ |
| `nn.TransformerDecoderLayer` | $$\text{DecoderLayer}(y) = \text{LayerNorm}(y + \text{SelfAttention}(y, y, y)) + \text{LayerNorm}(y + \text{MultiHeadAttention}(y, \text{Encoder}(x), \text{Encoder}(x))) + \text{FeedForward}(y)$$ |


---
$y = x_1^T A x_2 + b$
---
$y = xA^T + b$
---
$$\text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)} $$

     
     

$$\Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}$$

----
$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
- Input: :math:$(N, C)$ or :math:$(N, C, L)$, where $N$ is the batch size,
$C$ is the number of features or channels, and :math:$L$ is the sequence length
- Output: :math:$(N, C)$ or :math:$(N, C, L)$ (same shape as input)
----
## Norm2d
4D input
$$
 y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
5D input
- Input: :math:$(N, C, D, H, W)$
        - Output: :math:$(N, C, D, H, W)$ (same shape as input)
----
### InstanceNorm1d
Input: :math:$(N, C, L)$ or :math:$(C, L)$
        - Output: :math:$(N, C, L)$ or :math:$(C, L)$ (same shape as input)