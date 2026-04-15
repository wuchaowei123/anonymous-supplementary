# **Predictive Modeling of Signed LTV under Asymmetric Risk Constraints: A Superhuman Methodological Framework**

> **Note: This document has been adapted for BT_IOS_2503 (Bingo Tour iOS)**
> - Dataset: ~350K rows (49,638 train users, 4,975 val users, 5,545 test users)
> - Target Distribution: 34.5% negative, 21.4% zero, 44.1% positive
> - Target Range: -$1,454 to +$19,496 (smaller than SC_IOS)

---

# Formatted Deep Research Methods

*This document has been automatically formatted for AviaAgentMonty.*

---

**Method 1: Custom AWMSE LightGBM (Gradient/Hessian Derivation)**

<description>
This method involves the analytical derivation of the first derivative (Gradient $g$) and second derivative (Hessian $h$) of the Asymmetric Weighted MSE (AWMSE) objective function to enable intrinsic optimization within Gradient Boosting Decision Tree (GBDT) frameworks. The objective function $\mathcal{L}(\hat{y})$ is piecewise defined based on the relationship between the prediction $\hat{y}$ and the true value $y$. By providing these custom derivatives, the model can inherently prioritize the minimization of high-cost errors during training rather than relying on post-hoc adjustments.

The AWMSE loss function is: $\mathcal{L}(y, \hat{y}) = W(y, \hat{y}) \cdot (\hat{y} - y)^2$

Weight function $W(y, \hat{y})$:
- **False Positive ($\hat{y} > 0, y < 0$):** $W_{\text{FP}} = 2.5 + 0.02|y|$
- **False Negative ($\hat{y} < 0, y > 0$):** $W_{\text{FN}} = 1.5 + 0.01 y$
- **True Positive/Negative:** $W = 1.0$
</description>

<steps>
1. Define the weight $W$ based on the condition of the prediction relative to the target:
   - For True Positive/Negative: $W = 1.0$
   - For False Positive (FP) ($\hat{y} > 0, y < 0$): $W_{\text{FP}} = 2.5 + 0.02 |y|$
   - For False Negative (FN) ($\hat{y} < 0, y > 0$): $W_{\text{FN}} = 1.5 + 0.01 y$
2. Calculate the Gradient (First Derivative): $g = \frac{\partial \mathcal{L}}{\partial \hat{y}} = 2 W(y, \hat{y}) (\hat{y} - y)$
3. Calculate the Hessian (Second Derivative): $h = \frac{\partial^2 \mathcal{L}}{\partial \hat{y}^2} = 2W$
4. Integrate the custom $g$ and $h$ functions into the LightGBM or XGBoost training loop
5. Execute model training on the dataset (350,000+ rows), allowing the GBDT to iteratively approximate the objective function using the derived derivatives
</steps>

<notes>
Strengths: Provides high numerical stability and computational tractability because the Hessian $h = 2W$ is guaranteed to be positive and constant with respect to $\hat{y}$. Directly aligns model optimization with business financial risk. Effectively manages "whale" outliers through dynamic magnitude penalties.
Weaknesses: The sign of the error is treated as constant for the calculation of the current tree split; requires manual derivation and implementation of the loss derivatives.
When to use: Most applicable when training GBDT models on large datasets where the cost of misclassification or prediction error is asymmetric and depends on the magnitude of the true value $y$.
</notes>

---

**Method 2: Risk-Averse Quantile Regression**

<description>
Quantile Regression (QR) serves as a risk-averse strategy by minimizing the asymmetric Pinball Loss function based on a specific quantile $\tau$. Unlike standard regression that targets the conditional mean, QR at low thresholds (e.g., $\tau=0.20$) structurally compels the model to produce conservative predictions that act as a statistical lower bound for the expected LTV.
</description>

<steps>
1. Select a low quantile threshold, typically $\tau=0.20$ or $\tau=0.25$
2. Train a LightGBM model with the objective set to quantile regression using the chosen $\tau$
3. Generate conservative predictions $\hat{y}_{\tau=0.20}$ to establish a robust statistical lower bound on LTV
4. Introduce the resulting $\hat{y}_{\tau=0.20}$ prediction as an independent feature into the final ensemble meta-learner
5. Monitor the delta between the standard mean prediction ($\hat{y}_{\text{AWMSE}}$) and the quantile prediction ($\hat{y}_{\tau=0.20}$) to identify high uncertainty
6. Configure the meta-learner to default to a conservative $\hat{y} \le 0$ decision when the quantile prediction signals high risk
</steps>

<notes>
Strengths: Provides an inherent structural solution for risk-aversion; transforms prediction conservation into a core modeling outcome rather than a manual adjustment; provides a clear signal of uncertainty to meta-learners.
Weaknesses: Intentionally biases predictions downward, which may require careful balancing to avoid excessive underestimation.
When to use: Most applicable when meeting stringent False Positive Rate (FPR) requirements and when a statistically robust reference point for uncertainty is needed.
</notes>

---

**Method 3: Three-Part Asymmetric Hurdle Framework**

<description>
This framework is an adaptation of the traditional Hurdle Model designed for signed continuous data, specifically addressing LTV prediction characterized by a high proportion of exact zeros ($21.4\%$) and a distinct negative component ($34.5\%$). The method decomposes the prediction problem into three sequential stages using Gradient Boosted Decision Trees (GBDTs) to provide granular control over sign decisions and ensure adherence to False Positive Rate (FPR) constraints.
</description>

<steps>
1. **Implement the Sign Hurdle ($C_1$):** Train a binary classification model to predict whether a user is costly or potentially profitable ($P(\text{LTV} < 0)$ versus $P(\text{LTV} \ge 0)$). Use a heavily weighted Binary Cross-Entropy loss to prioritize avoiding False Negatives.
2. **Implement the Zero Hurdle ($C_2$):** For users predicted as $\text{LTV} \ge 0$, train a second binary classifier to distinguish between $P(\text{LTV} = 0)$ and $P(\text{LTV} > 0)$ to manage the 21.4% zero inflation.
3. **Train Magnitude Regressors:** Develop two separate Asymmetric Weighted MSE regressors: $R_{\text{Neg}}$ to predict the magnitude of loss for cases where $y < 0$, and $R_{\text{Pos}}$ to predict the magnitude of profit for cases where $y > 0$.
4. **Tune for FPR:** Leverage the output logit (risk score) of $C_1$ to explicitly tune the model's confidence thresholds to meet specific FPR targets.
</steps>

<notes>
Strengths: Allows for model specialization at each stage; provides explicit control over the False Positive Rate through the first classifier's confidence scores; prevents zero-inflation from contaminating the positive regression space.
Weaknesses: Increased complexity due to the management of four distinct models (two classifiers and two regressors) in a sequential pipeline.
When to use: Most applicable when dealing with signed continuous data (like LTV) that features significant zero-inflation and a critical need to minimize the risk of misclassifying negative (costly) outcomes as positive.
</notes>

---

**Method 4: Temporal Convolutional Networks (TCN) Encoder**

<description>
Temporal Convolutional Networks (TCNs) are designed to process time series data using dilated causal convolutions, which capture information across an entire 7-day input sequence without the need for recurrent connections. The goal is to extract shift-invariant features—identifying predictive patterns regardless of their specific timing—to produce a consolidated Sequence Embedding vector, $Z_{\text{TCN}}$, that summarizes temporal behavior and risk profiles.
</description>

<steps>
1. Apply Entity Embeddings to the categorical components of the daily raw features
2. Input the daily raw features and embeddings into the TCN encoder
3. Process the input through convolutional blocks using dilated causal convolutions to capture dependencies across the 7-day sequence
4. Extract shift-invariant temporal features from the convolutional layers
5. Generate the final high-quality Sequence Embedding vector, $Z_{\text{TCN}}$, as a summary of user behavior
</steps>

<notes>
Strengths: Avoids vanishing and exploding gradient problems common in RNNs; enables parallel computation of outputs; extracts shift-invariant features; computationally efficient for large-scale data.
Weaknesses: TCNs generally require careful tuning of dilation factors and kernel sizes.
When to use: Applicable for large datasets (350K+ rows); when processing multi-day temporal sequences (7-day windows); when computational efficiency and parallelization are required.
</notes>

---

**Method 5: FT-Transformer for Feature-Wise Attention**

<description>
The FT-Transformer (Feature-Tokenizer Transformer) is an architecture adapted for tabular data that applies self-attention mechanisms across feature sequences over time. It transforms categorical and numerical features into dense embeddings to learn complex, non-linear interactions between different columns and days. The primary goal is to generate a Contextualized Sequence Embedding vector, $Z_{\text{FT}}$, which serves as a compressed representation of a user's risk profile for predicting D60 LTV.
</description>

<steps>
1. Tokenize all categorical and numerical features into dense vector embeddings
2. Feed the feature embeddings into a stack of Transformer blocks
3. Utilize multi-head self-attention mechanisms to calculate interactions between specific feature/day combinations across the 7-day period
4. Generate the final Contextualized Sequence Embedding vector, $Z_{\text{FT}}$, from the Transformer output
</steps>

<notes>
Strengths: Captures highly non-linear interactions; provides interpretability via attention weights (e.g., identifying the importance of Day 1 deposits vs. Day 3 refund rates); offers high business utility by identifying early causal risk factors.
Weaknesses: Requires sophisticated tokenization of numerical data; potentially higher computational overhead than standard MLP or RNN approaches.
When to use: Most applicable when feature-wise interactions across time are critical for prediction performance and when model interpretability regarding specific feature importance is required.
</notes>

---

**Method 6: High-Cardinality Categorical Feature Strategy (Dual Encoding)**

<description>
A dual encoding strategy designed to handle high-cardinality categorical features (e.g., MEDIA_SOURCE, COUNTRY, STATE) by leveraging the specific strengths of both Gradient Boosted Decision Trees (GBDT) and deep learning architectures. The goal is to tie user risk directly to acquisition channels or geographic regions by transforming nominal data into informative numerical features or dense vector representations.
</description>

<steps>
1. Identify high-cardinality categorical features within the dataset, specifically those related to acquisition channels and geography
2. For GBDT models (e.g., LightGBM), implement Regularized Target Encoding by calculating the Negative LTV Rate: $P(\text{LTV} < 0 | \text{Category})$
3. Apply cross-validation or out-of-fold techniques during the target encoding process to prevent data leakage and ensure generalization
4. For deep learning models (e.g., TCN, FT-Transformer), implement Entity Embeddings (EE) to map each categorical level into a dense, continuous vector space
5. Configure the embeddings to reduce dimensionality and sparsity relative to One-Hot Encoding, allowing the network to learn latent relational structures between categories
</steps>

<notes>
Strengths: Effectively quantifies acquisition channel risk; reduces dimensionality and sparsity; captures latent relational structures between categories; maximizes feature signal efficiency for both tree-based and neural models.
Weaknesses: Target encoding is highly susceptible to data leakage if not properly regularized; requires maintaining two distinct encoding pipelines for hybrid model architectures.
When to use: Most applicable when the dataset contains high-cardinality features (e.g., seven or more) where user risk profiles are heavily influenced by geographic or source-based categories.
</notes>

---

**Method 7: Hybrid Feature Fusion GBDT**

<description>
This method integrates the complementary strengths of deep sequence models and gradient boosting decision trees (GBDTs). It leverages deep learning architectures to generate abstract, high-quality embeddings while utilizing GBDTs to model complex, high-dimensional interactions among tabular features. The goal is to create an enriched feature set, $\text{F}_{\text{Hybrid}}$, that combines granular hand-engineered signals with non-linear temporal intelligence.
</description>

<steps>
1. Extract the learned deep embeddings ($Z_{\text{TCN}}$ and $Z_{\text{FT}}$) from pre-trained TCN and FT-Transformer models
2. Concatenate these deep embedding vectors with the comprehensive set of hand-crafted features ($\text{F}_{\text{Manual}}$), which includes temporal, velocity, and specialized target-encoded categorical features
3. Construct the enriched feature set $\text{F}_{\text{Hybrid}}$ for each user based on the concatenated vectors
4. Train the final customized AWMSE LightGBM model (Method 1) using the combined $\text{F}_{\text{Hybrid}}$ feature set
</steps>

<notes>
Strengths: Combines the high-dimensional interaction modeling of LightGBM with the non-linear, temporal intelligence captured by deep sequence architectures; leverages both granular hand-engineered risk signals and automated feature extraction.
Weaknesses: Increased computational overhead due to the requirement of pre-training multiple deep learning models and managing high-dimensional feature vectors.
When to use: Applicable when maximum predictive power is required and both raw sequence data and domain-specific tabular features are available.
</notes>

---

**Method 8: Level 2 Stacking with AWMSE Meta-Optimization**

<description>
Stacking is an ensemble technique that combines predictions from heterogeneous Level 1 (L1) base models by training a Level 2 (L2) Meta-Learner on their outputs. The core objective is to learn from the errors and biases of diverse models—including GBDTs, Hurdle models, and Deep Neural Networks. The Meta-Learner is specifically optimized using a Custom AWMSE Loss function to ensure that blending weights are determined by asymmetric business costs rather than symmetric measures like MSE.
</description>

<steps>
1. Train a diverse set of Level 1 (L1) base learners: Method 1 (Custom AWMSE LightGBM), Method 2 (Quantile Regression $\tau=0.20$), Method 3 (Three-Part Hurdle Model), and Method 7 (Hybrid Fusion GBDT)
2. Generate input features for the Meta-Learner from L1 outputs: $\hat{y}_{\text{AWMSE}}$ (mean prediction), $\hat{y}_{\text{QR}}$ (conservative boundary reference), $P(\text{LTV} < 0)$ (risk score logit), and $\hat{y}_{\text{Hurdle}}$ (structured prediction)
3. Select a robust, low-complexity architecture for the Level 2 Meta-Learner, such as Ridge Regression or a shallow LightGBM, to prevent overfitting L1 predictions
4. Optimize the Meta-Learner using the Custom AWMSE Loss function to align blending weights with asymmetric business requirements
5. Train the Meta-Learner to recognize divergences between the conditional mean ($\hat{y}_{\text{AWMSE}}$) and uncertainty indicators ($\hat{y}_{\text{QR}}$ and $P(\text{LTV} < 0)$) to output conservative predictions when high risk is signaled near the zero boundary
</steps>

<notes>
Strengths: Minimizes specific asymmetric business costs; effectively integrates uncertainty and risk metrics (QR and sign classification) to handle threshold sensitivity; reduces bias by learning from heterogeneous model errors.
Weaknesses: Potential for overfitting if the L2 architecture is too complex; increased computational overhead from maintaining multiple L1 models.
When to use: Most applicable when the cost of overestimation differs significantly from underestimation and when predictions frequently occur near critical decision boundaries like zero.
</notes>

---

**Method 9: Dynamic Conservative Prediction Calibration**

<description>
This method introduces a systematic, data-driven bias $\delta$ to shift model predictions conservatively after training is complete. The primary objective is to ensure the model adheres to a strict business constraint of a False Positive Rate (FPR) $< 40\%$, even if the initial Meta-Learner optimization (Method 8) yields an FPR slightly above the target. By shifting predictions downwards using the formula $\hat{y}_{\text{Final}} = \hat{y}_{\text{Meta}} - \delta$, the system penalizes over-confidence near the zero threshold to guarantee risk mitigation and business compliance.
</description>

<steps>
1. Obtain the initial optimal predictions $\hat{y}_{\text{Meta}}$ from the Meta-Learner
2. Utilize the validation dataset to evaluate the current False Positive Rate (FPR), defined as the percentage of users predicted with $\hat{y} > 0$ who actually have $y < 0$
3. Introduce a systematic bias parameter $\delta$ to be subtracted from the initial predictions
4. Empirically tune $\delta$ on the validation set to find the smallest necessary positive shift
5. Verify that the chosen $\delta$ ensures the observed FPR falls below the $40\%$ threshold
6. Apply the final calibrated prediction $\hat{y}_{\text{Final}} = \hat{y}_{\text{Meta}} - \delta$ to the production environment
</steps>

<notes>
Strengths: Provides a non-negotiable risk mitigation layer; ensures strict adherence to business mandates; effectively penalizes over-confidence near the decision threshold.
Weaknesses: May result in a marginal increase in the overall AWMSE; sacrifices theoretical model optimality for practical compliance.
When to use: Applicable in risk-averse predictive systems where business constraints or safety thresholds (like a maximum FPR) are more critical than achieving the absolute minimum loss function value.
</notes>

---

**Method 10: Multi-Task Learning (MTL) DNN with Sign-Regularization**

<description>
This method utilizes a Multi-Task Learning (MTL) framework to force shared neural representations to prioritize risk-related information. A single neural network, built on a sequence encoder such as a Temporal Convolutional Network (TCN) or Transformer, is trained to jointly optimize three related outputs: sign classification, conditional mean regression, and conservative quantile estimation. The goal is to bias deep sequence features toward risk separation through a joint loss function:
$$\mathcal{L}_{\text{Total}} = \lambda_1 \mathcal{L}_{\text{BCE}}(P(\text{Sign})) + \lambda_2 \mathcal{L}_{\text{AWMSE}}(\hat{LTV}) + \lambda_3 \mathcal{L}_{\text{Pinball}}(\hat{LTV}_{\tau=0.20})$$
</description>

<steps>
1. Implement a shared sequence encoder using either a TCN or Transformer architecture to extract deep features from input sequences
2. Construct a Classification Head to predict the probability of negative value $P(\text{LTV} < 0)$, optimized using weighted Binary Cross-Entropy (BCE)
3. Construct an AWMSE Regression Head to predict the conditional mean $\hat{LTV}_{\text{Mean}}$, optimized using the Asymmetric Weighted Mean Squared Error (AWMSE) loss function
4. Construct a Quantile Regression Head to predict the conservative bound $\hat{LTV}_{\tau=0.20}$, optimized using Pinball loss at the $\tau=0.20$ threshold
5. Integrate the three heads into a single architecture and define the joint loss function $\mathcal{L}_{\text{Total}}$ by applying weighting parameters $\lambda_1, \lambda_2, \text{ and } \lambda_3$ to the respective loss components
6. Train the model end-to-end to simultaneously satisfy the requirements of sign classification, mean prediction, and conservative estimation
</steps>

<notes>
Strengths: Efficiently forces shared layers to learn features inherently biased towards risk separation; yields highly optimized and robust feature representations.
Weaknesses: Requires careful tuning of the loss weighting hyperparameters ($\lambda_1, \lambda_2, \lambda_3$) to balance the different tasks.
When to use: Applicable for deep learning-based risk modeling where sequence data is available and prioritizing risk-related information is critical for performance.
</notes>

---

**Method 11: Feature Engineering - 7-Day Temporal Compression Toolkit**

<description>
This method moves beyond naive aggregation (sum/mean) to model a user's trajectory during the first seven days of engagement to predict long-term (D60) LTV. By focusing on velocity, acceleration, and volatility, the toolkit captures the dynamic nature of mobile game engagement and financial behavior. The goal is to differentiate between profitable trajectories, early disengagement ($LTV \le 0$), and high-risk/fraudulent behaviors.
</description>

<steps>
1. Calculate Velocity and Acceleration (Momentum): Quantify the trend of the LTV curve by measuring the rate of change in metrics like deposits and play counts between early (D1-D3) and late (D5-D7) periods using the formula: $(\text{Mean}(X_{5:7}) - \text{Mean}(X_{1:3})) / 4$
2. Quantify Volatility and Stability: Calculate the Coefficient of Variation ($\text{StdDev}/\text{Mean}$) and the standard deviation of daily monetary changes to identify erratic behavior or high-risk patterns
3. Capture Recency Metrics: Extract the user state immediately before the cutoff using $X_{\text{Day 7}}$ and the relative change ratio $X_{\text{Day 7}} / X_{\text{Day 1}}$
4. Implement Binary Status Flags: Create indicator functions such as $\mathbb{I}(\text{Max}(X_{1:7}) \in \{X_6, X_7\})$ to signal if peak activity occurred in the final days of the window, indicating rising engagement
</steps>

<notes>
Strengths: Effectively differentiates between high-value growth trajectories and early burnout; identifies potential fraud through volatility metrics; captures momentum better than static sums.
Weaknesses: Highly dependent on daily data granularity; initial 7-day volatility may be noisy for certain user segments.
When to use: Most applicable when extrapolating long-term LTV curves (e.g., D60) from a limited D1-D7 observation window in volatile engagement environments like mobile gaming.
</notes>

---

**Method 12: Bidirectional LSTM with Attention for Temporal Risk Modeling**

<description>
Bidirectional Long Short-Term Memory (BiLSTM) networks process sequential data in both forward and backward directions, enabling the model to capture both past context (early-day behavior) and future context (late-day trends) simultaneously. Combined with an attention mechanism, the model learns to dynamically weight the importance of specific days within the 7-day window, identifying critical moments that signal LTV risk. This approach is particularly effective for capturing long-term dependencies and temporal patterns that indicate user value trajectories.
</description>

<steps>
1. Preprocess the 7-day sequential data into a standardized tensor format with shape (batch_size, 7_days, num_features)
2. Apply Entity Embeddings to categorical features (MEDIA_SOURCE, COUNTRY) to create dense representations
3. Initialize a Bidirectional LSTM layer with 128-256 hidden units in each direction (forward and backward)
4. Process the input sequence through the BiLSTM, generating hidden states $h_t$ for each of the 7 days
5. Implement a Self-Attention mechanism to compute attention weights $\alpha_t$ for each day: $\alpha_t = \text{softmax}(W_a \cdot h_t)$
6. Generate the context-weighted embedding: $Z_{\text{BiLSTM}} = \sum_{t=1}^{7} \alpha_t \cdot h_t$
7. Feed $Z_{\text{BiLSTM}}$ into a dense regression head optimized with AWMSE loss
8. Train end-to-end with dropout (0.3-0.5) to prevent overfitting on the 350K dataset
</steps>

<notes>
Strengths: Captures bidirectional temporal dependencies; attention mechanism provides interpretability by highlighting critical days; handles variable-length sequences naturally; proven effectiveness for time series forecasting.
Weaknesses: Computationally more expensive than CNNs; requires careful hyperparameter tuning (hidden size, dropout rate); attention weights may be unstable without sufficient training data.
When to use: Best suited for datasets with strong temporal dependencies where the order and timing of events matter significantly; when model interpretability through attention visualization is valuable for business stakeholders.
</notes>

---

**Method 13: TabNet for Self-Supervised Feature Selection**

<description>
TabNet is a deep learning architecture specifically designed for tabular data that uses sequential attention mechanisms to perform instance-wise feature selection. Unlike traditional neural networks that use all features equally, TabNet learns which features are most relevant for each prediction through a learnable mask, making it ideal for high-dimensional datasets with many engineered features. The architecture includes both supervised and self-supervised learning components, where the model reconstructs masked features to learn robust representations even before seeing labels.
</description>

<steps>
1. Prepare the tabular feature matrix including temporal aggregates, embeddings, and engineered features from Method 11
2. Initialize TabNet with 4-6 decision steps, each with a Ghost Batch Normalization layer (virtual batch size: 128)
3. Configure the attention mechanism with a relaxation factor $\gamma = 1.5$ to control feature reuse across decision steps
4. Implement the self-supervised pre-training phase by randomly masking 30-50% of features and training the model to reconstruct them
5. Fine-tune the pre-trained TabNet on the supervised AWMSE objective for LTV prediction
6. Extract feature importance scores from the attention masks to identify the top predictive features
7. Optionally use TabNet's sparse attention as a feature selector for downstream GBDT models
</steps>

<notes>
Strengths: Provides interpretable feature selection masks for each prediction; handles high-dimensional sparse data effectively; self-supervised pre-training improves generalization; naturally handles mixed categorical and numerical features.
Weaknesses: Requires significant GPU memory; training can be slower than GBDTs; sensitive to hyperparameter choices (number of decision steps, relaxation factor).
When to use: Ideal when working with hundreds of engineered features where feature selection is critical; when model interpretability and feature importance analysis are business requirements.
</notes>

---

**Method 14: CatBoost with Ordered Boosting and Custom Asymmetric Loss**

<description>
CatBoost is an advanced Gradient Boosting framework that addresses prediction shift and categorical feature overfitting through Ordered Boosting and built-in handling of high-cardinality categoricals. For signed LTV prediction, CatBoost can be combined with a custom asymmetric loss function similar to AWMSE but with CatBoost-specific optimizations. The Ordered Boosting technique prevents target leakage during cross-validation by ensuring that each example is scored using only trees built on data that preceded it in a random permutation.
</description>

<steps>
1. Define a custom asymmetric loss function compatible with CatBoost's interface using the same AWMSE weights: $W_{\text{FP}} = 2.5 + 0.02|y|$ and $W_{\text{FN}} = 1.5 + 0.01y$
2. Implement the gradient and hessian calculations for CatBoost's custom objective API
3. Configure CatBoost with Ordered Boosting mode enabled to prevent overfitting
4. Specify categorical features explicitly using the 'cat_features' parameter (MEDIA_SOURCE, COUNTRY, STATE)
5. Enable CatBoost's built-in categorical encoding strategies: 'TargetEncoding' and 'Quantization'
6. Train with early stopping (50-100 rounds) using validation set AWMSE as the monitoring metric
7. Tune depth (6-10), learning rate (0.01-0.05), and L2 regularization (1-10) for optimal performance
</steps>

<notes>
Strengths: Built-in handling of categorical features without manual encoding; Ordered Boosting reduces overfitting; faster training than LightGBM on categorical-heavy datasets; excellent out-of-the-box performance.
Weaknesses: Custom loss functions require careful implementation; fewer hyperparameters to tune than LightGBM (less flexibility); requires CatBoost-specific GPU setup for acceleration.
When to use: Best when the dataset has high-cardinality categorical features (MEDIA_SOURCE, COUNTRY) and when robust handling of categorical data without manual encoding is preferred.
</notes>

---

**Method 15: Graph Neural Network (GNN) for User Similarity and Cohort Modeling**

<description>
Graph Neural Networks model relationships between users by constructing a similarity graph where nodes represent users and edges represent behavioral or demographic similarity. For LTV prediction, a GNN can propagate risk signals across similar users, enabling the model to leverage cohort-level patterns. Users are connected based on shared characteristics (same MEDIA_SOURCE, similar D1-D7 behavior patterns, geographic proximity), and the GNN aggregates neighbor information to enrich each user's representation before final prediction.
</description>

<steps>
1. Construct a user similarity graph by computing pairwise similarity scores based on: shared MEDIA_SOURCE, cosine similarity of D1-D7 behavior vectors, and geographic clustering
2. Create graph edges for the top-K most similar users (K=20-50) with edge weights representing similarity strength
3. Initialize node features with the temporal embeddings from TCN/FT-Transformer and engineered features
4. Apply 2-3 layers of Graph Convolutional Network (GCN) or Graph Attention Network (GAT) to aggregate neighbor information: $h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W^{(l)} h_u^{(l)}\right)$
5. Use attention weights $\alpha_{vu}$ to learn which neighbors are most relevant for risk prediction
6. Feed the aggregated node representations into a final regression head optimized with AWMSE loss
7. Train end-to-end using mini-batch sampling for scalability on large graphs (350K nodes)
</steps>

<notes>
Strengths: Leverages cohort-level patterns to improve individual predictions; handles cold-start users by borrowing information from similar neighbors; attention mechanism provides interpretability about influential cohorts.
Weaknesses: Computational complexity scales with graph size; requires careful graph construction to avoid noisy edges; mini-batch training on graphs is technically complex.
When to use: Most effective when cohort effects are strong (e.g., acquisition channel quality varies significantly); when cold-start performance is critical for new users with limited data.
</notes>

---

**Method 16: Neural Oblivious Decision Trees (NODE) for Differentiable Tree Ensembles**

<description>
Neural Oblivious Decision Trees (NODE) combine the representational power of neural networks with the interpretability and feature interaction modeling of decision trees. Unlike standard neural networks, NODE uses oblivious decision trees as building blocks, where each tree uses the same splitting feature at each level across all branches. This architecture is differentiable and can be trained end-to-end with gradient descent while maintaining tree-like interpretability. For LTV prediction, NODE can capture complex feature interactions similar to GBDTs but with the flexibility of deep learning optimization.
</description>

<steps>
1. Initialize NODE with 2048-4096 oblivious trees, each with depth 6-8 levels
2. Configure the feature selection layer to choose splitting features at each tree level using learned attention weights
3. Implement soft decision nodes using sigmoid activation: $p_{\text{right}} = \sigma(\beta \cdot (x_i - \theta))$ where $\beta$ controls decision sharpness
4. Define the tree output as a weighted combination of leaf values based on the probabilities from all decision nodes
5. Train NODE end-to-end using Adam optimizer with the custom AWMSE loss function
6. Apply layer-wise pre-training by initializing tree parameters from a shallow decision tree trained on the data
7. Use ensemble averaging across all trees to generate final predictions: $\hat{y} = \frac{1}{M}\sum_{m=1}^{M} T_m(x)$
8. Extract feature importance by analyzing the learned attention weights at each tree level
</steps>

<notes>
Strengths: Combines the best of neural networks (end-to-end optimization, gradient-based training) and decision trees (interpretability, feature interactions); can outperform GBDTs on complex tabular problems; provides tree-based feature importance.
Weaknesses: Training requires significant GPU resources; sensitive to initialization; more hyperparameters than standard neural networks (tree depth, number of trees, decision sharpness).
When to use: Best for large-scale tabular problems where GBDT performance plateaus; when end-to-end differentiable training is beneficial for integration with other neural components (e.g., embedding layers).
</notes>

---

**Method 17: Temporal Fusion Transformer (TFT) for Multi-Horizon LTV Forecasting**

<description>
Temporal Fusion Transformer (TFT) is a specialized attention-based architecture designed for multi-horizon time series forecasting with interpretability. For LTV prediction, TFT can model the progression from D1 to D60 LTV by learning which historical time steps (D1-D7) are most predictive of the final outcome. TFT uses gating mechanisms to select relevant features, variable selection networks to identify important inputs, and temporal self-attention to capture long-range dependencies. This approach is particularly powerful when both static features (user demographics) and time-varying features (daily behavior) need to be integrated.
</description>

<steps>
1. Organize input data into three categories: static features (MEDIA_SOURCE, COUNTRY), historical time-varying features (D1-D7 behavior), and known future inputs (if applicable)
2. Apply variable selection networks (VSN) to each category to learn feature importance weights
3. Process static features through a fully connected layer to generate a static context vector
4. Encode historical sequences using LSTM layers to capture temporal dependencies
5. Apply multi-head self-attention across the 7-day window to identify critical time steps
6. Implement interpretable multi-horizon decoder that generates predictions for multiple future time points
7. Add gating mechanisms (GLU - Gated Linear Units) at each layer to control information flow
8. Train with quantile loss to generate prediction intervals: $\mathcal{L}_{\text{quantile}}(\tau) = \sum_t \max(\tau(y_t - \hat{y}_t), (\tau-1)(y_t - \hat{y}_t))$
9. Extract attention weights and variable importance scores for model interpretability
</steps>

<notes>
Strengths: State-of-the-art performance on time series forecasting benchmarks; provides rich interpretability through attention visualization and variable selection; naturally handles multi-horizon predictions; produces uncertainty estimates through quantile regression.
Weaknesses: Extremely complex architecture with many hyperparameters; requires significant computational resources; long training time compared to simpler models.
When to use: Best for production systems where both accuracy and interpretability are critical; when stakeholders need to understand which days and features drive predictions; when uncertainty quantification is required for risk management.
</notes>

---

# Part III: Essential Baselines and Emerging Methods

*The following methods complete the methodological coverage by adding classic baselines, cutting-edge architectures, and scenario-specific approaches.*

---

**Method 18: Ridge/Linear Regression Baseline**

<description>
Ridge Regression (L2-regularized linear regression) serves as the fundamental baseline for LTV prediction, establishing the linear relationship between features and the target variable. For signed LTV with asymmetric costs, the standard MSE can be weighted to approximate the AWMSE objective. This method validates whether the feature engineering captures sufficient linear signal and provides an interpretable coefficient-based feature importance analysis that is essential for business stakeholders.
</description>

<steps>
1. Prepare the feature matrix with all engineered features from Method 11, target-encoded categoricals, and temporal aggregates
2. Apply standard scaling (z-score normalization) to all features to ensure comparable coefficient magnitudes
3. Implement sample weighting using the AWMSE weight scheme: $W_{\text{FP}} = 2.5 + 0.02|y|$ for false positives
4. Train Ridge Regression with L2 regularization: $\mathcal{L} = \sum_i W_i(y_i - \hat{y}_i)^2 + \alpha ||\beta||^2_2$
5. Tune the regularization parameter $\alpha$ using cross-validation on validation AWMSE
6. Extract feature importance from the absolute values of learned coefficients: $|importance_j| = |\beta_j \cdot \sigma_j|$
7. Use predictions as a "linear signal" feature for downstream stacking ensembles
</steps>

<notes>
Strengths: Fully interpretable through learned coefficients; extremely fast training and inference; provides upper bound on what linear models can achieve; essential baseline for experimental comparisons.
Weaknesses: Cannot capture non-linear interactions; may underperform on complex feature relationships; coefficient interpretation assumes feature independence.
When to use: Always include as a baseline; useful when interpretability is paramount; provides fast initial predictions for online systems.
</notes>

---

**Method 19: XGBoost with Custom Asymmetric Loss**

<description>
XGBoost is the original gradient boosting framework that introduced regularized learning objectives and column subsampling. While LightGBM often outperforms on large datasets, XGBoost provides complementary predictions due to different tree-building strategies (level-wise vs. leaf-wise). For LTV prediction, XGBoost with custom AWMSE gradients/hessians creates a diverse base learner for ensemble stacking, often excelling on smaller or noisier subsets of the data.
</description>

<steps>
1. Define custom gradient and hessian functions for AWMSE compatible with XGBoost's `obj` parameter
2. Implement the gradient: $g_i = 2 W(y_i, \hat{y}_i) (\hat{y}_i - y_i)$ where $W$ follows the asymmetric weight scheme
3. Implement the hessian: $h_i = 2 W(y_i, \hat{y}_i)$ (constant with respect to prediction)
4. Configure XGBoost with tree parameters: max_depth=6-10, subsample=0.8, colsample_bytree=0.8
5. Enable L1 and L2 regularization (reg_alpha, reg_lambda) to control model complexity
6. Train with early stopping using validation AWMSE as the evaluation metric
7. Extract SHAP values for feature importance analysis: $\phi_j(x) = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[f(S \cup \{j\}) - f(S)]$
8. Use XGBoost predictions as diverse input to the Level 2 meta-learner (Method 8)
</steps>

<notes>
Strengths: Level-wise tree growth often more robust to overfitting than leaf-wise; excellent SHAP value support for interpretability; GPU acceleration available; proven industrial reliability.
Weaknesses: Generally slower than LightGBM on large datasets; requires more memory for histogram computation; custom loss function API slightly more complex than LightGBM.
When to use: As complementary GBDT base learner for stacking; when model diversity in ensembles is needed; when SHAP-based interpretability is required.
</notes>

---

**Method 20: Logistic Regression + Hurdle Framework**

<description>
A lightweight, interpretable alternative to the full Three-Part Hurdle (Method 3) using logistic regression for the classification stages. This approach decomposes the LTV prediction into: (1) Binary classification for $P(\text{LTV} < 0)$ using weighted logistic regression, and (2) Magnitude regression only for predicted positive cases. The simplicity enables fast iteration, A/B testing, and provides clear probability calibration through Platt scaling.
</description>

<steps>
1. Train a Logistic Regression classifier to predict $P(\text{LTV} < 0)$ vs. $P(\text{LTV} \geq 0)$
2. Apply class weighting to penalize false positives: class_weight={0: 1.0, 1: 2.5} (costly users are class 1)
3. Calibrate probabilities using Platt scaling or isotonic regression on the validation set
4. For users with $P(\text{LTV} \geq 0) > \tau$, apply a simple Ridge Regression for magnitude prediction
5. Tune the threshold $\tau$ on the validation set to achieve target FPR < 40%
6. Combine predictions: $\hat{y} = \mathbb{I}(P \geq \tau) \cdot \hat{y}_{\text{magnitude}} - \mathbb{I}(P < \tau) \cdot \hat{y}_{\text{neg\_magnitude}}$
7. Extract odds ratios from logistic coefficients for business-interpretable risk factors
</steps>

<notes>
Strengths: Extremely fast training and inference; fully interpretable through odds ratios; well-calibrated probabilities; easy to deploy in production; minimal hyperparameters.
Weaknesses: Limited capacity for non-linear relationships; requires high-quality feature engineering to perform well; may underperform on complex user segments.
When to use: As a fast baseline; for real-time scoring systems with latency constraints; when model interpretability and auditability are legal requirements.
</notes>

---

**Method 21: Random Forest with Feature Importance Analysis**

<description>
Random Forest is a bagging-based ensemble of decision trees that provides robust predictions through averaging and built-in out-of-bag (OOB) error estimation. Unlike gradient boosting methods that sequentially minimize a loss function, Random Forest builds independent trees on bootstrap samples, making it naturally resistant to overfitting. For LTV prediction, Random Forest excels at identifying important features through permutation importance and can handle the signed LTV distribution without requiring custom loss functions.
</description>

<steps>
1. Configure Random Forest with 500-1000 trees, max_depth=20-30, and min_samples_leaf=5-10
2. Enable bootstrap sampling and set max_features='sqrt' for column subsampling
3. Train on the full feature set including temporal, categorical, and engineered features
4. Compute OOB (Out-of-Bag) error as an unbiased estimate of generalization performance
5. Calculate Permutation Importance by shuffling each feature and measuring AWMSE increase: $I_j = \frac{1}{K}\sum_{k=1}^{K}[\text{AWMSE}_{\text{shuffled}_j} - \text{AWMSE}_{\text{original}}]$
6. Use Mean Decrease Impurity (MDI) as a complementary importance measure
7. Apply sample weights during training to approximate asymmetric loss: weight_i = W(y_i, median(y))
8. Use RF predictions as a diversity-enhancing input to stacking ensemble
</steps>

<notes>
Strengths: Robust to overfitting through bagging; provides OOB error without separate validation set; parallelizable training; excellent permutation importance analysis; handles mixed feature types naturally.
Weaknesses: Cannot directly optimize custom loss functions like AWMSE; predictions are bounded by training data range; high memory usage for large forests.
When to use: For feature importance analysis; as a diverse base learner for ensembles; when robustness to outliers is important; as a sanity check baseline.
</notes>

---

**Method 22: MLP-Mixer for Tabular Data**

<description>
MLP-Mixer is a pure-MLP architecture that achieves competitive performance without attention mechanisms by mixing information across both features (token-mixing) and samples within a batch (channel-mixing). For tabular LTV prediction, the Tabular MLP-Mixer applies two types of dense layers alternately: one that mixes across the feature dimension and another that mixes across the hidden dimension. This architecture is computationally efficient, avoiding the quadratic complexity of self-attention while maintaining representational capacity.
</description>

<steps>
1. Tokenize each input feature into a dense vector of dimension $d$ (e.g., $d=64$): $x_{\text{tokens}} \in \mathbb{R}^{F \times d}$ where $F$ is the number of features
2. Apply the Token-Mixing MLP across the feature dimension: $h_1 = \sigma(W_1 \cdot x_{\text{tokens}}^T) \cdot x_{\text{tokens}}$
3. Apply the Channel-Mixing MLP across the hidden dimension: $h_2 = \sigma(W_2 \cdot h_1)$
4. Stack 4-6 Mixer blocks with residual connections and LayerNorm
5. Apply Global Average Pooling across the feature dimension to obtain a fixed-size representation
6. Add a final MLP regression head optimized with AWMSE loss
7. Train with AdamW optimizer, learning rate warmup, and cosine annealing schedule
8. Apply dropout (0.1-0.3) between Mixer blocks to prevent overfitting
</steps>

<notes>
Strengths: Computationally efficient compared to Transformers; pure-MLP architecture is easy to implement and debug; competitive with attention-based models on tabular data; scalable to high-dimensional feature spaces.
Weaknesses: May require more training data than GBDTs to achieve similar performance; lacks the interpretability of attention weights; hyperparameter-sensitive architecture.
When to use: When seeking a lightweight alternative to Transformers; for rapid prototyping of deep learning baselines; when computational resources are limited.
</notes>

---

**Method 23: Retrieval-Augmented LTV (RALTV)**

<description>
Retrieval-Augmented LTV combines traditional feature-based prediction with retrieval of similar historical users to create a hybrid prediction system. For each target user, the model retrieves the K most similar users from the training set based on behavioral and demographic similarity, then aggregates their historical LTV outcomes to inform the prediction. This approach is particularly effective for cold-start users and captures cohort-level patterns that may not be learned by feature-based models alone.
</description>

<steps>
1. Build a user embedding index using the sequence embeddings from TCN/FT-Transformer (Method 4, 5)
2. Index all training users with their true D60 LTV values using approximate nearest neighbor search (FAISS, Annoy)
3. For each target user, retrieve the K=50-100 most similar historical users based on embedding cosine similarity
4. Compute retrieval-based features: $\bar{y}_{\text{retrieval}} = \frac{1}{K}\sum_{k=1}^{K} y_k^{(\text{hist})}$, $\sigma_{\text{retrieval}} = \text{std}(y_1, ..., y_K)$
5. Calculate retrieval confidence: $c = \frac{1}{K}\sum_{k=1}^{K} \text{sim}(u_{\text{target}}, u_k)$
6. Create signed LTV distribution features: $P(\text{LTV} < 0)_{\text{retrieval}} = \frac{1}{K}\sum_{k=1}^{K}\mathbb{I}(y_k < 0)$
7. Concatenate retrieval features with original features and pass to the final predictor
8. Implement a gating mechanism to blend retrieval predictions with model predictions: $\hat{y}_{\text{final}} = \alpha \cdot \hat{y}_{\text{model}} + (1-\alpha) \cdot \hat{y}_{\text{retrieval}}$
</steps>

<notes>
Strengths: Excellent cold-start performance by leveraging similar users; provides interpretable explanations ("your prediction is based on users like..."); captures long-tail patterns; robust to distribution shift.
Weaknesses: Requires efficient nearest neighbor infrastructure; retrieval quality depends on embedding quality; storage overhead for user index; may leak information if not carefully implemented.
When to use: For cold-start user prediction; when cohort-based patterns are strong; when interpretability through similar users is valuable; for A/B testing new user segments.
</notes>

---

**Method 24: Diffusion Model for LTV Distribution Estimation**

<description>
Diffusion Models learn to generate samples from a target distribution by reversing a gradual noising process. For LTV prediction, a conditional diffusion model can learn the full probability distribution $P(y|\mathbf{x})$ rather than just the point estimate, enabling uncertainty quantification and handling of the heavy-tailed, signed LTV distribution. The model is conditioned on user features and generates multiple LTV samples from which statistics (mean, quantiles, probability of negative) can be computed.
</description>

<steps>
1. Design a conditional U-Net denoiser that takes: noised LTV value $y_t$, noise level $t$, and user features $\mathbf{x}$
2. Implement the forward diffusion process: $y_t = \sqrt{\bar{\alpha}_t} y_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$
3. Train the denoiser to predict the noise: $\mathcal{L} = \mathbb{E}_{t, y_0, \epsilon}[||\epsilon - \epsilon_\theta(y_t, t, \mathbf{x})||^2]$
4. Apply classifier-free guidance by training with dropout on condition features (p=0.1)
5. At inference, sample 100-500 LTV values from the reverse diffusion process
6. Compute prediction statistics: $\hat{y}_{\text{mean}} = \mathbb{E}[\text{samples}]$, $\hat{y}_{0.2} = \text{quantile}_{0.2}(\text{samples})$
7. Estimate $P(\text{LTV} < 0) = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(y_i^{(\text{sample})} < 0)$
8. Use the sampled distribution to make risk-aware decisions with explicit uncertainty bounds
</steps>

<notes>
Strengths: Generates full probability distribution rather than point estimates; naturally handles multi-modal and heavy-tailed distributions; enables principled uncertainty quantification; state-of-the-art for density estimation.
Weaknesses: Computationally expensive at inference (multiple diffusion steps); requires significant training data and GPU resources; complex to implement and tune; slower than direct regression.
When to use: When uncertainty quantification is critical for decision-making; for heavy-tailed LTV distributions; when risk-sensitive predictions are needed; for scenario analysis and simulation.
</notes>

---

**Method 25: Mixture of Experts (MoE) for User Segmentation**

<description>
Mixture of Experts partitions the user population into distinct segments and trains specialized expert models for each segment. A gating network learns to route each user to the appropriate expert(s) based on their features. For LTV prediction, MoE naturally handles user heterogeneity—high-value whales, casual players, and potentially churning users require different prediction strategies. The architecture enables both hard routing (single expert) and soft routing (weighted combination of experts).
</description>

<steps>
1. Design the gating network: $g(\mathbf{x}) = \text{softmax}(W_g \cdot \mathbf{x} + b_g) \in \mathbb{R}^E$ where $E$ is the number of experts
2. Initialize $E=4-8$ expert networks, each a small MLP or LightGBM model
3. Implement sparse routing using top-k gating: only activate top-k experts per sample to reduce computation
4. Define expert outputs: $e_i(\mathbf{x}) = \text{Expert}_i(\mathbf{x})$ for $i \in \{1, ..., E\}$
5. Combine expert predictions: $\hat{y} = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot e_i(\mathbf{x})$
6. Add load balancing loss to ensure all experts are utilized: $\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$ where $f_i$ is fraction routed to expert $i$
7. Train end-to-end with combined loss: $\mathcal{L} = \mathcal{L}_{\text{AWMSE}} + \lambda \mathcal{L}_{\text{balance}}$
8. Analyze expert specialization by examining which user segments are routed to each expert
</steps>

<notes>
Strengths: Explicitly models user heterogeneity; enables specialized predictions for different user segments; scalable through sparse routing; provides interpretability through segment analysis; used by Google, Meta, and other tech giants.
Weaknesses: Gating network may be unstable during training; load balancing is challenging; requires careful initialization; more hyperparameters than single models.
When to use: When user population has distinct segments with different LTV patterns; for large-scale systems where scalability matters; when segment-specific insights are valuable for business.
</notes>

---

**Method 26: Survival Analysis for LTV with Time-to-Event Modeling**

<description>
Survival Analysis models the time until an event (user churn, first purchase, etc.) and can be extended to predict expected LTV by combining churn probability with conditional spending. For mobile gaming LTV, the Cox Proportional Hazards model estimates churn risk as a function of user features, while a separate model predicts spending conditional on survival. This approach is particularly powerful for subscription-based or retention-focused LTV where the lifetime component is as important as the value component.
</description>

<steps>
1. Define the survival outcome: time-to-churn (days until last activity) and event indicator (churned vs. censored)
2. Engineer time-varying covariates from D1-D7 data representing changing engagement levels
3. Fit a Cox Proportional Hazards model: $h(t|\mathbf{x}) = h_0(t) \exp(\beta^T \mathbf{x})$
4. Alternatively, use DeepSurv (neural network survival model) for non-linear hazard functions
5. Estimate survival probability at D60: $S(60|\mathbf{x}) = \exp(-\int_0^{60} h(t|\mathbf{x}) dt)$
6. Train a conditional spending model: $\mathbb{E}[\text{Spending}|T > 60, \mathbf{x}]$ for users who survive
7. Combine survival and spending: $\hat{\text{LTV}} = S(60|\mathbf{x}) \cdot \mathbb{E}[\text{Spending}|T > 60] - (1-S(60|\mathbf{x})) \cdot \mathbb{E}[\text{Loss}|\text{churn}]$
8. Use the hazard function for early warning: identify users with rapidly increasing churn risk
</steps>

<notes>
Strengths: Naturally handles censored data (users still active); provides churn probability curves over time; enables proactive intervention for at-risk users; theoretically grounded in survival statistics.
Weaknesses: Requires well-defined churn definition; may not directly optimize for LTV; censoring patterns affect model quality; Cox model assumes proportional hazards.
When to use: For subscription or retention-focused LTV; when churn prediction is a business priority; when time-to-event patterns are important; for proactive user intervention strategies.
</notes>

---

**Method 27: Reinforcement Learning for Intervention-Aware LTV**

<description>
Traditional LTV models predict "natural" LTV assuming no intervention, but real-world systems apply retention campaigns, bonuses, and targeted offers. Reinforcement Learning (RL) models the LTV as a function of both user state and actions taken by the system, enabling prediction of "intervention-aware" LTV. This approach uses historical intervention data to learn counterfactual LTV outcomes and optimize for maximum long-term value through sequential decision-making.
</description>

<steps>
1. Define the state space $\mathcal{S}$: user features, current engagement, historical LTV, and days since install
2. Define the action space $\mathcal{A}$: no action, bonus offer, push notification, personalized content
3. Model the transition dynamics: $P(s_{t+1}|s_t, a_t)$ representing how user state evolves given actions
4. Define the reward function: $r_t = \text{daily\_LTV}_t - \text{action\_cost}_t$
5. Train a Q-network to estimate action-value: $Q(s, a) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a]$
6. Use offline RL algorithms (CQL, IQL) to learn from logged historical intervention data
7. Estimate counterfactual LTV: $\hat{\text{LTV}}_{\text{no\_intervention}} = Q(s, a_{\text{null}})$
8. Optimize intervention policy: $\pi^*(s) = \arg\max_a Q(s, a)$
9. Predict intervention-aware LTV: $\hat{\text{LTV}}_{\pi^*} = \mathbb{E}_{a \sim \pi^*}[Q(s, a)]$
</steps>

<notes>
Strengths: Captures causal effect of interventions on LTV; enables optimal policy learning; provides counterfactual "what-if" analysis; aligns prediction with decision-making.
Weaknesses: Requires logged intervention data; offline RL is challenging with limited data; action space must be well-defined; distribution shift between logged and optimal policies.
When to use: When intervention policies significantly affect LTV; for optimizing retention campaigns; when counterfactual analysis is needed; for A/B test design.
</notes>

---

**Method 28: Self-Supervised Pre-training for Cold-Start Users**

<description>
Self-Supervised Pre-training learns representations from unlabeled user behavior data before fine-tuning on labeled LTV prediction. For cold-start users with limited history, pre-trained representations capture general behavioral patterns that transfer to LTV prediction. Contrastive learning objectives (SimCLR, MoCo) and masked feature prediction enable the model to learn robust features from the abundant unlabeled engagement data available before D60 labels are observed.
</description>

<steps>
1. Create data augmentations for user sequences: random feature masking (30%), temporal jittering, feature dropout
2. Implement contrastive learning: positive pairs are augmented views of the same user, negatives are other users
3. Define the contrastive loss: $\mathcal{L}_{\text{contrast}} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)}$
4. Alternatively, use masked feature prediction: randomly mask 30% of features and train to reconstruct
5. Pre-train the encoder on all available user data (including users without D60 labels yet)
6. Extract the learned encoder weights and freeze or fine-tune on the labeled LTV prediction task
7. For new users at D1, use the pre-trained encoder to generate embeddings for immediate LTV estimation
8. Continuously update pre-training with new user data to capture evolving behavioral patterns
</steps>

<notes>
Strengths: Leverages abundant unlabeled data; improves cold-start performance significantly; learned representations transfer across tasks; reduces labeling requirements.
Weaknesses: Pre-training is computationally expensive; augmentation design requires domain knowledge; may not improve much if labeled data is abundant; two-stage training adds complexity.
When to use: For cold-start LTV prediction with limited labels; when unlabeled behavioral data is abundant; for transfer learning across games/products; when D60 labels are sparse.
</notes>

---

**Method 29: Recommended Production Pipeline (Summary)**

<description>
The comprehensive production pipeline integrates all 28 methods into a unified, risk-aware system. It combines classical baselines (Methods 18-21) for interpretability, modern deep learning (Methods 22-25) for capacity, scenario-specific approaches (Methods 26-28) for business alignment, and the original core methods (1-17) for proven performance. The final system ensures maximum accuracy while guaranteeing FPR compliance through calibration.
</description>

<steps>
1. Implement baselines: Ridge Regression (18), XGBoost (19), Logistic+Hurdle (20), Random Forest (21)
2. Extract temporal features using the 7-Day Compression Toolkit (Method 11)
3. Train GBDT foundations: Custom AWMSE LightGBM (1), CatBoost (14), XGBoost (19)
4. Train deep sequence encoders: TCN (4), FT-Transformer (5), BiLSTM+Attention (12), TFT (17)
5. Apply modern architectures: TabNet (13), NODE (16), MLP-Mixer (22), MoE (25)
6. Implement specialized approaches: Diffusion Model (24), Survival Analysis (26), RL (27)
7. Add retrieval augmentation (23) and self-supervised pre-training (28) for cold-start users
8. Construct user similarity graph and train GNN (15) for cohort-level signals
9. Apply Hybrid Feature Fusion (7) combining all embeddings with manual features
10. Add Risk-Averse Quantile Regression (2) for uncertainty estimation
11. Build Three-Part Hurdle Framework (3) for explicit sign control
12. Train Multi-Task Learning DNN (10) with joint optimization
13. Combine all models in Level 2 Stacking (8) with AWMSE Meta-Optimization
14. Apply Dynamic Conservative Calibration (9) to ensure FPR $< 40\%$
15. Export $P(\text{LTV} < 0)$ logit score to UA system for cohort filtering
</steps>

<notes>
Strengths: Comprehensive coverage of classical, modern, and scenario-specific methods; provides baselines for comparison; handles cold-start and interventions; ensures business compliance.
Weaknesses: Extremely high complexity; requires significant computational resources; difficult to maintain; long training pipeline.
When to use: Production deployment requiring maximum accuracy and full methodological coverage.
</notes>

---
