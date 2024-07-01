# -----

## optimizer

### Adadelta

$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
\end{aligned}$$

### adagrad

$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow 0                             \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
\end{aligned}$$

### Adam algorithm

$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
 \end{aligned}$$

### Implements Adamax algorithm (a variant of Adam based on infinity norm).
$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}$$
### AdamW algorithm

$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
\end{aligned}$$
### NAdam
$$\begin{aligned}
&\rule{110mm}{0.4pt} \\
&\textbf{input} : \gamma t \text{ (lr)}, \ \beta 1, \ \beta 2 \text{ (betas)}, \ \theta 0 \text{ (params)}, \ f(\theta) \text{ (objective)} \\
&\hspace{13mm} \lambda \text{ (weight decay)}, \ \psi \text{ (momentum decay)}, \ \textit{decoupled weight decay} \\
&\textbf{initialize} : m0 \leftarrow 0 \text{ (first moment)}, \ v0 \leftarrow 0 \text{ (second moment)} \\
&\rule{110mm}{0.4pt} \\
&\textbf{for} \ t = 1 \ \textbf{to} \ \ldots \ \textbf{do} \\
&\hspace{5mm} g t \leftarrow \nabla_{\theta} f t (\theta_{t-1}) \\
&\hspace{5mm} \theta t \leftarrow \theta_{t-1} \\
&\hspace{5mm} \textbf{if} \ \lambda \neq 0 \\
&\hspace{10mm} \textbf{if} \ \textit{decoupled weight decay} \\
&\hspace{15mm} \theta t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1} \\
&\hspace{10mm} \textbf{else} \\
&\hspace{15mm} g t \leftarrow g t + \lambda \theta_{t-1} \\
&\hspace{5mm} \mu t \leftarrow \beta 1 \left(1 - \frac{1}{2} \cdot 0.96^{t \psi} \right) \\
&\hspace{5mm} \mu_{t+1} \leftarrow \beta 1 \left(1 - \frac{1}{2} \cdot 0.96^{(t+1) \psi} \right) \\
&\hspace{5mm} m t \leftarrow \beta 1 m_{t-1} + (1 - \beta 1) g t \\
&\hspace{5mm} v t \leftarrow \beta 2 v_{t-1} + (1 - \beta 2) g^2 t \\
&\hspace{5mm} \widehat{m t} \leftarrow \frac{\mu_{t+1} m t}{1 - \prod_{i=1}^{t+1} \mu i} + \frac{(1 - \mu t) g t}{1 - \prod_{i=1}^{t} \mu i} \\
&\hspace{5mm} \widehat{v t} \leftarrow \frac{v t}{1 - \beta 2^t} \\
&\hspace{5mm} \theta t \leftarrow \theta t - \gamma \frac{\widehat{m t}}{\sqrt{\widehat{v t}} + \epsilon} \\
&\rule{110mm}{0.4pt} \\
&\textbf{return} \ \theta t \\
&\rule{110mm}{0.4pt}
\end{aligned}$$

### RAdam

$$\begin{aligned}
&\rule{110mm}{0.4pt} \\
&\textbf{input} : \gamma \text{ (lr)}, \ \beta 1, \ \beta 2 \text{ (betas)}, \ \theta 0 \text{ (params)}, \ f(\theta) \text{ (objective)}, \ \lambda \text{ (weight decay)}, \ \epsilon \text{ (epsilon)}, \ \textit{decoupled weight decay} \\
&\textbf{initialize} : m0 \leftarrow 0 \text{ (first moment)}, \ v0 \leftarrow 0 \text{ (second moment)}, \ \rho_{\infty} \leftarrow \frac{2}{1 - \beta 2} - 1 \\
&\rule{110mm}{0.4pt} \\
&\textbf{for} \ t = 1 \ \textbf{to} \ \ldots \ \textbf{do} \\
&\hspace{6mm} g t \leftarrow \nabla_{\theta} f t (\theta_{t-1}) \\
&\hspace{6mm} \theta t \leftarrow \theta_{t-1} \\
&\hspace{6mm} \textbf{if} \ \lambda \neq 0 \\
&\hspace{12mm} \textbf{if} \ \textit{decoupled weight decay} \\
&\hspace{18mm} \theta t \leftarrow \theta t - \gamma \lambda \theta t \\
&\hspace{12mm} \textbf{else} \\
&\hspace{18mm} g t \leftarrow g t + \lambda \theta t \\
&\hspace{6mm} m t \leftarrow \beta 1 m_{t-1} + (1 - \beta 1) g t \\
&\hspace{6mm} v t \leftarrow \beta 2 v_{t-1} + (1 - \beta 2) g^2 t \\
&\hspace{6mm} \widehat{m t} \leftarrow \frac{m t}{1 - \beta 1^t} \\
&\hspace{6mm} \rho t \leftarrow \rho_{\infty} - \frac{2t \beta 2^t}{1 - \beta 2^t} \\
&\hspace{6mm} \textbf{if} \ \rho t > 5 \\
&\hspace{12mm} l t \leftarrow \frac{\sqrt{1 - \beta 2^t}}{\sqrt{v t} + \epsilon} \\
&\hspace{12mm} r t \leftarrow \sqrt{\frac{(\rho t - 4)(\rho t - 2) \rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} - 2) \rho t}} \\
&\hspace{12mm} \theta t \leftarrow \theta t - \gamma \widehat{m t} r t l t \\
&\hspace{6mm} \textbf{else} \\
&\hspace{12mm} \theta t \leftarrow \theta t - \gamma \widehat{m t} \\
&\rule{110mm}{0.4pt} \\
&\textbf{return} \ \theta t \\
&\rule{110mm}{0.4pt}
\end{aligned}$$

### RMSprop

$$\begin{aligned}
&\rule{110mm}{0.4pt} \\
&\textbf{input} : \alpha \text{ (alpha)}, \ \gamma \text{ (lr)}, \ \theta 0 \text{ (params)}, \ f(\theta) \text{ (objective)}, \ \lambda \text{ (weight decay)}, \ \mu \text{ (momentum)}, \ \textit{centered} \\
&\textbf{initialize} : v 0 \leftarrow 0 \text{ (square average)}, \ \textbf{b}0 \leftarrow 0 \text{ (buffer)}, \ g^{\text{ave}} 0 \leftarrow 0 \\
&\rule{110mm}{0.4pt} \\
&\textbf{for} \ t = 1 \ \textbf{to} \ \ldots \ \textbf{do} \\
&\hspace{5mm} g t \leftarrow \nabla_{\theta} f t (\theta t - 1) \\
&\hspace{5mm} \textbf{if} \ \lambda \neq 0 \\
&\hspace{10mm} g t \leftarrow g t + \lambda \theta t - 1 \\
&\hspace{5mm} v t \leftarrow \alpha v t - 1 + (1 - \alpha) g^2 t \\
&\hspace{5mm} \textbf{if} \ \textit{centered} \\
&\hspace{10mm} g^{\text{ave}} t \leftarrow \mu g^{\text{ave}} t - 1 + (1 - \mu) g t \\
&\hspace{10mm} \widehat{v} t \leftarrow v t - \left(g^{\text{ave}} t \right)^2 \\
&\hspace{5mm} \textbf{if} \ \mu > 0 \\
&\hspace{10mm} \textbf{b} t \leftarrow \mu \textbf{b} t - 1 + \frac{g t}{\sqrt{\widehat{v} t} + \epsilon} \\
&\hspace{10mm} \theta t \leftarrow \theta t - 1 - \gamma \textbf{b} t \\
&\hspace{5mm} \textbf{else} \\
&\hspace{10mm} \theta t \leftarrow \theta t - 1 - \gamma \frac{g t}{\sqrt{\widehat{v} t} + \epsilon} \\
&\rule{110mm}{0.4pt} \\
&\textbf{return} \ \theta t \\
&\rule{110mm}{0.4pt}
\end{aligned}$$

### resilientbackpropagation

$$\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
\end{aligned}$$
### stochastic gradient descent (optionally with momentum)
$$\begin{aligned}
&\rule{110mm}{0.4pt} \\
&\textbf{input} : \gamma \text{ (lr)}, \ \theta 0 \text{ (params)}, \ f(\theta) \text{ (objective)}, \ \lambda \text{ (weight decay)}, \ \mu \text{ (momentum)}, \ \tau \text{ (dampening)}, \ \textit{nesterov}, \ \textit{maximize} \\
&\rule{110mm}{0.4pt} \\
&\textbf{for} \ t = 1 \ \textbf{to} \ \ldots \ \textbf{do} \\
&\hspace{5mm} g t \leftarrow \nabla_{\theta} f t (\theta_{t-1}) \\
&\hspace{5mm} \textbf{if} \ \lambda \neq 0 \\
&\hspace{10mm} g t \leftarrow g t + \lambda \theta_{t-1} \\
&\hspace{5mm} \textbf{if} \ \mu \neq 0 \\
&\hspace{10mm} \textbf{if} \ t > 1 \\
&\hspace{15mm} b t \leftarrow \mu b t - 1 + (1 - \tau) g t \\
&\hspace{10mm} \textbf{else} \\
&\hspace{15mm} b t \leftarrow g t \\
&\hspace{10mm} \textbf{if} \ \textit{nesterov} \\
&\hspace{15mm} g t \leftarrow g t + \mu b t \\
&\hspace{10mm} \textbf{else} \\
&\hspace{15mm} g t \leftarrow b t \\
&\hspace{5mm} \textbf{if} \ \textit{maximize} \\
&\hspace{10mm} \theta t \leftarrow \theta t - 1 + \gamma g t \\
&\hspace{5mm} \textbf{else} \\
&\hspace{10mm} \theta t \leftarrow \theta t - 1 - \gamma g t \\
&\rule{110mm}{0.4pt} \\
&\textbf{return} \ \theta t \\
&\rule{110mm}{0.4pt}
\end{aligned}$$

>>>>>>> origin/main
