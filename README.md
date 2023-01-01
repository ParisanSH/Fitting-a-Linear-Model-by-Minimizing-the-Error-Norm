# Fitting-a-Linear-Model-by-Minimizing-the-Error-Norm
In this project, I have tried to fit a linear model (𝑤𝑥 + 𝑏) on my own generated dataset, by minimizing the error norm (𝑙0, 𝑙1, 𝑙2,𝑙 − 𝑖𝑛𝑓𝑖𝑛𝑖𝑡𝑦)

# How data has been created:
𝒏 numbers (float or integer) have been selected randomly from [1,200] as 𝑋. One arbitrary set of 𝒘, 𝒃 have been chosen, and the target values Y (𝑦 = 𝑤𝑥 + 𝑏) have been calculated, then some noise 𝜀 was added to the target values (E.g. 𝜀~𝒩(0,1)).

# Steps:
1. The 𝑤′, 𝑏′ have been calculated from data by reducing the error norms (𝑙0, 𝑙1, 𝑙2, 𝑙 − 𝑖𝑛𝑓𝑖𝑛𝑖𝑡𝑦).
2. 𝒏 has been chosen from [2, 20, 50, 100] (This is going to make 16 combinations regarding different kinds of norms and 𝒏).
3. Data was plotted, and the found model using error minimization has been shown.
4. The MSE (Mean Square Error) has been calculated for each configuration. (what we are taking as the target here is important).
