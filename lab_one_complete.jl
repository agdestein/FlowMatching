# %% [markdown]
# # Lab One: Simulating ODEs and SDEs

# %% [markdown]
# Welcome to lab one! In this lab, we will provide an intuitive and hands-on walk-through of ODEs and SDEs. If you find any mistakes, or have any other feedback, please feel free to email us at `erives@mit.edu` and `phold@mit.edu`. Enjoy!

# %%
# Imports

using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using WGLMakie

# %% [markdown]
# # Part 0: Introduction

# %% [markdown]
# First, let us make precise the central objects of study: *ordinary differential equations* (ODEs) and *stochastic differential equations* (SDEs). The basis of both ODEs and SDEs are time-dependent *vector fields*, which we recall from lecture as being functions $u$ defined by $$u:\mathbb{R}^d\times [0,1]\to \mathbb{R}^d,\quad (x,t)\mapsto u_t(x)$$
# That is, $u_t(x)$ takes in *where in space we are* ($x$) and *where in time we are* ($t$), and spits out the *direction we should be going in* $u_t(x)$. An ODE is then given by $$d X_t = u_t(X_t)dt, \quad \quad X_0 = x_0.$$
# Similarly, an SDE is of the form $$d X_t = u_t(X_t)dt + \sigma_t d W_t, \quad \quad X_0 = x_0,$$
# which can be thought of as starting with an ODE given by $u_t$, and adding noise via the *Brownian motion* $(W_t)_{0 \le t \le 1}$. The amount of noise that we add is given by the *diffusion coefficient* $\sigma_t$.

# %% [markdown]
# **Note**: One might consider an ODE to be a special case of SDEs with zero diffusion coefficient. This intuition is valid, however for pedagogical (and performance) reasons, we will treat them separately for the scope of this lab.

# %% [markdown]
# # Part 1: Numerical Methods for Simulating ODEs and SDEs
# We may think of ODEs and SDEs as describing the motion of a particle through space. Intuitively, the ODE above says "start at $X_0=x_0$", and move so that your instantaneous velocity is given by $u_t(X_t)$. Similarly, the SDE says "start at $X_0=x_0$", and move so that your instantaneous velocity is given by $u_t(X_t)$ plus a little bit of random noise given scaled by $\sigma_t$. Formally, these trajectories traced out by this intuitive descriptions are said to be *solutions* to the ODEs and SDEs, respectively. Numerical methods for computing these solutions are all essentially based on *simulating*, or *integrating*, the ODE or SDE.
#
# In this section we'll implement the *Euler* and *Euler-Maruyama* numerical simulation schemes for integrating ODEs and SDEs, respectively. Recall from lecture that the Euler simulation scheme corresponds to the discretization
# $$d X_t = u_t(X_t) dt  \quad \quad \rightarrow \quad \quad X_{t + h} = X_t + hu_t(X_t),$$
# where $h = \Delta t$ is the *step size*. Similarly, the Euler-Maruyama scheme corresponds to the discretization
# $$ dX_t = u(X_t,t) dt + \sigma_t d W_t  \quad \quad \rightarrow \quad \quad X_{t + h} = X_t + hu_t(X_t) + \sqrt{h} \sigma_t z_t, \quad z_t \sim N(0,I_d).$$
# Let's implement these!

# %%
function simulate(step, x, ts)
    for t_idx in eachindex(ts)
        t = ts[t_idx]
        h = ts[t_idx+1] - ts[t_idx]
        x = step(x, t, h)
    end
    x
end

function simulate_with_trajectory(step, x, ts)
    xs = [x]
    for t_idx = 1:length(ts)-1
        t = ts[t_idx]
        h = ts[t_idx+1] - ts[t_idx]
        x = step(x, t, h)
        push!(xs, x)
    end
    stack(xs)
end

# %% [markdown]
# ### Question 1.1: Integrate EulerSimulator and EulerMaruyamaSimulator

# %% [markdown]
# **Your job**: Fill in the `step` methods of `EulerSimulator` and `EulerMaruyamaSimulator`.

# %%
euler_step(drift, xt, t, h) = xt + drift(xt, t) * h

# %%
function euler_maruyama_step(drift, diff, xt, t, h)
    a = drift(xt, t)
    b = diff(xt, t)
    @. xt + a * h + b * sqrt(h) * randn()
end

# %% [markdown]
# **Note:** When the diffusion coefficient is zero, the Euler and Euler-Maruyama simulation are equivalent!

# %% [markdown]
# # Part 2: Visualizing Solutions to SDEs
# Let's get a feel for what the solutions to these SDEs look like in practice (we'll get to ODEs later...). To do so, we we'll implement and visualize two special choices of SDEs from lecture: a (scaled) *Brownian motion*, and an *Ornstein-Uhlenbeck* (OU) process.

# %% [markdown]
# ### Question 2.1: Implementing Brownian Motion
# First, recall that a Brownian motion is recovered (by definition) by setting $u_t = 0$ and $\sigma_t = \sigma$, viz.,
# $$ dX_t = \sigma dW_t, \quad \quad X_0 = 0.$$

# %% [markdown]
# **Your job**: Intuitively, what might be expect the trajectories of $X_t$ to look like when $\sigma$ is very large? What about when $\sigma$ is close to zero?
#
# **Your answer**:

# %% [markdown]
# **Your job**: Fill in the `drift_coefficient` and `difusion_coefficient` methods of the `BrownianMotion` class below.

# %% [markdown]
# Now let's plot! We'll make use of the following utility function.

# %%

let
    σ = 1.0
    ts = range(0.0, 5.0, 501) # simulation timesteps
    x0 = 0.0 # Initial values - let's start at zero
    trajectories = map(1:5) do i
        simulate_with_trajectory(
            (x, t, h) -> euler_maruyama_step((x, t) -> zero(x), Returns(σ), x, t, h),
            x0,
            ts,
        )
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title = "Trajectories of Brownian Motion with σ = $σ",
        xlabel = "Time (t)",
        ylabel = "Xt",
    )
    for x in trajectories
        lines!(ax, ts, x)
    end
    fig
end

# %% [markdown]
# **Your job**: What happens when you vary the value of `sigma`?
#
# **Your answer**:

# %% [markdown]
# ### Question 2.2: Implementing an Ornstein-Uhlenbeck Process
# An OU process is given by setting $u_t(X_t) = - \theta X_t$ and $\sigma_t = \sigma$, viz.,
# $$ dX_t = -\theta X_t\, dt + \sigma\, dW_t, \quad \quad X_0 = x_0.$$

# %% [markdown]
# **Your job**: Intuitively, what would the trajectory of $X_t$ look like for a very small value of $\theta$? What about a very large value of $\theta$?
#
# **Your answer**:

# %% [markdown]
# **Your job**: Fill in the `drift_coefficient` and `difusion_coefficient` methods of the `OUProcess` class below.

# %%
# Try comparing multiple choices side-by-side
let
    θσ = [(0.25, 0.0), (0.25, 0.25), (0.25, 0.5), (0.25, 1.0)]
    simulation_time = 20.0
    num_plots = length(θσ)
    fig = Figure(; size = (num_plots * 250, 300))
    for (idx, θσ) in enumerate(θσ)
        θ, σ = θσ
        ax = Axis(
            fig[1, idx];
            title = "θ = $θ, σ = $σ",
            xlabel = "Time (t)",
            ylabel = "Xt",
            ylabelvisible = idx == 1,
            yticklabelsvisible = idx == 1,
        )
        ylims!(ax, -11, 11)
        x0 = range(-10.0, 10.0, 10) |> collect # Initial values - let's start at zero
        ts = range(0.0, simulation_time, 1000) # simulation timesteps
        for x0 in x0
            x = simulate_with_trajectory(
                (x, t, h) ->
                    euler_maruyama_step((x, t) -> -θ * x, (x, t) -> σ, x, t, h),
                x0,
                ts,
            )
            lines!(ax, ts, x)
        end
    end
    Label(fig[0, :]; text = "Trajectories of OU Process")
    fig
end

# %% [markdown]
# **Your job**: What do you notice about the convergence of the solutions? Are they converging to a particular point? Or to a distribution? Your answer should be two *qualitative* sentences of the form: "When ($\theta$ or $\sigma$) goes (up or down), we see...".
#
# **Hint**: Pay close attention to the ratio $D \triangleq \frac{\sigma^2}{2\theta}$ (see the next few cells below!).
#
# **Your answer**:

# %%
# Let's try rescaling with time
let
    σ = [1.0, 2.0, 10.0]
    d = [0.25, 1.0, 4.0] # sigma**2 / 2t
    simulation_time = 10.0
    nd = length(d)
    nσ = length(σ)
    fig = Figure(; size = (nd * 250, nσ * 200))
    for (id, d) in enumerate(d), (iσ, σ) in enumerate(σ)
        θ = σ^2 / 2 / d
        time_scale = σ^2
        x0 = range(-20.0, 20.0, 20) |> collect # Initial values - let's start at zero
        ts = range(0.0, simulation_time / time_scale, 1000) # simulation timesteps
        trajectories = map(
            x0 -> simulate_with_trajectory(
                (x, t, h) ->
                    euler_maruyama_step((x, t) -> -θ * x, (x, t) -> σ, x, t, h),
                x0,
                ts,
            ),
            x0,
        )
        ax = Axis(
            fig[id, iσ];
            title = "d = $d, σ = $σ",
            xlabel = "Time (t / σ^2)",
            ylabel = "Xt",
            xlabelvisible = id == nd,
            xticklabelsvisible = id == nd,
            ylabelvisible = iσ == 1,
            yticklabelsvisible = iσ == 1,
        )
        ylims!(ax, -21, 21)
        for x in trajectories
            lines!(ax, ts * time_scale, x)
        end
    end
    Label(fig[0, :]; text = "Trajectories of OU Process")
    fig
end

# %% [markdown]
# **Your job**: What conclusion can we draw from the figure above? One qualitative sentence is fine. We'll revisit this in Section 3.2.
#
# **Your answer**:

# %% [markdown]
# # Part 3: Transforming Distributions with SDEs
# In the previous section, we observed how individual *points* are transformed by an SDE. Ultimately, we are interested in understanding how *distributions* are transformed by an SDE (or an ODE...). After all, our goal is to design ODEs and SDEs which transform a noisy distribution (such as the Gaussian $N(0, I_d)$), to the data distribution $p_{\text{data}}$ of interest. In this section, we will visualize how distributions are transformed by a very particular family of SDEs: *Langevin dynamics*.
#
# First, let's define some distributions to play around with. In practice, there are two qualities one might hope a distribution to have:
# 1. The first quality is that one can measure the *density* of a distribution $p(x)$. This ensures that we can compute the gradient $\nabla \log p(x)$ of the log density. This quantity is known as the *score* of $p$, and paints a picture of the local geometry of the distribution. Using the score, we will construct and simulate the *Langevin dynamics*, a family of SDEs which "drive" samples toward the distribution $\pi$. In particular, the Langevin dynamics *preserve* the distribution $p(x)$. In Lecture 2, we will make this notion of driving more precise.
# 2. The second quality is that we can draw samples from the distribution $p(x)$.
# For simple, toy distributions, such as Gaussians and simple mixture models, it is often true that both qualities are satisfied. For more complex choices of $p$, such as distributions over images, we can sample but cannot measure the density.

score(density, x) = ForwardDiff.gradient(x -> logpdf(density, x), x)

function random_2D(nmodel, std, scale = 10.0)
    dim = 2
    models = map(1:nmodel) do i
        r = rand(dim)
        mean = @. (r - 0.5) * scale
        cov = std^2 * I(dim)
        MvNormal(mean, cov)
    end
    MixtureModel(models)
end

function symmetric_2D(n, std, scale = 10.0)
    angles = range(0, 2π, n + 1)[2:end]
    dim = 2
    models = map(angles) do θ
        mean = scale * [cos(θ), sin(θ)]
        cov = Diagonal(fill(std^2, dim))
        MvNormal(mean, cov)
    end
    MixtureModel(models)
end

# %%
# Visualize densities
let
    distributions = [
        ("Gaussian", MvNormal(zeros(2), 10.0 * I(2))),
        ("Random Mixture", random_2D(5, 1.0, 20.0)),
        ("Symmetric Mixture", symmetric_2D(5, 1.0, 8.0)),
    ]
    fig = Figure(; size = (300 * length(distributions), 400))
    bins = 100
    scale = 15
    for (idx, distribution) in enumerate(distributions)
        title, distribution = distribution
        ax = Axis(fig[1, idx]; title)
        x = range(-scale, scale, bins)
        y = reshape(x, 1, :)
        p = broadcast((x, y) -> logpdf(distribution, [x, y]), x, y)
        m = maximum(p)
        heatmap!(ax, x, x, p; colorrange = (-15, m), colormap = :Blues)
        contour!(ax, x, x, p; color = :gray, levels = 20)
    end
    fig
end

# %% [markdown]
# ### Question 3.1: Implementing Langevin Dynamics

# %% [markdown]
# In this section, we'll simulate the (overdamped) Langevin dynamics $$dX_t = \frac{1}{2} \sigma^2\nabla \log p(X_t) dt + \sigma dW_t,$$.
#
# **Your job**: Fill in the `drift_coefficient` and `diffusion_coefficient` methods of the class `LangevinSDE` below.

# %%

# %% [markdown]
# Now, let's graph the results!

let
    # Construct the simulator
    target = random_2D(5, 0.75, 15.0)
    # Graph the results!
    num_samples = 1000
    source = MvNormal(zeros(2), 20.0 * I(2))
    density = target
    plot_every = 333
    nplot = 4
    timesteps = range(0.0, 5.0, (nplot - 1) * plot_every + 1)
    bins = 200
    scale = 15
    # Simulate
    σ = 0.6
    trajectories = map(1:num_samples) do i
        x0 = rand(source)
        simulate_with_trajectory(
            (x, t, h) -> euler_maruyama_step(
                (xt, t) -> 0.5 * σ^2 * score(target, xt),
                (xt, t) -> σ,
                x,
                t,
                h,
            ),
            x0,
            timesteps,
        )
    end |> stack
    fig = Figure(; size = (250 * nplot, 300))
    for (iplot, i) in enumerate(1:plot_every:length(timesteps))
        ax = Axis(fig[1, iplot])
        let
            bins = 100
            scale = 15
            x = range(-scale, scale, bins)
            y = reshape(x, 1, :)
            p = broadcast((x, y) -> logpdf(target, [x, y]), x, y)
            m = maximum(p)
            heatmap!(ax, x, x, p; colorrange = (-15, m), colormap = :Blues)
        end
        t = timesteps[i]
        xt = trajectories[:, i, :]
        scatter!(ax, xt[1, :], xt[2, :]; color = :black, markersize = 5, alpha = 0.75)
    end
    fig
end

# %% [markdown]
# **Your job**: Try varying the value of $\sigma$, the number and range of the simulation steps, the source distribution, and target density. What do you notice? Why?
#
# **Your answer**:

# %% [markdown]
# ### Question 3.2: Ornstein-Uhlenbeck as Langevin Dynamics
# In this section, we'll finish off with a brief mathematical exercise connecting Langevin dynamics and Ornstein-Uhlenbeck processes. Recall that for (suitably nice) distribution $p$, the *Langevin dynamics* are given by
# $$dX_t = \frac{1}{2} \sigma^2\nabla \log p(X_t) dt + \sigma\, dW_t, \quad \quad X_0 = x_0,$$
# while for given $\theta, \sigma$, the Ornstein-Uhlenbeck process is given by
# $$ dX_t = -\theta X_t\, dt + \sigma\, dW_t, \quad \quad X_0 = x_0.$$

# %% [markdown]
# **Your job**: Show that when $p(x) = N(0, \frac{\sigma^2}{2\theta})$, the score is given by $$\nabla \log p(x) = -\frac{2\theta}{\sigma^2}x.$$
#
# **Hint**: The probability density of the Gaussian $p(x) = N(0, \frac{\sigma^2}{2\theta})$ is given by $$p(x)  = \frac{\sqrt{\theta}}{\sigma\sqrt{\pi}} \exp\left(-\frac{x^2\theta}{\sigma^2}\right).$$
#
# **Your answer**: From the hint,
# $$\log p(x) = - \frac{x^2\theta}{\sigma^2} + C.$$
# Thus, $\nabla \log p(x) = \frac{d}{dx} \log p(x)$ is given by
# $$ \frac{d}{dx} \left(- \frac{\theta}{\sigma^2}x^2\right) = \boxed{- \frac{2\theta}{\sigma^2}x}.$$

# %% [markdown]
# **Your job**: Conclude that when $p(x) = N(0, \frac{\sigma^2}{2\theta})$, the Langevin dynamics
# $$dX_t = \frac{1}{2} \sigma^2\nabla \log p(X_t) dt + \sigma dW_t,$$
# is equivalent to the Ornstein-Uhlenbeck process
# $$ dX_t = -\theta X_t\, dt + \sigma\, dW_t, \quad \quad X_0 = 0.$$
#
# **Your answer**: Just plug in the previous result.
