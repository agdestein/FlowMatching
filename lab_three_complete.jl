# %% [markdown]
# # Lab 3: A Conditional Generative Model for Images
# Welcome to lab 3! In the previous lab, we studied *unconditional* generation, for toy, two-dimensional data distributions. In this lab, we will study *conditional* generation on *images* from the MNIST dataset of handwritten digits. Each such MNIST image is not two dimensions but $32\times 32 = 1024$ dimensions! The nature of our new, more challenging setting will require us to take special care:
# 1. To tackle *conditional* generation, we will employ *classifier-free guidance* (CFG) (see Part 2.1).
# 2. To parameterize our learned vector field for high-dimensional image-valued data, a simple MLP will not suffice. Instead, we will adopt the *U-Net* architecture (see part 2.2).
#
# If you find any mistakes, or have any other feedback, please feel free to email us at `erives@mit.edu` and `phold@mit.edu`. Enjoy!

using Adapt
using CairoMakie
using Distributions
using ForwardDiff
using LinearAlgebra
using Lux
# using LuxCUDA
using NNlib
using Optimisers
using Random
using WGLMakie
using Zygote
using MLDatasets
using MLUtils

outdir = joinpath(@__DIR__, "output", "lab_three") |> mkpath
rng = Random.Xoshiro(0)

# %% [markdown]
# # Part 1: Getting a Feel for MNIST
# In this section, we'll get a feel for MNIST. We'll then experiment with adding noise to MNIST.

# %% [markdown]
# Now let's view some samples under the conditional probability path.
#
let
    data = MNIST(:train)
    nstep = 5
    n = 3
    fig = Figure(; size = (nstep * 200, 200))
    x0 = [randn(28, 28) for j = 1:n, i = 1:n]
    for (istep, t) in enumerate(range(0, 1, nstep))
        g = GridLayout(fig[1, istep])
        for i = 1:n, j = 1:n
            ax = Axis(
                g[i, j];
                aspect = DataAspect(),
                xticksvisible = false,
                yticksvisible = false,
                xticklabelsvisible = false,
                yticklabelsvisible = false,
            )
            z = data[i+n*(j-1)].features
            @. z = (z - 0.5) / 0.5
            x0ij = x0[i, j]
            x = @. (1 - t) * x0ij + t * z
            x = reverse(x; dims = 2)
            image!(ax, x; interpolate = false, colorrange = (-1, 1))
        end
        colgap!(g, 0)
        rowgap!(g, 0)
    end
    fig
end

# %% [markdown]
# # Part 2: Classifier Free Guidance

# %% [markdown]
# ### Problem 2.1: Classifier Free Guidance

# %% [markdown]
# **Guidance**: Whereas for unconditional generation, we simply wanted to generate *any* digit, we would now like to be able to specify, or *condition*, on the identity of the digit we would like to generate. That is, we would like to be able to say "generate an image of the digit 8", rather than just "generate an image of a digit". We will henceforth refer to the digit we would like to generate as $x \in \mathbb{R}^{1 \times 32 \times 32}$, and the conditioning variable (in this case, a label), as $y \in \{0, 1, \dots, 9\}$. If we imagine fixing our choice of $y$, and take our data distribution as $p_{\text{simple}}(x|y)$, then we have recovered the unconditional generative problem, and we can construct a generative model using e.g., a conditional flow matching objective via $$\begin{align*}\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta;y) &= \,\,\mathbb{E}_{\square} \lVert u_t^{\theta}(x|y) - u_t^{\text{ref}}(x|z)\rVert^2\\ \square &= z \sim p_{\text{data}}(z|y), x \sim p_t(x|z)\end{align*}$$
# We may now then allow $y$ to vary by simply taking our conditional flow matching expectation to be over $y$ as well (rather than fixing $y$), and explicitly conditioning our learned approximation on $u_t^{\theta}(x|y)$ on the choice of $y$. We therefore obtain the the *guided* conditional flow matching objective $$\begin{align*}\mathcal{L}_{\text{CFM}}(\theta) &= \,\,\mathbb{E}_{\square} \lVert u_t^{\theta}(x|y) - u_t^{\text{ref}}(x|z)\rVert^2\\ \square &= z,y \sim p_{\text{data}}(z,y), x \sim p_t(x|z)\end{align*}$$
# Note that $(z,y) \sim p_{\text{simple}}(z,y)$ is obtained in practice by sampling an image $z$, and a label $y$, from our labelled (MNIST) dataset. This is all well and good, and we emphasize that if our goal was simply to sample from $p_{\text{data}}(x|y)$, our job would be done (at least in theory). In practice, one might argue that we care more about the *perceptual quality* of our images. To this end, we will a derive a procedure known as *classifier-free guidance*.

# %% [markdown]
# **Classifier-Free Guidance**: For the sake of intuition, we will develop guidance through the lense of Gaussian probability paths, although the final result might reasonably be applied to any probability path. Recall from the lecture that for $(a_t, b_t) = \left(\frac{\dot{\alpha}_t}{\alpha_t}, -\frac{\dot{\beta}_t \beta_t \alpha_t - \dot{\alpha}_t \beta_t^2}{\alpha_t}\right)$, we have $$u_t(x|y) = a_tx + b_t\nabla \log p_t(x|y).$$
# This identity allows us to relate the *conditional marginal velocity* $u_t(x|y)$ to the *conditional score* $\nabla \log p_t(x|y)$. However, notice that $$\nabla \log p_t(x|y) = \nabla \log \left(\frac{p_t(x)p_t(y|x)}{p_t(y)}\right) = \nabla \log p_t(x) + \nabla \log p_t(y|x),$$
# so that we may rewrite $$u_t(x|y) = a_tx + b_t(\nabla \log p_t(x) + \nabla \log p_t(y|x)) = u_t(x) + b_t \nabla \log p_t(y|x).$$
# An approximation of the term $\nabla \log p_t(y|x)$ could be considered as a sort of noisy classifier (and in fact this is the origin of *classifier guidance*, which we do not consider here). In practice, people have noticed that the conditioning seems to work better when we scale the contribution of this classifier term, yielding
# $$\tilde{u}_t(x|y) = u_t(x) + w b_t \nabla \log p_t(y|x)$$
# where $w > 1$ is known as the *guidance scale*. We may then plug in $b_t\log p_t(y|x) = u^{\text{target}}_t(x|y) - u^{\text{target}}_t(x)$ to obtain $$\begin{align}\tilde{u}_t(x|y) &= u_t(x) + w b_t \nabla \log p_t(y|x)\\
# &= u_t(x) + w (u^{\text{target}}_t(x|y) - u^{\text{target}}_t(x))\\
# &= (1-w) u_t(x) + w u_t(x|y). \end{align}$$
# The idea is thus to train both $u_t(x)$ as well as the conditional model $u_t(x|y)$, and then combine them *at inference time* to obtain $\tilde{u}_t(x|y)$. Our recipe will thus be:
# 1. Train $u_t^{\theta} \approx u_t(x)$ as well as the conditional model $u_t^{\theta}(x|y) \approx u_t(x|y)$ using conditional flow matching.
# 2. At inference time, sample using $\tilde{u}_t^{\theta}(x|y)$.
#
# "But wait!", you say, "why must we train two models?". Indeed, we can instead treat $u_t(x)$ as $u_t(x|y)$, where $y=\varnothing$ denotes *the absence of conditioning*. We may thus augment our label set with a new, additional $\varnothing$ label, so that $y \in \{0,1,\dots, 9, \varnothing\}$. This technique is known as **classifier-free guidance** (CFG). We thus arrive at
# $$\boxed{\tilde{u}_t(x|y) = (1-w) u_t(x|\varnothing) + w u_t(x|y)}.$$

# %% [markdown]
# **Training and CFG**: We must now amend our conditional flow matching objective to account for the possibility of $y = \varnothing$. Of course, when we sample $(z,y)$ from MNIST, we will never obtain $y = \varnothing$, so we must introduce the possibliity of this artificially. To do so, we will define some hyperparameter $\eta$ to be the *probability* that we discard the original label $y$, and replace it with $\varnothing$. In practice, we might set $\varnothing = 10$, for example, as it is sufficient to distinguish it from the other digit identities. When we go and implement our model, we need ony be able to index into some embedding, such as via `torch.nn.Embedding`. We thus arrive at our CFG conditional flow matching training objective:
# $$\begin{align*}\mathcal{L}_{\text{CFM}}(\theta) &= \,\,\mathbb{E}_{\square} \lVert u_t^{\theta}(x|y) - u_t^{\text{ref}}(x|z)\rVert^2\\
# \square &= z,y \sim p_{\text{data}}(z,y), x \sim p_t(x|z),\,\text{replace $y$ with $\varnothing$ with probability $\eta$}\end{align*}$$
# In plain English, this objective reads:
# 1. Sample an image $z$ and a label $y$ from $p_{\text{data}}$ (here, MNIST).
# 2. With probability $\eta$, replace the label $y$ with the null label $\varnothing \triangleq 10$.
# 3. Sample $t$ from $\mathcal{U}[0,1]$.
# 4. Sample $x$ from the conditional probability path $p_t(x|z)$.
# 5. Regress $u_t^{\theta}(x|y)$ against $u_t^{\text{ref}}(x|z)$.

# %% [markdown]
# ### Question 2.2: Training for Classifier-Free Guidance
# In this section, you'll the training objective $\mathcal{L}_{\text{CFM}}(\theta)$ in which $u_t^{\theta}(x|y)$ is an instance of the class `ConditionalVectorField` described below.

# %% [markdown]
# # Part 3: An Architecture for Images
# At this point, we have discussed classifier free guidance, and the necessary considerations that must be made on the part of our model and in training our model. What remains is to actually discuss the choice of model. In particular, our usual choice of an MLP, while fine for the simple distributions of the previous lab, will no longer suffice. To this end, we will a new convolutional architecture - the **U-Net** - which is specifically tailored toward images. A diagram of the U-Net we'll be using is shown below. ![image.png](attachment:bd703834-9239-4ed3-b8c1-9639fc971575.png)

# %% [markdown]
# ### Question 3.1: Building a U-Net

# %% [markdown]
# Below, we implement the U-Net shown in the diagram above.

# %%
silu(x) = @. x / (1 + exp(-x))

# Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
function FourierEncoder(dim)
    @assert dim % 2 == 0
    half_dim = div(dim, 2)
    weights = randn(Float32, half_dim) |> gpu_device()
    @compact(; weights) do t
        t = reshape(t, 1, :)
        freqs = @. 2 * t * weights
        sin_embed = @. sqrt(2.0f0) * sinpi(freqs)
        cos_embed = @. sqrt(2.0f0) * cospi(freqs)
        output = vcat(sin_embed, cos_embed)
        @return output
    end
end

ResidualLayer(n, nt, ny) =
    @compact(;
        block1 = Chain(silu, BatchNorm(n), Conv((3, 3), n => n; pad = 1)),
        block2 = Chain(silu, BatchNorm(n), Conv((3, 3), n => n; pad = 1)),
        time_adapter = Chain(Dense(nt => nt, silu), Dense(nt => n)),
        y_adapter = Chain(Dense(ny => ny, silu), Dense(ny => n)),
    ) do (x, t_embed, y_embed)
        res = copy(x)

        # Initial conv block
        x = block1(x)

        # Add time embedding
        t_embed = time_adapter(t_embed)
        t_embed = reshape(t_embed, 1, 1, size(t_embed)...)
        x = x .+ t_embed

        # Add y embedding (conditional embedding)
        y_embed = y_adapter(y_embed)
        y_embed = reshape(y_embed, 1, 1, size(y_embed)...)
        x = x .+ y_embed

        # Second conv block
        x = block2(x)

        # Add back residual
        x = x .+ res

        @return x
    end

Encoder(nin, nout, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nin, nt, ny), nresidual),
        downsample = Conv((3, 3), nin => nout; stride = 2, pad = 1),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        x = downsample(x)
        @return x
    end

Midcoder(nchannel, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nchannel, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

Decoder(nin, nout, nresidual, nt, ny) =
    @compact(;
        upsample = Chain(Upsample(2, :bilinear), Conv((3, 3), nin => nout; pad = 1)),
        res_blocks = fill(ResidualLayer(nout, nt, ny), nresidual),
    ) do (x, t, y)
        x = upsample(x)
        for block in res_blocks
            x = block((x, t, y))
        end
        @return x
    end

MNISTUNet(; channels, nresidual, nt, y_embed_dim) =
    @compact(;
        init_conv = Chain(
            Conv((3, 3), 1 => channels[1]; pad = 1),
            BatchNorm(channels[1]),
            silu,
        ),
        time_embedder = FourierEncoder(nt),
        y_embedder = Embedding(11 => y_embed_dim),
        encoders = map(
            i -> Encoder(channels[i], channels[i+1], nresidual, nt, y_embed_dim),
            1:length(channels)-1,
        ),
        decoders = map(
            i -> Decoder(channels[i], channels[i-1], nresidual, nt, y_embed_dim),
            length(channels):-1:2,
        ),
        midcoder = Midcoder(channels[end], nresidual, nt, y_embed_dim),
        final_conv = Conv((3, 3), channels[1] => 1; pad = 1),
    ) do (x, t, y)
        # Embed t and y
        t_embed = time_embedder(t)
        y_embed = y_embedder(y)

        # Initial convolution
        x = init_conv(x) # (bs, c_0, 32, 32)

        residuals = ()

        # Encoders
        for encoder in encoders
            x = encoder((x, t_embed, y_embed)) # 2w x 2h x c_i x bs -> w x h x c_{i+1} x bs
            residuals = residuals..., copy(x)
        end

        # Midcoder
        x = midcoder((x, t_embed, y_embed))

        # Decoders
        for decoder in decoders
            residuals..., res = residuals
            x = x + res
            x = decoder((x, t_embed, y_embed)) # w x h x c_i x bs -> 2w x 2h x c_{i-1} x bs
        end

        # Final convolution
        x = final_conv(x) # 32 x 32 x 1 x bs

        @return x
    end

# %% [markdown]
# **Your job**: Pick *two* components of the architecture above (each one of `FourierEncoder`, `ResidualLayer`, `Encoder`, `Decoder`, or `Midcoder`), and explain, in your own words, (1) their role in the U-Net, (2) their inputs and outputs, and (3) a brief description of how the inputs turn into outputs.
#
# **Your answer**:

# %% [markdown]
# ### Question 3.2: Training a U-Net for Classifier-Free Guidance

# %% [markdown]
# Now let's train!

# %%
function loadmnist(batchsize, split)
    dataset = MNIST(; split)
    imgs = dataset.features
    labels_raw = dataset.targets

    # Process images into (H,W,C,BS) batches
    y = labels_raw .+ 1 |> collect
    z = reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)) |> f32 |> collect

    # Normalize
    μ, σ = 5.0f-1, 5.0f-1
    @. z = (z - μ) / σ

    # Use DataLoader to automatically minibatch and shuffle the data
    DataLoader((y, z); batchsize, shuffle = true, partial = false)
end

# Initialize model
unet = let
    device = gpu_device()
    model =
        MNISTUNet(; channels = [32, 64, 128], nresidual = 2, nt = 40, y_embed_dim = 40)
    ps, st = Lux.setup(rng, model) |> device
    train_state = Training.TrainState(model, ps, st, Adam(1.0f-3))
    nepoch = 20
    nsample = 250
    eta = 0.1
    data_train = loadmnist(nsample, :train)
    loss = MSELoss()
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(data_train)
        y, z = batch |> device
        # Set each label to 10 (i.e., null) with probability eta
        @. y = ifelse(rand() > eta, y, 11)
        x0 = randn!(similar(z))
        t = rand!(similar(z, nsample))
        t .*= 0.999f0 # Definition of target `u` is not good for `t` close to 1
        tre = reshape(t, 1, 1, 1, :)
        x = @. tre * z + (1 - tre) * x0
        u = @. (z - x) / (1 - tre) # Linear conditional vector field
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, ((x, t, y), u), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"
    end
    ps_freeze = train_state.parameters
    st_freeze = train_state.states
    (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))
end

# %% [markdown]
# How well does our model do? Let's find out! We'll use the class `CFGVectorFieldODE` to wrap the UNet in an instance of `ode` so that we can integrate it!

# %%
let
    # Play with these!
    nsample = 10
    nstep = 100
    guidance_scales = [0.0f0, 3.0f0, 5.0f0]
    fig = Figure(; size = (800, 300))
    for (i, w) in enumerate(guidance_scales)
        labels = 1:11
        nlabel = length(labels)
        y = stack(fill(labels, nsample); dims = 1) |> vec |> gpu_device()
        x = randn(Float32, 28, 28, 1, nsample)
        x = reshape(stack(fill(x, nlabel)), 28, 28, 1, :) |> gpu_device()
        t = 0.0f0
        for j = 1:nstep
            @show t
            h = 1.0f0 / nstep
            tcast = fill(t, nsample * nlabel) |> gpu_device()
            uy = unet(x, tcast, y)
            ynull = fill!(copy(y), 11)
            unull = unet(x, tcast, ynull)
            u = (1 - w) * unull + w * uy
            @. x += h * u
            t += h
        end
        ax = Axis(
            fig[1, i];
            title = "Guidance: w = $w",
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        )
        x = x |> cpu_device()
        x = reverse(x; dims = 2) # Reorient for plotting
        x = reshape(x, 28, 28, nsample, nlabel)
        x = permutedims(x, (1, 3, 2, 4))
        x = reshape(x, 28 * nsample, 28 * nlabel)
        image!(ax, x; interpolate = false, colorrange = (-1, 1))
    end
    file = "$outdir/results.pdf"
    @info "Saving to $file"
    save(file, fig; backend = CairoMakie)
    fig
end
