#include <torch/extension.h>

__global__ void kernel_compute_weights_fwd(
    const float* __restrict__ sigmas,
    const float* __restrict__ steps,
    const int* __restrict__ info,
    const float threshold,
    float* weights,
    const int n_rays
) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > n_rays) return;

    const int ray_start = info[2*idx];
    const int n_samples = info[2*idx+1];
    if (n_samples == 0) return;

    float transmittance = 1.;
    float alpha;
    int k = 0;
    int _k;
    // early terminate ray if transmittance under threshold
    while (transmittance > threshold && k < n_samples) {
        _k = ray_start+k;
        alpha = expf(-sigmas[_k] * steps[_k]);
        weights[_k] = transmittance * (1. - alpha);
        transmittance *= alpha;
        k++;
    }
    return;
}

__global__ void kernel_compute_weights_bwd(
    const float* __restrict__ sigmas,
    const float* __restrict__ steps,
    const int* __restrict__ info,
    const float* __restrict__ weights,
    const float* __restrict__ grad_weights,
    float* grad_sigmas,
    const int n_rays
) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > n_rays) return;

    const int ray_start = info[2*idx];
    const int n_samples = info[2*idx+1];
    if (n_samples == 0) return;

    float acc = 0.;
    float alpha;
    int k = n_samples - 1;
    int _k;
    while (k >= 0) {
        _k = ray_start+k;
        alpha = expf(-sigmas[_k] * steps[_k]);
        grad_sigmas[_k] = steps[_k] * (acc + weights[_k] * alpha * grad_weights[_k] / (1. - alpha));
        acc -= weights[_k] * grad_weights[_k];
        k--;
    }
    return;
}

/* PYTORCH/PYBIND BOILERPLATE */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor compute_weights_fwd(
    torch::Tensor sigmas,
    torch::Tensor steps,
    torch::Tensor info,
    float threshold
) {
    CHECK_INPUT(sigmas);
    CHECK_INPUT(steps);
    CHECK_INPUT(info);
    TORCH_CHECK(sigmas.dim() == 1);
    TORCH_CHECK(steps.dim() == 1);
    TORCH_CHECK(info.dim() == 2 && info.size(1) == 2);

    int n_samples = sigmas.size(0);
    int n_rays = info.size(0);
    int n_threads = 1024;
    int n_blocks = (n_rays - 1) / n_threads + 1;

    torch::Tensor weights = torch::zeros_like(sigmas);

    kernel_compute_weights_fwd<<<n_threads, n_blocks>>>(
        sigmas.data_ptr<float>(),
        steps.data_ptr<float>(),
        info.data_ptr<int>(),
        threshold,
        weights.data_ptr<float>(),
        n_rays
    );
    return weights;
}

torch::Tensor compute_weights_bwd(
    torch::Tensor sigmas,
    torch::Tensor steps,
    torch::Tensor info,
    torch::Tensor weights,
    torch::Tensor grad_weights
) {
    CHECK_INPUT(sigmas);
    CHECK_INPUT(steps);
    CHECK_INPUT(info);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    TORCH_CHECK(sigmas.dim() == 1);
    TORCH_CHECK(steps.dim() == 1);
    TORCH_CHECK(info.dim() == 2 && info.size(1) == 2);
    TORCH_CHECK(weights.dim() == 1);
    TORCH_CHECK(grad_weights.dim() == 1);

    int n_samples = sigmas.size(0);
    int n_rays = info.size(0);
    int n_threads = 1024;
    int n_blocks = (n_rays - 1) / n_threads + 1;

    torch::Tensor grad_sigmas = torch::zeros_like(sigmas);

    kernel_compute_weights_bwd<<<n_threads, n_blocks>>>(
        sigmas.data_ptr<float>(),
        steps.data_ptr<float>(),
        info.data_ptr<int>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        grad_sigmas.data_ptr<float>(),
        n_rays
    );
    return sigmas;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_weights_fwd", &compute_weights_fwd);
    m.def("compute_weights_bwd", &compute_weights_bwd);
}