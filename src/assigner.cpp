#include <cmath>
#include <array>
#include <torch/extension.h>

namespace py = pybind11;

using box = std::array<float,4>;

float cx(const box& b)      { return 0.5 * (b[0] + b[2]); }
float cy(const box& b)      { return 0.5 * (b[1] + b[3]); }
float width(const box& b)   { return std::max(0.0f, b[2] - b[0]); }
float height(const box& b)  { return std::max(0.0f, b[3] - b[1]); }
float area(const box& b)    { return width(b) * height(b); }
bool  contains(const box& b, float cx, float cy) { return cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]; }

box inter(const box& b0, const box& b1) 
{ 
    return {std::max(b0[0], b1[0]), std::max(b0[1], b1[1]), std::min(b0[2], b1[2]), std::min(b0[3], b1[3])};
}

float iou(const box& b0, const box& b1) 
{ 
    return area(inter(b0, b1)) / (area(b0) + area(b1) - area(inter(b0, b1)) + 1e-8);
}

float dist(const box& b0, const box& b1)
{
    return std::sqrt(std::pow(cx(b0) - cx(b1), 2) + std::pow(cy(b0) - cy(b1), 2));
}

torch::Tensor assign_atss (
    torch::Tensor       anchors, // [N, 4]
    torch::Tensor       targets, // [B, D, 5]
    std::vector<long>   ps,      // [3]
    long                nc,      // Number of classes
    long                topk     // Number of candidates per pyramic level
)
{
    auto device    = anchors.device();
    anchors        = anchors.to(torch::TensorOptions(torch::Device("cpu")));
    targets        = targets.to(torch::TensorOptions(torch::Device("cpu")));
    auto anchors_a = anchors.accessor<float,2>();
    auto targets_a = targets.accessor<float,3>();
    const long B = targets_a.size(0);
    const long D = targets_a.size(1);
    const long N = anchors_a.size(0);
    std::vector<float>  ious(ps.size()*topk);
    std::vector<float>  dists(N);
    std::vector<long>   indices;
    std::vector<long>   candidates(ps.size()*topk);

    auto targets2   = torch::zeros({B, N, 5+nc});
    auto targets2_a = targets2.accessor<float,3>();

    for (long b = 0 ; b < B ; ++b)
    {
        for (long d = 0 ; d < D ; ++d)
        {
            if (targets_a[b][d][4] > -1)
            {
                const box bt = {targets_a[b][d][0], targets_a[b][d][1], targets_a[b][d][2], targets_a[b][d][3]};
                const int cls = targets_a[b][d][4];

                // 1. Calculate L2s
                for (long n = 0 ; n < N ; ++n)
                {
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};
                    dists[n] = dist(ba, bt);
                }

                // 2. Select topk from each level
                long start = 0;
                for (long l = 0 ; l < ps.size() ; ++l)
                {
                    indices.resize(ps[l]);
                    std::iota(begin(indices), end(indices), start);
                    std::partial_sort(begin(indices), begin(indices) + topk, end(indices), [&](long i, long j) {return dists[i] < dists[j];});
                    std::copy(begin(indices), begin(indices) + topk, begin(candidates) + l*topk);
                    start += ps[l];
                }

                // 3. Calculate IOUs and thresh
                float x  = 0.0f;
                float x2 = 0.0f;
                for (size_t i = 0 ; i < candidates.size() ; ++i)
                {
                    const long n = candidates[i];
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};
                    ious[i] = iou(ba, bt);
                    x  += ious[i];
                    x2 += ious[i] * ious[i];
                }

                const float mu      = x / candidates.size();
                const float stdev   = std::sqrt(x2 / candidates.size() - mu*mu);
                const float thresh  = mu + stdev;

                // 5. Add to targets2
                for (size_t i = 0 ; i < candidates.size() ; ++i)
                {
                    const long n = candidates[i];
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};
                    
                    if (ious[i] > thresh && contains(bt, cx(ba), cy(ba)) && ious[i] > targets2_a[b][n][4])
                    {                            
                        targets2_a[b][n][0]     = bt[0];
                        targets2_a[b][n][1]     = bt[1];
                        targets2_a[b][n][2]     = bt[2];
                        targets2_a[b][n][3]     = bt[3];
                        targets2_a[b][n][4]     = ious[i];
                        targets2_a[b][n][5+cls] = 1;
                    }
                }
            }
        }
    }

    // 6. Set confidence back to 1
    for (long b = 0 ; b < B ; ++b)
        for (long n = 0 ; n < N ; ++n)
            if (targets2_a[b][n][4] > 0)
                targets2_a[b][n][4] = 1.0;

    return targets2.to(device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("atss", &assign_atss, "ATSS assigner");
}