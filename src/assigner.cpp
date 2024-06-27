#include <cmath>
#include <torch/extension.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct box 
{
    float x0{0.0f}; 
    float y0{0.0f}; 
    float x1{0.0f}; 
    float y1{0.0f}; 
    float cx() const    { return 0.5 * (x0 + x1); }
    float cy() const    { return 0.5 * (y0 + y1); }
    float w() const     { return std::max(0.0f, x1 - x0); }
    float h() const     { return std::max(0.0f, y1 - y0); }
    float area() const  { return w() * h(); }
    bool  contains(const float cx, const float cy) const { return cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1; }
};

box inter(const box& b0, const box& b1) 
{ 
    return {.x0 = std::max(b0.x0, b1.x0),
            .y0 = std::max(b0.y0, b1.y0),
            .x1 = std::min(b0.x1, b1.x1),
            .y1 = std::min(b0.y1, b1.y1)};
}

float iou(const box& b0, const box& b1) 
{ 
    return inter(b0, b1).area() / (b0.area() + b1.area() - inter(b0, b1).area() + 1e-8);
}

float dist(const box& b0, const box& b1)
{
    return std::sqrt(std::pow(b0.cx() - b1.cx(), 2) + std::pow(b0.cy() - b1.cy(), 2));
}

torch::Tensor assign_atss (
    torch::Tensor       anchors, // [N, 4]
    torch::Tensor       targets, // [B, D, 5]
    std::vector<long>   ps,      // [3]
    long                nc,      // Number of classes
    long                topk     // Number of candidates per pyramic level
)
{
    auto anchors_a = anchors.accessor<float,2>();
    auto targets_a = targets.accessor<float,3>();
    const long B = targets_a.size(0);
    const long D = targets_a.size(1);
    const long N = anchors_a.size(0);
    std::vector<float>  ious(N);
    std::vector<float>  dists(N);
    std::vector<long>   indices;
    std::vector<long>   candidates;

    auto targets2   = torch::zeros({B, N, 5+nc});
    auto targets2_a = targets2.accessor<float,3>();

    for (long b = 0 ; b < B ; ++b)
    {
        for (long d = 0 ; d < D ; ++d)
        {
            if (targets_a[b][d][4] > -1)
            {
                // 1. Calculate IOUs and L2s
                for (long n = 0 ; n < N ; ++n)
                {
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};
                    const box bt = {targets_a[b][d][0], targets_a[b][d][1], targets_a[b][d][2], targets_a[b][d][3]};
                    dists[n] = dist(ba, bt);
                    ious[n]  = iou(ba, bt);
                }

                // 2. Select topk from each level
                long start = 0;
                candidates.clear();
                candidates.reserve(ps.size() * topk);
                for (long l = 0 ; l < ps.size() ; ++l)
                {
                    indices.resize(ps[l]);
                    std::iota(begin(indices), end(indices), start);
                    std::partial_sort(begin(indices), begin(indices) + topk, end(indices), [&](long i, long j) {return dists[i] < dists[j];});
                    candidates.insert(end(candidates), begin(indices), begin(indices) + topk);
                    start += ps[l];
                }

                // 3. Calculate thresh
                float x  = 0.0f;
                float x2 = 0.0f;
                for (long n : candidates)
                {
                    x  += ious[n];
                    x2 += ious[n] * ious[n];
                }

                const float mu      = x / candidates.size();
                const float stdev   = std::sqrt(x2 / candidates.size() - mu*mu);
                const float thresh  = mu + stdev;

                // printf("target[%ld][%ld] iou mu %f std %f thresh %f - ncandidates %zu\n", b, d, mu, stdev, thresh, candidates.size());

                // 5. Add to targets2
                for (long n : candidates)
                {
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};
                    const box bt = {targets_a[b][d][0], targets_a[b][d][1], targets_a[b][d][2], targets_a[b][d][3]};
                    
                    if (ious[n] > thresh && bt.contains(ba.cx(), ba.cy()) && ious[n] > targets2_a[b][n][4])
                    {                            
                        const int cls           = targets_a[b][d][4];
                        targets2_a[b][n][0]     = targets_a[b][d][0];
                        targets2_a[b][n][1]     = targets_a[b][d][1];
                        targets2_a[b][n][2]     = targets_a[b][d][2];
                        targets2_a[b][n][3]     = targets_a[b][d][3];
                        targets2_a[b][n][4]     = ious[n];
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

    return targets2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("atss", &assign_atss, "ATSS assigner");
}