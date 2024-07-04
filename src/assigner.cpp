#include <cmath>
#include <array>
#include <limits>
#include <torch/extension.h>

namespace py = pybind11;

using std::pow;
using std::min;
using std::max;
using box = std::array<float,4>;

constexpr float cx(const box& b)      { return 0.5 * (b[0] + b[2]); }
constexpr float cy(const box& b)      { return 0.5 * (b[1] + b[3]); }
constexpr float width(const box& b)   { return std::max(0.0f, b[2] - b[0]); }
constexpr float height(const box& b)  { return std::max(0.0f, b[3] - b[1]); }
constexpr float area(const box& b)    { return width(b) * height(b); }
constexpr bool  contains(const box& b, const float cx, const float cy) { return cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]; }

constexpr box inter(const box& b0, const box& b1) 
{ 
    return {std::max(b0[0], b1[0]), std::max(b0[1], b1[1]), std::min(b0[2], b1[2]), std::min(b0[3], b1[3])};
}

constexpr box convex(const box& b0, const box& b1)
{
    return {std::min(b0[0], b1[0]), std::min(b0[1], b1[1]), std::max(b0[2], b1[2]), std::max(b0[3], b1[3])};
}

enum iou_type {IOU, GIOU, DIOU, CIOU};

constexpr float iou(const box& b0, const box& b1, const iou_type type = IOU, const double eps = 1e-8) 
{ 
    const float inter_ = area(inter(b0, b1));
    const float union_ = area(b0) + area(b1) - area(inter(b0, b1));
    const float iou_   = inter_ / (union_ + eps);

    if (type == IOU)
        return iou_;

    const box convex_   = convex(b0, b1);
    const float cw      = width(convex_);
    const float ch      = height(convex_);
    const float c_area  = cw*ch;

    if (type == GIOU)
        return iou_ - (c_area - union_) / (c_area + eps);

    const float c2      = std::pow(cw, 2) + std::pow(ch, 2);
    const float rho2    = std::pow(cx(b0) - cx(b1), 2) + std::pow(cy(b0) - cy(b1), 2);
    const float diou_   = iou_ - rho2 / (c2 + eps);

    if (type == DIOU)
        return diou_;

    const float v       = (4 / std::pow(M_PI, 2)) * std::pow(std::atan(width(b1)/height(b1)) - std::atan(width(b0)/height(b0)), 2);
    const float alpha   = v / (1 - iou_ + v + eps);

    return diou_ - alpha * v;
}

constexpr float dist(const box& b0, const box& b1)
{
    return std::sqrt(std::pow(cx(b0) - cx(b1), 2) + std::pow(cy(b0) - cy(b1), 2));
}

constexpr float centreness(const box& b, const float cx, const float cy)
{
    const float left    = cx - b[0];
    const float right   = b[2] - cx;
    const float top     = cy - b[1];
    const float bottom  = b[3] - cy;
    return std::sqrt((std::min(left,right) * std::min(top,bottom)) / (std::max(left,right) * std::max(top,bottom)));                
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> atss_fcos (
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
    const long B    = targets_a.size(0);
    const long D    = targets_a.size(1);
    const long N    = anchors_a.size(0);
    std::vector<float>  ious(ps.size()*topk);
    std::vector<float>  dists(N);
    std::vector<long>   indices;
    std::vector<long>   candidates(ps.size()*topk);
    std::vector<float>  best_ious(N);
    std::vector<long>   best_cls(N);

    auto boxes      = torch::zeros({B, N, 4});
    auto scores     = torch::zeros({B, N});
    auto classes    = torch::zeros({B, N, nc});
    auto boxes_a    = boxes.accessor<float,3>();
    auto scores_a   = scores.accessor<float,2>();
    auto classes_a  = classes.accessor<float,3>();

    for (long b = 0 ; b < B ; ++b)
    {
        // Running track of best IOU and best class for an anchor point
        std::fill(begin(best_ious), end(best_ious), 0.0f);
        std::fill(begin(best_cls), end(best_cls), 0);

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
                    ious[i] = iou(ba, bt, CIOU);
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
                    
                    if (ious[i] > thresh && contains(bt, cx(ba), cy(ba)) && ious[i] > best_ious[n])
                    {     
                        boxes_a[b][n][0]        = bt[0];                       
                        boxes_a[b][n][1]        = bt[1];
                        boxes_a[b][n][2]        = bt[2];
                        boxes_a[b][n][3]        = bt[3];
                        scores_a[b][n]          = centreness(bt, cx(ba), cy(ba));
                        classes_a[b][n][best_cls[n]] = 0; // Reset last class
                        classes_a[b][n][cls]    = 1;
                        best_ious[n]            = ious[i];
                        best_cls[n]             = cls;
                    }
                }
            }
        }
    }

    return std::make_tuple(boxes.to(device), scores.to(device), classes.to(device));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tal (
    torch::Tensor       pred_boxes,     // [B, N, 4]
    torch::Tensor       pred_scores,    // [B, N, nc]
    torch::Tensor       sxy,            // [N, 2]
    torch::Tensor       targets,        // [B, D, 5]
    long                topk,           // Number of candidates per pyramic level
    float               alpha,
    float               beta
)
{
    // Store device for later then put everything on CPU
    auto device    = pred_boxes.device();
    pred_boxes     = pred_boxes.to(torch::TensorOptions(torch::Device("cpu")));
    pred_scores    = pred_scores.to(torch::TensorOptions(torch::Device("cpu")));
    sxy            = sxy.to(torch::TensorOptions(torch::Device("cpu")));
    targets        = targets.to(torch::TensorOptions(torch::Device("cpu")));

    // Accessors
    auto pred_boxes_a   = pred_boxes.accessor<float,3>();
    auto pred_scores_a  = pred_scores.accessor<float,3>();
    auto sxy_a          = sxy.accessor<float,2>();
    auto targets_a      = targets.accessor<float,3>();

    // Shapes
    const auto [B, D, N, nc] = std::make_tuple(targets_a.size(0), targets_a.size(1), pred_scores_a.size(1), pred_scores_a.size(2));

    // Outputs 
    auto boxes      = torch::zeros({B, N, 4});
    auto scores     = torch::zeros({B, N});
    auto classes    = torch::zeros({B, N, nc});
    auto boxes_a    = boxes.accessor<float,3>();
    auto scores_a   = scores.accessor<float,2>();
    auto classes_a  = classes.accessor<float,3>();

    // Temporaries
    std::vector<float>  metric(N);
    std::vector<float>  ious(N);
    std::vector<long>   indices(N);
    std::vector<float>  best_ious(N);
    std::vector<long>   best_cls(N);

    for (long b = 0 ; b < B ; ++b)
    {
        // Running track of best IOU and best class for an anchor point
        std::fill(begin(best_ious), end(best_ious), 0.0f);
        std::fill(begin(best_cls), end(best_cls), 0);

        for (long d = 0 ; d < D ; ++d)
        {
            if (targets_a[b][d][4] > -1)
            {
                const box tbox = {targets_a[b][d][0], targets_a[b][d][1], targets_a[b][d][2], targets_a[b][d][3]};
                const int tcls = targets_a[b][d][4];

                // 1. Calculate metric
                for (long n = 0 ; n < N ; ++n)
                {
                    const box   pbox   = {pred_boxes_a[b][n][0], pred_boxes_a[b][n][1], pred_boxes_a[b][n][2], pred_boxes_a[b][n][3]};
                    const float pscore = pred_scores_a[b][n][tcls];
                    ious[n]   = iou(tbox, pbox, CIOU);
                    metric[n] = pow(pscore, alpha) *  pow(ious[n], beta);
                }

                // 2. Select topk
                std::iota(begin(indices), end(indices), 0);
                std::partial_sort(begin(indices), begin(indices) + topk, end(indices), [&](long i, long j) {return metric[i] < metric[j];});

                // 3. Normalize by max(IOU) / max(t)
                float max_iou       = std::numeric_limits<float>::lowest();
                float max_metric    = std::numeric_limits<float>::lowest();

                for (long i = 0 ; i < topk ; ++i)
                {
                    const long n = indices[i];
                    max_iou     = max(max_iou, ious[n]);
                    max_metric  = max(max_metric, metric[n]);
                }

                const float g = max_iou / max_metric;

                // 4. Add to targets
                for (size_t i = 0 ; i < topk ; ++i)
                {
                    const long n = indices[i];
                    const box ba = {anchors_a[n][0], anchors_a[n][1], anchors_a[n][2], anchors_a[n][3]};

                    if (contains(bt, cx(ba), cy(ba)))
                    {
                        boxes_a[b][n][0]        = bt[0];                       
                        boxes_a[b][n][1]        = bt[1];
                        boxes_a[b][n][2]        = bt[2];
                        boxes_a[b][n][3]        = bt[3];
                        scores_a[b][n]          = metric[n];
                        classes_a[b][n][best_cls[n]] = 0; // Reset last class
                        classes_a[b][n][cls]    = 1;
                        best_ious[n]            = ious[n];
                        best_cls[n]             = cls;
                    }
                }
            }
        }
    }

    return std::make_tuple(boxes.to(device), scores.to(device), classes.to(device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("atss", &atss_fcos, "ATSS assigner");
    m.def("tal",  &tal,       "TAL assigner");
}