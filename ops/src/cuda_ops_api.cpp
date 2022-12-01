#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "dot_product/dot_product_kernel.h"
#include "scalar_attention/scalar_attention_kernel.h"
#include "utils/boxes_kernel.h"
#include "index_pooling/index_pooling_kernel.h"
#include "iou_nms/iou3d_cpu.h"
#include "iou_nms/iou3d_nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_forward", &dot_product_forward, "dot_product_forward");
    m.def("dot_product_backward", &dot_product_backward, "dot_product_backward");
    m.def("scalar_attention_forward", &scalar_attention_forward, "scalar_attention_forward");
    m.def("scalar_attention_backward", &scalar_attention_backward, "scalar_attention_backward");
    m.def("index_pooling_forward", &index_pooling_forward, "index_pooling_forward");
    m.def("index_pooling_backward", &index_pooling_backward, "index_pooling_backward");
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");
    m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
	m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
	m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
	m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
}