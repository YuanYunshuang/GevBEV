from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='cuda_ops',
    author='xxx',
    version="0.1.0",
    ext_modules=[
        CUDAExtension('cuda_ops', [
            'src/cuda_ops_api.cpp',
            'src/dot_product/dot_product.cpp',
            'src/dot_product/dot_product_kernel.cu',
            'src/scalar_attention/scalar_attention.cpp',
            'src/scalar_attention/scalar_attention_kernel.cu',
            'src/index_pooling/index_pooling.cpp',
            'src/index_pooling/index_pooling_kernel.cu',
            'src/utils/boxes.cpp',
            'src/utils/boxes_kernel.cu',
            'src/iou_nms/iou3d_cpu.cpp',
            'src/iou_nms/iou3d_nms.cpp',
            'src/iou_nms/iou3d_nms_kernel.cu',
        ],
                      extra_compile_args={
                          'cxx': ['-g'],
                          'nvcc': ['-O2']
                      }),

        ],
    cmdclass={'build_ext': BuildExtension})
