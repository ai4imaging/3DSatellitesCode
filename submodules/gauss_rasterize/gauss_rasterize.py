import torch

class GaussRasterizerSetting:
    def __init__(self, image_height, image_width, tanfovx, tanfovy, scale_modifier, sh_degree, prefiltered, viewmatrix, projmatrix, campos, bg):
        self.image_height = image_height  #int
        self.image_width = image_width  #int 
        self.tanfovx = tanfovx  #float
        self.tanfovy = tanfovy  #float
        self.scale_modifier = scale_modifier  #float
        self.sh_degree = sh_degree  #int
        self.prefiltered = prefiltered  #bool
        self.viewmatrix = viewmatrix  #torch.Tensor
        self.projmatrix = projmatrix  #torch.Tensor
        self.campos = campos  #torch.Tensor
        self.bg = bg  #torch.Tensor

class GaussRasterizer(torch.nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

    def forward(self, means3D, means2D, opacities, shs, scales, rotations):
        return _GaussRasterizerFunction.apply(means3D, means2D, opacities, shs, scales, rotations, self.setting)

class _GaussRasterizerFunction(torch.autograd.Function):
    import os
    gf = os.path.dirname(__file__)
    from torch.utils.cpp_extension import load  #~/.cache/torch_extensions/py310_cu117/GaussRasterize/
    _C = load(name='GaussRasterize', sources=[gf+'/gauss_rasterize.cpp', gf+'/gauss_rasterize.cu', gf+'/cuda_rasterizer/forward.cu',gf+'/cuda_rasterizer/backward.cu',gf+'/cuda_rasterizer/rasterizer_impl.cu'], extra_include_paths=[gf+'/opengl_mathematics/'], extra_cflags=[''], verbose=False)

    @staticmethod
    def forward(ctx, means3D, means2D, opacities, sh, scales, rotations, setting, colors_precomp=torch.Tensor([]), cov3Ds_precomp=torch.Tensor([])):
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _GaussRasterizerFunction._C.rasterize_gaussians(setting.bg, means3D, colors_precomp, opacities, scales, rotations, setting.scale_modifier, cov3Ds_precomp, setting.viewmatrix, setting.projmatrix, setting.tanfovx, setting.tanfovy, setting.image_height, setting.image_width, sh, setting.sh_degree, setting.campos, setting.prefiltered, False)
        ctx.raster_settings = setting
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        setting = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _GaussRasterizerFunction._C.rasterize_gaussians_backward(setting.bg, means3D, radii, colors_precomp, scales, rotations, setting.scale_modifier, cov3Ds_precomp, setting.viewmatrix, setting.projmatrix, setting.tanfovx, setting.tanfovy, grad_out_color, sh, setting.sh_degree, setting.campos, geomBuffer, num_rendered, binningBuffer, imgBuffer, False)
        return (grad_means3D, grad_means2D, grad_opacities, grad_sh, grad_scales, grad_rotations, None)

'''
from gauss_rasterize.gauss_rasterize import GaussRasterizerSetting, GaussRasterizer
class GsRender:
    def render(self, viewpoint_camera, pc, bg_color, device, scale_modifier=1.0, is_train=True):
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=is_train, device=device)
        rasterizer = GaussRasterizer(setting=GaussRasterizerSetting(image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width), tanfovx=math.tan(viewpoint_camera.FovX * 0.5), tanfovy=math.tan(viewpoint_camera.FovY * 0.5), scale_modifier=scale_modifier, sh_degree=pc.now_sh_degree, prefiltered=False, viewmatrix=viewpoint_camera.world_view_transform, projmatrix=viewpoint_camera.full_proj_transform, campos=viewpoint_camera.camera_center, bg=bg_color))
        rendered_image, radii = rasterizer(means3D=pc.get_xyz, means2D=screenspace_points, opacities=pc.get_opacity, shs=pc.get_features, scales=pc.get_scaling, rotations=pc.get_rotation)
        return rendered_image, screenspace_points, radii
'''
