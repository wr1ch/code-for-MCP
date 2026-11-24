This is the official code for MCP-Net. 
MCP-Net: Multidirectional correlation-propagating network for progressive 3D medical image registration,
Pattern Recognition,
Volume 183, Part A,
2027,
114973,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2026.114973.
Full model code will be available later. You can also contact us at the email: jiangfengwang@wust.edu.cn.
You can adjust the scale of the model by configuring these parameters：
                 num_subspaces=(8, 4, 2, 1),
                 head_dim=6,
                 neighbourhood_size=3,
                 window_size=(5, 5, 5),
                 num_transformer_layers=4,
                 diffeomorphic=False,
                 int_steps=7,
                 qk_scale=1,
                 use_checkpoint=True
