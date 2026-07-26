class Mymodel(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_channel=1,
                 channels=4,
                 head_dim=6,
                 num_heads=[8, 8, 8, 8, 1],
                 scale=None,
                 use_checkpoint=True):
        super(Mymodel, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_1 = Conv_block(c*4, c*4, use_checkpoint)
        self.upsample_1 = PatchExpanding_block(embed_dim=8*c)
        self.corrmerging1=CorrMLP_block(4*c+4*c, 2*c, use_checkpoint=use_checkpoint,single3=False)
        self.reghead_1 = DeformHead_block(2*c, use_checkpoint)


        self.projblock2 = ProjectionLayer(4*c, dim=head_dim*num_heads[3])
        self.DIR2 = DIRPRIOR(head_dim*num_heads[3], num_heads[3], qk_scale=scale)
        self.dir2=CorrMLP_block(8*c+3 * num_heads[3], 8*c, use_checkpoint=use_checkpoint,single3=True)
        self.corr_trans2 = SwinTrans_stage_block(embed_dim=8*c,
                                             num_layers=4,
                                             num_heads=num_heads[3],
                                             window_size=[5,5,5],
                                             use_checkpoint=True)
        self.upsample_2 = PatchExpanding_block(embed_dim=16*c)
        self.corrmerging2=CorrMLP_block(8*c+8*c, 8*c, use_checkpoint=use_checkpoint,single3=False)
        self.reghead_2 = DeformHead_block(8*c, use_checkpoint)

        self.projblock3 = ProjectionLayer(8*c, dim=head_dim*num_heads[2])
        self.DIR3 = DIRPRIOR(head_dim*num_heads[2], num_heads[2], qk_scale=scale)
        self.dir3=CorrMLP_block(16*c+3 * num_heads[2], 16*c, use_checkpoint=use_checkpoint,single3=True)
        self.corr_trans3 = SwinTrans_stage_block(embed_dim=16*c,
                                             num_layers=4,
                                             num_heads=num_heads[2],
                                             window_size=[5,5,5],
                                             use_checkpoint=True)
        self.upsample_3 = PatchExpanding_block(embed_dim=32*c)
        self.corrmerging3=CorrMLP_block(16*c+16*c, 16*c, use_checkpoint=use_checkpoint,single3=False)
        self.reghead_3 = DeformHead_block(16*c, use_checkpoint)

        self.projblock4 = ProjectionLayer(16*c, dim=head_dim*num_heads[1])
        self.DIR4 = DIRPRIOR(head_dim*num_heads[1], num_heads[1], qk_scale=scale)
        self.dir4=CorrMLP_block(32*c+3 * num_heads[1], 32*c, use_checkpoint=use_checkpoint,single3=True)
        self.corr_trans4 = SwinTrans_stage_block(embed_dim=32*c,
                                             num_layers=4,
                                             num_heads=num_heads[1],
                                             window_size=[5,5,5],
                                             use_checkpoint=True)
        self.upsample_4 = PatchExpanding_block(embed_dim=64*c)
        self.corrmerging4=CorrMLP_block(32*c+32*c, 32*c, use_checkpoint=use_checkpoint,single3=False)
        self.reghead_4 = DeformHead_block(32*c, use_checkpoint)

        self.projblock5 = ProjectionLayer(32*c, dim=head_dim*num_heads[0])
        self.DIR5 = DIRPRIOR(head_dim*num_heads[0], num_heads[0], qk_scale=scale)
        self.dir5=CorrMLP_block(64*c+3 * num_heads[0], 64*c, use_checkpoint=use_checkpoint,single3=True)
        self.corr_trans5 = SwinTrans_stage_block(embed_dim=64*c,
                                             num_layers=4,
                                             num_heads=num_heads[0],
                                             window_size=[5,5,5],
                                             use_checkpoint=True)
        self.reghead_5 = DeformHead_block(64*c, use_checkpoint)
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        self.integrate = nn.ModuleList()
        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.integrate.append(VecInt([s // 2 ** i for s in inshape], nsteps=self.step))


    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        q5, k5 = self.projblock5(F5), self.projblock5(M5)
        w = self.DIR5(q5, k5)
        dirF5=self.dir5(torch.cat((F5,w,M5),dim=1))
        Corr5=self.corr_trans5(dirF5)
        flow5=self.reghead_5(Corr5)
        flow5=self.integrate[4](flow5)
        flow5=self.ResizeTransformer(flow5)

        M4 = self.transformer[3](M4, flow5)
        q4,k4 = self.projblock4(F4), self.projblock4(M4)
        w=self.DIR4(q4, k4)
        dirF4=self.dir4(torch.cat((F4,w,M4),dim=1))
        Corr4=self.corr_trans4(dirF4)       
        Corr4=self.corrmerging4(torch.cat((Corr4,self.upsample_4(Corr5)),dim=1))
        flow4 = self.reghead_4(Corr4)
        flow5=self.integrate[3](flow4)
#        flow4 = flow5+flow4
        flow4=self.ResizeTransformer(flow4)

        M3 = self.transformer[2](M3, flow4)
        q3, k3 = self.projblock3(F3), self.projblock3(M3)
        w = self.DIR3(q3, k3)
        dirF3=self.dir3(torch.cat((F3,w,M3),dim=1))
        Corr3=self.corr_trans3(dirF3)
        Corr3=self.corrmerging3(torch.cat((Corr3,self.upsample_3(Corr4)),dim=1))
        flow3 = self.reghead_3(Corr3)
        flow5=self.integrate[2](flow3)
#        flow3 = flow4+flow3
        flow3=self.ResizeTransformer(flow3)

        M2 = self.transformer[1](M2, flow3)
        q2,k2 = self.projblock2(F2), self.projblock2(M2)
        w=self.DIR2(q2, k2)
        dirF2=self.dir2(torch.cat((F2,w,M2),dim=1))
        Corr2=self.corr_trans2(dirF2)
        Corr2=self.corrmerging2(torch.cat((Corr2,self.upsample_2(Corr3)),dim=1))
        flow2 = self.reghead_2(Corr2)
        flow5=self.integrate[1](flow2)
#        flow2 = flow3+flow2
        flow2=self.ResizeTransformer(flow2)

        M1 = self.transformer[0](M1, flow2)
        Corr1=self.conv_1(torch.cat((F1,M1),dim=1))
        Corr1=self.corrmerging1(torch.cat((Corr1,self.upsample_1(Corr2)),dim=1))
        flow1 = self.reghead_1(Corr1)
        flow1=self.integrate[0](flow1)
#        flow1 = flow2+flow1

        y_moved = self.transformer[0](moving, flow1)

        return y_moved, flow1


